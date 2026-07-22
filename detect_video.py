# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
# See Through Project Version 2.0
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlpackage          # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import serial
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 定義 SiLU 函數
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 2. 強制注入到 PyTorch 核心的 activation 模組中
import torch.nn.modules.activation
if not hasattr(torch.nn.modules.activation, 'SiLU'):
    torch.nn.modules.activation.SiLU = SiLU

# 3. 同時注入到全域 nn 模組防禦
if not hasattr(nn, 'SiLU'):
    nn.SiLU = SiLU

from collections import deque
import argparse
import csv
import json
import os
import platform
import sys
from glob import glob, has_magic
from pathlib import Path

import torch
import numpy as np
import cv2
from norfair import Detection, Tracker

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils.plots import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
# =========================================================
# Nano → STM32 UART
# =========================================================

UART_PORT = "/dev/ttyUSB0"
UART_BAUD_RATE = 115200

# 狀態未改變時的定期重送時間
UART_RESEND_INTERVAL_SECONDS = 0.5
# =========================================================
# 動態 ROI 垂直位置設定
# =========================================================

# 前五幀尚未完成校正時使用的預設底部
DEFAULT_ROI_BOTTOM_Y = 0.88

# ROI 垂直總高度固定
ROI_TOTAL_HEIGHT = 0.24

# near／mid／far 在 ROI 總高度中的比例
NEAR_HEIGHT_RATIO = 0.57
MID_HEIGHT_RATIO = 0.25
FAR_HEIGHT_RATIO = 0.18

# 使用影片前五幀估計引擎蓋上緣
HOOD_CALIBRATION_FRAMES = 5

# 至少要有三筆有效量測，才使用自動校正結果
HOOD_MIN_VALID_SAMPLES = 3

# 偵測到引擎蓋上緣後，ROI 底部再往上保留一點距離
HOOD_SAFETY_MARGIN_RATIO = 0.012

# 動態 ROI 底部允許範圍
MIN_DYNAMIC_ROI_BOTTOM_Y = 0.63
MAX_DYNAMIC_ROI_BOTTOM_Y = 0.94


# =========================================================
# 引擎蓋上緣偵測範圍
# =========================================================

# 只檢查畫面中央部分，避免道路兩側影響
HOOD_SEARCH_LEFT_RATIO = 0.25
HOOD_SEARCH_RIGHT_RATIO = 0.75

# 只在畫面下半部搜尋道路與引擎蓋交界
HOOD_SEARCH_TOP_RATIO = 0.52
HOOD_SEARCH_BOTTOM_RATIO = 0.98

# 某一列中央區域有至少 20% 為道路，就視為道路列
HOOD_ROAD_ROW_THRESHOLD = 0.20

# 垂直方向平滑，必須是奇數
HOOD_ROW_SMOOTH_KERNEL = 21

# 道路區段至少要連續這麼多列，才不是小雜訊
HOOD_MIN_ROAD_RUN_ROWS = 10

# 中央紅色：主要行車走廊，稍微加寬
CENTER_BOTTOM_L = 0.29
CENTER_BOTTOM_R = 0.71
CENTER_TOP_L = 0.44
CENTER_TOP_R = 0.56

# 左側黃色：稍微加寬
LEFT_BOTTOM_L = 0.14
LEFT_BOTTOM_R = 0.29
LEFT_TOP_L = 0.38
LEFT_TOP_R = 0.44

# 右側黃色：稍微加寬
RIGHT_BOTTOM_L = 0.71
RIGHT_BOTTOM_R = 0.86
RIGHT_TOP_L = 0.56
RIGHT_TOP_R = 0.62
# =========================================================
# 前方車輛固定追蹤範圍
#
# 中央紅色 near／mid／far 永遠追蹤車輛，
# 不受 OCR 車速與 active_depths 影響。
# =========================================================
VEHICLE_TRACK_DEPTHS = [
    "near",
    "mid",
    "far",
]
# =========================================================
# LEAD 候選與切換設定
# =========================================================

# LEAD 候選只使用中央紅色 ROI 中間 70% 的寬度
# 行人警示 ROI 不受影響
LEAD_CORRIDOR_WIDTH_RATIO = 0.70

# 允許近車 BBOX 底部超過 ROI 底線約 2.5% 畫面高度
LEAD_BOTTOM_TOLERANCE_RATIO = 0.025

# LEAD 分數中，偏離畫面中央的扣分權重
LEAD_CENTER_SCORE_WEIGHT = 0.50

# 新候選至少要比目前 LEAD 高出多少分，才考慮切換
LEAD_SWITCH_SCORE_MARGIN = 0.04

# 新候選必須連續勝出幾幀才正式切換
LEAD_SWITCH_CONFIRM_FRAMES = 4
# =========================================================
# LEAD 實際距離與 TTC 分析設定
# =========================================================

# 距離歷史最多保留時間
DISTANCE_HISTORY_SECONDS = 1.20

# 最近幾幀距離取中位數，降低車框底部抖動
DISTANCE_SMOOTH_FRAMES = 5

# 使用約 0.35～0.60 秒前的距離計算接近速度
CLOSING_BASELINE_MIN_SECONDS = 0.35
CLOSING_BASELINE_MAX_SECONDS = 0.60

# 基準區間至少需要幾筆資料
CLOSING_BASELINE_MIN_SAMPLES = 3

# 接近速度低於此值時，不計算 TTC
MIN_CLOSING_SPEED_MPS = 0.20
# =========================================================
# 緊急煞車事件判斷設定
# =========================================================

# TTC 小於等於此值，視為進入危險範圍
EMERGENCY_TTC_THRESHOLD_SECONDS = 6.0

# 煞車燈 ONSET 發生後，在此時間內仍視為有效事件
BRAKE_ONSET_WINDOW_SECONDS = 3.0

# TTC 與煞車燈條件需連續成立幾幀
EMERGENCY_CONFIRM_FRAMES = 3

# 確認後維持事件顯示，避免只閃一下
EMERGENCY_HOLD_SECONDS = 1.0
# =========================================================
# 煞車狀態分類模型設定
# =========================================================

# 分類模型輸入大小
BRAKE_CLASSIFIER_IMAGE_SIZE = 224

# 判定為 brake_on 的最低信心
# 目前模型容易把部分 OFF 誤判為 ON，因此先設得較嚴格
BRAKE_ON_MIN_CONFIDENCE = 0.80

# 判定為 brake_off 的最低信心
BRAKE_OFF_MIN_CONFIDENCE = 0.55

# ON 必須連續出現幾幀才確認
BRAKE_ON_CONFIRM_FRAMES = 3

# OFF 必須連續出現幾幀才確認
BRAKE_OFF_CONFIRM_FRAMES = 2

# LEAD 車框向外保留的比例
BRAKE_CROP_PADDING_RATIO = 0.05
# =========================================================
# 車速 OCR 設定
# =========================================================

# 每幾幀執行一次 OCR
OCR_INTERVAL_FRAMES = 3

# OCR 結果超過此時間沒有更新，就視為過期
OCR_SPEED_MAX_AGE_SECONDS = 0.50
POSITION_COLORS = {
    "center": (80, 80, 255),
    "left": (90, 190, 255),
    "right": (90, 190, 255),
}

DEPTH_ALPHA = {
    "near": 0.38,
    "mid": 0.28,
    "far": 0.20,
}
# 未被目前車速啟用的 ROI，仍然淡淡顯示
INACTIVE_DEPTH_ALPHA = {
    "near": 0.10,
    "mid": 0.07,
    "far": 0.05,
}

# ROI 外框透明度
ACTIVE_LINE_ALPHA = 0.75
INACTIVE_LINE_ALPHA = 0.28
# =========================================================
# LEAD 煞車燈突變分析設定
# =========================================================

# 車框太小時，尾燈區像素不足，暫時不分析
BRAKE_MIN_BBOX_WIDTH = 55
BRAKE_MIN_BBOX_HEIGHT = 35


# 中央車身亮度參考區
BRAKE_BODY_X1_RATIO = 0.38
BRAKE_BODY_X2_RATIO = 0.62
BRAKE_BODY_Y1_RATIO = 0.28
BRAKE_BODY_Y2_RATIO = 0.60

# 歷史基準與目前狀態的比較時間
BRAKE_BASELINE_SECONDS = 0.50
BRAKE_RECENT_SECONDS = 0.10

# 暫定突變門檻，後面依實測結果調整
BRAKE_CHANGE_THRESHOLD = 0.06

# 整台車一起變亮時，扣除部分車身亮度變化
BRAKE_BODY_COMPENSATION = 0.60

# 左右變化不能相差太多
BRAKE_MIN_SYMMETRY_RATIO = 0.35

# =========================================================
# 動態紅色尾燈搜尋與追蹤
# =========================================================

# 在 LEAD 車框內搜尋紅色燈組的範圍
LAMP_SEARCH_X1_RATIO = 0.00
LAMP_SEARCH_X2_RATIO = 1.00
LAMP_SEARCH_Y1_RATIO = 0.15
LAMP_SEARCH_Y2_RATIO = 0.72

# OpenCV HSV：
# 紅色橫跨 Hue 的頭尾兩端
RED_HUE1_LOW = 0
RED_HUE1_HIGH = 10

RED_HUE2_LOW = 170
RED_HUE2_HIGH = 179

# 排除灰色、白色與低亮度雜訊
RED_SATURATION_MIN = 65
RED_VALUE_MIN = 45

# 紅色連通區面積限制
# 以整個 LEAD 車框面積為基準
LAMP_MIN_COMPONENT_AREA_RATIO = 0.0005
LAMP_MAX_COMPONENT_AREA_RATIO = 0.0400

# 左右燈配對條件
LAMP_MAX_VERTICAL_DIFF_RATIO = 0.15
LAMP_MIN_HORIZONTAL_GAP_RATIO = 0.25
LAMP_MIN_SIZE_SIMILARITY = 0.12

# 動態燈框 EMA 平滑
# 越小越穩定，但跟隨速度越慢
LAMP_TRACK_EMA_ALPHA = 0.35

# 單幀找不到紅燈時，沿用前一次位置幾幀
LAMP_TRACK_MAX_MISSES = 6

# =========================================================
# 行人／機車／腳踏車軌跡警示設定
# =========================================================

# 中央紅色 ROI 中，真正視為本車行駛路徑的寬度
VRU_EGO_PATH_WIDTH_RATIO = 1.00

# 保存最近幾秒的橫向移動歷史
VRU_MOTION_HISTORY_SECONDS = 0.50

# 至少觀察這麼久才判斷移動方向
VRU_MOTION_MIN_DURATION_SECONDS = 0.25

# 與本車路徑的正規化距離至少縮短多少
VRU_INWARD_MIN_PROGRESS = 0.05

# 每秒至少要向中央靠近多少，才算向內移動
VRU_INWARD_MIN_SPEED = 0.12

# 連續成立幾幀後才觸發黃色區警示
VRU_INWARD_CONFIRM_FRAMES = 3
# =========================================================
# 實際距離校正
# =========================================================

def load_distance_calibration(
    calibration_path,
):
    """
    讀取 pick_distance_y.py 產生的 JSON。

    回傳：
        calibration_y：
            由畫面上方到下方排列的 y 座標

        calibration_m：
            每個 y 座標對應的實際距離

        frame_width、frame_height：
            建立校正檔時的影片解析度
    """
    if calibration_path is None:
        raise ValueError(
            "必須提供 --distance-calibration"
        )

    calibration_path = Path(
        calibration_path
    )

    if not calibration_path.exists():
        raise FileNotFoundError(
            "找不到距離校正檔："
            f"{calibration_path}"
        )

    try:
        calibration_data = json.loads(
            calibration_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "距離校正 JSON 格式錯誤："
            f"{calibration_path}"
        ) from error

    if "distance_y" not in calibration_data:
        raise KeyError(
            "距離校正檔缺少 distance_y"
        )

    distance_y_data = (
        calibration_data["distance_y"]
    )

    calibration_pairs = []

    for (
        distance_text,
        y_value,
    ) in distance_y_data.items():
        calibration_pairs.append(
            (
                float(y_value),
                float(distance_text),
            )
        )

    # 影像 y 由上到下增加：
    # 通常為 20m、15m、10m、5m
    calibration_pairs.sort(
        key=lambda item: item[0]
    )

    if len(calibration_pairs) < 2:
        raise ValueError(
            "距離校正點至少需要兩筆"
        )

    calibration_y = np.asarray(
        [
            pair[0]
            for pair in calibration_pairs
        ],
        dtype=np.float32,
    )

    calibration_m = np.asarray(
        [
            pair[1]
            for pair in calibration_pairs
        ],
        dtype=np.float32,
    )

    if np.any(
        np.diff(calibration_y) <= 0
    ):
        raise ValueError(
            "距離校正的 y 座標不可重複，"
            "且必須由小到大排列"
        )

    frame_width = int(
        calibration_data.get(
            "frame_width",
            0,
        )
    )

    frame_height = int(
        calibration_data.get(
            "frame_height",
            0,
        )
    )

    print(
        "✅ 已載入距離校正："
        f"{calibration_path}"
    )

    for (
        calibration_y_value,
        calibration_distance,
    ) in calibration_pairs:
        print(
            "   "
            f"{calibration_distance:.0f} m "
            f"→ y={calibration_y_value:.0f}"
        )

    return {
        "y": calibration_y,
        "distance_m": calibration_m,
        "frame_width": frame_width,
        "frame_height": frame_height,
    }


def estimate_distance_from_y(
    contact_y,
    calibration_y,
    calibration_m,
):
    """
    將 LEAD 車框底部中心 y 座標，
    內插成 5～20 公尺內的連續距離。
    """
    contact_y = float(contact_y)

    # 超過最遠校正範圍
    if contact_y < calibration_y[0]:
        return None

    # 進入最近校正範圍內側
    if contact_y > calibration_y[-1]:
        return None

    return float(
        np.interp(
            contact_y,
            calibration_y,
            calibration_m,
        )
    )
def lerp(a, b, t):
    return a + (b - a) * t


def x_on_edge(
    bottom_x,
    top_x,
    y_ratio,
    roi_bottom_y,
    roi_top_y,
):
    """
    在固定大梯形左右邊線上，
    根據目前 y 位置計算 x。
    """
    denominator = roi_bottom_y - roi_top_y

    if denominator <= 0:
        raise ValueError(
            "roi_bottom_y 必須大於 roi_top_y"
        )

    t = (
        (roi_bottom_y - y_ratio)
        / denominator
    )

    return lerp(
        bottom_x,
        top_x,
        t,
    )


def make_layer_trapezoid(
    frame_w,
    frame_h,
    bottom_l,
    bottom_r,
    top_l,
    top_r,
    y_bottom,
    y_top,
    roi_bottom_y,
    roi_top_y,
):
    left_bottom_x = x_on_edge(
        bottom_l,
        top_l,
        y_bottom,
        roi_bottom_y,
        roi_top_y,
    )

    left_top_x = x_on_edge(
        bottom_l,
        top_l,
        y_top,
        roi_bottom_y,
        roi_top_y,
    )

    right_bottom_x = x_on_edge(
        bottom_r,
        top_r,
        y_bottom,
        roi_bottom_y,
        roi_top_y,
    )

    right_top_x = x_on_edge(
        bottom_r,
        top_r,
        y_top,
        roi_bottom_y,
        roi_top_y,
    )

    points = np.array(
        [
            [
                int(np.clip(
                    left_bottom_x * frame_w,
                    0,
                    frame_w - 1,
                )),
                int(np.clip(
                    y_bottom * frame_h,
                    0,
                    frame_h - 1,
                )),
            ],
            [
                int(np.clip(
                    left_top_x * frame_w,
                    0,
                    frame_w - 1,
                )),
                int(np.clip(
                    y_top * frame_h,
                    0,
                    frame_h - 1,
                )),
            ],
            [
                int(np.clip(
                    right_top_x * frame_w,
                    0,
                    frame_w - 1,
                )),
                int(np.clip(
                    y_top * frame_h,
                    0,
                    frame_h - 1,
                )),
            ],
            [
                int(np.clip(
                    right_bottom_x * frame_w,
                    0,
                    frame_w - 1,
                )),
                int(np.clip(
                    y_bottom * frame_h,
                    0,
                    frame_h - 1,
                )),
            ],
        ],
        dtype=np.int32,
    )

    return points.reshape((-1, 1, 2))
def point_in_vehicle_corridor(
    zone_points,
    point_x,
    point_y,
    width_ratio=1.0,
    bottom_tolerance_px=0,
):
    
    """
    判斷車框底部中心是否位於梯形車輛走廊。

    width_ratio：
        1.0  = 使用完整中央紅色 ROI
        0.70 = 只使用中央 70% 寬度

    bottom_tolerance_px：
        允許近車底部稍微超過 ROI 底線。
    """

    points = np.asarray(
        zone_points,
        dtype=np.float32,
    ).reshape(4, 2)

    # make_layer_trapezoid 的四點順序：
    # 左下、左上、右上、右下
    left_bottom = points[0]
    left_top = points[1]
    right_top = points[2]
    right_bottom = points[3]

    top_y = float(
        min(left_top[1], right_top[1])
    )

    bottom_y = float(
        max(left_bottom[1], right_bottom[1])
    )

    # 超出梯形頂部
    if point_y < top_y:
        return False

    # 超過底部容許範圍
    if point_y > bottom_y + bottom_tolerance_px:
        return False

    # 底部超出的點，壓回梯形底線計算左右範圍
    test_y = float(
        np.clip(
            point_y,
            top_y,
            bottom_y,
        )
    )

    vertical_span = max(
        bottom_y - top_y,
        1.0,
    )

    interpolation_ratio = (
        (test_y - top_y)
        / vertical_span
    )

    # 計算該高度的原始左右邊界
    left_x = (
        left_top[0]
        + interpolation_ratio
        * (left_bottom[0] - left_top[0])
    )

    right_x = (
        right_top[0]
        + interpolation_ratio
        * (right_bottom[0] - right_top[0])
    )

    corridor_center_x = (
        left_x + right_x
    ) / 2.0

    corridor_half_width = (
        (right_x - left_x)
        / 2.0
        * width_ratio
    )

    corridor_left_x = (
        corridor_center_x
        - corridor_half_width
    )

    corridor_right_x = (
        corridor_center_x
        + corridor_half_width
    )

    return (
        corridor_left_x
        <= point_x
        <= corridor_right_x
    )

def get_zone_horizontal_bounds(
    zone_points,
    point_y,
):
    """
    計算梯形在指定 y 高度的左右邊界。

    回傳：
        (left_x, right_x)

    如果 y 不在梯形垂直範圍內，回傳 None。
    """

    points = np.asarray(
        zone_points,
        dtype=np.float32,
    ).reshape(4, 2)

    # 點的順序：左下、左上、右上、右下
    left_bottom = points[0]
    left_top = points[1]
    right_top = points[2]
    right_bottom = points[3]

    top_y = float(
        min(left_top[1], right_top[1])
    )

    bottom_y = float(
        max(left_bottom[1], right_bottom[1])
    )

    if not top_y <= point_y <= bottom_y:
        return None

    vertical_span = max(
        bottom_y - top_y,
        1.0,
    )

    interpolation_ratio = (
        (float(point_y) - top_y)
        / vertical_span
    )

    left_x = (
        left_top[0]
        + interpolation_ratio
        * (left_bottom[0] - left_top[0])
    )

    right_x = (
        right_top[0]
        + interpolation_ratio
        * (right_bottom[0] - right_top[0])
    )

    return (
        float(left_x),
        float(right_x),
    )


def calculate_vru_warning_level(
    position,
    depth,
    current_speed,
):
    """
    保留原本行人／機車／腳踏車的警示等級規則。
    """

    if position == "center" and depth == "near":
        return 3

    if (
        position == "center"
        and depth == "mid"
        and current_speed >= 51
    ):
        return 3

    if position in ("left", "right"):
        if depth == "near" and current_speed >= 51:
            return 3

        if depth == "near":
            return 2

        if depth == "mid":
            return 2

        if depth == "far":
            return 1

        return 1

    if position == "center" and depth == "mid":
        return 2

    if position == "center" and depth == "far":
        return 2

    return 1

def build_dynamic_roi_zones(
    frame_w,
    frame_h,
    roi_bottom_y,
):
    """
    根據引擎蓋上緣決定 ROI 底部。

    ROI 總高度保持固定，
    near／mid／far 比例保持 57%／25%／18%。
    """
    roi_bottom_y = float(
        np.clip(
            roi_bottom_y,
            MIN_DYNAMIC_ROI_BOTTOM_Y,
            MAX_DYNAMIC_ROI_BOTTOM_Y,
        )
    )

    roi_top_y = (
        roi_bottom_y
        - ROI_TOTAL_HEIGHT
    )

    near_height = (
        ROI_TOTAL_HEIGHT
        * NEAR_HEIGHT_RATIO
    )

    mid_height = (
        ROI_TOTAL_HEIGHT
        * MID_HEIGHT_RATIO
    )

    near_bottom_y = roi_bottom_y
    near_top_y = (
        near_bottom_y
        - near_height
    )

    mid_bottom_y = near_top_y
    mid_top_y = (
        mid_bottom_y
        - mid_height
    )

    far_bottom_y = mid_top_y
    far_top_y = roi_top_y

    layer_y = {
        "near": (
            near_bottom_y,
            near_top_y,
        ),
        "mid": (
            mid_bottom_y,
            mid_top_y,
        ),
        "far": (
            far_bottom_y,
            far_top_y,
        ),
    }

    roi_zones = {}

    for depth, (
        y_bottom,
        y_top,
    ) in layer_y.items():

        roi_zones[
            ("center", depth)
        ] = make_layer_trapezoid(
            frame_w,
            frame_h,
            CENTER_BOTTOM_L,
            CENTER_BOTTOM_R,
            CENTER_TOP_L,
            CENTER_TOP_R,
            y_bottom,
            y_top,
            roi_bottom_y,
            roi_top_y,
        )

        roi_zones[
            ("left", depth)
        ] = make_layer_trapezoid(
            frame_w,
            frame_h,
            LEFT_BOTTOM_L,
            LEFT_BOTTOM_R,
            LEFT_TOP_L,
            LEFT_TOP_R,
            y_bottom,
            y_top,
            roi_bottom_y,
            roi_top_y,
        )

        roi_zones[
            ("right", depth)
        ] = make_layer_trapezoid(
            frame_w,
            frame_h,
            RIGHT_BOTTOM_L,
            RIGHT_BOTTOM_R,
            RIGHT_TOP_L,
            RIGHT_TOP_R,
            y_bottom,
            y_top,
            roi_bottom_y,
            roi_top_y,
        )

    return roi_zones

def find_true_runs(boolean_array):
    """
    找出布林陣列中所有連續 True 區間。

    回傳：
        [(start, end), ...]
    """

    # 必須使用有號整數。
    # 若使用 uint8，1 -> 0 的差值會變成 255，
    # 導致 changes == -1 永遠找不到結束位置。
    values = np.asarray(
        boolean_array,
        dtype=np.int8,
    )

    padded = np.pad(
        values,
        (1, 1),
        mode="constant",
        constant_values=0,
    )

    changes = np.diff(
        padded
    )

    starts = np.flatnonzero(
        changes == 1
    )

    ends = (
        np.flatnonzero(
            changes == -1
        )
        - 1
    )

    return list(
        zip(
            starts.tolist(),
            ends.tolist(),
        )
    )


def detect_hood_top_ratio(road_mask):
    """
    從 road mask 的主要道路白色帶，
    找出其最下方邊緣，作為引擎蓋上緣。

    road_mask：
        1 = 道路
        0 = 非道路
    """
    frame_h, frame_w = road_mask.shape

    # 搜尋中央較寬區域，避免中央前車遮住所有道路
    x1 = int(0.15 * frame_w)
    x2 = int(0.85 * frame_w)

    # 搜尋畫面下半部
    y1 = int(0.42 * frame_h)
    y2 = int(0.98 * frame_h)

    x1 = int(np.clip(x1, 0, frame_w - 1))
    x2 = int(np.clip(x2, x1 + 1, frame_w))

    y1 = int(np.clip(y1, 0, frame_h - 1))
    y2 = int(np.clip(y2, y1 + 1, frame_h))

    search_area = (
        road_mask[y1:y2, x1:x2] > 0
    ).astype(np.uint8)

    if search_area.size == 0:
        print("[HOOD DEBUG] 搜尋區域為空")
        return None

    # 填補 mask 的小洞
    close_kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    search_area = cv2.morphologyEx(
        search_area,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    # 計算每一橫列中，白色道路所占比例
    row_road_ratio = np.mean(
        search_area,
        axis=1,
    ).astype(np.float32)

    # 垂直方向平滑
    kernel_size = 15

    smooth_kernel = np.ones(
        kernel_size,
        dtype=np.float32,
    ) / kernel_size

    smooth_ratio = np.convolve(
        row_road_ratio,
        smooth_kernel,
        mode="same",
    )

    max_row_ratio = float(
        np.max(smooth_ratio)
    )

    print(
        "[HOOD DEBUG] "
        f"search_white={np.mean(search_area):.4f}, "
        f"max_row_ratio={max_row_ratio:.4f}"
    )

    # 搜尋範圍內幾乎沒有道路
    if max_row_ratio < 0.01:
        print("[HOOD DEBUG] 搜尋區域內道路像素太少")
        return None

    # 自適應門檻：
    # 至少 2%，或最高道路比例的 25%
    adaptive_threshold = max(
        0.02,
        max_row_ratio * 0.25,
    )

    valid_road_rows = (
        smooth_ratio
        >= adaptive_threshold
    )

    road_runs = find_true_runs(
        valid_road_rows
    )

    # 至少連續 5 列
    road_runs = [
        (start, end)
        for start, end in road_runs
        if (
            end - start + 1
            >= 5
        )
    ]

    print(
        "[HOOD DEBUG] "
        f"threshold={adaptive_threshold:.4f}, "
        f"runs={road_runs}"
    )

    if not road_runs:
        print("[HOOD DEBUG] 找不到連續道路區段")
        return None

    # 選擇最長的主要道路帶；
    # 若長度相同，選平均道路比例較高者
    selected_start, selected_end = max(
        road_runs,
        key=lambda run: (
            run[1] - run[0] + 1,
            float(
                np.mean(
                    smooth_ratio[
                        run[0]:run[1] + 1
                    ]
                )
            ),
        ),
    )

    # 道路帶最下緣 = 推估的引擎蓋上緣
    hood_top_row = (
        y1
        + selected_end
        + 1
    )

    hood_top_ratio = (
        hood_top_row
        / frame_h
    )

    # 避免極端錯誤值
    if not (
        0.55
        <= hood_top_ratio
        <= 0.96
    ):
        print(
            "[HOOD DEBUG] "
            f"hood={hood_top_ratio:.3f} 超出合理範圍"
        )
        return None

    print(
        "[HOOD DEBUG] "
        f"selected_run=({selected_start}, "
        f"{selected_end}), "
        f"hood={hood_top_ratio:.3f}"
    )

    return float(
        hood_top_ratio
    )

def draw_ground_clipped_nine_grid(
    img,
    roi_zones,
    active_depths=None,
):
    if active_depths is None:
        active_depths = [
            "near",
            "mid",
            "far",
        ]

    result = img.astype(
        np.float32
    ).copy()

    frame_h, frame_w = img.shape[:2]

    for (
        position,
        depth,
    ), points in roi_zones.items():

        # 啟用層正常顯示
        # 未啟用層淡色顯示
        is_active = (
            depth in active_depths
        )

        if is_active:
            zone_alpha = (
                DEPTH_ALPHA[depth]
            )
            line_alpha = (
                ACTIVE_LINE_ALPHA
            )

        else:
            zone_alpha = (
                INACTIVE_DEPTH_ALPHA[
                    depth
                ]
            )
            line_alpha = (
                INACTIVE_LINE_ALPHA
            )

        zone_mask = np.zeros(
            (frame_h, frame_w),
            dtype=np.uint8,
        )

        cv2.fillPoly(
            zone_mask,
            [points],
            1,
        )

        visible_alpha = (
            zone_mask.astype(
                np.float32
            )
            * zone_alpha
        )

        color = np.array(
            POSITION_COLORS[position],
            dtype=np.float32,
        )

        alpha_3d = (
            visible_alpha[..., None]
        )

        result = (
            result
            * (1.0 - alpha_3d)
            + color
            * alpha_3d
        )

        line_mask = np.zeros(
            (frame_h, frame_w),
            dtype=np.uint8,
        )

        cv2.polylines(
            line_mask,
            [points],
            isClosed=True,
            color=1,
            thickness=2,
        )

        visible_line_mask = (
            line_mask > 0
        )

        if np.any(
            visible_line_mask
        ):
            result[
                visible_line_mask
            ] = (
                result[
                    visible_line_mask
                ]
                * (1.0 - line_alpha)
                + color
                * line_alpha
            )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

def draw_text_right(
    img,
    text,
    y,
    right_margin=20,
    font_scale=0.7,
    color=(0, 0, 0),
    thickness=2,
):
    """
    將文字靠畫面右側對齊。
    OpenCV 使用 BGR，黑色為 (0, 0, 0)。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    x = max(
        0,
        img.shape[1] - right_margin - text_width,
    )

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
    )
def make_relative_box(
    bbox,
    frame_shape,
    x1_ratio,
    y1_ratio,
    x2_ratio,
    y2_ratio,
):
    """
    根據 LEAD 車框建立相對位置的小區域。
    """
    frame_h, frame_w = frame_shape[:2]

    box_x1, box_y1, box_x2, box_y2 = bbox

    box_x1 = int(np.clip(box_x1, 0, frame_w - 1))
    box_y1 = int(np.clip(box_y1, 0, frame_h - 1))
    box_x2 = int(np.clip(box_x2, box_x1 + 1, frame_w))
    box_y2 = int(np.clip(box_y2, box_y1 + 1, frame_h))

    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1

    region_x1 = int(
        box_x1 + box_width * x1_ratio
    )
    region_y1 = int(
        box_y1 + box_height * y1_ratio
    )
    region_x2 = int(
        box_x1 + box_width * x2_ratio
    )
    region_y2 = int(
        box_y1 + box_height * y2_ratio
    )

    region_x1 = int(
        np.clip(region_x1, 0, frame_w - 1)
    )
    region_y1 = int(
        np.clip(region_y1, 0, frame_h - 1)
    )
    region_x2 = int(
        np.clip(region_x2, region_x1 + 1, frame_w)
    )
    region_y2 = int(
        np.clip(region_y2, region_y1 + 1, frame_h)
    )

    return (
        region_x1,
        region_y1,
        region_x2,
        region_y2,
    )


def crop_box_region(
    frame,
    box,
):
    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def build_red_hsv_mask(
    crop,
):
    """
    建立真正的紅色色相遮罩。

    黃色／橘色方向燈通常不在這兩段 Hue 內，
    因此不會直接被當成紅色煞車燈。
    """
    if crop is None or crop.size == 0:
        return None, None

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    lower_red_1 = np.array(
        [
            RED_HUE1_LOW,
            RED_SATURATION_MIN,
            RED_VALUE_MIN,
        ],
        dtype=np.uint8,
    )

    upper_red_1 = np.array(
        [
            RED_HUE1_HIGH,
            255,
            255,
        ],
        dtype=np.uint8,
    )

    lower_red_2 = np.array(
        [
            RED_HUE2_LOW,
            RED_SATURATION_MIN,
            RED_VALUE_MIN,
        ],
        dtype=np.uint8,
    )

    upper_red_2 = np.array(
        [
            RED_HUE2_HIGH,
            255,
            255,
        ],
        dtype=np.uint8,
    )

    red_mask_1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1,
    )

    red_mask_2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2,
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2,
    )

    # 填補紅燈內部的小洞，但不做過強的開運算，
    # 避免遠方小型尾燈被消除。
    if (
        red_mask.shape[0] >= 5
        and red_mask.shape[1] >= 5
    ):
        red_mask = cv2.medianBlur(
            red_mask,
            3,
        )

        close_kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

    return hsv, red_mask


def calculate_red_lamp_score(
    crop,
):
    """
    根據真正紅色色相的面積與亮度計算分數。
    """
    hsv, red_mask = build_red_hsv_mask(
        crop
    )

    if hsv is None or red_mask is None:
        return None

    red_pixels = red_mask > 0

    if not np.any(red_pixels):
        return 0.0

    red_ratio = float(
        np.mean(red_pixels)
    )

    value_channel = (
        hsv[:, :, 2].astype(np.float32)
        / 255.0
    )

    red_values = value_channel[
        red_pixels
    ]

    mean_red_value = float(
        np.mean(red_values)
    )

    high_red_value = float(
        np.percentile(
            red_values,
            90,
        )
    )

    red_brightness = (
        0.50 * mean_red_value
        + 0.50 * high_red_value
    )

    return float(
        red_ratio
        * (
            0.25
            + 0.75 * red_brightness
        )
    )

def calculate_region_brightness(
    crop,
):
    """
    計算車身參考區的整體亮度。
    """
    if crop is None or crop.size == 0:
        return None

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    value_channel = hsv[:, :, 2]

    return float(
        np.mean(value_channel)
        / 255.0
    )
def normalized_box_center(
    norm_box,
):
    """
    取得相對車框座標下的小框中心。
    """
    return (
        (norm_box[0] + norm_box[2]) * 0.5,
        (norm_box[1] + norm_box[3]) * 0.5,
    )


def smooth_normalized_box(
    previous_box,
    current_box,
    alpha,
):
    """
    使用 EMA 平滑相對車框座標。
    """
    if previous_box is None:
        return current_box

    smoothed = tuple(
        (
            (1.0 - alpha) * previous_value
            + alpha * current_value
        )
        for previous_value, current_value
        in zip(previous_box, current_box)
    )

    return tuple(
        float(np.clip(value, 0.0, 1.0))
        for value in smoothed
    )


def normalized_box_to_frame(
    vehicle_bbox,
    frame_shape,
    normalized_box,
):
    """
    將相對於車框的座標轉成畫面座標。
    """
    return make_relative_box(
        bbox=vehicle_bbox,
        frame_shape=frame_shape,
        x1_ratio=normalized_box[0],
        y1_ratio=normalized_box[1],
        x2_ratio=normalized_box[2],
        y2_ratio=normalized_box[3],
    )


def detect_symmetric_red_lamp_pair(
    frame,
    vehicle_bbox,
    previous_pair=None,
):
    """
    在 LEAD 車框內找紅色候選區，並配對左右尾燈。

    previous_pair 格式：
    {
        "left": 左燈相對座標,
        "right": 右燈相對座標,
    }
    """
    frame_h, frame_w = frame.shape[:2]

    vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = (
        vehicle_bbox
    )

    vehicle_x1 = int(
        np.clip(
            vehicle_x1,
            0,
            frame_w - 1,
        )
    )

    vehicle_y1 = int(
        np.clip(
            vehicle_y1,
            0,
            frame_h - 1,
        )
    )

    vehicle_x2 = int(
        np.clip(
            vehicle_x2,
            vehicle_x1 + 1,
            frame_w,
        )
    )

    vehicle_y2 = int(
        np.clip(
            vehicle_y2,
            vehicle_y1 + 1,
            frame_h,
        )
    )

    vehicle_width = max(
        1,
        vehicle_x2 - vehicle_x1,
    )

    vehicle_height = max(
        1,
        vehicle_y2 - vehicle_y1,
    )

    vehicle_area = float(
        vehicle_width
        * vehicle_height
    )

    search_box = make_relative_box(
        bbox=(
            vehicle_x1,
            vehicle_y1,
            vehicle_x2,
            vehicle_y2,
        ),
        frame_shape=frame.shape,
        x1_ratio=LAMP_SEARCH_X1_RATIO,
        y1_ratio=LAMP_SEARCH_Y1_RATIO,
        x2_ratio=LAMP_SEARCH_X2_RATIO,
        y2_ratio=LAMP_SEARCH_Y2_RATIO,
    )

    search_crop = crop_box_region(
        frame,
        search_box,
    )

    hsv, red_mask = build_red_hsv_mask(
        search_crop
    )

    result = {
        "search_box": search_box,
        "candidate_boxes": [],
        "pair": None,
        "red_mask": red_mask,
    }

    if hsv is None or red_mask is None:
        return result

    (
        label_count,
        labels,
        stats,
        centroids,
    ) = cv2.connectedComponentsWithStats(
        red_mask,
        connectivity=8,
    )

    search_x1, search_y1, _, _ = (
        search_box
    )

    minimum_area = max(
        2,
        int(
            round(
                vehicle_area
                * LAMP_MIN_COMPONENT_AREA_RATIO
            )
        ),
    )

    maximum_area = max(
        minimum_area + 1,
        int(
            round(
                vehicle_area
                * LAMP_MAX_COMPONENT_AREA_RATIO
            )
        ),
    )

    candidates = []

    for label_index in range(
        1,
        label_count,
    ):
        component_x = int(
            stats[
                label_index,
                cv2.CC_STAT_LEFT,
            ]
        )

        component_y = int(
            stats[
                label_index,
                cv2.CC_STAT_TOP,
            ]
        )

        component_width = int(
            stats[
                label_index,
                cv2.CC_STAT_WIDTH,
            ]
        )

        component_height = int(
            stats[
                label_index,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        component_area = int(
            stats[
                label_index,
                cv2.CC_STAT_AREA,
            ]
        )

        if not (
            minimum_area
            <= component_area
            <= maximum_area
        ):
            continue

        if (
            component_width <= 0
            or component_height <= 0
        ):
            continue

        aspect_ratio = (
            component_width
            / max(
                component_height,
                1,
            )
        )

        if not (
            0.15
            <= aspect_ratio
            <= 8.0
        ):
            continue

        bounding_area = (
            component_width
            * component_height
        )

        fill_ratio = (
            component_area
            / max(
                bounding_area,
                1,
            )
        )

        if fill_ratio < 0.10:
            continue

        absolute_x1 = (
            search_x1
            + component_x
        )

        absolute_y1 = (
            search_y1
            + component_y
        )

        absolute_x2 = (
            absolute_x1
            + component_width
        )

        absolute_y2 = (
            absolute_y1
            + component_height
        )

        center_x = float(
            centroids[
                label_index,
                0,
            ]
            + search_x1
        )

        center_y = float(
            centroids[
                label_index,
                1,
            ]
            + search_y1
        )

        center_x_norm = (
            center_x - vehicle_x1
        ) / vehicle_width

        center_y_norm = (
            center_y - vehicle_y1
        ) / vehicle_height

        normalized_box = (
            (absolute_x1 - vehicle_x1)
            / vehicle_width,

            (absolute_y1 - vehicle_y1)
            / vehicle_height,

            (absolute_x2 - vehicle_x1)
            / vehicle_width,

            (absolute_y2 - vehicle_y1)
            / vehicle_height,
        )

        component_pixels = (
            labels == label_index
        )

        component_brightness = float(
            np.mean(
                hsv[:, :, 2][
                    component_pixels
                ]
            )
            / 255.0
        )

        candidate = {
            "box": (
                absolute_x1,
                absolute_y1,
                absolute_x2,
                absolute_y2,
            ),
            "norm_box": normalized_box,
            "cx_norm": center_x_norm,
            "cy_norm": center_y_norm,
            "area": component_area,
            "brightness": component_brightness,
        }

        candidates.append(
            candidate
        )

    result["candidate_boxes"] = [
        candidate["box"]
        for candidate in candidates
    ]

    left_candidates = [
        candidate
        for candidate in candidates
        if (
            0.05
            <= candidate["cx_norm"]
            < 0.48
        )
    ]

    right_candidates = [
        candidate
        for candidate in candidates
        if (
            0.52
            < candidate["cx_norm"]
            <= 0.95
        )
    ]

    previous_left_center = None
    previous_right_center = None

    if previous_pair is not None:
        previous_left_center = (
            normalized_box_center(
                previous_pair["left"]
            )
        )

        previous_right_center = (
            normalized_box_center(
                previous_pair["right"]
            )
        )

    best_pair = None
    best_cost = float("inf")

    for left_candidate in left_candidates:
        for right_candidate in right_candidates:
            horizontal_gap = (
                right_candidate["cx_norm"]
                - left_candidate["cx_norm"]
            )

            if (
                horizontal_gap
                < LAMP_MIN_HORIZONTAL_GAP_RATIO
            ):
                continue

            vertical_error = abs(
                left_candidate["cy_norm"]
                - right_candidate["cy_norm"]
            )

            if (
                vertical_error
                > LAMP_MAX_VERTICAL_DIFF_RATIO
            ):
                continue

            size_similarity = (
                min(
                    left_candidate["area"],
                    right_candidate["area"],
                )
                / max(
                    left_candidate["area"],
                    right_candidate["area"],
                    1,
                )
            )

            if (
                size_similarity
                < LAMP_MIN_SIZE_SIMILARITY
            ):
                continue

            left_center_distance = (
                0.5
                - left_candidate["cx_norm"]
            )

            right_center_distance = (
                right_candidate["cx_norm"]
                - 0.5
            )

            symmetry_error = abs(
                left_center_distance
                - right_center_distance
            )

            tracking_error = 0.0

            if (
                previous_left_center
                is not None
                and previous_right_center
                is not None
            ):
                tracking_error = (
                    abs(
                        left_candidate["cx_norm"]
                        - previous_left_center[0]
                    )
                    + abs(
                        left_candidate["cy_norm"]
                        - previous_left_center[1]
                    )
                    + abs(
                        right_candidate["cx_norm"]
                        - previous_right_center[0]
                    )
                    + abs(
                        right_candidate["cy_norm"]
                        - previous_right_center[1]
                    )
                )

            brightness_bonus = (
                left_candidate["brightness"]
                + right_candidate["brightness"]
            )

            pair_cost = (
                3.0 * vertical_error
                + 2.0 * symmetry_error
                + 0.8
                * (
                    1.0
                    - size_similarity
                )
                + 1.5 * tracking_error
                - 0.20 * brightness_bonus
            )

            if pair_cost < best_cost:
                best_cost = pair_cost

                best_pair = {
                    "left": (
                        left_candidate[
                            "norm_box"
                        ]
                    ),
                    "right": (
                        right_candidate[
                            "norm_box"
                        ]
                    ),
                    "cost": pair_cost,
                }

    result["pair"] = best_pair

    return result
def preprocess_brake_crop(
    vehicle_crop,
    image_size=BRAKE_CLASSIFIER_IMAGE_SIZE,
):
    """
    將 OpenCV BGR 車尾裁切圖轉成
    YOLOv5 Classification 所需的張量。

    流程：
    1. 中央裁切成正方形
    2. 縮放成 224 x 224
    3. BGR 轉 RGB
    4. 轉成 CHW
    5. ImageNet normalization
    """

    if vehicle_crop is None or vehicle_crop.size == 0:
        return None

    crop_height, crop_width = vehicle_crop.shape[:2]

    square_size = min(
        crop_height,
        crop_width,
    )

    if square_size <= 0:
        return None

    crop_top = (
        crop_height - square_size
    ) // 2

    crop_left = (
        crop_width - square_size
    ) // 2

    square_crop = vehicle_crop[
        crop_top:crop_top + square_size,
        crop_left:crop_left + square_size,
    ]

    resized_crop = cv2.resize(
        square_crop,
        (
            image_size,
            image_size,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # OpenCV BGR → RGB
    rgb_crop = resized_crop[
        :,
        :,
        ::-1,
    ].astype(np.float32)

    rgb_crop /= 255.0

    imagenet_mean = np.asarray(
        [
            0.485,
            0.456,
            0.406,
        ],
        dtype=np.float32,
    ).reshape(
        1,
        1,
        3,
    )

    imagenet_std = np.asarray(
        [
            0.229,
            0.224,
            0.225,
        ],
        dtype=np.float32,
    ).reshape(
        1,
        1,
        3,
    )

    normalized_crop = (
        rgb_crop - imagenet_mean
    ) / imagenet_std

    chw_crop = np.ascontiguousarray(
        normalized_crop.transpose(
            2,
            0,
            1,
        )
    )

    brake_tensor = torch.from_numpy(
        chw_crop
    ).unsqueeze(0)

    return brake_tensor


def classify_lead_brake_state(
    brake_model,
    frame,
    vehicle_bbox,
):
    """
    根據 Norfair 選出的 LEAD 車框，
    裁切完整車尾並分類 brake_on / brake_off。

    Returns:
        predicted_label:
            brake_on、brake_off 或 None

        confidence:
            0.0 ～ 1.0
    """

    if frame is None or vehicle_bbox is None:
        return None, 0.0

    frame_height, frame_width = (
        frame.shape[:2]
    )

    x1, y1, x2, y2 = [
        int(round(value))
        for value in vehicle_bbox
    ]

    bbox_width = max(
        x2 - x1,
        1,
    )

    bbox_height = max(
        y2 - y1,
        1,
    )

    padding_x = int(
        round(
            bbox_width
            * BRAKE_CROP_PADDING_RATIO
        )
    )

    padding_y = int(
        round(
            bbox_height
            * BRAKE_CROP_PADDING_RATIO
        )
    )

    x1 = max(
        0,
        x1 - padding_x,
    )

    y1 = max(
        0,
        y1 - padding_y,
    )

    x2 = min(
        frame_width,
        x2 + padding_x,
    )

    y2 = min(
        frame_height,
        y2 + padding_y,
    )

    if x2 <= x1 or y2 <= y1:
        return None, 0.0

    vehicle_crop = frame[
        y1:y2,
        x1:x2,
    ]

    brake_tensor = preprocess_brake_crop(
        vehicle_crop=vehicle_crop,
        image_size=(
            BRAKE_CLASSIFIER_IMAGE_SIZE
        ),
    )

    if brake_tensor is None:
        return None, 0.0

    brake_tensor = brake_tensor.to(
        brake_model.device,
        non_blocking=True,
    )

    brake_tensor = (
        brake_tensor.half()
        if brake_model.fp16
        else brake_tensor.float()
    )

    brake_result = brake_model(
        brake_tensor
    )

    # 部分後端可能回傳 list 或 tuple
    if isinstance(
        brake_result,
        (
            list,
            tuple,
        ),
    ):
        brake_result = brake_result[0]

    brake_probabilities = F.softmax(
        brake_result,
        dim=1,
    )[0]

    predicted_index = int(
        brake_probabilities.argmax().item()
    )

    predicted_confidence = float(
        brake_probabilities[
            predicted_index
        ].item()
    )

    brake_names = brake_model.names

    if isinstance(
        brake_names,
        dict,
    ):
        predicted_label = brake_names.get(
            predicted_index,
            str(predicted_index),
        )

    else:
        predicted_label = brake_names[
            predicted_index
        ]

    return (
        str(predicted_label),
        predicted_confidence,
    )

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    brake_weights=ROOT / "weights/brake_yolov5n.pt",
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    road_masks_dir=None,
    distance_calibration=None,
):
    """Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL. Default is 'yolov5s.pt'.
        source (str | Path): Input source, which can be a file, directory, URL, glob pattern, screen capture, or webcam
            index. Default is 'data/images'.
        data (str | Path): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        imgsz (tuple[int, int]): Inference image size as a tuple (height, width). Default is (640, 640).
        conf_thres (float): Confidence threshold for detections. Default is 0.25.
        iou_thres (float): Intersection Over Union (IOU) threshold for non-max suppression. Default is 0.45.
        max_det (int): Maximum number of detections per image. Default is 1000.
        device (str): CUDA device identifier (e.g., '0' or '0,1,2,3') or 'cpu'. Default is an empty string, which uses
            the best available device.
        view_img (bool): If True, display inference results using OpenCV. Default is False.
        save_txt (bool): If True, save results in a text file. Default is False.
        save_csv (bool): If True, save results in a CSV file. Default is False.
        save_conf (bool): If True, include confidence scores in the saved results. Default is False.
        save_crop (bool): If True, save cropped prediction boxes. Default is False.
        nosave (bool): If True, do not save inference images or videos. Default is False.
        classes (list[int]): List of class indices to filter detections by. Default is None.
        agnostic_nms (bool): If True, perform class-agnostic non-max suppression. Default is False.
        augment (bool): If True, use augmented inference. Default is False.
        visualize (bool): If True, visualize feature maps. Default is False.
        update (bool): If True, update all models' weights. Default is False.
        project (str | Path): Directory to save results. Default is 'runs/detect'.
        name (str): Name of the current experiment; used to create a subdirectory within 'project'. Default is 'exp'.
        exist_ok (bool): If True, existing directories with the same name are reused instead of being incremented.
            Default is False.
        line_thickness (int): Thickness of bounding box lines in pixels. Default is 3.
        hide_labels (bool): If True, do not display labels on bounding boxes. Default is False.
        hide_conf (bool): If True, do not display confidence scores on bounding boxes. Default is False.
        half (bool): If True, use FP16 half-precision inference. Default is False.
        dnn (bool): If True, use OpenCV DNN backend for ONNX inference. Default is False.
        vid_stride (int): Stride for processing video frames, to skip frames between processing. Default is 1.

    Returns:
        None

    Examples:
        ```python
        from ultralytics import run

        # Run inference on an image
        run(source='data/images/example.jpg', weights='yolov5s.pt', device='0')

        # Run inference on a video with specific confidence threshold
        run(source='data/videos/example.mp4', weights='yolov5s.pt', conf_thres=0.4, device='0')
        ```
    """
    import pytesseract
    import re
    import os

    if os.path.exists(
        r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ):
        pytesseract.pytesseract.tesseract_cmd = (
            r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        )
    else:
        pytesseract.pytesseract.tesseract_cmd = (
            r'/usr/bin/tesseract'
        )

    print(
        "✅ OCR 車速讀取模式啟用："
        f"{pytesseract.pytesseract.tesseract_cmd}"
    )
 
    source = str(source)
    # =========================================================
    # 讀取實際距離校正資料
    # =========================================================

    distance_calibration_data = (
        load_distance_calibration(
            distance_calibration
        )
    )

    distance_calibration_y = (
        distance_calibration_data["y"]
    )

    distance_calibration_m = (
        distance_calibration_data[
            "distance_m"
        ]
    )

    distance_calibration_width = (
        distance_calibration_data[
            "frame_width"
        ]
    )

    distance_calibration_height = (
        distance_calibration_data[
            "frame_height"
        ]
    )
    # =========================================================
    # 離線 road mask 資料夾
    # =========================================================
    road_masks_path = Path(road_masks_dir)

    if not road_masks_path.exists():
        raise FileNotFoundError(
            f"找不到 road mask 資料夾：{road_masks_path}"
        )

    if not road_masks_path.is_dir():
        raise NotADirectoryError(
            f"road mask 路徑不是資料夾：{road_masks_path}"
        )

    mask_files = sorted(
        road_masks_path.glob("*.png")
    )

    if len(mask_files) == 0:
        raise RuntimeError(
            f"road mask 資料夾內沒有 PNG：{road_masks_path}"
        )

    print(
        f"✅ Road mask 資料夾：{road_masks_path}"
    )

    print(
        f"✅ Road mask 數量：{len(mask_files)}"
    )
    # 自動讀取影片 FPS
    import cv2 as _cv2
    _cap = _cv2.VideoCapture(source)
    fps = _cap.get(_cv2.CAP_PROP_FPS) if _cap.isOpened() else 30
    _cap.release()
    print(f"✅ 影片 FPS：{fps}")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")

    if not (webcam or screenshot or is_url) and not (
        Path(source).exists() or (has_magic(source) and glob(source, recursive=True))
    ):
        raise FileNotFoundError(f"Source path '{source}' does not exist")

    save_img = not nosave and not source.endswith(".txt")  # save inference images

    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
        # =========================================================
    # 載入煞車狀態分類模型
    # =========================================================

    brake_weights_path = Path(
        brake_weights
    )

    if not brake_weights_path.is_file():
        raise FileNotFoundError(
            "找不到煞車分類模型："
            f"{brake_weights_path}"
        )

    brake_model = DetectMultiBackend(
        str(brake_weights_path),
        device=device,
        dnn=False,
        data=data,
        fp16=half,
    )

    if isinstance(
        brake_model.names,
        dict,
    ):
        loaded_brake_names = [
            str(class_name).lower()
            for class_name
            in brake_model.names.values()
        ]

    else:
        loaded_brake_names = [
            str(class_name).lower()
            for class_name
            in brake_model.names
        ]

    if (
        "brake_on"
        not in loaded_brake_names
        or "brake_off"
        not in loaded_brake_names
    ):
        raise ValueError(
            "煞車分類模型類別錯誤，"
            "必須包含 brake_on 與 brake_off。"
            f"目前類別：{brake_model.names}"
        )

    print(
        "✅ 煞車分類模型："
        f"{brake_weights_path}"
    )

    print(
        "✅ 煞車分類類別："
        f"{brake_model.names}"
    )

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    brake_model.warmup(
        imgsz=(
            1,
            3,
            BRAKE_CLASSIFIER_IMAGE_SIZE,
            BRAKE_CLASSIFIER_IMAGE_SIZE,
        )
    )
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    
    danger_hold_frames = 0  # 警報維持計數器
    current_speed = 0
    # 離線 road mask 從 000000.png 開始
    road_mask_frame_index = 0
    # =========================================================
    # 每支影片的動態 ROI 校正狀態
    # =========================================================

    dynamic_roi_bottom_y = (
        DEFAULT_ROI_BOTTOM_Y
    )

    hood_measurements = []

    hood_calibration_frame_count = 0

    roi_calibrated = False

    current_roi_source = None
    # 保留 OCR 畫面實際讀到的數值與單位
    current_speed_raw = 0
    current_speed_unit = "KMH"

    current_level = 0
    # =========================================================
    # 開啟 Nano → STM32 UART
    # =========================================================

    stm32_uart = None
    last_uart_packet = None
    last_uart_send_time = 0.0

    try:
        stm32_uart = serial.Serial(
            port=UART_PORT,
            baudrate=UART_BAUD_RATE,
            timeout=0.1,
            write_timeout=0.1,
        )

        print(
            f"✅ STM32 UART 已連線："
            f"{UART_PORT}, "
            f"{UART_BAUD_RATE} baud"
        )

    except serial.SerialException as uart_error:
        print(
            f"⚠️ STM32 UART 連線失敗："
            f"{uart_error}"
        )

    # =========================================================
    # Norfair 前方車輛追蹤器
    # =========================================================

    vehicle_tracker = Tracker(
        distance_function="iou",
        distance_threshold=0.7,
        hit_counter_max=8,
        initialization_delay=2,
    )

    # =========================================================
    # 行人／機車／腳踏車 Tracker
    # =========================================================

    vru_tracker = Tracker(
        distance_function="iou",
        distance_threshold=0.7,
        hit_counter_max=6,
        initialization_delay=1,
    )

    # 每個 VRU Track ID 的移動歷史
    vru_motion_history = {}

    # 每個 Track ID 已連續向內移動幾幀
    vru_inward_counter = {}

    # 目前鎖定的主要前車 Track ID
    lead_vehicle_id = None
    # 目前正在等待切換的新候選 ID
    lead_switch_candidate_id = None

    # 新候選已經連續勝出幾幀
    lead_switch_counter = 0
    # =========================================================
    # LEAD 實際距離歷史
    # =========================================================
        # =========================================================
    # 煞車分類時間序列狀態
    # =========================================================

    # 這組狀態目前屬於哪一台 LEAD
    brake_state_lead_id = None

    # 尚在等待確認的候選狀態
    # ON、OFF 或 None
    brake_candidate_state = None

    # 候選狀態已連續出現幾幀
    brake_candidate_counter = 0

    # 已確認的穩定狀態
    # ON、OFF 或 None
    brake_stable_state = None

    # =========================================================
    # 緊急煞車事件狀態
    # =========================================================

    # 最近一次 OFF → ON 發生時間
    last_brake_onset_time_sec = None

    # 最近一次 OFF → ON 屬於哪一台 LEAD
    last_brake_onset_lead_id = None

    # TTC + B 連續成立幀數
    emergency_confirm_counter = 0

    # 緊急事件維持到哪個影片時間
    emergency_hold_until_time_sec = -1.0
    # 保存原始距離與平滑距離
    lead_distance_history = deque()
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            # =========================================================
            # 如果切換到另一支影片，重新進行 ROI 校正
            # =========================================================

            roi_source_key = str(
                Path(path)
            )

            if current_roi_source != roi_source_key:
                current_roi_source = roi_source_key

                dynamic_roi_bottom_y = (
                    DEFAULT_ROI_BOTTOM_Y
                )

                hood_measurements = []

                hood_calibration_frame_count = 0

                roi_calibrated = False

                print(
                    "🔄 開始進行新影片的 ROI 校正"
                )
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop

            # Write results
            danger_detected_now = False 
            danger_level_now = 0
            vru_counter = {"person": 0, "motorcycle": 0, "bicycle": 0}
            # =========================================================
            # 本幀要送入 Norfair 的車輛偵測結果
            # 只有中央紅色 ROI 內的車才會加入
            # =========================================================
            norfair_vehicle_detections = []
            tracked_vehicles = []
            # 本幀要送入 VRU Tracker 的偵測結果
            norfair_vru_detections = []
            tracked_vrus = []
            
            h, w, _ = im0.shape
            # 本幀影片時間，供煞車燈事件與緊急狀態使用
            current_time_sec = (
                float(frame)
                / max(float(fps), 1.0)
            )

            # 每幀先預設沒有新的 OFF → ON
            # 只有真正完成狀態轉換時才設為 True
            brake_onset_detected = False
            # 煞車燈分析保留原始畫面。
            # 必須放在繪製彩色 ROI 之前。
            brake_analysis_frame = im0.copy()
            # =========================================================
            # 讀取目前影片幀所對應的 road mask
            # =========================================================
            road_mask_path = (
                road_masks_path
                / f"{road_mask_frame_index:06d}.png"
            )

            # Windows 下避免 cv2.imread 無法讀取中文路徑
            mask_data = np.fromfile(
                str(road_mask_path),
                dtype=np.uint8,
            )

            road_mask = cv2.imdecode(
                mask_data,
                cv2.IMREAD_GRAYSCALE,
            )

            if road_mask is None:
                raise FileNotFoundError(
                    "找不到目前幀的 road mask："
                    f"{road_mask_path}"
                )

            # 尺寸不同時，縮放回目前 YOLO 畫面的大小
            if road_mask.shape != (h, w):
                road_mask = cv2.resize(
                    road_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 同時支援：
            # 0/1 mask
            # 0/255 mask
            road_mask = (
                road_mask > 0
            ).astype(np.uint8)
            print(
                f"[MASK DEBUG] frame={road_mask_frame_index}, "
                f"shape={road_mask.shape}, "
                f"white_pixels={np.count_nonzero(road_mask)}, "
                f"white_ratio={np.mean(road_mask):.4f}"
            )
            # =========================================================
            # 使用前五幀估計引擎蓋上緣
            # 前五幀仍使用預設 ROI，不會空白
            # =========================================================

            if (
                not roi_calibrated
                and hood_calibration_frame_count
                < HOOD_CALIBRATION_FRAMES
            ):
                hood_calibration_frame_count += 1

                detected_hood_top = (
                    detect_hood_top_ratio(
                        road_mask
                    )
                )

                if detected_hood_top is not None:
                    hood_measurements.append(
                        detected_hood_top
                    )

                    print(
                        "引擎蓋量測："
                        f"{hood_calibration_frame_count}/"
                        f"{HOOD_CALIBRATION_FRAMES}"
                        f"  hood={detected_hood_top:.3f}"
                    )

                else:
                    print(
                        "引擎蓋量測："
                        f"{hood_calibration_frame_count}/"
                        f"{HOOD_CALIBRATION_FRAMES}"
                        "  無有效結果"
                    )

                # 前五幀處理完成後才結算
                if (
                    hood_calibration_frame_count
                    >= HOOD_CALIBRATION_FRAMES
                ):
                    if (
                        len(hood_measurements)
                        >= HOOD_MIN_VALID_SAMPLES
                    ):
                        median_hood_top = float(
                            np.median(
                                hood_measurements
                            )
                        )

                        calibrated_bottom_y = (
                            median_hood_top
                            - HOOD_SAFETY_MARGIN_RATIO
                        )

                        dynamic_roi_bottom_y = float(
                            np.clip(
                                calibrated_bottom_y,
                                MIN_DYNAMIC_ROI_BOTTOM_Y,
                                MAX_DYNAMIC_ROI_BOTTOM_Y,
                            )
                        )

                        print(
                            "✅ 動態 ROI 校正完成："
                            f"有效量測={len(hood_measurements)}, "
                            f"hood中位數={median_hood_top:.3f}, "
                            f"ROI底部={dynamic_roi_bottom_y:.3f}"
                        )

                    else:
                        dynamic_roi_bottom_y = (
                            DEFAULT_ROI_BOTTOM_Y
                        )

                        print(
                            "⚠️ 有效引擎蓋量測不足，"
                            "使用預設 ROI："
                            f"bottom={dynamic_roi_bottom_y:.3f}"
                        )

                    roi_calibrated = True
            # if view_img:
            #     cv2.imshow(
            #         "Loaded Road Mask",
            #         road_mask * 255,
            #     )

            # OCR 讀取車速（每10幀讀一次，降低運算量）
            if frame:
                try:
                    # 搜尋畫面下方，不依賴固定座標
                    search_region = im0[int(h * 0.8):h, 0:w]
                    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                    config = '--psm 6 --oem 3'
                    ocr_text = pytesseract.image_to_string(binary, config=config)
                    # =================================================
                    # 同時支援：
                    # 30 KM/H
                    # 30 KMH
                    # 30 KPH
                    # 30 MPH
                    # =================================================
                    # 不刪除數字與單位之間的空白，
                    # 避免日期、時間和車速全部黏在一起
                    ocr_search_text = ocr_text.upper()

                    # 特殊符號轉為空白，但保留 / 和 :
                    ocr_search_text = re.sub(
                        r"[^A-Z0-9/:\s]",
                        " ",
                        ocr_search_text,
                    )

                    # 多個空白整理成一個空白
                    ocr_search_text = re.sub(
                        r"\s+",
                        " ",
                        ocr_search_text,
                    ).strip()

                    # 同時支援：
                    # 55 MPH
                    # 55MPH
                    # 55 KM/H
                    # 55 KMH
                    # 55 KPH
                    match = re.search(
                        r"(?<!\d)(\d{1,3})\s*(MPH|KM/?H|KPH)\b",
                        ocr_search_text,
                    )
                    if match:
                        speed_val_raw = int(match.group(1))
                        detected_unit = match.group(2)

                        # MPH 轉換成 km/h
                        if detected_unit == "MPH":
                            speed_val_kmh = round(
                                speed_val_raw * 1.609344
                            )
                            speed_unit = "MPH"

                        else:
                            # KM/H、KMH、KPH 全部視為 km/h
                            speed_val_kmh = speed_val_raw
                            speed_unit = "KMH"

                        # 統一檢查換算後的合理車速
                        if 0 <= speed_val_kmh <= 200:
                            current_speed_raw = speed_val_raw
                            current_speed_unit = speed_unit

                            # 後續所有判斷統一使用 km/h
                            current_speed = speed_val_kmh
                except:
                    pass
 
            # 依車速決定哪些深度層要啟用
            if current_speed == 0:
                active_depths = []                        # 靜止不預警
            elif 1 <= current_speed <= 30:
                active_depths = ["near"]                  # 低速：只開近端
            elif 31 <= current_speed <= 50:
                active_depths = ["near", "mid"]           # 中速：近端＋中端
            else:
                active_depths = ["near", "mid", "far"]    # 高速：全開
            # =========================================================
            # 顯示影片原始車速單位
            # MPH 同時顯示換算後的 km/h，方便檢查門檻
            # =========================================================
            if current_speed_unit == "MPH":
                speed_display_text = (
                    f"{current_speed_raw} MPH "
                    f"({current_speed} km/h)"
                )
            else:
                speed_display_text = (
                    f"{current_speed_raw} km/h"
                )
            # =========================================================
            # 建立新版固定九宮格
            # =========================================================
            roi_zones = build_dynamic_roi_zones(
                frame_w=w,
                frame_h=h,
                roi_bottom_y=dynamic_roi_bottom_y,
            )

            # =========================================================
            # 使用 road mask，只排除底部引擎蓋
            # active_depths 由目前車速決定
            # =========================================================
            im0 = draw_ground_clipped_nine_grid(
                img=im0,
                roi_zones=roi_zones,
                active_depths=active_depths,
            )

            # ROI 畫完後才建立 YOLO Annotator
            # 確保偵測框與文字會畫在 ROI 上方
            annotator = Annotator(
                im0,
                line_width=line_thickness,
            )
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    # 💡 先抓取原廠 AI 算出來的名字
                    original_class_name = names[c] 
                    final_class_name = original_class_name 
                    
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    xywh_ratio = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    x_center = xywh_ratio[0]
                    y_center = xywh_ratio[1]
                    w_ratio = xywh_ratio[2]  # 物件寬度比例
                    h_ratio = xywh_ratio[3]  # 物件高度比例
                    
                    # 💡 計算物件「寬高比」= 寬度 / 高度
                    aspect_ratio = w_ratio / h_ratio 
                    
                    y_bottom = y_center + (h_ratio / 2) # 接地點

                    # 💡 定點清除擋風玻璃反光
                    if original_class_name == 'car' and (x_center < 0.30 and y_center > 0.70):
                        continue 

                    # =================================================================
                    # 🚀 幾何學【視覺 Override】補丁
                    # =================================================================
                    if final_class_name == 'motorcycle' and aspect_ratio > 1.2: 
                        final_class_name = 'car' 
                        bbox_color = (255, 105, 180) # 粉紅色標註
                    else:
                        bbox_color = colors(c, True)
                    
                    # 預設不顯示 BBOX
                    # 只有進入目前車速啟用的實際警戒區才改成 True

                    # =========================================================
                    # 前方車輛 ROI 判斷與 Norfair Detection 建立
                    #
                    # 只有車框底部中心位於目前啟用的中央紅色 ROI，
                    # 才會交給 Norfair 追蹤。
                    # =========================================================
                    if final_class_name in ["car", "bus", "truck"]:
                        bottom_center_x = int(x_center * w)
                        bottom_center_y = min(
                            int(y_bottom * h),
                            h - 1,
                        )

                        vehicle_depth = None

                       # 緊急煞車前車追蹤固定檢查中央三層，
                        # 不受目前車速影響。
                        for depth in VEHICLE_TRACK_DEPTHS:
                            center_red_zone = roi_zones[
                                ("center", depth)
                            ]

                            # 只有 near 層允許底部稍微超出 ROI
                            if depth == "near":
                                bottom_tolerance_px = int(
                                    h
                                    * LEAD_BOTTOM_TOLERANCE_RATIO
                                )
                            else:
                                bottom_tolerance_px = 0

                            inside_tracking_roi = (
                                point_in_vehicle_corridor(
                                    zone_points=center_red_zone,
                                    point_x=bottom_center_x,
                                    point_y=bottom_center_y,
                                    width_ratio=1.0,
                                    bottom_tolerance_px=(
                                        bottom_tolerance_px
                                    ),
                                )
                            )

                            if inside_tracking_roi:
                                vehicle_depth = depth
                                break

                        # 只有位於紅色 ROI 內才送給 Norfair
                        if vehicle_depth is not None:
                            x1 = float(xyxy[0])
                            y1 = float(xyxy[1])
                            x2 = float(xyxy[2])
                            y2 = float(xyxy[3])

                            # Norfair 的 bbox 格式：
                            # 左上角與右下角
                            bbox_points = np.array(
                                [
                                    [x1, y1],
                                    [x2, y2],
                                ],
                                dtype=np.float32,
                            )

                            bbox_scores = np.array(
                                [
                                    confidence,
                                    confidence,
                                ],
                                dtype=np.float32,
                            )

                            norfair_vehicle_detections.append(
                                Detection(
                                    points=bbox_points,
                                    scores=bbox_scores,
                                    label="vehicle",
                                    data={
                                        "class_name": final_class_name,
                                        "confidence": confidence,
                                        "depth": vehicle_depth,

                                        # 記錄 Detection 所屬幀數
                                        # 用來排除 Norfair 的歷史預測
                                        "frame": int(frame),
                                    },
                                )
                            )

                    # =================================================================
                    # 🔥 畫框邏輯
                    # =================================================================
                    
                    
                    if save_csv: write_to_csv(p.name, final_class_name, confidence_str)
                    if save_txt:  
                        if save_format == 0: coords = ((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist())
                        else: coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                        with open(f"{txt_path}.txt", "a") as f: f.write(("%g " * len(line)).rstrip() % line + "\n")
                    if save_crop: save_one_box(xyxy, imc, file=save_dir / "crops" / final_class_name / f"{p.stem}.jpg", BGR=True)

                    # =================================================================
                    # 🚀 ADAS 深度與「新梯形」範圍判定 (使用 pointPolygonTest 連動)
                    # =================================================================
                    # =================================================================
                    # 💡 駕駛過濾邏輯：區分真實行人與摩托車/腳踏車駕駛
                    # =================================================================
                    def calc_iou(boxA, boxB):
                        """計算兩個框的 IOU（Intersection over Union）"""
                        xA = max(boxA[0], boxB[0])
                        yA = max(boxA[1], boxB[1])
                        xB = min(boxA[2], boxB[2])
                        yB = min(boxA[3], boxB[3])
                        interW = max(0, xB - xA)
                        interH = max(0, yB - yA)
                        interArea = interW * interH
                        if interArea == 0:
                            return 0.0
                        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
                        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
                        return interArea / float(areaA + areaB - interArea)

                    def is_rider(person_box, vehicle_box):
                        """
                        判斷一個 person 框是否為摩托車/腳踏車的駕駛
                        條件一：person 框底部在 vehicle 框中線以上（位置判斷）
                        條件二：IOU 大於 0.2（重疊率判斷）
                        兩個條件同時成立才判定為駕駛
                        """
                        person_bottom = person_box[3]
                        vehicle_mid_y = (vehicle_box[1] + vehicle_box[3]) / 2
                        position_ok = person_bottom <= vehicle_mid_y * 1.2  # 加一點緩衝
                        iou_ok = calc_iou(person_box, vehicle_box) >= 0.2
                        return position_ok or iou_ok

                    # 先整理這一幀所有偵測到的物件
                    # det_boxes 格式：{類別名稱: [list of (x1,y1,x2,y2)]}
                    # 注意：這段要放在 for *xyxy, conf, cls in reversed(det): 迴圈外面
                    # 所以這裡只處理當前這個目標物

                    # 判斷當前目標是否為騎乘者（person 且跟附近車輛重疊）
                    if final_class_name == 'person':
                        is_driver = False
                        for *v_xyxy, v_conf, v_cls in reversed(det):
                            v_class = model.names[int(v_cls)]
                            if v_class in ['motorcycle', 'bicycle']:
                                v_box = [int(v_xyxy[0]), int(v_xyxy[1]),
                                         int(v_xyxy[2]), int(v_xyxy[3])]
                                p_box = [int(xyxy[0]), int(xyxy[1]),
                                         int(xyxy[2]), int(xyxy[3])]
                                if is_rider(p_box, v_box):
                                    is_driver = True
                                    break
                        if is_driver:
                            final_class_name = 'rider'  # 標記為騎乘者，跳過弱勢用路人判定

                    # =================================================
                    # 將行人／機車／腳踏車送入 VRU Tracker
                    #
                    # 這裡只建立追蹤資料，
                    # 不再直接根據單幀位置觸發警示。
                    # =================================================
                    vulnerable_road_users = [
                        "person",
                        "motorcycle",
                        "bicycle",
                    ]

                    if (
                        final_class_name
                        in vulnerable_road_users
                    ):
                        vru_x1 = float(xyxy[0])
                        vru_y1 = float(xyxy[1])
                        vru_x2 = float(xyxy[2])
                        vru_y2 = float(xyxy[3])

                        vru_bbox_points = np.array(
                            [
                                [vru_x1, vru_y1],
                                [vru_x2, vru_y2],
                            ],
                            dtype=np.float32,
                        )

                        vru_bbox_scores = np.array(
                            [
                                confidence,
                                confidence,
                            ],
                            dtype=np.float32,
                        )

                        norfair_vru_detections.append(
                            Detection(
                                points=vru_bbox_points,
                                scores=vru_bbox_scores,
                                label=final_class_name,
                                data={
                                    "class_name": (
                                        final_class_name
                                    ),
                                    "confidence": confidence,
                                    "frame": int(frame),
                                },
                            )
                        )
            # =========================================================
            # 更新 VRU Tracker
            # =========================================================
            tracked_vrus = vru_tracker.update(
                detections=(
                    norfair_vru_detections
                    if norfair_vru_detections
                    else None
                )
            )

            live_vru_ids = set()

            for tracked_vru in tracked_vrus:
                if tracked_vru.id is None:
                    continue

                track_id = tracked_vru.id
                live_vru_ids.add(track_id)

                if tracked_vru.last_detection is None:
                    continue

                vru_data = (
                    tracked_vru.last_detection.data
                    or {}
                )

                # 不使用 Norfair 暫時保留的歷史預測
                if (
                    vru_data.get("frame")
                    != int(frame)
                ):
                    continue

                detection_points = np.asarray(
                    tracked_vru.last_detection.points,
                    dtype=np.float32,
                )

                estimate = np.asarray(
                    tracked_vru.estimate,
                    dtype=np.float32,
                )

                if (
                    detection_points.shape != (2, 2)
                    or estimate.shape != (2, 2)
                ):
                    continue

                vru_class_name = vru_data.get(
                    "class_name",
                    "vru",
                )

                # 本幀偵測框的底部中心
                detected_x1 = int(
                    detection_points[0][0]
                )
                detected_y1 = int(
                    detection_points[0][1]
                )
                detected_x2 = int(
                    detection_points[1][0]
                )
                detected_y2 = int(
                    detection_points[1][1]
                )

                bottom_center_x = int(
                    (
                        detected_x1
                        + detected_x2
                    ) / 2
                )

                bottom_center_y = min(
                    detected_y2,
                    h - 1,
                )

                # Norfair 平滑後的框用於畫面顯示
                tracked_x1 = int(estimate[0][0])
                tracked_y1 = int(estimate[0][1])
                tracked_x2 = int(estimate[1][0])
                tracked_y2 = int(estimate[1][1])

                # =============================================
                # 找出目前所在的啟用 ROI
                # 中央優先，避免共用邊界重複判定
                # =============================================
                matched_position = None
                matched_depth = None

                for depth in active_depths:
                    for position in (
                        "center",
                        "left",
                        "right",
                    ):
                        zone_points = roi_zones[
                            (position, depth)
                        ]

                        inside_zone = cv2.pointPolygonTest(
                            zone_points,
                            (
                                bottom_center_x,
                                bottom_center_y,
                            ),
                            False,
                        )

                        if inside_zone >= 0:
                            matched_position = position
                            matched_depth = depth
                            break

                    if matched_position is not None:
                        break

                # 不在目前車速啟用的 ROI，不顯示也不警示
                if (
                    matched_position is None
                    or matched_depth is None
                ):
                    continue

                # =============================================
                # 計算目標相對於本車行駛走廊的位置
                # =============================================
                center_zone_points = roi_zones[
                    ("center", matched_depth)
                ]

                horizontal_bounds = (
                    get_zone_horizontal_bounds(
                        zone_points=center_zone_points,
                        point_y=bottom_center_y,
                    )
                )

                if horizontal_bounds is None:
                    continue

                (
                    center_left_x,
                    center_right_x,
                ) = horizontal_bounds

                center_line_x = (
                    center_left_x
                    + center_right_x
                ) / 2.0

                center_half_width = max(
                    (
                        center_right_x
                        - center_left_x
                    ) / 2.0,
                    1.0,
                )

                # 正規化橫向位置：
                # 中央線 = 0
                # 紅色 ROI 左右邊界約為 -1、+1
                normalized_lateral_position = (
                    (
                        bottom_center_x
                        - center_line_x
                    )
                    / center_half_width
                )

                inside_ego_path = (
                    abs(
                        normalized_lateral_position
                    )
                    <= VRU_EGO_PATH_WIDTH_RATIO
                )

                distance_to_ego_path = max(
                    0.0,
                    abs(
                        normalized_lateral_position
                    )
                    - VRU_EGO_PATH_WIDTH_RATIO,
                )

                if inside_ego_path:
                    motion_side = "center"
                elif normalized_lateral_position < 0:
                    motion_side = "left"
                else:
                    motion_side = "right"

                # =============================================
                # 保存最近約 0.5 秒的橫向距離
                # =============================================
                history = vru_motion_history.setdefault(
                    track_id,
                    deque(),
                )

                # 目標跨越中央或換到另一側時，
                # 舊方向歷史不可繼續使用
                if (
                    history
                    and history[-1][2]
                    != motion_side
                ):
                    history.clear()
                    vru_inward_counter[track_id] = 0

                history.append(
                    (
                        current_time_sec,
                        distance_to_ego_path,
                        motion_side,
                    )
                )

                while (
                    history
                    and (
                        current_time_sec
                        - history[0][0]
                    )
                    > VRU_MOTION_HISTORY_SECONDS
                ):
                    history.popleft()

                # =============================================
                # 判斷是否持續向本車路徑靠近
                # =============================================
                moving_inward = False

                if len(history) >= 2:
                    history_duration = (
                        history[-1][0]
                        - history[0][0]
                    )

                    sample_count = min(
                        3,
                        len(history),
                    )

                    old_distance = float(
                        np.median(
                            [
                                sample[1]
                                for sample
                                in list(history)[
                                    :sample_count
                                ]
                            ]
                        )
                    )

                    new_distance = float(
                        np.median(
                            [
                                sample[1]
                                for sample
                                in list(history)[
                                    -sample_count:
                                ]
                            ]
                        )
                    )

                    inward_progress = (
                        old_distance
                        - new_distance
                    )

                    if history_duration > 0:
                        inward_speed = (
                            inward_progress
                            / history_duration
                        )
                    else:
                        inward_speed = 0.0

                    moving_inward = (
                        history_duration
                        >= (
                            VRU_MOTION_MIN_DURATION_SECONDS
                        )
                        and inward_progress
                        >= VRU_INWARD_MIN_PROGRESS
                        and inward_speed
                        >= VRU_INWARD_MIN_SPEED
                    )

                if moving_inward:
                    vru_inward_counter[track_id] = (
                        vru_inward_counter.get(
                            track_id,
                            0,
                        )
                        + 1
                    )
                else:
                    vru_inward_counter[track_id] = 0

                inward_confirmed = (
                    vru_inward_counter.get(
                        track_id,
                        0,
                    )
                    >= VRU_INWARD_CONFIRM_FRAMES
                )

                # =============================================
                # 最終威脅條件
                #
                # 1. 已經進入本車中央走廊
                # 2. 或從外側持續向中央靠近
                # =============================================
                is_vru_threat = (
                    inside_ego_path
                    or inward_confirmed
                )

                if is_vru_threat:

                    # 已經進入本車行駛走廊，使用 center 等級
                    if inside_ego_path:
                        warning_position = "center"

                    # 尚在走廊外側，視為左／右側侵入
                    else:
                        warning_position = motion_side

                    detected_level = (
                        calculate_vru_warning_level(
                            position=warning_position,
                            depth=matched_depth,
                            current_speed=current_speed,
                        )
                    )

                    danger_level_now = max(
                        danger_level_now,
                        detected_level,
                    )

                    if (
                        vru_class_name
                        in vru_counter
                    ):
                        vru_counter[
                            vru_class_name
                        ] += 1

                    # BBOX 顏色對應警示等級
                    warning_bbox_colors = {
                        1: (0, 255, 255),
                        2: (0, 165, 255),
                        3: (0, 0, 255),
                    }

                    vru_bbox_color = (
                        warning_bbox_colors[
                            detected_level
                        ]
                    )

                else:
                    # 偵測得到，但目前不具威脅
                    vru_bbox_color = (
                        150,
                        150,
                        150,
                    )

                # 顯示追蹤後的 BBOX
                annotator.box_label(
                    [
                        tracked_x1,
                        tracked_y1,
                        tracked_x2,
                        tracked_y2,
                    ],
                    None,
                    color=vru_bbox_color,
                )

            # 清除已經完全消失的 Track ID 歷史
            for stored_track_id in list(
                vru_motion_history.keys()
            ):
                if stored_track_id not in live_vru_ids:
                    vru_motion_history.pop(
                        stored_track_id,
                        None,
                    )

                    vru_inward_counter.pop(
                        stored_track_id,
                        None,
                    )
            # =========================================================
            # 每一幀更新 Norfair Tracker
            # 即使這一幀沒有車，也要更新 Tracker
            # =========================================================
            tracked_vehicles = vehicle_tracker.update(
                detections=(
                    norfair_vehicle_detections
                    if norfair_vehicle_detections
                    else None
                )
            )

            # =========================================================
            # 從本幀有效的追蹤結果建立 LEAD 候選
            #
            # 必須同時符合：
            # 1. 已經取得 Track ID
            # 2. 本幀真的被 YOLO 偵測到
            # 3. 底部中心仍然在中央紅色 ROI
            # =========================================================
            lead_candidates = []

            for tracked_vehicle in tracked_vehicles:
                if tracked_vehicle.id is None:
                    continue

                if tracked_vehicle.last_detection is None:
                    continue

                detection_data = (
                    tracked_vehicle.last_detection.data or {}
                )

                # Norfair 可能會短暫保留消失車輛的位置。
                # 不是本幀真的偵測到的車，不可以成為 LEAD。
                if detection_data.get("frame") != int(frame):
                    continue

                # 本幀 YOLO Detection 的原始框
                detection_points = np.asarray(
                    tracked_vehicle.last_detection.points,
                    dtype=np.float32,
                )

                # Norfair 平滑後的追蹤框
                estimate = np.asarray(
                    tracked_vehicle.estimate,
                    dtype=np.float32,
                )

                if (
                    detection_points.shape != (2, 2)
                    or estimate.shape != (2, 2)
                ):
                    continue

                # =====================================================
                # 使用本幀 YOLO 車框的底部中心重新檢查 ROI
                # =====================================================
                detected_x1 = int(detection_points[0][0])
                detected_y1 = int(detection_points[0][1])
                detected_x2 = int(detection_points[1][0])
                detected_y2 = int(detection_points[1][1])

                bottom_center_x = int(
                    (detected_x1 + detected_x2) / 2
                )
                bottom_center_y = min(
                    detected_y2,
                    h - 1,
                )

                candidate_depth = None
                # LEAD 候選固定使用中央三層，
                # 不受目前車速影響。
                for depth in VEHICLE_TRACK_DEPTHS:
                    center_red_zone = roi_zones[
                        ("center", depth)
                    ]

                    # 只有 near 層允許底部超出
                    if depth == "near":
                        bottom_tolerance_px = int(
                            h
                            * LEAD_BOTTOM_TOLERANCE_RATIO
                        )
                    else:
                        bottom_tolerance_px = 0

                    # LEAD 只接受中央紅色 ROI 中間 70%
                    inside_lead_corridor = (
                        point_in_vehicle_corridor(
                            zone_points=center_red_zone,
                            point_x=bottom_center_x,
                            point_y=bottom_center_y,
                            width_ratio=(
                                LEAD_CORRIDOR_WIDTH_RATIO
                            ),
                            bottom_tolerance_px=(
                                bottom_tolerance_px
                            ),
                        )
                    )

                    if inside_lead_corridor:
                        candidate_depth = depth
                        break
                # 車輛離開中央紅色 ROI，立即失去候選資格
                if candidate_depth is None:
                    continue

                # 使用 Norfair 平滑後的位置顯示追蹤框
                x1 = int(estimate[0][0])
                y1 = int(estimate[0][1])
                x2 = int(estimate[1][0])
                y2 = int(estimate[1][1])

                # 車框底部越靠近畫面下方，分數越高
                bottom_score = (
                    bottom_center_y / max(h, 1)
                )

                # 越靠近畫面中央，分數越高
                center_error = (
                    abs(bottom_center_x - (w / 2))
                    / max(w / 2, 1)
                )

                lead_score = (
                    bottom_score
                    - LEAD_CENTER_SCORE_WEIGHT
                    * center_error
                )

                lead_candidates.append(
                    {
                        "id": tracked_vehicle.id,
                        "vehicle": tracked_vehicle,
                        "class_name": detection_data.get(
                            "class_name",
                            "vehicle",
                        ),
                        "confidence": detection_data.get(
                            "confidence",
                            0.0,
                        ),
                        "depth": candidate_depth,
                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                        "bottom_center": (
                            bottom_center_x,
                            bottom_center_y,
                        ),
                        "score": lead_score,
                    }
                )

            # 方便依 Track ID 尋找候選車
            candidate_by_id = {
                candidate["id"]: candidate
                for candidate in lead_candidates
            }

            # =========================================================
            # LEAD 選擇與連續幀切換
            # =========================================================

            # 目前 LEAD 已經離開中央子走廊
            if lead_vehicle_id not in candidate_by_id:
                lead_vehicle_id = None
                lead_switch_candidate_id = None
                lead_switch_counter = 0

            # ---------------------------------------------------------
            # 目前沒有 LEAD：立即選擇最佳候選
            # ---------------------------------------------------------
            if lead_vehicle_id is None:

                if lead_candidates:
                    best_candidate = max(
                        lead_candidates,
                        key=lambda vehicle: vehicle["score"],
                    )

                    lead_vehicle_id = (
                        best_candidate["id"]
                    )

                lead_switch_candidate_id = None
                lead_switch_counter = 0

            # ---------------------------------------------------------
            # 已經有 LEAD：每幀與最佳候選重新比較
            # ---------------------------------------------------------
            elif lead_candidates:

                current_lead_candidate = (
                    candidate_by_id[lead_vehicle_id]
                )

                best_candidate = max(
                    lead_candidates,
                    key=lambda vehicle: vehicle["score"],
                )

                best_candidate_id = (
                    best_candidate["id"]
                )

                current_lead_score = (
                    current_lead_candidate["score"]
                )

                best_candidate_score = (
                    best_candidate["score"]
                )

                # 其他車必須明顯優於目前 LEAD
                better_candidate_found = (
                    best_candidate_id
                    != lead_vehicle_id
                    and best_candidate_score
                    >= (
                        current_lead_score
                        + LEAD_SWITCH_SCORE_MARGIN
                    )
                )

                if better_candidate_found:

                    # 同一台新候選持續勝出
                    if (
                        lead_switch_candidate_id
                        == best_candidate_id
                    ):
                        lead_switch_counter += 1

                    # 換成另一台新候選，重新計數
                    else:
                        lead_switch_candidate_id = (
                            best_candidate_id
                        )
                        lead_switch_counter = 1

                    # 連續勝出足夠幀數後才切換
                    if (
                        lead_switch_counter
                        >= LEAD_SWITCH_CONFIRM_FRAMES
                    ):
                        lead_vehicle_id = (
                            best_candidate_id
                        )

                        lead_switch_candidate_id = None
                        lead_switch_counter = 0

                # 沒有其他車明顯更好，取消等待切換
                else:
                    lead_switch_candidate_id = None
                    lead_switch_counter = 0
            # 取得目前真正鎖定的主要前車
            lead_vehicle = candidate_by_id.get(
                lead_vehicle_id
            )

                        # =========================================================
            # 本幀 LEAD 實際距離與 TTC 分析結果
            # =========================================================

            lead_distance_m = None
            lead_smoothed_distance_m = None
            lead_closing_speed_mps = None
            lead_ttc = None

            lead_distance_text = "DIST:--"
            lead_motion_text = "CLOSING:-- TTC:--"

            if lead_vehicle is not None:
                lead_status_text = (
                    f"LEAD ID:{lead_vehicle_id} "
                    f"{lead_vehicle['class_name']} "
                    f"{lead_vehicle['depth']}"
                )

                lead_status_color = (0, 255, 255)

                # =====================================================
                # 將距離校正 y 座標縮放至目前影片解析度
                # =====================================================

                if distance_calibration_height <= 0:
                    raise ValueError(
                        "距離校正檔缺少有效的 frame_height"
                    )

                distance_y_scale = (
                    float(h)
                    / float(distance_calibration_height)
                )

                current_distance_calibration_y = (
                    distance_calibration_y
                    * distance_y_scale
                )

                # LEAD 車框底部中心
                lead_contact_x, lead_contact_y = (
                    lead_vehicle["bottom_center"]
                )

                # 由底部中心 y 換算原始距離
                lead_distance_m = (
                    estimate_distance_from_y(
                        contact_y=lead_contact_y,
                        calibration_y=(
                            current_distance_calibration_y
                        ),
                        calibration_m=(
                            distance_calibration_m
                        ),
                    )
                )

                lead_time_sec = (
                    float(frame)
                    / max(float(fps), 1.0)
                )

                # =====================================================
                # LEAD ID 改變時，舊車距離歷史不可沿用
                # =====================================================

                if (
                    lead_distance_history
                    and lead_distance_history[-1]["id"]
                    != lead_vehicle_id
                ):
                    lead_distance_history.clear()

                # =====================================================
                # 刪除超過保存時間的舊距離
                # =====================================================

                while (
                    lead_distance_history
                    and (
                        lead_time_sec
                        - lead_distance_history[0]["time_sec"]
                    )
                    > DISTANCE_HISTORY_SECONDS
                ):
                    lead_distance_history.popleft()

                # =====================================================
                # 只有 5～20 公尺內才計算距離、Closing 與 TTC
                # =====================================================

                if lead_distance_m is not None:

                    # -------------------------------------------------
                    # 1. 最近五幀距離取中位數
                    # -------------------------------------------------

                    distance_history_list = list(
                        lead_distance_history
                    )

                    previous_sample_count = max(
                        DISTANCE_SMOOTH_FRAMES - 1,
                        0,
                    )

                    if previous_sample_count > 0:
                        previous_distance_samples = (
                            distance_history_list[
                                -previous_sample_count:
                            ]
                        )
                    else:
                        previous_distance_samples = []

                    recent_raw_distances = [
                        sample["raw_distance_m"]
                        for sample
                        in previous_distance_samples
                        if sample["raw_distance_m"] is not None
                    ]

                    recent_raw_distances.append(
                        lead_distance_m
                    )

                    lead_smoothed_distance_m = float(
                        np.median(
                            recent_raw_distances
                        )
                    )

                    # -------------------------------------------------
                    # 2. 取得約 0.35～0.60 秒前的距離
                    # -------------------------------------------------

                    closing_baseline_samples = [
                        sample
                        for sample in lead_distance_history
                        if (
                            CLOSING_BASELINE_MIN_SECONDS
                            <= (
                                lead_time_sec
                                - sample["time_sec"]
                            )
                            <= CLOSING_BASELINE_MAX_SECONDS
                            and sample[
                                "smoothed_distance_m"
                            ]
                            is not None
                        )
                    ]

                    if (
                        len(closing_baseline_samples)
                        >= CLOSING_BASELINE_MIN_SAMPLES
                    ):
                        baseline_distance_m = float(
                            np.median(
                                [
                                    sample[
                                        "smoothed_distance_m"
                                    ]
                                    for sample
                                    in closing_baseline_samples
                                ]
                            )
                        )

                        baseline_time_sec = float(
                            np.mean(
                                [
                                    sample["time_sec"]
                                    for sample
                                    in closing_baseline_samples
                                ]
                            )
                        )

                        distance_delta_time = (
                            lead_time_sec
                            - baseline_time_sec
                        )

                        if distance_delta_time > 0:
                            # 正值：距離縮短，正在接近
                            # 負值：距離增加，正在遠離
                            lead_closing_speed_mps = (
                                baseline_distance_m
                                - lead_smoothed_distance_m
                            ) / distance_delta_time

                            # 只有明確接近時才計算 TTC
                            if (
                                lead_closing_speed_mps
                                > MIN_CLOSING_SPEED_MPS
                            ):
                                lead_ttc = (
                                    lead_smoothed_distance_m
                                    / lead_closing_speed_mps
                                )

                    # -------------------------------------------------
                    # 3. 計算完成後才保存目前這一幀
                    # -------------------------------------------------

                    lead_distance_history.append(
                        {
                            "id": lead_vehicle_id,
                            "frame": int(frame),
                            "time_sec": lead_time_sec,
                            "raw_distance_m": (
                                lead_distance_m
                            ),
                            "smoothed_distance_m": (
                                lead_smoothed_distance_m
                            ),
                        }
                    )

                    # DIST 顯示平滑後距離
                    lead_distance_text = (
                        f"DIST:"
                        f"{lead_smoothed_distance_m:.1f}m "
                        f"Y:{lead_contact_y}"
                    )

                    # -------------------------------------------------
                    # 4. 建立 Closing 與唯一 TTC 顯示
                    # -------------------------------------------------

                    if lead_closing_speed_mps is None:
                        lead_motion_text = (
                            "CLOSING:WAIT TTC:WAIT"
                        )

                    elif lead_ttc is None:
                        lead_motion_text = (
                            f"CLOSING:"
                            f"{lead_closing_speed_mps:+.2f}m/s "
                            f"TTC:--"
                        )

                    else:
                        display_ttc = min(
                            lead_ttc,
                            99.9,
                        )

                        lead_motion_text = (
                            f"CLOSING:"
                            f"{lead_closing_speed_mps:+.2f}m/s "
                            f"TTC:{display_ttc:.1f}s"
                        )

                else:
                    # 超出 5～20 公尺範圍時，
                    # 不沿用之前的距離歷史
                    lead_distance_history.clear()

                    if (
                        lead_contact_y
                        < current_distance_calibration_y[0]
                    ):
                        maximum_distance = float(
                            np.max(
                                distance_calibration_m
                            )
                        )

                        lead_distance_text = (
                            f"DIST:>{maximum_distance:.0f}m "
                            f"Y:{lead_contact_y}"
                        )

                    else:
                        minimum_distance = float(
                            np.min(
                                distance_calibration_m
                            )
                        )

                        lead_distance_text = (
                            f"DIST:<{minimum_distance:.0f}m "
                            f"Y:{lead_contact_y}"
                        )

                    lead_motion_text = (
                        "CLOSING:-- TTC:--"
                    )

            else:
                lead_status_text = "LEAD: NONE"
                lead_status_color = (180, 180, 180)

                lead_distance_m = None
                lead_smoothed_distance_m = None
                lead_closing_speed_mps = None
                lead_ttc = None

                lead_distance_text = "DIST:--"
                lead_motion_text = (
                    "CLOSING:-- TTC:--"
                )

                # LEAD 消失後清除距離歷史
                lead_distance_history.clear()
                        # =========================================================
            # LEAD 煞車狀態分類
            # =========================================================

            brake_light_text = "BRAKE: NONE"

            # 舊的紅燈候選框除錯畫面暫時停用
            brake_debug_boxes = None

            if lead_vehicle is None:
                # 沒有 LEAD，所有分類狀態清除
                brake_state_lead_id = None
                brake_candidate_state = None
                brake_candidate_counter = 0
                brake_stable_state = None

            else:
                # -----------------------------------------------------
                # LEAD 換車後不可沿用上一台車的狀態
                # -----------------------------------------------------

                if (
                    brake_state_lead_id
                    != lead_vehicle_id
                ):
                    brake_state_lead_id = (
                        lead_vehicle_id
                    )

                    brake_candidate_state = None
                    brake_candidate_counter = 0
                    brake_stable_state = None

                (
                    brake_x1,
                    brake_y1,
                    brake_x2,
                    brake_y2,
                ) = lead_vehicle["bbox"]

                brake_bbox_width = max(
                    0,
                    brake_x2 - brake_x1,
                )

                brake_bbox_height = max(
                    0,
                    brake_y2 - brake_y1,
                )

                # -----------------------------------------------------
                # LEAD 車框太小時，先不做煞車分類
                # -----------------------------------------------------

                if (
                    brake_bbox_width
                    < BRAKE_MIN_BBOX_WIDTH
                    or brake_bbox_height
                    < BRAKE_MIN_BBOX_HEIGHT
                ):
                    brake_light_text = (
                        "BRAKE: TOO SMALL"
                    )

                    brake_candidate_state = None
                    brake_candidate_counter = 0
                    brake_stable_state = None

                else:
                    (
                        predicted_brake_label,
                        predicted_brake_confidence,
                    ) = classify_lead_brake_state(
                        brake_model=brake_model,
                        frame=brake_analysis_frame,
                        vehicle_bbox=(
                            lead_vehicle["bbox"]
                        ),
                    )

                    raw_brake_state = None

                    if (
                        predicted_brake_label
                        is not None
                    ):
                        normalized_brake_label = (
                            predicted_brake_label
                            .strip()
                            .lower()
                        )

                        if (
                            normalized_brake_label
                            == "brake_on"
                            and
                            predicted_brake_confidence
                            >= BRAKE_ON_MIN_CONFIDENCE
                        ):
                            raw_brake_state = "ON"

                        elif (
                            normalized_brake_label
                            == "brake_off"
                            and
                            predicted_brake_confidence
                            >= BRAKE_OFF_MIN_CONFIDENCE
                        ):
                            raw_brake_state = "OFF"

                    # -------------------------------------------------
                    # 信心不足時，不改變穩定狀態
                    # -------------------------------------------------

                    if raw_brake_state is None:
                        brake_candidate_state = None
                        brake_candidate_counter = 0

                        stable_text = (
                            brake_stable_state
                            if brake_stable_state
                            is not None
                            else "WAIT"
                        )

                        brake_light_text = (
                            "BRAKE:? "
                            f"{predicted_brake_label} "
                            f"{predicted_brake_confidence:.2f} "
                            f"STABLE:{stable_text}"
                        )

                    else:
                        # ---------------------------------------------
                        # 計算 ON 或 OFF 連續出現幾幀
                        # ---------------------------------------------

                        if (
                            brake_candidate_state
                            == raw_brake_state
                        ):
                            brake_candidate_counter += 1

                        else:
                            brake_candidate_state = (
                                raw_brake_state
                            )

                            brake_candidate_counter = 1

                        required_confirm_frames = (
                            BRAKE_ON_CONFIRM_FRAMES
                            if raw_brake_state == "ON"
                            else BRAKE_OFF_CONFIRM_FRAMES
                        )

                        # ---------------------------------------------
                        # 達到連續幀數後，更新穩定狀態
                        # ---------------------------------------------

                        if (
                            brake_candidate_counter
                            >= required_confirm_frames
                            and brake_stable_state
                            != raw_brake_state
                        ):
                            previous_stable_state = (
                                brake_stable_state
                            )

                            brake_stable_state = (
                                raw_brake_state
                            )

                            # 真正事件只接受：
                            # 穩定 OFF → 穩定 ON
                            if (
                                previous_stable_state
                                == "OFF"
                                and brake_stable_state
                                == "ON"
                            ):
                                brake_onset_detected = True

                        stable_text = (
                            brake_stable_state
                            if brake_stable_state
                            is not None
                            else "WAIT"
                        )

                        brake_light_text = (
                            f"BRAKE:{raw_brake_state} "
                            f"{predicted_brake_confidence:.2f} "
                            f"COUNT:{brake_candidate_counter}/"
                            f"{required_confirm_frames} "
                            f"STABLE:{stable_text}"
                        )

                        if brake_onset_detected:
                            brake_light_text += (
                                " ONSET"
                            )
            # =========================================================
            # TTC + 煞車燈 ONSET 緊急煞車事件判斷
            # =========================================================

            # ---------------------------------------------------------
            # 1. LEAD 消失或換車時清除舊事件
            # ---------------------------------------------------------

            if lead_vehicle_id is None:
                last_brake_onset_time_sec = None
                last_brake_onset_lead_id = None

                emergency_confirm_counter = 0
                emergency_hold_until_time_sec = -1.0

            elif (
                last_brake_onset_lead_id
                is not None
                and last_brake_onset_lead_id
                != lead_vehicle_id
            ):
                last_brake_onset_time_sec = None
                last_brake_onset_lead_id = None

                emergency_confirm_counter = 0
                emergency_hold_until_time_sec = -1.0

            # ---------------------------------------------------------
            # 2. C：本幀是否剛發生穩定 OFF → ON
            # ---------------------------------------------------------

            if (
                brake_onset_detected
                and lead_vehicle_id is not None
            ):
                last_brake_onset_time_sec = (
                    current_time_sec
                )

                last_brake_onset_lead_id = (
                    lead_vehicle_id
                )

            # ---------------------------------------------------------
            # 3. T：TTC 是否小於等於 6 秒
            # ---------------------------------------------------------

            ttc_critical = (
                lead_ttc is not None
                and 0.0 < lead_ttc
                <= EMERGENCY_TTC_THRESHOLD_SECONDS
            )

            # ---------------------------------------------------------
            # 4. B：最近 1 秒是否發生過同一台車的 ONSET
            # ---------------------------------------------------------

            brake_onset_recent = (
                last_brake_onset_time_sec
                is not None
                and last_brake_onset_lead_id
                == lead_vehicle_id
                and 0.0
                <= (
                    current_time_sec
                    - last_brake_onset_time_sec
                )
                <= BRAKE_ONSET_WINDOW_SECONDS
            )

            # ---------------------------------------------------------
            # 5. 緊急事件條件：T 與 B 同時成立
            # ---------------------------------------------------------

            emergency_condition = (
                ttc_critical
                and brake_onset_recent
            )

            # ---------------------------------------------------------
            # 6. 連續 3 幀確認
            # ---------------------------------------------------------

            if emergency_condition:
                emergency_confirm_counter += 1

            else:
                emergency_confirm_counter = 0

            # ---------------------------------------------------------
            # 7. 確認後維持 1 秒
            # ---------------------------------------------------------

            if (
                emergency_confirm_counter
                >= EMERGENCY_CONFIRM_FRAMES
            ):
                emergency_hold_until_time_sec = (
                    current_time_sec
                    + EMERGENCY_HOLD_SECONDS
                )

            emergency_brake_confirmed = (
                current_time_sec
                <= emergency_hold_until_time_sec
            )

            # ---------------------------------------------------------
            # 8. 建立畫面文字
            # ---------------------------------------------------------

            if emergency_brake_confirmed:
                emergency_state_text = (
                    "EVENT: EMERGENCY_BRAKE "
                    f"T:{int(ttc_critical)} "
                    f"C:{int(brake_onset_detected)} "
                    f"B:{int(brake_onset_recent)}"
                )

                emergency_state_color = (
                    0,
                    0,
                    255,
                )

            elif emergency_condition:
                emergency_state_text = (
                    "EVENT: SUSPECT "
                    f"{emergency_confirm_counter}/"
                    f"{EMERGENCY_CONFIRM_FRAMES} "
                    f"T:{int(ttc_critical)} "
                    f"C:{int(brake_onset_detected)} "
                    f"B:{int(brake_onset_recent)}"
                )

                emergency_state_color = (
                    0,
                    140,
                    255,
                )

            else:
                emergency_state_text = (
                    "EVENT: NORMAL "
                    f"T:{int(ttc_critical)} "
                    f"C:{int(brake_onset_detected)} "
                    f"B:{int(brake_onset_recent)}"
                )

                emergency_state_color = (
                    0,
                    0,
                    0,
                )
            # 防抖結算：保留最高等級
            if danger_level_now > 0:
                danger_hold_frames = 15
                current_level = danger_level_now
            else:
                if danger_hold_frames > 0:
                    danger_hold_frames -= 1
                    # current_level 維持上一幀的值，不動（防抖期間保持警示）
                else:
                    current_level = 0

            # =========================================================
            # 建立 4-bit 事件編碼：T1T0 L1L0
            #
            # 00 = 安全
            # 01 = 弱勢用路人
            # 10 = 前前車緊急煞車
            # 11 = 系統錯誤
            # =========================================================

                        # =========================================================
            # 一次只輸出一種事件
            #
            # 優先順序：
            # 緊急煞車 > 弱勢用路人 > SAFE
            # =========================================================

            if emergency_brake_confirmed:
                # 緊急煞車固定為 Level 1
                # 事件種類 10、等級 01 → 1001
                event_type = 0b10
                event_level = 0b01
                led_color = (0, 255, 255)

            elif current_level == 1:
                # 弱勢用路人 Level 1
                # 事件種類 01、等級 01 → 0101
                event_type = 0b01
                event_level = 0b01
                led_color = (0, 255, 255)

            elif current_level == 2:
                # 弱勢用路人 Level 2
                # 事件種類 01、等級 10 → 0110
                event_type = 0b01
                event_level = 0b10
                led_color = (0, 140, 255)

            elif current_level >= 3:
                # 弱勢用路人 Level 3
                # 事件種類 01、等級 11 → 0111
                event_type = 0b01
                event_level = 0b11
                led_color = (0, 0, 255)

            else:
                # SAFE
                # 事件種類 00、無等級 00 → 0000
                event_type = 0b00
                event_level = 0b00
                led_color = (0, 255, 0)
            # 將前兩位事件種類與後兩位等級合併
            uart_value = (
                (event_type << 2)
                | event_level
            )

            # 終端機顯示，例如緊急煞車為 1001
            uart_code = f"{uart_value:04b}"

            # 只傳送 4-bit 事件編碼
            # UART 實體使用一個 Byte，高四位固定為 0000
            uart_packet = bytes([
                uart_value & 0b1111
            ])
            current_uart_time = time.monotonic()

            should_send_uart = (
                uart_packet != last_uart_packet
                or
                current_uart_time - last_uart_send_time
                >= UART_RESEND_INTERVAL_SECONDS
            )

            if stm32_uart is not None and should_send_uart:
                try:
                    stm32_uart.write(uart_packet)
                    stm32_uart.flush()

                    last_uart_packet = uart_packet
                    last_uart_send_time = current_uart_time

                    print(
                        f"📤 Nano → STM32：{uart_code}"
                    )

                except serial.SerialException as uart_error:
                    print(
                        f"⚠️ UART 傳送失敗：{uart_error}"
                    )

                    try:
                        stm32_uart.close()
                    except Exception:
                        pass

                    stm32_uart = None

            # 終端機印出 UART 編碼
            # 整理 VRU 統計字串
            vru_summary = " | " + " ".join(
                f"{v} {k}{'s' if v > 1 else ''}"
                for k, v in vru_counter.items() if v > 0
            ) if any(v > 0 for v in vru_counter.values()) else ""

            print(
                f"UART 4-bit: {uart_code}"
                f"  |  Level: {current_level}"
                f"  |  Speed: {speed_display_text}"
                f"{vru_summary}"
            )

            # 取得已畫好 YOLO 偵測框的結果
            im0 = annotator.result()
            # =========================================================
            # 顯示煞車燈分析區域
            # =========================================================
            if brake_debug_boxes is not None:
                search_box = (
                    brake_debug_boxes["search"]
                )

                candidate_boxes = (
                    brake_debug_boxes[
                        "candidates"
                    ]
                )

                left_box = (
                    brake_debug_boxes["left"]
                )

                right_box = (
                    brake_debug_boxes["right"]
                )

                body_box = (
                    brake_debug_boxes["body"]
                )

                lamp_source = (
                    brake_debug_boxes["source"]
                )

                # 藍色：整個動態紅燈搜尋範圍
                cv2.rectangle(
                    im0,
                    (
                        search_box[0],
                        search_box[1],
                    ),
                    (
                        search_box[2],
                        search_box[3],
                    ),
                    (255, 0, 0),
                    1,
                )

                # 紫色：所有 HSV 紅色候選區塊
                for candidate_box in candidate_boxes:
                    cv2.rectangle(
                        im0,
                        (
                            candidate_box[0],
                            candidate_box[1],
                        ),
                        (
                            candidate_box[2],
                            candidate_box[3],
                        ),
                        (255, 0, 255),
                        1,
                    )

                # 只有動態找到或暫時追蹤的燈組，
                # 才畫左右綠色框。
                if (
                    lamp_source in (
                        "DETECTED",
                        "TRACKED",
                    )
                    and left_box is not None
                    and right_box is not None
                ):
                    selected_color = (
                        0,
                        255,
                        0,
                    )

                    cv2.rectangle(
                        im0,
                        (
                            left_box[0],
                            left_box[1],
                        ),
                        (
                            left_box[2],
                            left_box[3],
                        ),
                        selected_color,
                        2,
                    )

                    cv2.rectangle(
                        im0,
                        (
                            right_box[0],
                            right_box[1],
                        ),
                        (
                            right_box[2],
                            right_box[3],
                        ),
                        selected_color,
                        2,
                    )

                else:
                    # 找不到可靠燈組時不畫左右框
                    # 白色只用於 LAMP:NONE 文字
                    selected_color = (
                        255,
                        255,
                        255,
                    )

                cv2.putText(
                    im0,
                    f"LAMP:{lamp_source}",
                    (
                        search_box[0],
                        max(
                            20,
                            search_box[1] - 5,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    selected_color,
                    1,
                )
            # =========================================================
            # 顯示本幀有效的 Track ID
            #
            # 主要前車：黃色粗框
            # 其他紅色 ROI 內車輛：白色框
            # =========================================================
            for tracked_candidate in lead_candidates:
                track_id = tracked_candidate["id"]

                x1, y1, x2, y2 = tracked_candidate["bbox"]

                contact_x, contact_y = tracked_candidate[
                    "bottom_center"
                ]

                is_lead = (
                    lead_vehicle_id is not None
                    and track_id == lead_vehicle_id
                )

                if is_lead:
                    track_color = (0, 255, 255)
                    track_thickness = 4
                    track_prefix = "LEAD "
                else:
                    track_color = (255, 255, 255)
                    track_thickness = 2
                    track_prefix = ""

                # 畫出追蹤框
                cv2.rectangle(
                    im0,
                    (x1, y1),
                    (x2, y2),
                    track_color,
                    track_thickness,
                )

                                # =====================================================
                # BBOX 上方資訊
                # =====================================================

                label_x = x1

                # BBOX 太靠近畫面頂端時，改顯示在框內
                if y1 >= 70:
                    lead_label_y = y1 - 10
                    emergency_label_y = y1 - 40
                else:
                    lead_label_y = y1 + 25
                    emergency_label_y = y1 + 55

                # Track／LEAD 資訊
                cv2.putText(
                    im0,
                    (
                        f"{track_prefix}"
                        f"ID:{track_id} "
                        f"{tracked_candidate['class_name']} "
                        f"{tracked_candidate['depth']}"
                    ),
                    (label_x, lead_label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    track_color,
                    2,
                    cv2.LINE_AA,
                )
                # 畫出實際用來判斷 ROI 的底部中心點
                cv2.circle(
                    im0,
                    (
                        contact_x,
                        contact_y,
                    ),
                    5,
                    track_color,
                    -1,
                )

        
            # =========================================================
            # 右上角最終展示資訊
            #
            # 7 km/h     ●
            # LEVEL 1
            # =========================================================

            dashboard_width = 285
            dashboard_height = 145
            dashboard_margin = 12

            dashboard_left = (
                im0.shape[1]
                - dashboard_width
                - dashboard_margin
            )

            dashboard_top = dashboard_margin

            dashboard_right = (
                im0.shape[1]
                - dashboard_margin
            )

            dashboard_bottom = (
                dashboard_top
                + dashboard_height
            )

            # 半透明黑色背景
            dashboard_overlay = im0.copy()

            cv2.rectangle(
                dashboard_overlay,
                (dashboard_left, dashboard_top),
                (dashboard_right, dashboard_bottom),
                (20, 20, 20),
                -1,
            )

            cv2.addWeighted(
                dashboard_overlay,
                0.68,
                im0,
                0.32,
                0,
                im0,
            )

            # 警示圓燈
            led_center = (
                im0.shape[1] - 45,
                42,
            )

            cv2.circle(
                im0,
                led_center,
                20,
                led_color,
                -1,
            )

            cv2.circle(
                im0,
                led_center,
                20,
                (255, 255, 255),
                2,
            )

            # 車速：資訊欄內左對齊
            cv2.putText(
                im0,
                f"{current_speed} km/h",
                (
                    dashboard_left + 18,
                    50,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.90,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # =====================================================
            # 右上角警示等級
            # 緊急煞車成立時，不顯示互相矛盾的 SAFE
            # =====================================================

            if current_speed == 0 or current_level == 0:
                if emergency_brake_confirmed:
                    level_display_text = None
                else:
                    level_display_text = "SAFE"
            else:
                level_display_text = (
                    f"LEVEL {current_level}"
                )

            # 有文字時才畫出來
            if level_display_text is not None:
                cv2.putText(
                    im0,
                    level_display_text,
                    (
                        dashboard_left + 18,
                        85,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            # LEAD 穩定煞車燈狀態
            if brake_stable_state == "ON":
                brake_display_text = "LEAD BRAKE: ON"
                brake_display_color = (0, 0, 255)

            elif brake_stable_state == "OFF":
                brake_display_text = "LEAD BRAKE: OFF"
                brake_display_color = (210, 210, 210)

            else:
                brake_display_text = "LEAD BRAKE: --"
                brake_display_color = (150, 150, 150)

            cv2.putText(
                im0,
                brake_display_text,
                (
                    dashboard_left + 18,
                    120,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                brake_display_color,
                2,
                cv2.LINE_AA,
            )
            # Stream results
            if view_img:
                display_frame = cv2.resize(
                    im0,
                    (1280, 720),
                    interpolation=cv2.INTER_AREA,
                )

                cv2.imshow(
                    str(p),
                    display_frame,
                )

                cv2.waitKey(1)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            _fps_out = vid_cap.get(cv2.CAP_PROP_FPS)  # ← 改這行，用 _fps_out 存輸出用的fps
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            _fps_out, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), _fps_out, (w, h))
                    vid_writer[i].write(im0)
            # 下一幀讀取下一張 road mask
            road_mask_frame_index += 1

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")
        # =========================================================
    # 影片處理結束，關閉 UART
    # =========================================================

    if stm32_uart is not None:
        try:
            stm32_uart.close()
            print("✅ STM32 UART 已關閉")

        except Exception:
            pass
    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to
            None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Examples:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument(
        "--brake-weights",
        type=str,
        default=ROOT / "weights/brake_yolov5n.pt",
        help="brake_on / brake_off classification model",
    )
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--road-masks-dir", type=str, required=True, help="TwinLiteNet+ 產生的逐幀 road mask 資料夾")
    parser.add_argument("--distance-calibration", type=str, required=True, help="pick_distance_y.py 產生的 distance_calibration.json")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Notes:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
   if __name__ == "__main__":
    print(
        "目前執行程式：",
        Path(__file__).resolve(),
    )

    opt = parse_opt()
    main(opt)
    ```
    """
    #check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)