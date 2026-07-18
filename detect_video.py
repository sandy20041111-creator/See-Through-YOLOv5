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

import sys
from xml.parsers.expat import model
import torch
import torch.nn as nn

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

import argparse
import csv
import os
import platform
import sys
from glob import glob, has_magic
from pathlib import Path

import torch
import numpy as np
import cv2

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


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
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
    if os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    else:
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    print("✅ OCR 車速讀取模式啟用")
 
    source = str(source)

    import threading

    ocr_result = {"speed": 0}
    ocr_lock = threading.Lock()

    def ocr_worker(frame_img, h, w):
        try:
            search_region = frame_img[int(h * 0.75):h, 0:w]
            gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            config = '--psm 12 --oem 3'
            ocr_text = pytesseract.image_to_string(binary, config=config)
            match = re.search(r'(\d{1,3})\s*KM', ocr_text.upper())
            if match:
                speed_val = int(match.group(1))
                if 0 <= speed_val <= 120:
                    with ocr_lock:
                        ocr_result["speed"] = speed_val
        except Exception as e:
            print(f"OCR錯誤: {e}")
            pass

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

    if str(weights[0]).endswith('.pth'):
        from torch2trt import TRTModule
        model = TRTModule()
        model.load_state_dict(torch.load(weights[0]))
        model = model.cuda().half()
        stride, names, pt = 32, [f'class{i}' for i in range(85)], False
        # 設定 TRTModule 缺少的屬性
        model.fp16 = True
        model.xml = False
        model.triton = False
        model.device = device
    else:
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
    
    imgsz = check_img_size(imgsz, s=stride)  # check image size

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
    if not str(weights[0]).endswith('.pth'):
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    danger_hold_frames = 0  # 警報維持計數器
    current_speed = 0
    # 預先同步跑一次 OCR，避免開始時車速為 0
    try:
        import cv2 as _cv2_pre
        print(f"預先OCR source: {source}")
        _cap_pre = _cv2_pre.VideoCapture(source)
        _cap_pre.set(cv2.CAP_PROP_POS_FRAMES, 30)  # ← 加這行，跳到第30幀
        _ret, _frame_pre = _cap_pre.read()
        print(f"預先OCR _ret: {_ret}")
        _cap_pre.release()
        if _ret:
            _h, _w = _frame_pre.shape[:2]
            print(f"幀尺寸: {_h}x{_w}")  # ← 加這行
            _region = _frame_pre[int(_h * 0.75):_h, 0:_w]
            _gray = cv2.cvtColor(_region, cv2.COLOR_BGR2GRAY)
            _gray = cv2.resize(_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, _binary = cv2.threshold(_gray, 150, 255, cv2.THRESH_BINARY)
            _text = pytesseract.image_to_string(_binary, config='--psm 12 --oem 3')
            print(f"預先OCR文字: {repr(_text.strip()[:80])}")
            _match = re.search(r'(\d{1,3})\s*KM', _text.upper())
            if _match:
                speed_val = int(_match.group(1))
                if 0 <= speed_val <= 120:
                    current_speed = speed_val
                    ocr_result["speed"] = speed_val
                    print(f"✅ 初始車速: {current_speed} km/h")
    except Exception as e:
        print(f"預先OCR錯誤: {e}")
    current_level = 0
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if (model.fp16 if hasattr(model, 'fp16') else half) else im.float()
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
                if str(weights[0]).endswith('.pth'):
                    pred = model(im)
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
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness)

            # Write results
            danger_detected_now = False 
            danger_level_now = 0
            vru_counter = {"person": 0, "motorcycle": 0, "bicycle": 0}
            
            h, w, _ = im0.shape

            # OCR 讀取車速（每10幀讀一次，降低運算量）
            print(f"frame: {frame}")
            if (int(frame) % 10 == 0 or frame == 1):
                if not hasattr(run, 'ocr_thread') or not run.ocr_thread.is_alive():
                    run.ocr_thread = threading.Thread(target=ocr_worker, args=(im0.copy(), h, w))
                    run.ocr_thread.daemon = True
                    run.ocr_thread.start()

            with ocr_lock:
                current_speed = ocr_result["speed"]
                if current_speed > 0:
                    print(f"車速更新: {current_speed} km/h")
 
            # 依車速決定哪些深度層要啟用
            if current_speed == 0:
                active_depths = []                        # 靜止不預警
            elif 1 <= current_speed <= 30:
                active_depths = ["near"]                  # 低速：只開近端
            elif 31 <= current_speed <= 50:
                active_depths = ["near", "mid"]           # 中速：近端＋中端
            else:
                active_depths = ["near", "mid", "far"]    # 高速：全開

            # =================================================================
            # 💡 新邏輯：先定義一個「大梯形」的左右邊界斜線方程式，
            # 再用水平線在不同高度切出近/中/遠三層，確保所有層級的
            # 左右邊界都精確落在同一條斜線上，不會有轉折/上翹的問題
            # =================================================================

            # 大梯形的四個關鍵控制點（最寬的底部 與 最窄的頂部）
            BOTTOM_Y = 1.00   # 畫面最底部
            TOP_Y    = 0.52   # 整個ROI系統的最頂端（消失點附近）→ 比上次收回一點，避免拉太遠上翹

            # 中央車道：底部寬、頂部窄 → 再放寬一點
            CENTER_BOTTOM_L, CENTER_BOTTOM_R = 0.20, 0.80
            CENTER_TOP_L,    CENTER_TOP_R    = 0.45, 0.55

            # 左側相鄰車道：底部寬、頂部窄（接在中央車道左邊）
            LEFT_BOTTOM_L, LEFT_BOTTOM_R = -0.12, 0.20
            LEFT_TOP_L,    LEFT_TOP_R    = 0.37, 0.45

            # 右側相鄰車道：底部寬、頂部窄（接在中央車道右邊）
            RIGHT_BOTTOM_L, RIGHT_BOTTOM_R = 0.80, 1.12
            RIGHT_TOP_L,    RIGHT_TOP_R    = 0.55, 0.63

            # 三層的水平切割線（高度比例），數字越小越接近消失點
            # 近端再拉長（1.00→0.68，原本只到0.75）
            LAYER_Y = {
                "near": (1.00, 0.68),   # 近端：底部 -> 0.68h（再拉長）
                "mid":  (0.68, 0.58),   # 中端：0.68h -> 0.58h
                "far":  (0.58, 0.52),   # 遠端：0.58h -> 0.52h
            }

            def lerp(a, b, t):
                """線性插值：在 a 到 b 之間，依比例 t 取值"""
                return a + (b - a) * t

            def x_on_edge(bottom_x, top_x, y_ratio):
                """
                給定一條從(bottom_x, BOTTOM_Y)到(top_x, TOP_Y)的斜線，
                求這條線在某個 y_ratio 高度時的 x 座標
                """
                t = (BOTTOM_Y - y_ratio) / (BOTTOM_Y - TOP_Y)
                return lerp(bottom_x, top_x, t)

            def make_layer_trapezoid(bottom_l, bottom_r, top_l, top_r, y_bottom, y_top):
                """
                依「左右兩條邊界斜線」與「指定的上下高度」，算出該層梯形的四個頂點。
                這樣同一個方向(中/左/右)的三層，左右邊界永遠落在同一條斜線上。
                """
                left_bottom_x  = x_on_edge(bottom_l, top_l, y_bottom)
                left_top_x     = x_on_edge(bottom_l, top_l, y_top)
                right_bottom_x = x_on_edge(bottom_r, top_r, y_bottom)
                right_top_x    = x_on_edge(bottom_r, top_r, y_top)

                pts = np.array([
                    [int(left_bottom_x * w),  int(y_bottom * h)],  # 左下
                    [int(left_top_x * w),     int(y_top * h)],     # 左上
                    [int(right_top_x * w),    int(y_top * h)],     # 右上
                    [int(right_bottom_x * w), int(y_bottom * h)],  # 右下
                ], np.int32)
                return pts.reshape((-1, 1, 2))

            # ---- 依三層高度，分別產生中央/左側/右側的梯形 ----
            roi_zones = {}
            for depth, (y_b, y_t) in LAYER_Y.items():
                roi_zones[("center", depth)] = make_layer_trapezoid(
                    CENTER_BOTTOM_L, CENTER_BOTTOM_R, CENTER_TOP_L, CENTER_TOP_R, y_b, y_t
                )
                roi_zones[("left", depth)] = make_layer_trapezoid(
                    LEFT_BOTTOM_L, LEFT_BOTTOM_R, LEFT_TOP_L, LEFT_TOP_R, y_b, y_t
                )
                roi_zones[("right", depth)] = make_layer_trapezoid(
                    RIGHT_BOTTOM_L, RIGHT_BOTTOM_R, RIGHT_TOP_L, RIGHT_TOP_R, y_b, y_t
                )

            # 🎨 先複製一份乾淨的 im0 用來製作科技感陰影
            overlay = im0.copy()

            pos_colors = {
                "center": (0, 0, 255),     # 紅色 (BGR)
                "left":   (0, 140, 255),   # 橘色
                "right":  (0, 140, 255),   # 橘色
            }
            depth_alpha = {
                "near": 0.65,
                "mid":  0.45,
                "far":  0.30,
            }

            for (pos, depth), pts in roi_zones.items():
                if depth not in active_depths:
                    continue  # 這層沒啟用，跳過不畫
 
                color = pos_colors[pos]
                alpha = depth_alpha[depth]
                temp = overlay.copy()
                cv2.fillPoly(temp, [pts], color)
                cv2.addWeighted(temp, alpha, overlay, 1 - alpha, 0, overlay)
                cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=1)


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

                    # =================================================================
                    # 🔥 畫框邏輯
                    # =================================================================
                    display_label = None if hide_labels else (final_class_name if hide_conf else f"{final_class_name} {conf:.2f}")
                    annotator.box_label(xyxy, display_label, color=bbox_color) 
                    
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

                    vulnerable_road_users = ['person', 'motorcycle', 'bicycle']
                    if final_class_name in vulnerable_road_users:
                        bottom_center_x = int(x_center * w)
                        bottom_center_y = int(y_bottom * h)

                        for (pos, depth), zone_pts in roi_zones.items():
                            if depth not in active_depths:
                                continue
                            if cv2.pointPolygonTest(zone_pts, (bottom_center_x, bottom_center_y), False) >= 0:
                                if pos == "center" and depth == "near":
                                    detected_level = 3
                                elif pos == "center" and depth == "mid" and current_speed >= 51:
                                    detected_level = 3
                                elif pos == "left" or pos == "right":
                                    if depth == "near" and current_speed >= 51:
                                        detected_level = 3
                                    elif depth == "near" and current_speed <= 50:
                                        detected_level = 2
                                    elif depth == "mid":
                                        detected_level = 2
                                    elif depth == "far":
                                        detected_level = 1
                                    else:
                                        detected_level = 1
                                elif pos == "center" and depth == "mid":
                                    detected_level = 2
                                elif pos == "center" and depth == "far":
                                    detected_level = 2
                                else:
                                    detected_level = 1
                                if detected_level > danger_level_now:
                                    danger_level_now = detected_level
                                vru_counter[final_class_name] += 1 

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

            # 依等級決定燈號顏色與 UART 編碼
            if current_speed == 0:
                led_color = (0, 255, 0)       # 綠色：靜止
                uart_code = "AA 00"
            elif current_level == 0:
                led_color = (0, 255, 0)       # 綠色：無危險
                uart_code = "AA 10"
            elif current_level == 1:
                led_color = (0, 255, 255)     # 黃色 (BGR)
                uart_code = "AA 11"
            elif current_level == 2:
                led_color = (0, 140, 255)     # 橘色 (BGR)
                uart_code = "AA 12"
            else:
                led_color = (0, 0, 255)       # 紅色 (BGR)
                uart_code = "AA 13"

            # 終端機印出 UART 編碼
            # 整理 VRU 統計字串
            vru_summary = " | " + " ".join(
                f"{v} {k}{'s' if v > 1 else ''}"
                for k, v in vru_counter.items() if v > 0
            ) if any(v > 0 for v in vru_counter.values()) else ""

            print(f"UART: {uart_code}  |  Level: {current_level}  |  Speed: {current_speed} km/h{vru_summary}")
            print(f"推論時間: {dt[1].dt * 1E3:.1f}ms  |  FPS: {1/(dt[1].dt + 1e-9):.1f}")

            # HUD 疊合 overlay
            im0 = annotator.result()
            cv2.addWeighted(overlay, 0.25, im0, 0.75, 0, im0)

            # 畫面左上角：圓形燈號
            led_center = (40, 40)
            cv2.circle(im0, led_center, 22, led_color, -1)           # 實心圓
            cv2.circle(im0, led_center, 22, (255, 255, 255), 2)      # 白色外框

            # 畫面左上角：車速文字（燈號右側）
            cv2.putText(im0, f"{current_speed} km/h",
                        (75, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2)

            # Stream results
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

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

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

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
        opt = parse_opt()
        main(opt)
    ```
    """
    #check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
