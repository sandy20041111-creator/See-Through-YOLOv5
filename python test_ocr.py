import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('test_frame.png')

# 往左移，抓數字+KM/H整個區域，再讓OCR只取數字
speed_region = img[600:650, 630:770]
cv2.imwrite('speed_region.png', speed_region)

# 前處理
gray = cv2.cvtColor(speed_region, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv2.imwrite('speed_binary.png', binary)

# OCR 這次允許讀字母，再用正則抽出數字前的那組數字
config = '--psm 7 --oem 3'
text = pytesseract.image_to_string(binary, config=config)

# 抓 KM/H 前面的數字
match = re.search(r'(\d+)\s*KM', text.upper())
speed = int(match.group(1)) if match else 0

print(f"OCR 原始輸出：{repr(text)}")
print(f"讀到的車速：{speed} km/h")