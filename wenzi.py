import os
import cv2
from ultralytics import YOLO

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from paddleocr import PaddleOCR

MODEL_PATH   = r"D:\1\EzYOLO-main\runs\detect\runs\train\exp_2\weights\best.pt" # 使用原始字符串避免转义
#IMAGE_FOLDER = r"D:\1000\pliers_images" # 使用原始字符串
IMAGE_FOLDER = r"D:\1\pliers_images" # 使用原始字符串

CONF_THRESH  = 0.4
OUTPUT_LOG   = os.path.join(os.path.dirname(__file__), "inference_log.txt")
# ====================================================

# 初始化 OCR
ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)
model = YOLO(MODEL_PATH)

with open(OUTPUT_LOG, "w", encoding="utf-8") as log_file:
    def log_line(message=""):
        print(message)
        log_file.write(f"{message}\n")

    log_line("===== 配件文字识别（真实输出）=====")
    log_line()

    # 遍历图片
    for filename in os.listdir(IMAGE_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(IMAGE_FOLDER, filename)
        img = cv2.imread(img_path)
        if img is None:
            log_line("[WARN] 图片读取失败，已跳过")
            continue
        results = model(img_path, conf=CONF_THRESH, verbose=False)

        log_line(f"========== {filename} ==========")

        # 逐个检测框识别文字
        for idx, box in enumerate(results[0].boxes):
            class_id = int(box.cls.item()) if box.cls is not None else -1
            cls_name = results[0].names.get(class_id, str(class_id)) if class_id >= 0 else "unknown"

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                log_line(f"[{cls_name} {idx+1}] 检测框无效，已跳过")
                continue

            crop = img[y1:y2, x1:x2]

            # OCR 识别
            res = list(ocr.predict(crop))
            text = ""
            if res:
                first = res[0]
                if isinstance(first, dict) and first.get("rec_texts"):
                    text = " ".join(first["rec_texts"])
                elif hasattr(first, "rec_texts") and getattr(first, "rec_texts"):
                    text = " ".join(first.rec_texts)
                elif isinstance(first, list):
                    text = " ".join([line[1][0] for line in first if len(line) > 1 and len(line[1]) > 0])

            log_line(f"[{cls_name} {idx+1}] 识别文字：{text.strip()}")

        log_line()

print(f"识别结果已保存到：{OUTPUT_LOG}")