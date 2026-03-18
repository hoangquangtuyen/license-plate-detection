import sys
import cv2
import argparse
import subprocess
from pathlib import Path
import easyocr
import pytesseract

ROOT = Path(__file__).resolve().parent.parent
YOLOV5_DIR = ROOT / "yolov5"


def run_detection(weights, source):
    weights = str((ROOT / weights).resolve())
    source = str((ROOT / source).resolve())

    cmd = [
        sys.executable,
        str((YOLOV5_DIR / "detect.py").resolve()),
        "--weights", weights,
        "--source", source,
        "--save-txt",
        "--conf", "0.25",
        "--project", str((ROOT / "runs/detect").resolve()),
        "--exist-ok"
    ]

    subprocess.run(cmd, cwd=str(YOLOV5_DIR), check=True)


def get_latest_run():
    detect_dir = ROOT / "runs/detect"
    runs = sorted(detect_dir.glob("exp*"), key=lambda x: x.stat().st_mtime)
    return runs[-1]


def ocr_on_crops(run_dir):
    reader = easyocr.Reader(['en'])

    for img_path in run_dir.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        label_file = run_dir / "labels" / (img_path.stem + ".txt")
        if not label_file.exists():
            continue

        print(f"\n🖼 Image: {img_path.name}")

        with open(label_file) as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.split())

                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                crop = img[y1:y2, x1:x2]

                # EasyOCR
                easy_text = ""
                result = reader.readtext(crop)
                if result:
                    easy_text = result[0][-2]

                # Tesseract
                tess_text = pytesseract.image_to_string(crop, config='--psm 7')

                print(f"📌 EasyOCR    : {easy_text}")
                print(f"📌 Tesseract  : {tess_text.strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", default="runs/train/yolov5s_aug_ep100/weights/best.pt")
    parser.add_argument("--source", default="data/test/images")

    args = parser.parse_args()

    print("🚀 Running detection...")
    run_detection(args.weights, args.source)

    print("🔍 Running OCR...")
    run_dir = get_latest_run()
    ocr_on_crops(run_dir)