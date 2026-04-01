import sys
import cv2
import argparse
import subprocess
from pathlib import Path
import easyocr
import pytesseract

ROOT = Path(__file__).resolve().parent.parent
YOLOV5_DIR = ROOT / "yolov5"

# Weights đã lưu trên Google Drive sau khi train
DRIVE_WEIGHTS_DIR = Path("/content/drive/MyDrive/license_plate_weights")

# 4 thí nghiệm tương ứng với tên đã train
EXPERIMENTS = {
    "1": "exp1_yolov5s_aug",
    "2": "exp2_yolov5s_no_aug",
    "3": "exp3_yolov5m_aug",
    "4": "exp4_yolov5m_no_aug",
}


def get_weights_path(exp_id):
    """
    Lấy đường dẫn weights theo thứ tự ưu tiên:
      1. Từ Google Drive (nếu đã lưu)
      2. Từ runs/train/ (nếu vẫn còn trong session)
    """
    name = EXPERIMENTS[exp_id]

    # Ưu tiên Drive
    drive_path = DRIVE_WEIGHTS_DIR / f"{name}_best.pt"
    if drive_path.exists():
        print(f"💾 Load weights từ Drive: {drive_path}")
        return str(drive_path)

    # Fallback: trong session
    local_path = ROOT / "runs/train" / name / "weights/best.pt"
    if local_path.exists():
        print(f"📁 Load weights từ local: {local_path}")
        return str(local_path)

    raise FileNotFoundError(
        f"❌ Không tìm thấy weights cho thí nghiệm {exp_id} ({name}).\n"
        f"   Kiểm tra Drive: {drive_path}\n"
        f"   Hoặc local   : {local_path}"
    )


def run_detection(weights_path, source, conf, name):
    source = str((ROOT / source).resolve()) if not Path(source).is_absolute() else source

    cmd = [
        sys.executable,
        str((YOLOV5_DIR / "detect.py").resolve()),
        "--weights", weights_path,
        "--source",  source,
        "--save-txt",
        "--conf",    str(conf),
        "--project", str((ROOT / "runs/detect").resolve()),
        "--name",    name,
        "--exist-ok",
    ]

    subprocess.run(cmd, cwd=str(YOLOV5_DIR), check=True)


def get_latest_run(name):
    run_dir = ROOT / "runs/detect" / name
    if run_dir.exists():
        return run_dir
    # fallback: lấy run mới nhất
    detect_dir = ROOT / "runs/detect"
    runs = sorted(detect_dir.glob("exp*"), key=lambda x: x.stat().st_mtime)
    return runs[-1]


def ocr_on_crops(run_dir):
    reader = easyocr.Reader(['en'])

    img_paths = list(run_dir.glob("*.jpg")) + list(run_dir.glob("*.png"))
    if not img_paths:
        print("⚠️  Không tìm thấy ảnh kết quả detect.")
        return

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        label_file = run_dir / "labels" / (img_path.stem + ".txt")
        if not label_file.exists():
            print(f"⚠️  Không có label cho: {img_path.name}")
            continue

        print(f"\n🖼  Image: {img_path.name}")

        with open(label_file) as f:
            for i, line in enumerate(f, 1):
                cls, x, y, bw, bh = map(float, line.split())

                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # EasyOCR
                easy_result = reader.readtext(crop)
                easy_text = easy_result[0][-2] if easy_result else "(không đọc được)"

                # Tesseract
                tess_text = pytesseract.image_to_string(crop, config='--psm 7').strip()
                if not tess_text:
                    tess_text = "(không đọc được)"

                print(f"  Biển số #{i}:")
                print(f"    📌 EasyOCR   : {easy_text}")
                print(f"    📌 Tesseract : {tess_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict – Nhận diện biển số xe")

    parser.add_argument(
        "--exp", default="1", choices=["1", "2", "3", "4"],
        help="Chọn thí nghiệm: 1=yolov5s+aug | 2=yolov5s | 3=yolov5m+aug | 4=yolov5m"
    )
    parser.add_argument("--source",  default="data/test/images", help="Ảnh hoặc video đầu vào")
    parser.add_argument("--conf",    type=float, default=0.25,   help="Ngưỡng confidence")

    args = parser.parse_args()

    exp_name = EXPERIMENTS[args.exp]
    print(f"\n{'='*60}")
    print(f"🔬 Thí nghiệm : {args.exp} – {exp_name}")
    print(f"{'='*60}\n")

    # Lấy weights
    weights_path = get_weights_path(args.exp)

    # Detect
    print("🚀 Đang chạy detection...")
    run_detection(weights_path, args.source, args.conf, name=exp_name)

    # OCR
    print("\n🔍 Đang chạy OCR...")
    run_dir = get_latest_run(exp_name)
    ocr_on_crops(run_dir)

    print(f"\n✅ Xong! Kết quả lưu tại: {run_dir}")