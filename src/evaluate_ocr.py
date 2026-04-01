import cv2
import easyocr
import pytesseract
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# File ground truth: mỗi dòng gồm  image_name,plate_text
# Ví dụ: 00001.jpg,51F12345
GT_FILE = ROOT / "data/test/labels.txt"


def normalize(text):
    """Chuẩn hóa chuỗi trước khi so sánh: bỏ khoảng trắng, in hoa."""
    return text.strip().upper().replace(" ", "").replace("-", "")


def load_gt():
    gt = {}
    if not GT_FILE.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy file ground truth: {GT_FILE}")
    with open(GT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, text = line.split(",", maxsplit=1)
            gt[name.strip()] = normalize(text)
    return gt


def compare_ocr(image_dir):
    reader = easyocr.Reader(['en'])
    gt = load_gt()

    easy_correct = 0
    tess_correct = 0
    total = 0

    img_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))

    if not img_paths:
        print(f"⚠️  Không tìm thấy ảnh trong: {image_dir}")
        return

    for img_path in img_paths:
        true_text = gt.get(img_path.name)
        if true_text is None:
            print(f"⚠️  Không có ground truth cho: {img_path.name}, bỏ qua.")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Không đọc được ảnh: {img_path.name}, bỏ qua.")
            continue

        # EasyOCR
        easy_result = reader.readtext(img)
        easy_text = normalize(easy_result[0][-2]) if easy_result else ""

        # Tesseract
        tess_text = normalize(pytesseract.image_to_string(img, config='--psm 7'))

        if easy_text == true_text:
            easy_correct += 1
        if tess_text == true_text:
            tess_correct += 1

        total += 1

        print(f"  {img_path.name} | GT: {true_text:12} | Easy: {easy_text:12} | Tess: {tess_text}")

    if total == 0:
        print("⚠️  Không có ảnh nào khớp với ground truth.")
        return

    print(f"\n{'='*50}")
    print(f"📊 OCR EVALUATION  ({total} ảnh)")
    print(f"{'='*50}")
    print(f"  EasyOCR accuracy   : {easy_correct/total:.4f}  ({easy_correct}/{total})")
    print(f"  Tesseract accuracy : {tess_correct/total:.4f}  ({tess_correct}/{total})")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    compare_ocr(ROOT / "data/test/images")