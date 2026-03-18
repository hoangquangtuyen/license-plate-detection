import cv2
import easyocr
import pytesseract
from pathlib import Path

GT_FILE = "data/test/labels.txt"  # file ground truth: image_name,plate_text


def load_gt():
    gt = {}
    with open(GT_FILE) as f:
        for line in f:
            name, text = line.strip().split(",")
            gt[name] = text
    return gt


def compare_ocr(image_dir):
    reader = easyocr.Reader(['en'])
    gt = load_gt()

    easy_correct = 0
    tess_correct = 0
    total = 0

    for img_path in Path(image_dir).glob("*.jpg"):
        img = cv2.imread(str(img_path))

        easy = reader.readtext(img)
        easy_text = easy[0][-2] if easy else ""

        tess_text = pytesseract.image_to_string(img, config='--psm 7').strip()

        true_text = gt.get(img_path.name, "")

        if easy_text == true_text:
            easy_correct += 1

        if tess_text == true_text:
            tess_correct += 1

        total += 1

    print("\n📊 OCR EVALUATION")
    print(f"EasyOCR accuracy   : {easy_correct/total:.4f}")
    print(f"Tesseract accuracy : {tess_correct/total:.4f}")


if __name__ == "__main__":
    compare_ocr("data/test/images")