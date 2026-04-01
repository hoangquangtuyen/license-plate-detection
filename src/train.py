import sys
import shutil
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLOV5_DIR = ROOT / "yolov5"
CONFIG = (ROOT / "configs" / "license_plate.yaml").resolve()

# ── Google Drive ──────────────────────────────────────────────────────────────
# Thư mục lưu weights trên Drive sau khi train xong.
# Thay đổi đường dẫn nếu muốn lưu vào thư mục khác.
DRIVE_SAVE_DIR = Path("/content/drive/MyDrive/license_plate_weights")


def mount_drive():
    """Mount Google Drive (chỉ chạy được trên Colab)."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("✅ Đã mount Google Drive\n")
        return True
    except ImportError:
        print("⚠️  Không phải môi trường Colab, bỏ qua bước mount Drive.\n")
        return False


def save_weights_to_drive(exp_names):
    """Sao chép file best.pt của từng thí nghiệm lên Google Drive."""
    if not Path("/content/drive").exists():
        print("⚠️  Drive chưa được mount, bỏ qua bước lưu weights.\n")
        return

    DRIVE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Đang lưu weights lên Drive: {DRIVE_SAVE_DIR}")

    for name in exp_names:
        best = ROOT / "runs/train" / name / "weights/best.pt"
        if best.exists():
            dst = DRIVE_SAVE_DIR / f"{name}_best.pt"
            shutil.copy(best, dst)
            print(f"   ✅ {name}_best.pt")
        else:
            print(f"   ❌ Không tìm thấy weights: {best}")


# ── Train ─────────────────────────────────────────────────────────────────────

def train(epochs, batch, img_size, weights, augment, name, device):
    if not YOLOV5_DIR.exists():
        raise FileNotFoundError("❌ Chưa có yolov5 folder")

    cmd = [
        sys.executable,
        str((YOLOV5_DIR / "train.py").resolve()),

        "--img",    str(img_size),
        "--batch",  str(batch),
        "--epochs", str(epochs),

        "--data",    str(CONFIG),
        "--weights", weights,

        "--project", str((ROOT / "runs/train").resolve()),
        "--name",    name,

        "--device",  device,
        "--workers", "4",
        # "--cache",
        "--exist-ok",
    ]

    if not augment:
        hyp = (YOLOV5_DIR / "data/hyps/hyp.scratch-low.yaml").resolve()
        cmd += ["--hyp", str(hyp)]

    print(f"\n{'='*60}")
    print(f"🚀 Training: {name}")
    print(f"   Weights : {weights}")
    print(f"   Augment : {'✅ Có' if augment else '❌ Không'}")
    print(f"   Epochs  : {epochs}  |  Batch: {batch}  |  Img: {img_size}")
    print(f"{'='*60}\n")

    subprocess.run(cmd, check=True, cwd=str(YOLOV5_DIR))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device",   type=str, default="0")
    args = parser.parse_args()

    # Mount Drive trước khi train để sẵn sàng lưu sau
    mount_drive()

    # 4 thí nghiệm theo đề cương: so sánh YOLOv5s/m x Augmentation/Không
    experiments = [
        ("yolov5s.pt", True,  "exp1_yolov5s_aug"),
        ("yolov5s.pt", False, "exp2_yolov5s_no_aug"),
        ("yolov5m.pt", True,  "exp3_yolov5m_aug"),
        ("yolov5m.pt", False, "exp4_yolov5m_no_aug"),
    ]

    success_names = []
    results = []

    for idx, (weights, augment, name) in enumerate(experiments, 1):
        print(f"\n[{idx}/4] Bắt đầu thí nghiệm: {name}")
        try:
            train(
                epochs=args.epochs,
                batch=args.batch,
                img_size=args.img_size,
                weights=weights,
                augment=augment,
                name=name,
                device=args.device,
            )
            results.append((name, "✅ Thành công"))
            success_names.append(name)
        except subprocess.CalledProcessError as e:
            results.append((name, f"❌ Lỗi: {e}"))
            print(f"⚠️  Thí nghiệm {name} thất bại, tiếp tục thí nghiệm tiếp theo...\n")

    # Lưu toàn bộ weights lên Drive sau khi train xong
    save_weights_to_drive(success_names)

    # Tổng kết
    print(f"\n{'='*60}")
    print("📊 TỔNG KẾT 4 THÍ NGHIỆM")
    print(f"{'='*60}")
    for name, status in results:
        print(f"  {status}  {name}")
    print(f"\n💡 Kết quả train : {ROOT / 'runs/train'}")
    print(f"💾 Weights trên Drive: {DRIVE_SAVE_DIR}")
    print(f"{'='*60}\n")