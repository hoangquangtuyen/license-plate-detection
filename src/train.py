import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLOV5_DIR = ROOT / "yolov5"
CONFIG = (ROOT / "configs" / "license_plate.yaml").resolve()


def train(epochs, batch, img_size, weights, augment, name, device):

    if not YOLOV5_DIR.exists():
        raise FileNotFoundError("❌ Chưa có yolov5 folder")

    cmd = [
        sys.executable,
        str((YOLOV5_DIR / "train.py").resolve()),

        "--img", str(img_size),
        "--batch", str(batch),
        "--epochs", str(epochs),

        "--data", str(CONFIG),
        "--weights", weights,

        "--project", str((ROOT / "runs/train").resolve()),
        "--name", name,

        "--device", device,
        "--workers", "4",
        # "--cache",
        "--exist-ok",
    ]

    if not augment:
        hyp = (YOLOV5_DIR / "data/hyps/hyp.scratch-low.yaml").resolve()
        cmd += ["--hyp", str(hyp)]

    print(f"\n🚀 Training: {name}")
    subprocess.run(cmd, check=True, cwd=str(YOLOV5_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--weights", type=str, default="yolov5s.pt")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="")

    args = parser.parse_args()

    if not args.name:
        aug = "no_aug" if args.no_augment else "aug"
        args.name = f"{args.weights}_{aug}_ep{args.epochs}"

    train(
        args.epochs,
        args.batch,
        args.img_size,
        args.weights,
        not args.no_augment,
        args.name,
        args.device
    )