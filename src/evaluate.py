import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs/train"


def load(run):
    file = RUNS / run / "results.csv"

    if not file.exists():
        print(f"❌ Không tìm thấy: {file}")
        return None

    return pd.read_csv(file)


def get_metrics(df):
    col_p = "metrics/precision(B)"
    col_r = "metrics/recall(B)"
    col_map = "metrics/mAP_0.5(B)"

    best = df.loc[df[col_map].idxmax()]

    precision = best[col_p]
    recall = best[col_r]
    map50 = best[col_map]

    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1, map50


def compare(runs):
    print("\n📊 EVALUATION\n")

    results = []

    for run_name in runs:
        df = load(run_name)
        if df is None:
            continue

        try:
            p, rcl, f1, map50 = get_metrics(df)

            results.append({
                "Model": run_name,
                "Precision": round(p, 4),
                "Recall": round(rcl, 4),
                "F1-score": round(f1, 4),
                "mAP@0.5": round(map50, 4)
            })

        except Exception as e:
            print(f"❌ Lỗi {run_name}: {e}")

    if results:
        df_result = pd.DataFrame(results)
        print(df_result.to_string(index=False))


if __name__ == "__main__":
    runs = [
        "yolov5s_aug_ep100",
        "yolov5s_no_aug_ep100",
        "yolov5m_aug_ep100",
        "yolov5m_no_aug_ep100"
    ]

    compare(runs)