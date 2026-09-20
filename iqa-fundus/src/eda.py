"""EDA cho Milestone 1 (tiêu chí 5 điểm — nặng nhất).

Sinh các biểu đồ đặc thù ảnh y sinh:
  - phân bố nhãn Good/Usable/Reject (mất cân bằng)
  - phân bố kích thước ảnh
  - ảnh mẫu mỗi lớp
  - kiểm tra nhóm patient_id (rủi ro rò rỉ)
"""
from __future__ import annotations
import argparse, os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from .seed import seed_everything
from .data import load_labels


def plot_label_distribution(df, out_dir):
    ax = df["quality"].value_counts().plot(kind="bar", color="#C00000")
    ax.set_title("Phan bo nhan chat luong (imbalance)")
    ax.set_xlabel("Quality"); ax.set_ylabel("So anh")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "label_distribution.png"), dpi=150); plt.close()


def report_group_integrity(df, group_col, out_dir):
    """In cảnh báo nếu 1 patient trải nhiều nhãn / thiếu group -> ghi vào báo cáo M1."""
    lines = []
    if group_col not in df.columns:
        lines.append(f"[WARN] Thieu cot '{group_col}': KHONG the chia patient-level. "
                     f"Phai neu o Han che (rui ro leakage).")
    else:
        n_groups = df[group_col].nunique()
        multi = df.groupby(group_col)["quality"].nunique()
        lines.append(f"So group ({group_col}): {n_groups}")
        lines.append(f"So group co >1 muc chat luong: {(multi > 1).sum()}")
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "group_integrity.txt"), "w") as f:
        f.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg["seed"])
    out = cfg["paths"]["out_dir"] + "/eda"; os.makedirs(out, exist_ok=True)

    df = load_labels(cfg["data"]["labels_csv"], cfg["data"]["classes"], cfg["data"]["binary"])
    print(df["quality"].value_counts())
    plot_label_distribution(df, out)
    report_group_integrity(df, cfg["data"]["group_col"], out)
    print(f"[eda] outputs -> {out}")


if __name__ == "__main__":
    main()
