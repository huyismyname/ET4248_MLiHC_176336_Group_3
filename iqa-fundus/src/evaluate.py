"""Hệ thống đa độ đo + bootstrap 95% CI + confusion matrix (Milestone 4).

Cung cấp sẵn:
  - metrics_multiclass(): Sensitivity/Specificity theo lớp, Macro-F1, (PR-AUC nếu có score)
  - bootstrap_ci(): khoảng tin cậy 95% bằng resampling
  - confusion(): ma trận nhầm lẫn (nhấn mạnh ranh giới Usable<->Reject)
"""
from __future__ import annotations
import argparse
import numpy as np
import yaml
from sklearn.metrics import (
    f1_score, recall_score, confusion_matrix, average_precision_score,
)


def specificity_per_class(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    spec = []
    total = cm.sum()
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return np.array(spec)


def metrics_multiclass(y_true, y_pred, n_classes):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "sensitivity_per_class": recall_score(
            y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0
        ).tolist(),
        "specificity_per_class": specificity_per_class(y_true, y_pred, n_classes).tolist(),
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000, ci=0.95, seed=42):
    """CI 95% cho một hàm metric scalar (vd. macro-F1)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true); stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 + ci) / 2 * 100)
    return float(np.mean(stats)), float(lo), float(hi)


def confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    print("Confusion matrix (hang=that, cot=du doan):")
    print("      " + "  ".join(f"{c[:6]:>6}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[:6]:>6} " + "  ".join(f"{v:6d}" for v in cm[i]))
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    classes = ["gradable", "ungradable"] if cfg["data"]["binary"] else cfg["data"]["classes"]

    # DEMO số liệu giả để kiểm tra pipeline metric (thay bằng dự đoán thật ở M4).
    rng = np.random.default_rng(cfg["seed"])
    y_true = rng.integers(0, len(classes), 400)
    y_pred = y_true.copy()
    flip = rng.random(400) < 0.2
    y_pred[flip] = rng.integers(0, len(classes), flip.sum())

    print(metrics_multiclass(y_true, y_pred, len(classes)))
    mean, lo, hi = bootstrap_ci(
        y_true, y_pred, lambda a, b: f1_score(a, b, average="macro"),
        n_boot=cfg["eval"]["n_bootstrap"], ci=cfg["eval"]["ci"], seed=cfg["seed"],
    )
    print(f"Macro-F1 = {mean:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    confusion(y_true, y_pred, classes)


if __name__ == "__main__":
    main()
