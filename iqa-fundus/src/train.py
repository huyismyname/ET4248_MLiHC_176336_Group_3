"""Baseline + mô hình đề xuất, có W&B tracking (Milestone 2/3).

Chạy: python -m src.train --config configs/default.yaml
Đây là khung tối thiểu — điền phần train loop khi vào M2/M3.
"""
from __future__ import annotations
import argparse
import yaml

from .seed import seed_everything
from .data import load_labels, make_group_split


def build_model(name: str, num_classes: int):
    import timm
    return timm.create_model(name, pretrained=True, num_classes=num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg["seed"])

    num_classes = 2 if cfg["data"]["binary"] else len(cfg["data"]["classes"])

    df = load_labels(cfg["data"]["labels_csv"], cfg["data"]["classes"], cfg["data"]["binary"])
    train_df, val_df = make_group_split(
        df, cfg["data"]["group_col"], cfg["data"]["n_splits"], cfg["data"]["val_fold"]
    )

    if cfg["wandb"]["enabled"]:
        import wandb
        wandb.init(project=cfg["wandb"]["project"], config=cfg)

    model = build_model(cfg["train"]["model"], num_classes)
    print(f"[train] model={cfg['train']['model']} classes={num_classes} "
          f"train={len(train_df)} val={len(val_df)}")

    # TODO (M2/M3): DataLoader (albumentations fit-on-train), class-weighted loss,
    #   early stopping trên val loss, cosine LR, checkpoint ra ./checkpoints (Drive).
    # TODO (M3): ablation (backbone / ordinal head / augmentation on-off).
    raise SystemExit("Khung train — dien loop o Milestone 2/3.")


if __name__ == "__main__":
    main()
