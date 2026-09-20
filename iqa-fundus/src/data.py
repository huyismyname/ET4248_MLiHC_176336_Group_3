"""Dữ liệu EyeQ + chia tập CHỐNG RÒ RỈ ở mức bệnh nhân/mắt.

Điểm chấm cốt lõi (Milestone 2):
  - GroupKFold theo `patient_id` (KHÔNG train_test_split ngẫu nhiên).
  - `assert` chứng minh Train/Test độc lập 100% theo group.
  - Transform/normalize fit trên Train, chỉ apply sang Val/Test.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


# ----------------------------- Split -----------------------------
def make_group_split(df: pd.DataFrame, group_col: str, n_splits: int, val_fold: int):
    """Trả về (train_df, val_df) tách theo group, kèm assert chống rò rỉ."""
    if group_col not in df.columns:
        raise KeyError(
            f"Thiếu cột group '{group_col}'. EyeQ phải có patient_id/eye id để chia "
            f"patient-level. Nếu không có, PHẢI nêu ở Hạn chế (rủi ro rò rỉ)."
        )
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values
    splits = list(gkf.split(df, groups=groups))
    tr_idx, va_idx = splits[val_fold]
    train_df, val_df = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()

    # --- Bằng chứng chống rò rỉ (đưa vào PR Milestone 2) ---
    overlap = set(train_df[group_col]) & set(val_df[group_col])
    assert not overlap, f"DATA LEAKAGE: {len(overlap)} group ở cả Train và Val!"
    print(f"[split] train={len(train_df)}  val={len(val_df)}  "
          f"groups: train={train_df[group_col].nunique()} val={val_df[group_col].nunique()} "
          f"overlap=0  ✔")
    return train_df, val_df


def load_labels(labels_csv: str, classes, binary: bool = False) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if binary:
        # Gộp Good+Usable -> gradable(0), Reject -> ungradable(1). Chỉ là ánh xạ nhãn có sẵn.
        df["label"] = (df["quality"] == "Reject").astype(int)
    else:
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        df["label"] = df["quality"].map(cls_to_idx)
    if df["label"].isna().any():
        raise ValueError("Có nhãn 'quality' không khớp danh sách classes trong config.")
    return df


# --------------------------- Torch Dataset ---------------------------
try:
    import cv2
    from torch.utils.data import Dataset

    class EyeQDataset(Dataset):
        """Đọc ảnh + nhãn. `transform` (albumentations) fit trên train, apply sang val/test."""
        def __init__(self, df: pd.DataFrame, root: str, transform=None):
            self.df = df.reset_index(drop=True)
            self.root = root
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i]
            img = cv2.cvtColor(cv2.imread(os.path.join(self.root, row["image"])), cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=img)["image"]
            return img, int(row["label"])
except ImportError:
    EyeQDataset = None  # môi trường chưa cài torch/cv2 (vd. lúc chạy EDA nhẹ)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
