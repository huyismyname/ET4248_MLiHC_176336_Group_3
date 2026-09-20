"""Tải/chuẩn bị dữ liệu EyeQ về thư mục ./data (KHÔNG commit dữ liệu lên Git).

EyeQ: https://github.com/HzFu/EyeQ  (nhãn Good/Usable/Reject dựa trên EyePACS).
Ảnh gốc EyePACS lấy từ Kaggle (cần tài khoản). Script này chỉ tạo khung thư mục
và kiểm tra file nhãn — điền bước tải cụ thể theo nguồn nhóm dùng (Kaggle API/Drive).
"""
from __future__ import annotations
import argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="./data")
    args = ap.parse_args()
    root = os.path.join(args.dest, "eyeq")
    os.makedirs(root, exist_ok=True)
    print(f"[data] thu muc dich: {root}")
    print("[data] Cac buoc:")
    print("  1. Lay nhan EyeQ tu https://github.com/HzFu/EyeQ (Label_EyeQ_*.csv).")
    print("  2. Tai anh EyePACS tuong ung (Kaggle) hoac dung ban da co tren Drive.")
    print("  3. Tao ./data/eyeq/labels.csv voi cot: image, quality, patient_id.")
    print("  4. KHONG commit anh -> da nam trong .gitignore.")

    labels = os.path.join(root, "labels.csv")
    if not os.path.exists(labels):
        with open(labels, "w") as f:
            f.write("image,quality,patient_id\n")
            f.write("# vi du: 10_left.jpeg,Good,10\n")
        print(f"[data] da tao khung {labels} (dien du lieu that vao).")
    else:
        print(f"[data] da co {labels}.")


if __name__ == "__main__":
    main()
