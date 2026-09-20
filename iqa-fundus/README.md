# Fundus Image Quality Assessment (IQA) — ET4248 Đề tài 2.7

Cổng đánh giá chất lượng ảnh đáy mắt (Good / Usable / Reject) làm bước tiền chẩn đoán
cho pipeline sàng lọc võng mạc. Dữ liệu: **EyeQ** (28.792 ảnh, nhãn có sẵn) + kiểm chứng
ngoại **DRIMDB**.

> Học phần: Học máy trong Y tế (ET4248) · Milestone-based, chấm liên tục qua Git.

---

## 0. Nguyên tắc bất di bất dịch (theo Rubric)

- **Zero data leakage**: chia tập ở mức **bệnh nhân/mắt** bằng `GroupKFold` trên `patient_id`
  (không bao giờ `train_test_split` ngẫu nhiên theo bản ghi). Scaler/augmentation **fit trên Train**,
  chỉ `transform` sang Val/Test. Xem `src/data.py`.
- **Reproducibility**: seed cố định (`src/seed.py`), `requirements.txt` ghim phiên bản, tái lập bằng 1 lệnh (`make`).
- **Không commit dữ liệu/nặng**: ảnh EyeQ và weights KHÔNG lên Git (xem `.gitignore`). Dùng `scripts/download_data.py`.
- **Zero-Hallucination**: mọi DOI trong báo cáo xác minh qua Crossref trước khi thêm vào `references.bib`.
- **AI disclosure**: ghi log prompt quan trọng vào `docs/ai_disclosure_log.md`.

## 1. Cấu trúc repo

```
iqa-fundus/
├── src/                # code chính (import được, KHÔNG để logic trong notebook)
│   ├── seed.py         # seed_everything()
│   ├── data.py         # patient/eye-level split + Dataset/loader
│   ├── eda.py          # hàm vẽ EDA
│   ├── train.py        # baseline + W&B tracking
│   └── evaluate.py     # Sens/Spec/MacroF1/PR-AUC + bootstrap 95% CI + confusion matrix
├── scripts/download_data.py
├── configs/default.yaml
├── notebooks/01_eda.ipynb     # notebook MỎNG: chỉ gọi hàm trong src/
├── docs/                      # branch_plan, ai_disclosure_log
├── .github/                   # template Issue báo cáo tuần + PR nghiệm thu
├── requirements.txt
├── Makefile
└── .gitignore
```

## 2. Chạy trên Google Colab Pro (repo là nguồn chân lý, Colab chỉ là GPU)

Dán ô này ở đầu notebook Colab:

```python
# 1) Clone repo
!git clone https://github.com/<user>/iqa-fundus.git
%cd iqa-fundus
# 2) Cài đặt (ghim phiên bản)
!pip install -q -r requirements.txt
# 3) Định danh commit đúng người (cho hệ số K_peer)
!git config user.name "Ho Ten"
!git config user.email "huy14122005@gmail.com"
# 4) Mount Drive để chứa DATA + WEIGHTS (không lên Git)
from google.colab import drive; drive.mount('/content/drive')
```

Commit **thường xuyên, nhiều commit nhỏ** vào đúng nhánh `gdN` ngay trong Colab.
Checkpoint model ra Drive (session Colab có thể ngắt bất cứ lúc nào).

## 3. Tái lập bằng 1 lệnh

```bash
make setup      # cài requirements
make data       # tải EyeQ về ./data (bỏ qua nếu đã có trên Drive)
make eda        # sinh biểu đồ EDA -> ./outputs/eda
make baseline   # train baseline + log W&B
make eval       # đa độ đo + bootstrap CI + confusion matrix
```

## 4. Quy ước Git

- Branch theo milestone: `gd1`…`gd7` (xem `docs/branch_plan.md`).
- Commit message: `<type>(<scope>): <subject>` — vd. `feat(data): patient-level GroupKFold split`.
- Mỗi milestone mở 1 Pull Request theo template, đính kèm Reproducibility Checklist.
- Issue báo cáo tuần: 1 issue/tuần theo template.
