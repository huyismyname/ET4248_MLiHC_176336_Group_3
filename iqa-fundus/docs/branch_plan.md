# Kế hoạch nhánh Git & Milestone (ET4248)

Nguồn chân lý là repo; Colab chỉ là GPU. Commit nhỏ, thường xuyên, đúng người.

| Nhánh              | Milestone            | Tuần | Nội dung chính                                                        |
|--------------------|----------------------|------|----------------------------------------------------------------------|
| `gd1-proposal`     | M1 (15%)             | 1–3  | Đề cương, phát biểu bài toán, ≥10 bài + ma trận so sánh, khoảng trống |
| `gd2-dataset-eda`  | M1 (15%)             | 2–3  | Tải EyeQ, EDA (phân bố nhãn, kích thước, kiểm tra group)             |
| `gd3-preprocessing`| M2 (20%)             | 4–6  | Patient/eye-level split + assert chống rò rỉ, tiền xử lý, augment    |
| `gd4-baseline`     | M2 (20%)             | 4–6  | Baseline ResNet-18, bảng độ đo cơ sở                                 |
| `gd5-proposed`     | M3 (25%)             | 7–10 | Mô hình đề xuất, ablation, W&B tracking, chống overfitting           |
| `gd6-evaluation-xai`| M4 (15%)            | 11–12| Đa độ đo + bootstrap CI, Grad-CAM (lý do reject), error analysis     |
| `gd7-ieee-paper`   | Chung kết (25%)      | 13–15| Paper IEEE, dọn repo, README tái lập, slide, vấn đáp                 |

## Quy ước

- Commit: `<type>(<scope>): <subject>` — type ∈ {feat, fix, docs, refactor, test, chore}.
  - vd. `feat(data): GroupKFold patient-level split + assert`
- Mỗi milestone: mở 1 Pull Request theo `.github/pull_request_template.md`, đính Reproducibility Checklist.
- Mỗi tuần: 1 Issue báo cáo tuần theo `.github/ISSUE_TEMPLATE/weekly_report.md`.
- Vai trò nhóm (CRediT): Project Lead/ML · Data & Preprocess · Eval & XAI · Docs & MLOps.
