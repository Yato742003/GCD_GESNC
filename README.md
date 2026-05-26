# GESNC — GEN-Augmented Semi-Supervised SNC for Generalized Category Discovery

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pipeline thực nghiệm cho bài toán **Generalized Category Discovery (GCD)** trên **CIFAR-100** (80 known / 20 unknown) và **CUB-200-2011** (100 known / 100 unknown).

GESNC kết hợp:
- **ViT-B/16 DINO** làm feature extractor (backbone đóng băng hoàn toàn).
- **Linear probing** trên các lớp known để sinh logits.
- **GEN score filtering** + **ReAct clipping** để chọn pseudo-label chất lượng cao.
- **Semi-supervised Selective Neighbor Clustering (SNC)** để phân cụm đồng thời known + unknown.

> **Luận văn Tốt nghiệp** — Trường Đại học Bách Khoa TP.HCM (HCMUT), HK 25/2.

---

## Mục lục

1. [Tổng quan pipeline](#1-tổng-quan-pipeline)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Chuẩn bị dữ liệu & checkpoint](#5-chuẩn-bị-dữ-liệu--checkpoint)
6. [Trích xuất đặc trưng (Feature Extraction)](#6-trích-xuất-đặc-trưng-feature-extraction)
7. [Đánh giá đơn lẻ (Single-Run Evaluation)](#7-đánh-giá-đơn-lẻ-single-run-evaluation)
8. [Đánh giá đa hạt giống & kiểm định thống kê (Multi-Seed + Paired T-Test)](#8-đánh-giá-đa-hạt-giống--kiểm-định-thống-kê)
9. [Phân tích trường hợp thất bại (Failure Analysis)](#9-phân-tích-trường-hợp-thất-bại)
10. [Ablation Study](#10-ablation-study)
11. [Kết quả tham khảo](#11-kết-quả-tham-khảo)
12. [Tham số dòng lệnh chi tiết](#12-tham-số-dòng-lệnh-chi-tiết)
13. [Khắc phục sự cố](#13-khắc-phục-sự-cố)

---

## 1. Tổng quan pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Bước 1: Extract Features                                       │
│  ViT-B/16 DINO (frozen) → [CLS] token 768-dim cho mỗi ảnh     │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Bước 2: Train Linear Classifier Head                           │
│  768 → N_KNOWN (80 cho CIFAR-100, 100 cho CUB-200)             │
│  Huấn luyện 100 epochs, Adam lr=1e-3, trên labeled samples     │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Bước 3: (Tùy chọn) ReAct Clipping                             │
│  Clip feature tại quantile q=0.99 của labeled set               │
│  → Giảm ảnh hưởng outlier activations cho OOD detection         │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Bước 4: GEN Score Pseudo-Labeling                              │
│  Tính GEN entropy (M, γ) trên logits                            │
│  → Chọn top PCT% mẫu tự tin nhất làm pseudo-label anchors      │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Bước 5: Semi-Supervised SNC Clustering                         │
│  Real labels + Pseudo labels → SNC (K clusters)                 │
│  → Gán nhãn cụm cho toàn bộ tập dữ liệu                        │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Bước 6: Hungarian Matching Evaluation                          │
│  Ánh xạ cluster → ground truth → tính All/Old/New ACC, H-score │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Cấu trúc thư mục

```text
GCD_GESNC/
├── checkpoints/
│   ├── cifar100/
│   │   └── final.pth              # ViT-B/16 DINO checkpoint (~435 MB)
│   └── cub200/
│       └── final.pth              # ViT-B/16 DINO checkpoint (~435 MB)
├── data/                          # CIFAR-100 sẽ tự download vào đây
├── features/                      # Feature files được sinh bởi extract_features.py
│   ├── cifar100_train_feat.pt
│   ├── cifar100_test_feat.pt
│   ├── cub200_cipr_train_feat.pt
│   └── cub200_cipr_test_feat.pt
├── figures/                       # Đồ thị phân tích lỗi (sinh bởi analyze_failures.py)
├── results/                       # CSV kết quả multi-seed và báo cáo lỗi
├── src/
│   ├── models/
│   │   └── vision_transformer.py  # ViT-B/16 architecture
│   ├── pipeline/
│   │   └── snc_wrapper.py         # SNC clustering wrapper
│   ├── snc/                       # Selective Neighbor Clustering core
│   └── utils/
│       ├── gen_entropy.py         # GEN score computation
│       └── metrics.py             # Accuracy metrics, l2_normalize, ...
├── extract_features.py            # Trích xuất feature CIFAR-100
├── main_eval.py                   # Đánh giá CIFAR-100
├── eval_cub_gesnc.py              # Đánh giá CUB-200 (trích xuất + eval)
├── run_multi_seed.py              # Chạy đa hạt giống + paired t-test
├── analyze_failures.py            # Phân tích lỗi + confusion matrix 20×80
├── ablation_cifar_full.sh         # Shell script ablation study
├── ablation_m_gamma.sh            # Shell script ablation M×γ
├── plot_pct_curve.py              # Vẽ đồ thị PCT sensitivity
├── requirements.txt               # Thư viện Python (phiên bản pinned)
└── README.md
```

> **Lưu ý:** `checkpoints/` và `features/` không được đẩy lên GitHub do dung lượng lớn. Khi clone repo trên máy mới, cần copy hoặc tải lại.

---

## 3. Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|---|---|---|
| **GPU** | NVIDIA GPU ≥ 12 GB VRAM | Tesla T4 (16 GB) / L4 (24 GB) |
| **RAM** | ≥ 12 GB | ≥ 16 GB |
| **Ổ cứng** | ≥ 3 GB trống | ≥ 5 GB (checkpoint + features + data) |
| **Python** | 3.9+ | 3.10 |
| **CUDA** | 11.8+ | 12.1 |
| **OS** | Ubuntu 20.04+ / Windows 10+ | Ubuntu 22.04 |

> CPU cũng chạy được `main_eval.py` nếu đã có feature files. Tuy nhiên `extract_features.py` và `eval_cub_gesnc.py` chậm hơn nhiều trên CPU.

---

## 4. Cài đặt môi trường

### 4.1. Clone repo

```bash
git clone https://github.com/Yato742003/GCD_GESNC.git
cd GCD_GESNC
```

### 4.2. Tạo virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
pip install --upgrade pip setuptools wheel
```

### 4.3. Cài PyTorch (chọn đúng phiên bản CUDA)

**GPU — CUDA 12.1:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

**GPU — CUDA 11.8:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

**CPU only:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4.4. Cài các thư viện còn lại

```bash
pip install -r requirements.txt
```

### 4.5. Kiểm tra cài đặt

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

Kết quả mong đợi (ví dụ trên Tesla T4):
```text
PyTorch: 2.1.0+cu121
CUDA available: True
GPU: Tesla T4
VRAM: 15.8 GB
```

---

## 5. Chuẩn bị dữ liệu & checkpoint

### 5.1. Checkpoint ViT-B/16 DINO

Đặt checkpoint vào đúng thư mục theo dataset:

```bash
# CIFAR-100
mkdir -p checkpoints/cifar100
# Copy hoặc tải final.pth vào checkpoints/cifar100/final.pth

# CUB-200
mkdir -p checkpoints/cub200
# Copy hoặc tải final.pth vào checkpoints/cub200/final.pth
```

Nếu checkpoint trên Google Drive, có thể dùng `gdown`:
```bash
python -m gdown <GOOGLE_DRIVE_FILE_ID> -O checkpoints/cifar100/final.pth
```

Kiểm tra:
```bash
ls -lh checkpoints/cifar100/   # Cần có final.pth (~435 MB)
ls -lh checkpoints/cub200/     # Cần có final.pth (~435 MB)
```

### 5.2. Dữ liệu CIFAR-100

CIFAR-100 sẽ được **tự động download** bởi torchvision khi chạy `extract_features.py` lần đầu. Dữ liệu lưu vào thư mục `data/`.

### 5.3. Dữ liệu CUB-200-2011

Tải thủ công bộ CUB-200-2011 và giải nén:
```bash
# Đặt tại ~/cipr_cub200/data/CUB_200_2011/
# Cấu trúc bên trong: images/, images.txt, image_class_labels.txt, train_test_split.txt
```

CUB class split file (từ CiPR):
```bash
# Đặt tại data/splits/cub_osr_splits.pkl
# Hoặc tại ~/CiPR/data/splits/cub_osr_splits.pkl (tự phát hiện)
```

---

## 6. Trích xuất đặc trưng (Feature Extraction)

### 6.1. CIFAR-100

```bash
python extract_features.py \
  --pretrain checkpoints/cifar100/final.pth \
  --output_dir features/ \
  --seed 0
```

**Output:**
```text
features/cifar100_train_feat.pt   # 50,000 samples × 768-dim
features/cifar100_test_feat.pt    # 10,000 samples × 768-dim
```

**Thời gian:** ~2 phút trên Tesla T4.

### 6.2. CUB-200

CUB-200 được trích xuất tự động bên trong `eval_cub_gesnc.py` (bước đầu tiên). Nếu muốn trích xuất riêng hoặc dùng cached features, xem `--refresh_cache`.

---

## 7. Đánh giá đơn lẻ (Single-Run Evaluation)

### 7.1. CIFAR-100 — Cấu hình tốt nhất (GESNC)

```bash
python main_eval.py \
  --feat_dir features/ \
  --protocol transductive \
  --pct 10 --m 8 --gamma 0.2 \
  --react --react_q 0.99 \
  --seed 0
```

### 7.2. CIFAR-100 — Pure SNC Baseline (CiPR)

```bash
python main_eval.py \
  --feat_dir features/ \
  --protocol transductive \
  --pct 0 \
  --seed 0
```

### 7.3. CIFAR-100 — Lưu kết quả vào CSV

```bash
python main_eval.py \
  --feat_dir features/ \
  --protocol transductive \
  --pct 10 --m 8 --gamma 0.2 \
  --react --react_q 0.99 \
  --seed 0 \
  --output_csv results/cifar100_single.csv
```

### 7.4. CUB-200 — Cấu hình tốt nhất (GESNC)

```bash
python eval_cub_gesnc.py \
  --data_root ~/cipr_cub200/data/CUB_200_2011 \
  --pretrain checkpoints/cub200/final.pth \
  --feat_dir features/ \
  --protocol transductive \
  --pct 10 --m 8 --gamma 0.1 \
  --seed 0
```

### 7.5. CUB-200 — Pure SNC Baseline

```bash
python eval_cub_gesnc.py \
  --data_root ~/cipr_cub200/data/CUB_200_2011 \
  --pretrain checkpoints/cub200/final.pth \
  --feat_dir features/ \
  --protocol transductive \
  --pct 0 \
  --seed 0
```

---

## 8. Đánh giá đa hạt giống & kiểm định thống kê

Script `run_multi_seed.py` tự động:
1. Chạy cả **CiPR baseline** (PCT=0) và **GESNC** (PCT=10 + GEN + ReAct) qua nhiều hạt giống.
2. Tính **mean ± std** (sample standard deviation, ddof=1).
3. Thực hiện **paired t-test** (`scipy.stats.ttest_rel`) để đánh giá ý nghĩa thống kê.
4. Xuất bảng **LaTeX** sẵn sàng chèn vào luận văn.

### 8.1. Chạy cả CIFAR-100 (5 seeds) và CUB-200 (3 seeds)

```bash
python run_multi_seed.py --dataset both
```

### 8.2. Chỉ chạy CIFAR-100

```bash
python run_multi_seed.py --dataset cifar100
```

### 8.3. Chỉ chạy CUB-200

```bash
python run_multi_seed.py --dataset cub200
```

**Hạt giống sử dụng:**
- CIFAR-100: `[0, 1, 2, 42, 123]` (5 seeds)
- CUB-200: `[0, 1, 2]` (3 seeds)

**Thời gian ước tính:**
- CIFAR-100: ~5 seeds × 2 methods × ~3 min = **~30 phút** trên Tesla T4
- CUB-200: ~3 seeds × 2 methods × ~5 min = **~30 phút** trên Tesla T4

**Output:**
```text
results/cifar100_cipr_temp.csv             # Kết quả thô CiPR từng seed
results/cifar100_gesnc_temp.csv            # Kết quả thô GESNC từng seed
results/cifar100_multi_seed_final.csv      # Thống kê tổng hợp: mean, std, t-stat, p-value
results/cub200_cipr_temp.csv
results/cub200_gesnc_temp.csv
results/cub200_multi_seed_final.csv
```

Script sẽ in ra terminal:
- Bảng so sánh mean ± std cho mỗi metric.
- Giá trị t-stat và p-value kèm diễn giải ý nghĩa thống kê.
- **Mã nguồn bảng LaTeX** sẵn sàng copy-paste vào luận văn.

---

## 9. Phân tích trường hợp thất bại

Script `analyze_failures.py` phân tích lỗi trên CIFAR-100 (cấu hình tốt nhất, seed=0):

```bash
python analyze_failures.py
```

**Output:**
```text
figures/fig_confusion_matrix_20x80.png     # Heatmap nhầm lẫn 20 Unknown × 25 Known
figures/fig_per_class_acc.png              # Biểu đồ cột accuracy 20 lớp Unknown
results/failure_analysis_report.txt        # Báo cáo dạng text
```

Script thực hiện:
1. Chạy lại pipeline GESNC tốt nhất (PCT=10, M=8, γ=0.2, ReAct q=0.99).
2. Ánh xạ nhãn cụm → nhãn thực tế qua **Hungarian matching**.
3. Xây dựng **ma trận nhầm lẫn 20×80** (20 lớp Unknown thực tế vs 80 lớp Known bị nhầm).
4. Tìm **Top-5 cặp nhầm lẫn ngữ nghĩa** nhiều nhất.
5. Tính **per-class accuracy** cho 20 lớp Unknown.

> **Yêu cầu:** Phải có `features/cifar100_train_feat.pt` và `features/cifar100_test_feat.pt` (xem [Bước 6](#6-trích-xuất-đặc-trưng-feature-extraction)).

---

## 10. Ablation Study

### 10.1. PCT sensitivity (M=8, γ=0.1 cố định)

```bash
for PCT in 0 5 10 15 20; do
  echo ">>> PCT=$PCT"
  python main_eval.py --protocol transductive --pct $PCT --m 8 --gamma 0.1 --feat_dir features/
done
```

### 10.2. M × γ grid search (PCT=10 cố định)

```bash
for M in 4 8 16; do
  for GAMMA in 0.05 0.1 0.2; do
    echo ">>> M=$M | gamma=$GAMMA"
    python main_eval.py --protocol transductive --pct 10 --m $M --gamma $GAMMA --feat_dir features/
  done
done
```

### 10.3. Chạy full ablation bằng script

```bash
bash ablation_cifar_full.sh 2>&1 | tee results/ablation_cifar.log
```

---

## 11. Kết quả tham khảo

### 11.1. CIFAR-100 — Single run (seed=0)

| Cấu hình | All ACC | Old ACC | New ACC | H-score |
|---|---|---|---|---|
| Pure SNC (CiPR baseline, PCT=0) | 81.50% | 82.40% | 79.70% | 81.03% |
| **GESNC (PCT=10, M=8, γ=0.2, ReAct)** | **84.82%** | **87.18%** | **80.10%** | **83.49%** |

### 11.2. CUB-200 — Single run (seed=0)

| Cấu hình | All ACC | Old ACC | New ACC | H-score |
|---|---|---|---|---|
| Pure SNC (CiPR baseline, PCT=0) | ~56% | ~58% | ~52% | ~55% |
| **GESNC (PCT=10, M=8, γ=0.1)** | **~59%** | **~61%** | **~55%** | **~58%** |

> Kết quả chính xác hơn sẽ được báo cáo sau khi chạy multi-seed (`run_multi_seed.py`).

### 11.3. Phần cứng thực nghiệm

| | CiPR (bài báo gốc) | GESNC (luận văn) |
|---|---|---|
| GPU | RTX 3090 (24 GB) | Tesla T4 (16 GB) / L4 (24 GB) |
| Backbone | Fine-tuned ViT-B/16 DINO | Frozen ViT-B/16 DINO |
| Augmentation | Random Crop + Flip | CenterCrop only |

---

## 12. Tham số dòng lệnh chi tiết

### `main_eval.py` (CIFAR-100)

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--feat_dir` | `~/GCD_GESNC/features` | Thư mục chứa feature files |
| `--protocol` | `transductive` | `train_only` / `transductive` / `both` |
| `--pseudo_scope` | `all` | Pseudo anchors từ `all` hoặc chỉ `train` |
| `--pct` | `10` | Top PCT% mẫu GEN score thấp nhất dùng làm pseudo-label |
| `--m` | `8` | Tham số M trong GEN score |
| `--gamma` | `0.1` | Tham số γ trong GEN score |
| `--react` | `False` | Bật ReAct clipping |
| `--react_q` | `0.99` | Quantile clipping threshold |
| `--seed` | `0` | Hạt giống cho linear head training |
| `--output_csv` | `""` | Đường dẫn file CSV để lưu kết quả (append mode) |

### `eval_cub_gesnc.py` (CUB-200)

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--data_root` | `~/cipr_cub200/data/CUB_200_2011` | Thư mục gốc CUB-200-2011 |
| `--pretrain` | `~/GCD_GESNC/checkpoints/cub200/final.pth` | Checkpoint ViT-B/16 |
| `--feat_dir` | `~/GCD_GESNC/features` | Cache features |
| `--split_path` | Auto-detect | Đường dẫn CiPR class split file |
| `--protocol` | `both` | `train_only` / `transductive` / `both` |
| `--pct` | `10` | Top PCT% cho GEN pseudo-labeling |
| `--m` | `8` | GEN parameter M |
| `--gamma` | `0.1` | GEN parameter γ |
| `--react` | `False` | Bật ReAct clipping |
| `--react_q` | `0.99` | ReAct quantile threshold |
| `--seed` | `0` | Random seed |
| `--batch_size` | `64` | Feature extraction batch size |
| `--num_workers` | `4` | DataLoader workers |
| `--refresh_cache` | `False` | Buộc trích xuất lại features |
| `--output_csv` | `""` | Đường dẫn CSV output |

### `run_multi_seed.py`

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--dataset` | `both` | `cifar100` / `cub200` / `both` |

---

## 13. Khắc phục sự cố

### Features chưa có

```text
Error: missing features at features/cifar100_train_feat.pt
```
→ Chạy `extract_features.py` trước (xem [Bước 6](#6-trích-xuất-đặc-trưng-feature-extraction)).

### Checkpoint không tìm thấy

```text
[WARNING] Pretrain not found at ./checkpoints/final.pth
```
→ Kiểm tra đường dẫn `--pretrain`. CIFAR-100 dùng `checkpoints/cifar100/final.pth`, CUB-200 dùng `checkpoints/cub200/final.pth`.

### CUDA out of memory

→ Giảm `--batch_size` (cho CUB-200) hoặc đảm bảo không có process khác chiếm GPU:
```bash
nvidia-smi
```

### CUB split file không tìm thấy

```text
[WARN] CUB split file not found. Falling back to natural labels
```
→ Copy file `cub_osr_splits.pkl` vào `data/splits/` hoặc chỉ định qua `--split_path`.

### Kết quả khác so với báo cáo

Kết quả có thể dao động nhẹ (~0.5%) do:
- Phiên bản thư viện khác nhau (đặc biệt PyTorch, pynndescent).
- Kiến trúc GPU khác nhau (FP32 precision).
- Thứ tự truy xuất dữ liệu.

Để tái tạo chính xác nhất, sử dụng đúng phiên bản trong `requirements.txt` và cài PyTorch `2.1.0+cu121`.

---

## Tài liệu tham khảo

- **CiPR**: Hao et al., "CiPR: An Efficient Framework with Cross-Instance Positive Relations for GCD", 2024.
- **DINO**: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers", ICCV 2021.
- **SNC**: Selective Neighbor Clustering for semi-supervised clustering.
- **GEN**: Liu et al., "Generalized ENtropy score for OOD detection", NeurIPS 2023.
- **ReAct**: Sun et al., "ReAct: Out-of-Distribution Detection with Rectified Activations", NeurIPS 2021.

---

## Giấy phép

MIT License — xem file [LICENSE](LICENSE).
