# GCD_GESNC — GEN-Augmented Semi-Supervised SNC for CIFAR-100

Pipeline thực nghiệm cho bài toán **Generalized Category Discovery (GCD)** trên **CIFAR-100** với thiết lập **80 known classes / 20 unknown classes**.

Project kết hợp:

- **ViT-B/16 feature extractor** đã fine-tune.
- **Linear probing** trên 80 lớp known.
- **GEN score filtering** để chọn pseudo-label có độ tin cậy cao.
- **Selective Neighbor Clustering (SNC)** để phân cụm bán giám sát.
- Tùy chọn **React clipping** để giảm ảnh hưởng của outlier activations trước khi tính GEN score.

---

## 1. Tổng quan pipeline

Luồng chạy chính:

1. **Extract features**
   - Dùng checkpoint `final.pth` để trích xuất feature CIFAR-100.
   - Sinh ra:
     - `features/cifar100_train_feat.pt`
     - `features/cifar100_test_feat.pt`

2. **Train linear classifier head**
   - Huấn luyện classifier `768 -> 80` trên 20,000 mẫu labeled của CIFAR-100 train.

3. **GEN pseudo-labeling**
   - Tính GEN score với tham số `M` và `gamma`.
   - Chọn top `PCT%` mẫu tự tin nhất để gán pseudo-label.

4. **SNC clustering**
   - Chạy SNC với `K=100`.
   - Hỗ trợ hai protocol:
     - `train_only`: chỉ dùng train features.
     - `transductive`: gộp train + test để clustering.

5. **Evaluation**
   - In ra:
     - `All ACC`
     - `Old ACC`
     - `New ACC`
     - `H-score`

---

## 2. Cấu trúc thư mục

```text
GCD_GESNC/
├── checkpoints/
│   └── final.pth
├── features/
│   ├── cifar100_train_feat.pt
│   └── cifar100_test_feat.pt
├── src/
│   ├── pipeline/
│   │   └── snc_wrapper.py
│   ├── snc/
│   └── utils/
│       ├── gen_entropy.py
│       └── metrics.py
├── extract_features.py
├── main_eval.py
├── requirements.txt
└── README.md
```

Lưu ý:

- `checkpoints/` và `features/` thường không nên push lên GitHub vì dung lượng lớn.
- Khi clone repo trên VM mới, cần tải/copy lại checkpoint và feature files vào đúng vị trí.

---

## 3. Chuẩn bị VM

### 3.1. Cài công cụ hệ thống

```bash
sudo apt update
sudo apt install -y git python3-pip python3-venv python3-full build-essential
```

### 3.2. Clone repository

```bash
cd ~
git clone https://github.com/Yato742003/GCD_GESNC
cd GCD_GESNC
```

### 3.3. Tạo virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

---

## 4. Cài thư viện Python

### Trường hợp dùng GPU NVIDIA L4 / CUDA 12.1

Cài PyTorch CUDA trước:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Sau đó cài các thư viện còn lại:

```bash
python -m pip install -r requirements.txt
```

Kiểm tra GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Kết quả mong muốn:

```text
True
NVIDIA L4
```

### Trường hợp chỉ dùng CPU

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

CPU vẫn chạy được `main_eval.py` nếu đã có feature files. Tuy nhiên, bước `extract_features.py` sẽ chậm hơn nhiều so với GPU.

---

## 5. Chuẩn bị checkpoint và features

### 5.1. Checkpoint

Đặt checkpoint CIFAR-100 vào:

```text
GCD_GESNC/checkpoints/final.pth
```

Nếu tải từ Google Drive bằng `gdown`:

```bash
mkdir -p ~/GCD_GESNC/checkpoints

python -m gdown <GOOGLE_DRIVE_FILE_ID> \
  -O ~/GCD_GESNC/checkpoints/final.pth
```

Ví dụ:

```bash
python -m gdown 16VIfujkJ3PrJR83znHiiAL8ZLp6U_eHk \
  -O ~/GCD_GESNC/checkpoints/final.pth
```

Kiểm tra:

```bash
ls -lh ~/GCD_GESNC/checkpoints
```

### 5.2. Feature files

Nếu đã có feature files, đặt vào:

```text
GCD_GESNC/features/cifar100_train_feat.pt
GCD_GESNC/features/cifar100_test_feat.pt
```

Kiểm tra:

```bash
ls -lh ~/GCD_GESNC/features
```

### 5.3. Extract features nếu chưa có

```bash
cd ~/GCD_GESNC
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python3 extract_features.py \
  --pretrain checkpoints/final.pth \
  --output_dir features/
```

Sau khi chạy xong, kiểm tra:

```bash
ls -lh features/
```

Cần có:

```text
cifar100_train_feat.pt
cifar100_test_feat.pt
```

---

## 6. Chạy evaluation

File chính:

```bash
main_eval.py
```

Các tham số quan trọng:

| Tham số          | Ý nghĩa                                                                |
| ---------------- | ---------------------------------------------------------------------- |
| `--protocol`     | `train_only`, `transductive`, hoặc `both`                              |
| `--pseudo_scope` | Trong transductive mode, chọn pseudo anchors từ `all` hoặc chỉ `train` |
| `--pct`          | Tỷ lệ mẫu tự tin nhất dùng làm pseudo-label                            |
| `--m`            | Tham số `M` trong GEN score                                            |
| `--gamma`        | Tham số `gamma` trong GEN score                                        |
| `--react`        | Bật React clipping                                                     |
| `--react_q`      | Quantile dùng để clip feature khi bật React                            |

---

## 7. Lệnh chạy chính thức

### 7.1. Transductive — GEN no React

```bash
cd ~/GCD_GESNC
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python3 main_eval.py \
  --protocol transductive \
  --pseudo_scope all \
  --pct 10 \
  --m 8 \
  --gamma 0.1
```

Khi chạy đúng no React, log đầu chương trình sẽ có:

```text
React=False
```

### 7.2. Transductive — GEN + React

```bash
cd ~/GCD_GESNC
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python3 main_eval.py \
  --protocol transductive \
  --pseudo_scope all \
  --pct 10 \
  --m 8 \
  --gamma 0.1 \
  --react \
  --react_q 0.99
```

Khi chạy đúng React, log sẽ có:

```text
React=True
[React:transductive] clip_q=0.9900
```

### 7.3. Strict train-only

```bash
cd ~/GCD_GESNC
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python3 main_eval.py \
  --protocol train_only \
  --pct 10 \
  --m 8 \
  --gamma 0.1
```

### 7.4. Chạy và lưu log

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_eval.py \
  --protocol transductive \
  --pseudo_scope all \
  --pct 10 \
  --m 8 \
  --gamma 0.1 | tee ~/cifar_no_react.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_eval.py \
  --protocol transductive \
  --pseudo_scope all \
  --pct 10 \
  --m 8 \
  --gamma 0.1 \
  --react \
  --react_q 0.99 | tee ~/cifar_react.log
```

Xem lại kết quả:

```bash
tail -n 40 ~/cifar_no_react.log
tail -n 40 ~/cifar_react.log
```

---

## 8. Kết quả tham khảo

### 8.1. GEN no React — transductive, PCT=10, M=8, gamma=0.1

Kết quả mẫu trên unlabeled train:

```text
All ACC : 84.89%
Old ACC : 87.14%
New ACC : 80.39%
H-score : 83.63%
```

Kết quả mẫu trên test set:

```text
All ACC : 81.18%
Old ACC : 83.93%
New ACC : 70.20%
H-score : 76.45%
```

### 8.2. GEN + React — transductive, PCT=10, M=8, gamma=0.1, react_q=0.99

Kết quả có thể dao động nhẹ theo môi trường, seed, phiên bản thư viện và file feature. Khi báo cáo, nên ghi rõ:

```text
Protocol: transductive
Pseudo scope: all
PCT: 10
M: 8
gamma: 0.1
React: True
react_q: 0.99
```

---

## 9. Checklist quay video demo

Nên quay các bước sau:

### Bước 1: Kiểm tra GPU

```bash
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### Bước 2: Kiểm tra project, checkpoint và features

```bash
cd ~/GCD_GESNC
ls
ls -lh checkpoints
ls -lh features
```

### Bước 3: Chạy no React

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_eval.py \
  --protocol transductive \
  --pseudo_scope all \
  --pct 10 \
  --m 8 \
  --gamma 0.1
```

### Bước 4: Chạy React

```bash
CUDA_VISIBLE_DEVICES=0 python3 main_eval.py \
  --protocol transductive \
  --pseudo_scope all \
  --pct 10 \
  --m 8 \
  --gamma 0.1 \
  --react \
  --react_q 0.99
```

### Bước 5: Zoom vào kết quả

Kết quả cuối cần thể hiện rõ:

```text
All ACC
Old ACC
New ACC
H-score
```

---

## 10. Troubleshooting

### Lỗi `ModuleNotFoundError: No module named 'torch'`

Nguyên nhân: virtual environment chưa cài PyTorch.

Cách sửa:

```bash
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Lỗi `externally-managed-environment`

Nguyên nhân: cài package vào Python hệ thống thay vì virtual environment.

Cách sửa:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### PyTorch không nhận GPU

Kiểm tra:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu ra `False`, cài lại PyTorch CUDA:

```bash
pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Lỗi thiếu feature files

Nếu `main_eval.py` báo thiếu:

```text
features/cifar100_train_feat.pt
features/cifar100_test_feat.pt
```

thì cần copy feature vào đúng thư mục hoặc chạy lại:

```bash
python3 extract_features.py \
  --pretrain checkpoints/final.pth \
  --output_dir features/
```

### Lỗi Google Drive `gdown --fuzzy`

Một số bản `gdown` không hỗ trợ `--fuzzy`. Dùng trực tiếp file ID:

```bash
python -m gdown <FILE_ID> -O checkpoints/final.pth
```

---

## 11. Ghi chú

- Không commit/push `checkpoints/` và `features/` nếu file quá lớn.
- Khi chuyển VM, có thể copy project bằng `rsync`, `scp`, hoặc attach disk snapshot.
- Không nên dùng lại `.venv` từ VM cũ; hãy tạo lại `.venv` trên VM mới.
- Nếu chỉ chạy `main_eval.py` và đã có feature files, CPU vẫn chạy được.
- Nếu cần chạy `extract_features.py`, nên dùng GPU để tiết kiệm thời gian.
