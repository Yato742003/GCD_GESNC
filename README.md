# GCD_GESNC (GEN-Augmented Semi-Supervised SNC)

Kiến trúc SOTA cho bài toán **Generalized Category Discovery (GCD)**.

Đạt **80.61% All ACC** trên tập CIFAR-100 (80 Known / 20 Unknown), sánh ngang với các mô hình top đầu thế giới (SimGCD, PromptCAL) bằng cách thay thế hoàn toàn cấu trúc Hard Gate cổ điển bằng phương pháp **Soft Pseudo-labeling qua GEN Entropy**.

## Điểm Cốt Lõi Của Kiến Trúc
1. **Linear Probing**: Xây dựng một classifier (768 -> 80) bằng 20,000 nhãn Labeled của tập Train.
2. **GEN Score Filtering (Soft Pseudo-labeling)**: Tính điểm Generalized Entropy (M=8, gamma=0.1) cho toàn bộ tập dữ liệu. Lọc ra **Top 10% (p10)** các mẫu tự tin nhất để gán nhãn dự đoán (`argmax`).
3. **Combined SNC (Transductive Bridge)**: Gộp chung 50k Train + 10k Test = 60,000 mẫu vào thuật toán phân cụm Selective Neighbor Clustering. 30k mẫu Unlabeled Train đóng vai trò làm "cầu nối mật độ" giúp truyền dẫn nhãn từ Anchors (20k gốc + 6k pseudo) sang tập Test một cách chính xác.

## Cấu trúc Thư Mục
- `data/`: Nơi chứa dataset CIFAR-100 gốc (nếu chạy từ đầu).
- `features/`: Nơi chứa 2 file vector đặc trưng `.pt`.
- `checkpoints/`: Nơi chứa trọng số ViT-B/16 `final.pth`.
- `src/snc/`: Thuật toán phân cụm Selective Neighbor Clustering (kế thừa từ CiPR).
- `src/pipeline/`: Lớp Wrapper quản lý vòng lặp và điều kiện Early Exit của SNC.
- `src/utils/`: Công cụ tính điểm GEN và đo đạc Metrics (Hungarian split accuracy).
- `main_eval.py`: Kịch bản thực thi End-to-End.

## Hướng dẫn chạy trên máy ảo (VM)

> **Lưu ý:** Thư mục `features/` và `checkpoints/` trên Git ban đầu sẽ rỗng (do các file quá lớn không nên push lên Github). Bạn cần copy data vào đúng thư mục trước khi chạy.

### Bước 1: Clone source code về VM
```bash
git clone <đường-dẫn-repo-của-bạn>
cd GCD_GESNC
```

### Bước 2: Chuẩn bị Môi trường (Virtual Environment)
Đảm bảo bạn đang sử dụng Python 3.8 trở lên.
```bash
# Tạo môi trường ảo (Tuỳ chọn)
python3 -m venv venv
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### Bước 3: Chuẩn bị Dữ liệu (Rất quan trọng)
Di chuyển hoặc copy các file features đã được trích xuất từ Backbone ViT-B/16 (bản fine-tuned của CiPR) vào thư mục `features/`.

Cấu trúc yêu cầu:
```text
GCD_GESNC/
└── features/
    ├── cifar100_train_feat.pt   (50k mẫu Train)
    └── cifar100_test_feat.pt    (10k mẫu Test)
```
> [!WARNING]
> **Nếu bạn không có sẵn 2 file `.pt` này:**
> Bạn bắt buộc phải trích xuất (extract) chúng từ tập ảnh gốc CIFAR-100 bằng trọng số mô hình tốt nhất (`final.pth`). Đặt file `final.pth` vào thư mục `checkpoints/` và chạy lệnh sau (đã được tích hợp sẵn trong repo):
> ```bash
> python extract_features.py --pretrain checkpoints/final.pth --output_dir features/
> ```

*(Lưu ý: Nếu file `.pt` của bạn đang nằm ở `~/features/` trên VM, script `main_eval.py` đã tự động tìm đến thư mục đó. Tuy nhiên, tốt nhất là copy thẳng vào thư mục `features/` của project này và sửa lại đường dẫn trong `main_eval.py` cho đồng bộ)*.

### Bước 4: Chạy Pipeline SOTA End-to-End
Chỉ cần chạy file kịch bản chính:
```bash
python main_eval.py
```

### Kết quả Benchmark Cần Đạt (Tập Test 10k)
| Metric | Baseline (Không GEN) | **Ours (GEN p10)** | SOTA So Sánh |
|---|:---:|:---:|:---:|
| **All ACC** | 80.00% | **80.61%** | SimGCD (80.1%), PromptCAL (81.2%) |
| **Old ACC** | 84.33% | 83.15% | CiPR (77.1%) |
| **New ACC** | 62.70% | **70.45%** | - |
