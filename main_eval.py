import os
import torch
import numpy as np
import torch.nn as nn

# Import các module từ kiến trúc SOTA
from src.utils.gen_entropy import compute_gen_score
from src.utils.metrics import _to_numpy, l2_normalize, split_cluster_acc
from src.pipeline.snc_wrapper import run_snc

import random

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(0) 
    print("="*65)
    print("GCD_GESNC Pipeline: GEN-Augmented Semi-Supervised SNC")
    print("="*65)

    # 1. Khai báo đường dẫn tuyệt đối (Để đảm bảo 100% không nhầm lẫn)
    train_feat_path = '/home/chaukietnguyen74/features/cifar100_train_feat.pt'
    test_feat_path  = '/home/chaukietnguyen74/features/cifar100_test_feat.pt'

    if not os.path.exists(train_feat_path):
        print(f"Error: Không tìm thấy file features tại {train_feat_path}")
        print("Vui lòng đảm bảo bạn đã copy file .pt về đúng thư mục.")
        return

    # 2. Load Dữ Liệu
    print("Loading extracted features...")
    train_data = torch.load(train_feat_path, weights_only=False)
    test_data  = torch.load(test_feat_path, weights_only=False)

    # Chuyển đổi tensor sang numpy array để xử lý
    train_feat   = _to_numpy(train_data['features'], None).astype('float32')
    train_labels = _to_numpy(train_data['labels'],   None).astype('int64')
    train_mask   = _to_numpy(train_data['mask'],     None).astype('int64')
    test_feat    = _to_numpy(test_data['features'],  None).astype('float32')
    test_labels  = _to_numpy(test_data['labels'],    None).astype('int64')

    # Lọc ra danh sách index của các mẫu có nhãn thật (Labeled mask = 1)
    labeled_mask = (train_mask == 1)
    d_l = np.where(labeled_mask)[0]

    n_train = len(train_feat)
    n_test  = len(test_feat)
    
    print(f"Loaded {n_train} Train samples ({labeled_mask.sum()} Labeled anchors) và {n_test} Test samples.")

    # 3. Huấn luyện Linear Classifier (Tạo "Bộ đo độ tự tin")
    print(f"\n[Phase 1] Huấn luyện Linear Classifier Head (768 -> 80) trên thiết bị: cpu...")
    head = nn.Linear(768, 80) # Không gọi .to(device) để chạy trên CPU y như hôm qua
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Chỉ lấy các mẫu có nhãn (20k anchors) để train classifier
    X_lab = torch.from_numpy(train_feat[d_l]).float()
    y_lab = torch.from_numpy(train_labels[d_l]).long()
    
    head.train()
    epochs = 100
    batch_size = 256
    for ep in range(epochs):
        for i in range(0, len(X_lab), batch_size):
            loss = criterion(head(X_lab[i:i+batch_size]), y_lab[i:i+batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()

    # 4. Quét toàn bộ 60k mẫu qua Classifier để lấy Logits
    combined_feat   = np.concatenate([train_feat, test_feat])
    combined_labels = np.concatenate([train_labels, test_labels])

    with torch.no_grad():
        combined_tensor = torch.from_numpy(combined_feat).float()
        all_logits = head(combined_tensor).numpy()

    # 5. Pseudo-labeling thông qua GEN Entropy
    # Thiết lập bộ tham số SOTA (Liu2023)
    M_PARAM = 8        # Chỉ tính entropy trên top 8 class xác suất cao nhất
    GAMMA_PARAM = 0.1  # Hệ số làm sắc nét phân phối (gamma)
    PCT = 10           # Chỉ lấy top 10% mẫu cực kỳ tự tin (p10)

    print(f"\n[Phase 2] Tính điểm GEN (M={M_PARAM}, gamma={GAMMA_PARAM}) và lọc top {PCT}% tự tin nhất...")
    all_gen = compute_gen_score(all_logits, M=M_PARAM, gamma=GAMMA_PARAM)
    pseudo_pred = all_logits.argmax(axis=1) # Lấy class có xác suất cao nhất làm nhãn dự đoán

    # Lọc mẫu: GEN score càng thấp càng tự tin thuộc Known Class
    thresh = np.percentile(all_gen, PCT)
    confident = (all_gen < thresh)

    # 6. Chuẩn bị Anchors cho SNC (Sự kết hợp giữa nhãn thật & nhãn ảo)
    sl = np.full(len(combined_labels), -101, dtype=np.int64) # -101 = Unlabeled
    
    # Bước 6a: Neo bằng nhãn thật (20,000 Labeled Train)
    sl[:n_train][labeled_mask] = train_labels[labeled_mask]
    
    # Bước 6b: Bơm thêm Pseudo-labels vào làm neo phụ
    orig_labeled = np.concatenate([labeled_mask, np.zeros(n_test, dtype=bool)])
    aug = confident & ~orig_labeled # Chỉ lấy những mẫu tự tin mà chưa có nhãn
    sl[aug] = pseudo_pred[aug]
    
    # Tạo mask: 1.0 cho các điểm làm neo, 0.0 cho unlabeled
    sm = (orig_labeled | aug).astype(np.float32)

    n_orig = int(labeled_mask.sum())
    n_pseudo = int(aug.sum())
    print(f"Anchors cung cấp cho mô hình: {n_orig} Nhãn thật + {n_pseudo} Nhãn ảo = {int(sm.sum())} Neo.")

    # 7. Phân cụm Bán giám sát (Semi-supervised SNC) trên 60,000 mẫu
    print("\n[Phase 3] Đang chạy Combined Semi-Supervised SNC (60,000 samples)...")
    _, _, req = run_snc(
        data=l2_normalize(combined_feat),
        req_clust=100,               # K=100 cụm (80 cũ + 20 mới)
        distance='cosine',           # Dùng Cosine Similarity
        ensure_early_exit=True,      # Tối ưu tốc độ gom cụm
        verbose=False,
        labeled=sl,                  # Truyền Anchors (Nhãn thật + Ảo)
        mask=sm
    )

    # 8. Đánh giá kết quả trên 10,000 Test Set
    test_pred = req[n_train:] # Cắt lấy phần dự đoán của 10k Test (Nửa sau của array)
    old_mask = (test_labels < 80) # Mask xác định Old Class
    
    # Tính toán Accuracy dựa trên thuật toán Hungarian Matching
    a, o, n = split_cluster_acc(test_labels, test_pred, old_mask)
    h = 2 * o * n / max(o + n, 1e-12) # H-Score (Harmonic Mean)

    print("\n" + "="*50)
    print(" SOTA TEST SET EVALUATION RESULTS ")
    print("="*50)
    print(f"All ACC : {a:.2%}")
    print(f"Old ACC : {o:.2%}")
    print(f"New ACC : {n:.2%}")
    print(f"H-score : {h:.2%}")
    print("="*50)

if __name__ == "__main__":
    main()
