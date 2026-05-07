import os
import argparse
import torch
import numpy as np
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from src.utils.gen_entropy import compute_gen_score
from src.utils.metrics import _to_numpy, l2_normalize
from src.pipeline.snc_wrapper import run_snc

import random


def split_cluster_acc_v2(y_true, y_pred, mask):
    """CiPR-standard Hungarian evaluation (same as eval_snc.py)."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = list(map(list, zip(*ind)))
    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    old_acc, total_old = 0, 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old += sum(w[:, i])
    old_acc /= total_old
    new_acc, total_new = 0, 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new += sum(w[:, i])
    new_acc /= total_new
    return total_acc, old_acc, new_acc

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='GCD_GESNC CIFAR-100 Evaluation')
    parser.add_argument('--feat_dir', type=str,
                        default=os.path.expanduser('~/GCD_GESNC/features'),
                        help='Thư mục chứa cifar100_train_feat.pt và cifar100_test_feat.pt')
    parser.add_argument('--pct', type=int, default=10,
                        help='Top PCT%% confident samples dùng làm pseudo-labels (0=pure SNC)')
    parser.add_argument('--m', type=int, default=8, help='GEN parameter M')
    parser.add_argument('--gamma', type=float, default=0.1, help='GEN parameter gamma')
    parser.add_argument('--react', action='store_true',
                        help='GEN+React: clip features tại quantile trước khi tính GEN score')
    parser.add_argument('--react_q', type=float, default=0.99,
                        help='Quantile threshold cho React clipping (default=0.99)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    print("="*65)
    print("GCD_GESNC Pipeline: GEN-Augmented Semi-Supervised SNC (CIFAR-100)")
    print("="*65)

    train_feat_path = os.path.join(args.feat_dir, 'cifar100_train_feat.pt')
    test_feat_path  = os.path.join(args.feat_dir, 'cifar100_test_feat.pt')

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

    # 3. Huấn luyện Linear Classifier Head (768 -> 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Phase 1] Huấn luyện Linear Classifier Head (768 -> 80) trên thiết bị: {device}...")
    head = nn.Linear(768, 80).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Chỉ lấy các mẫu có nhãn (20k anchors) để train classifier
    X_lab = torch.from_numpy(train_feat[d_l]).float().to(device)
    y_lab = torch.from_numpy(train_labels[d_l]).long().to(device)
    
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

    # GEN + React: clip features trước khi tính logits (chỉ dùng để chọn pseudo-labels)
    if args.react:
        clip_thresh = np.quantile(train_feat[d_l], args.react_q)
        feat_for_gen = np.clip(combined_feat, a_min=None, a_max=clip_thresh)
        print(f"  [React] clip_q={args.react_q:.4f}, thresh={clip_thresh:.4f}")
    else:
        feat_for_gen = combined_feat

    with torch.no_grad():
        all_logits = []
        for i in range(0, len(feat_for_gen), 512):
            batch = torch.from_numpy(feat_for_gen[i:i+512]).float().to(device)
            all_logits.append(head(batch).cpu().numpy())
        all_logits = np.concatenate(all_logits)

    # 5. GEN Pseudo-labeling
    M_PARAM    = args.m
    GAMMA_PARAM= args.gamma
    PCT        = args.pct

    react_tag = f" + React(q={args.react_q})" if args.react else ""
    print(f"\n[Phase 2] GEN{react_tag} score (M={M_PARAM}, gamma={GAMMA_PARAM}), Top {PCT}%...")
    pseudo_pred = all_logits.argmax(axis=1)
    if PCT > 0:
        all_gen = compute_gen_score(all_logits, M=M_PARAM, gamma=GAMMA_PARAM)
        thresh  = np.percentile(all_gen, PCT)
        confident = (all_gen < thresh)
    else:
        print("  PCT=0 → Pure SNC (chỉ dùng nhãn thật).")
        confident = np.zeros(len(combined_labels), dtype=bool)

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

    # 8a. Evaluation — CiPR protocol: unlabeled TRAIN data (mask=0)
    unlb_mask = (train_mask == 0)  # unlabeled trong train set
    train_pred = req[:n_train]
    trg_unlb = train_labels[unlb_mask]
    pred_unlb = train_pred[unlb_mask]
    old_mask_unlb = (trg_unlb < 80)

    a_u, o_u, n_u = split_cluster_acc_v2(trg_unlb, pred_unlb, old_mask_unlb)
    h_u = 2 * o_u * n_u / max(o_u + n_u, 1e-12)

    # 8b. Evaluation — Test set (10k standard CIFAR-100 test)
    test_pred = req[n_train:]
    old_mask_test = (test_labels < 80)
    a_t, o_t, n_t = split_cluster_acc_v2(test_labels, test_pred, old_mask_test)
    h_t = 2 * o_t * n_t / max(o_t + n_t, 1e-12)

    print("\n" + "="*60)
    print(" CIFAR-100 RESULTS — CiPR Protocol (Unlabeled Train) ")
    print("="*60)
    print(f"  All ACC : {a_u:.4f}  ({a_u:.2%})")
    print(f"  Old ACC : {o_u:.4f}  ({o_u:.2%})")
    print(f"  New ACC : {n_u:.4f}  ({n_u:.2%})")
    print(f"  H-score : {h_u:.4f}  ({h_u:.2%})")
    print("="*60)
    print(f"  CiPR Paper Baseline: All=80.00%, Old=84.33%, New=62.70%")
    print(f"  Delta All (vs paper): {(a_u - 0.80)*100:+.2f}%")

    print("\n" + "="*60)
    print(" CIFAR-100 RESULTS — Test Set (10k Standard) ")
    print("="*60)
    print(f"  All ACC : {a_t:.4f}  ({a_t:.2%})")
    print(f"  Old ACC : {o_t:.4f}  ({o_t:.2%})")
    print(f"  New ACC : {n_t:.4f}  ({n_t:.2%})")
    print(f"  H-score : {h_t:.4f}  ({h_t:.2%})")

if __name__ == "__main__":
    main()
