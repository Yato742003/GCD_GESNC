import os
import argparse
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

# Import modules từ kiến trúc SOTA của bạn
from src.utils.gen_entropy import compute_gen_score
from src.utils.metrics import l2_normalize
from src.pipeline.snc_wrapper import run_snc
from src.models.vision_transformer import vit_base

# dataset_cub200.py nằm ở ~/cipr_cub200/
import sys
sys.path.append(os.path.expanduser('~/cipr_cub200'))
from dataset_cub200 import CUB200GCDDataset, get_transform


def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Hàm đánh giá chuẩn của CiPR (eval_snc.py L12-55).
    Tính All/Old/New ACC dùng Hungarian matching trên toàn bộ data,
    rồi tách riêng theo mask Old/New.
    """
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
    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances
    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances
    return total_acc, old_acc, new_acc

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def extract_features(model, dataloader, device, has_mask=True):
    """
    Trích xuất features từ CUB-200.
    CiPR cubloader trả về (img, label, mask, index) — 4 outputs.
    dataset_cub200.py có thể trả về (img, label, mask) — 3 outputs.
    has_mask=True nếu dataset trả về mask, False nếu chỉ (img, label).
    """
    model.eval()
    all_feats, all_labels, all_masks = [], [], []

    for batch in tqdm(dataloader, desc="Extracting"):
        imgs = batch[0].to(device)
        labels = batch[1]
        feats = model(imgs)  # ViT_Linear.forward trả về (embed, backbone_feat)
        # CiPR model trả về tuple (embed, y) — ta cần backbone features (y)
        if isinstance(feats, tuple):
            feats = feats[1]  # Lấy raw backbone CLS token (768-dim)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
        if has_mask and len(batch) >= 3:
            all_masks.append(batch[2].numpy())
        else:
            all_masks.append(np.zeros(len(labels), dtype=np.int64))

    return (np.concatenate(all_feats).astype('float32'),
            np.concatenate(all_labels).astype('int64'),
            np.concatenate(all_masks).astype('int64'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.expanduser('~/cipr_cub200/data/CUB_200_2011'))
    parser.add_argument('--pretrain', type=str, default=os.path.expanduser('~/cipr_cub200/CiPR/checkpoints/run/cipr_cub/final.pth'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pct', type=int, default=10,
                        help='Top PCT%% confident samples dùng làm pseudo-labels (0 = tắt GEN, pure SNC)')
    parser.add_argument('--m', type=int, default=8, help='GEN parameter M')
    parser.add_argument('--gamma', type=float, default=0.1, help='GEN parameter gamma')
    parser.add_argument('--react', action='store_true',
                        help='GEN+React: clip features tại quantile trước khi tính GEN score')
    parser.add_argument('--react_q', type=float, default=0.99,
                        help='Quantile threshold cho React clipping (default=0.99)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for feature extraction')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*65)
    print("GCD_GESNC Pipeline on CUB-200-2011 (100 Known / 100 Unknown)")
    print("="*65)

    # 1. Load Model
    # CiPR lưu bằng: torch.save(model.state_dict()) với model = DataParallel(ViT_Linear)
    # Keys trong file: 'module.backbone.xxx' (backbone) và 'module.head.xxx' (DINOHead)
    # Ta chỉ cần backbone (vit_base) → filter prefix 'module.backbone.'
    print(f"Loading ViT-B/16 backbone from: {args.pretrain}")
    model = vit_base()
    ckpt = torch.load(args.pretrain, map_location='cpu')

    backbone_state = {}
    loaded = 0
    for k, v in ckpt.items():
        if k.startswith('module.backbone.'):
            new_key = k[len('module.backbone.'):]  # strip 'module.backbone.'
            backbone_state[new_key] = v
            loaded += 1
        elif k.startswith('backbone.'):
            new_key = k[len('backbone.'):]  # strip 'backbone.'
            backbone_state[new_key] = v
            loaded += 1

    if loaded == 0:
        # Fallback: thử strip 'module.' chung (single-GPU checkpoint)
        for k, v in ckpt.items():
            if not k.startswith('module.head') and not k.startswith('head'):
                new_key = k.replace('module.', '')
                backbone_state[new_key] = v
                loaded += 1
        print(f"  [WARN] No 'module.backbone.*' keys found. Fallback: loaded {loaded} keys.")
    else:
        print(f"  Loaded {loaded} backbone keys from checkpoint.")

    msg = model.load_state_dict(backbone_state, strict=False)
    print(f"  Missing: {len(msg.missing_keys)} | Unexpected: {len(msg.unexpected_keys)}")
    model = model.to(device)

    # 2. Datasets & Loaders
    transform = get_transform('test', size=224)
    train_dataset = CUB200GCDDataset(args.data_root, mode='train_unlabeled', transform=transform, seed=args.seed)
    test_dataset  = CUB200GCDDataset(args.data_root, mode='test', transform=transform, seed=args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Extract or Load Features
    feat_dir = os.path.expanduser('~/GCD_GESNC/features')
    os.makedirs(feat_dir, exist_ok=True)
    train_feat_path = os.path.join(feat_dir, 'cub200_train_feat.pt')
    test_feat_path  = os.path.join(feat_dir, 'cub200_test_feat.pt')

    print("\n[Phase 1] Trích xuất Features...")
    if os.path.exists(train_feat_path) and os.path.exists(test_feat_path):
        print("  Loading pre-extracted features...")
        train_data = torch.load(train_feat_path, map_location='cpu', weights_only=False)
        test_data  = torch.load(test_feat_path, map_location='cpu', weights_only=False)
        train_feat, train_labels, train_mask = train_data['features'].numpy(), train_data['labels'].numpy(), train_data['mask'].numpy()
        test_feat, test_labels = test_data['features'].numpy(), test_data['labels'].numpy()
    else:
        print("  Extracting features from scratch...")
        train_feat, train_labels, train_mask = extract_features(model, train_loader, device)
        test_feat, test_labels, _            = extract_features(model, test_loader, device)
        torch.save({'features': torch.tensor(train_feat), 'labels': torch.tensor(train_labels), 'mask': torch.tensor(train_mask)}, train_feat_path)
        torch.save({'features': torch.tensor(test_feat), 'labels': torch.tensor(test_labels)}, test_feat_path)
        print("  Saved features to disk.")

    labeled_mask = (train_mask == 1)
    d_l = np.where(labeled_mask)[0]
    n_train = len(train_feat)
    n_test  = len(test_feat)

    # 4. Huấn luyện Linear Classifier Head (768 -> 100)
    print(f"\n[Phase 2] Huấn luyện Linear Classifier (768 -> 100)...")
    head = nn.Linear(768, 100).to(device) # CUB-200 có 100 Known
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    X_lab = torch.from_numpy(train_feat[d_l]).to(device)
    y_lab = torch.from_numpy(train_labels[d_l]).to(device)
    
    head.train()
    for ep in range(100):
        for i in range(0, len(X_lab), 256):
            loss = criterion(head(X_lab[i:i+256]), y_lab[i:i+256])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()

    # 5. Pseudo-labeling với GEN Entropy
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
        feat_tensor = torch.from_numpy(feat_for_gen).to(device)
        all_logits = head(feat_tensor).cpu().numpy()

    M_PARAM = args.m
    GAMMA_PARAM = args.gamma
    PCT = args.pct

    react_tag = f" + React(q={args.react_q})" if args.react else ""
    print(f"\n[Phase 3] GEN{react_tag} Pseudo-labeling (M={M_PARAM}, gamma={GAMMA_PARAM}, Top {PCT}%)...")

    # pseudo_pred khởi tạo trước để tránh UnboundLocalError khi PCT=0
    pseudo_pred = all_logits.argmax(axis=1)

    if PCT > 0:
        all_gen = compute_gen_score(all_logits, M=M_PARAM, gamma=GAMMA_PARAM)
        thresh = np.percentile(all_gen, PCT)
        confident = (all_gen < thresh)
    else:
        print("  PCT=0 → Bỏ qua GEN, chạy Pure SNC (chỉ dùng nhãn thật).")
        confident = np.zeros(len(combined_labels), dtype=bool)

    # 6. Prepare SNC Anchors
    sl = np.full(len(combined_labels), -101, dtype=np.int64)
    sl[:n_train][labeled_mask] = train_labels[labeled_mask]
    
    orig_labeled = np.concatenate([labeled_mask, np.zeros(n_test, dtype=bool)])
    aug = confident & ~orig_labeled
    sl[aug] = pseudo_pred[aug]
    sm = (orig_labeled | aug).astype(np.float32)

    print(f"Anchors: {int(orig_labeled.sum())} Real + {int(aug.sum())} Pseudo = {int(sm.sum())} Total.")

    # 7. SNC Clustering
    print("\n[Phase 4] Combined Semi-Supervised SNC (K=200)...")
    _, _, req = run_snc(
        data=l2_normalize(combined_feat),
        req_clust=200, # CUB-200 có 200 clusters
        distance='cosine',
        ensure_early_exit=True,
        verbose=False,
        labeled=sl,
        mask=sm
    )

    # 8. Evaluation — theo chuẩn CiPR (split_cluster_acc_v2)
    # CiPR eval_snc.py đánh giá trên unlabeled data (mask==0) thay vì toàn bộ test
    # Ở đây test_dataset là tập test riêng biệt → toàn bộ đều unlabeled
    test_pred = req[n_train:]
    old_mask = (test_labels < 100)  # CUB-200: 100 Known classes

    a, o, n = split_cluster_acc_v2(test_labels, test_pred, old_mask)
    h = 2 * o * n / max(o + n, 1e-12)

    print("\n" + "="*55)
    print(" 🚀 GESNC CUB-200 TEST RESULTS (CiPR-standard eval) ")
    print("="*55)
    print(f"  All ACC : {a:.4f}  ({a:.2%})")
    print(f"  Old ACC : {o:.4f}  ({o:.2%})")
    print(f"  New ACC : {n:.4f}  ({n:.2%})")
    print(f"  H-score : {h:.4f}  ({h:.2%})")
    print("="*55)
    print(f"\n  CiPR Baseline (epoch 60): All=62.22%, Old=63.58%, New=61.54%")
    print(f"  Delta All: {(a - 0.6222)*100:+.2f}%")

if __name__ == "__main__":
    main()
