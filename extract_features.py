"""
Feature Extraction — ViT-B/16 DINO
====================================
Extract [CLS] token (768-dim) cho toàn bộ CIFAR-100 bằng ViT-B/16 DINO pretrained.
Backbone hoàn toàn FROZEN.

Output files:
    features/cifar100_train_feat.pt  — dict {'features': (N,768), 'labels': (N,), 'mask': (N,)}
    features/cifar100_test_feat.pt   — dict {'features': (N,768), 'labels': (N,)}

mask: 1 = labeled (known class sample được chọn làm labeled), 0 = unlabeled
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Load local backbone architecture
from src.models.vision_transformer import vit_base

# ─── Config ───────────────────────────────────────────────────────────────────
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

N_KNOWN_CLASSES   = 80
N_UNKNOWN_CLASSES = 20
LABELED_FRACTION  = 0.5   # 50% của known class samples làm labeled

# ─── Transforms ───────────────────────────────────────────────────────────────
def get_transform():
    """Khớp chính xác với cấu trúc transform của CiPR: Resize(256) -> CenterCrop(224)"""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

# ─── GCD Split ────────────────────────────────────────────────────────────────
def make_gcd_split(
    targets: np.ndarray,
    n_known: int = N_KNOWN_CLASSES,
    labeled_fraction: float = LABELED_FRACTION,
    seed: int = 0,
) -> np.ndarray:
    """Tạo labeled mask theo GCD split chuẩn."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(targets), dtype=int)

    for c in range(n_known):
        indices = np.where(targets == c)[0]
        n_labeled = max(1, int(len(indices) * labeled_fraction))
        chosen = rng.choice(indices, size=n_labeled, replace=False)
        mask[chosen] = 1

    return mask

# ─── Extraction ───────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Extract [CLS] token features từ ViT."""
    all_features = []
    for images, _ in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract DINO ViT-B/16 features for CIFAR-100")
    parser.add_argument('--data_root', type=str, default='./data',
                        help="Thư mục chứa CIFAR-100")
    parser.add_argument('--output_dir', type=str, default='./features',
                        help="Thư mục lưu features")
    parser.add_argument('--pretrain', type=str, default='./checkpoints/final.pth',
                        help="Path tới weights")
    parser.add_argument('--batch_size', type=int, default=250) # Đã đổi từ 256 sang 250 để chia hết cho 50k và 10k
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load model ──
    print(f"Loading ViT-B/16 DINO from: {args.pretrain}")
    model = vit_base()
    if os.path.exists(args.pretrain):
        state_dict = torch.load(args.pretrain, map_location='cpu')
        # Nếu là dict chứa 'model', lấy phần model ra
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Xử lý tiền tố 'module.backbone.' hoặc 'backbone.'
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith('module.backbone.'):
                new_key = k[len('module.backbone.'):]
            elif k.startswith('backbone.'):
                new_key = k[len('backbone.'):]
            elif k.startswith('module.'):
                new_key = k[len('module.'):]
            
            # Chỉ lấy các trọng số thuộc về backbone ViT
            clean_state_dict[new_key] = v
            
        msg = model.load_state_dict(clean_state_dict, strict=False)
        print(f"Loaded pretrained DINO weights. Matched keys: {len(clean_state_dict)}")
    else:
        print(f"[WARNING] Pretrain not found at {args.pretrain}. Sử dụng random init. Yêu cầu copy file vào!")
    model = model.to(device).eval()

    # Freeze tất cả parameters
    for p in model.parameters():
        p.requires_grad = False

    transform = get_transform()

    # ── Train set ──
    print("\n[1/2] Processing CIFAR-100 TRAIN set...")
    train_dataset = datasets.CIFAR100(args.data_root, train=True, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, pin_memory=True)

    train_labels = np.array(train_dataset.targets)
    train_mask   = make_gcd_split(train_labels, n_known=N_KNOWN_CLASSES,
                                   labeled_fraction=LABELED_FRACTION, seed=args.seed)

    train_features = extract_features(model, train_loader, device)

    train_data = {
        'features': train_features,
        'labels':   torch.LongTensor(train_labels),
        'mask':     torch.LongTensor(train_mask),
        'n_known':  N_KNOWN_CLASSES,
        'n_unknown': N_UNKNOWN_CLASSES,
    }
    out_path = os.path.join(args.output_dir, 'cifar100_train_feat.pt')
    torch.save(train_data, out_path)
    print(f"Saved: {out_path} | shape={train_features.shape}")

    # ── Test set ──
    print("\n[2/2] Processing CIFAR-100 TEST set...")
    test_dataset = datasets.CIFAR100(args.data_root, train=False, download=True, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_labels   = np.array(test_dataset.targets)
    test_features = extract_features(model, test_loader, device)

    test_data = {
        'features': test_features,
        'labels':   torch.LongTensor(test_labels),
        'n_known':  N_KNOWN_CLASSES,
        'n_unknown': N_UNKNOWN_CLASSES,
    }
    out_path = os.path.join(args.output_dir, 'cifar100_test_feat.pt')
    torch.save(test_data, out_path)
    print(f"Saved: {out_path} | shape={test_features.shape}")
    print("\n✅ Feature extraction hoàn tất!")

if __name__ == "__main__":
    main()
