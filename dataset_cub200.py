"""
dataset_cub200.py
=================
CUB-200-2011 Dataset với GCD Split chuẩn.

Theo CiPR paper (và GCD paper gốc của Vaze et al. 2022):
  - 200 classes tổng cộng
  - Known (Old): class 0–99  (100 classes)
  - Unknown (New): class 100–199 (100 classes)
  - Labeled fraction: 50% của Known class training samples
  - K = 200 clusters cho SNC

Cấu trúc CUB-200-2011:
  CUB_200_2011/
  ├── images/           (11788 ảnh)
  ├── images.txt        (index → path)
  ├── image_class_labels.txt (index → class_id 1-indexed)
  ├── train_test_split.txt   (index → 1=train / 0=test)
  └── classes.txt
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── GCD Split Constants (theo CiPR paper) ───────────────────
N_KNOWN_CLASSES   = 100   # Classes 1–100 (1-indexed in CUB)
N_UNKNOWN_CLASSES = 100   # Classes 101–200
TOTAL_CLASSES     = 200
LABELED_FRACTION  = 0.5   # 50% of known class train samples


# ── Transforms ──────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_transform(split='train', size=224):
    """
    Transforms theo CiPR / standard GCD setup cho CUB-200.
    Train: random crop + flip (augmentation cho contrastive).
    Test:  center crop (deterministic).
    """
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.08, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif split == 'train_plain':
        # Không augment — dùng để extract features
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # 'test'
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_two_crop_transform(size=224):
    """
    Two-view transform cho contrastive loss (SimCLR style).
    Dùng trong CiPR training loop cho unsupervised loss.
    """
    aug = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * size) | 1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return TwoCropTransform(aug)


class TwoCropTransform:
    """Trả về 2 augmented views của cùng một ảnh."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


# ── CUB-200 Base Parser ──────────────────────────────────────
def parse_cub200(root):
    """
    Parse các file metadata của CUB-200-2011.

    Returns:
        img_paths: list of absolute image paths
        labels:    np.ndarray, 0-indexed class labels (0–199)
        is_train:  np.ndarray bool, True = train split
    """
    images_file = os.path.join(root, 'images.txt')
    labels_file = os.path.join(root, 'image_class_labels.txt')
    split_file  = os.path.join(root, 'train_test_split.txt')

    # images.txt: "1 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
    with open(images_file) as f:
        img_paths = [
            os.path.join(root, 'images', line.strip().split(' ', 1)[1])
            for line in f
        ]

    # image_class_labels.txt: "1 1"  (img_id class_id 1-indexed)
    with open(labels_file) as f:
        labels = np.array([int(line.split()[1]) - 1 for line in f])  # → 0-indexed

    # train_test_split.txt: "1 1" (img_id is_train)
    with open(split_file) as f:
        is_train = np.array([int(line.split()[1]) == 1 for line in f])

    return img_paths, labels, is_train


def make_gcd_split(labels, is_train, n_known=N_KNOWN_CLASSES,
                   labeled_fraction=LABELED_FRACTION, seed=0):
    """
    Tạo GCD split chuẩn cho CUB-200.

    Returns:
        labeled_mask: np.ndarray bool len(train_indices)
                      True = sample này có label (dùng làm anchor)
    """
    rng = np.random.default_rng(seed)

    # Chỉ làm việc với training set
    train_idx = np.where(is_train)[0]
    train_labels = labels[train_idx]

    labeled_mask = np.zeros(len(train_idx), dtype=bool)

    # 50% của Known class samples → labeled
    for c in range(n_known):
        pos = np.where(train_labels == c)[0]
        n_labeled = max(1, int(len(pos) * labeled_fraction))
        chosen = rng.choice(pos, size=n_labeled, replace=False)
        labeled_mask[chosen] = True

    return labeled_mask


# ── Main Dataset Classes ─────────────────────────────────────
class CUB200GCDDataset(Dataset):
    """
    Dataset cho training CiPR trên CUB-200.
    Trả về (image, label, is_labeled) tuple.

    Mode:
        'train_labeled'   : chỉ labeled Known class samples
        'train_unlabeled' : toàn bộ train set (Known + Unknown)
        'test'            : toàn bộ test set
    """
    def __init__(self, root, mode='train_unlabeled',
                 transform=None, seed=0):
        """
        Args:
            root:      đường dẫn đến thư mục CUB_200_2011/
            mode:      'train_labeled' | 'train_unlabeled' | 'test'
            transform: torchvision transform
            seed:      random seed cho GCD split
        """
        self.root = root
        self.mode = mode

        # Parse raw data
        all_paths, all_labels, is_train = parse_cub200(root)

        # GCD split mask (trên training set)
        labeled_mask = make_gcd_split(all_labels, is_train, seed=seed)

        # Train indices
        train_idx = np.where(is_train)[0]
        test_idx  = np.where(~is_train)[0]

        if mode == 'train_labeled':
            # Chỉ lấy labeled Known class samples
            selected = train_idx[labeled_mask]
            self.img_paths = [all_paths[i] for i in selected]
            self.labels    = all_labels[selected]
            self.is_labeled = np.ones(len(selected), dtype=bool)

        elif mode == 'train_unlabeled':
            # Toàn bộ training set (50k Known labeled + Unknown unlabeled)
            self.img_paths = [all_paths[i] for i in train_idx]
            self.labels    = all_labels[train_idx]
            self.is_labeled = labeled_mask  # True cho labeled Known samples

        elif mode == 'test':
            self.img_paths = [all_paths[i] for i in test_idx]
            self.labels    = all_labels[test_idx]
            self.is_labeled = np.zeros(len(test_idx), dtype=bool)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.transform = transform

        print(f"[CUB200 {mode}] {len(self)} samples | "
              f"Known: {(self.labels < N_KNOWN_CLASSES).sum()} | "
              f"Unknown: {(self.labels >= N_KNOWN_CLASSES).sum()} | "
              f"Labeled: {self.is_labeled.sum()}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label     = int(self.labels[idx])
        is_labeled = int(self.is_labeled[idx])

        return img, label, is_labeled


class CUB200ContrastiveDataset(Dataset):
    """
    Dataset cho training CiPR contrastive phase.
    Trả về (two_views, label, is_labeled).
    Dùng TwoCropTransform.
    """
    def __init__(self, root, mode='train_unlabeled',
                 size=224, seed=0):
        self.base = CUB200GCDDataset(
            root, mode=mode,
            transform=get_two_crop_transform(size),
            seed=seed
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        views, label, is_labeled = self.base[idx]
        return views[0], views[1], label, is_labeled


# ── Utility functions ────────────────────────────────────────
def get_datasets_for_training(data_root, seed=0):
    """
    Tạo đầy đủ các dataset objects cho CiPR training loop.

    Returns dict với:
        train_dataset       : contrastive dataset (unlabeled=full train)
        test_dataset        : test set (plain transform)
        train_labeled_dataset: labeled subset (plain transform)
    """
    datasets = {
        'train_dataset': CUB200ContrastiveDataset(
            data_root, mode='train_unlabeled', seed=seed
        ),
        'test_dataset': CUB200GCDDataset(
            data_root, mode='test',
            transform=get_transform('test'),
            seed=seed
        ),
        'train_labeled_dataset': CUB200GCDDataset(
            data_root, mode='train_labeled',
            transform=get_transform('train_plain'),
            seed=seed
        ),
    }
    return datasets


def get_class_splits():
    """
    Trả về known/unknown class indices cho CUB-200.
    Theo CiPR paper: sorted split, first 100 = known.
    """
    known_classes   = list(range(0, N_KNOWN_CLASSES))        # 0–99
    unknown_classes = list(range(N_KNOWN_CLASSES, TOTAL_CLASSES))  # 100–199
    return known_classes, unknown_classes


if __name__ == '__main__':
    # Quick sanity check
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset_cub200.py /path/to/CUB_200_2011")
        sys.exit(1)

    root = sys.argv[1]
    print("\n=== Dataset Sanity Check ===")

    for mode in ['train_labeled', 'train_unlabeled', 'test']:
        ds = CUB200GCDDataset(root, mode=mode,
                               transform=get_transform('train_plain'))
        img, label, is_lab = ds[0]
        print(f"  {mode}: img shape={img.shape}, label={label}, "
              f"is_labeled={is_lab}")

    known, unknown = get_class_splits()
    print(f"\nKnown classes: {len(known)} ({known[0]}–{known[-1]})")
    print(f"Unknown classes: {len(unknown)} ({unknown[0]}–{unknown[-1]})")
    print("\nDataset OK!")
