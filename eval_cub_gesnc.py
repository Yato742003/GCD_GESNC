import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.models.vision_transformer import vit_base
from src.pipeline.snc_wrapper import run_snc
from src.utils.gen_entropy import compute_gen_score
from src.utils.metrics import l2_normalize


N_KNOWN_CLASSES = 100
N_TOTAL_CLASSES = 200
LABELED_FRACTION = 0.5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def split_cluster_acc_v2(y_true, y_pred, mask):
    """CiPR-style Hungarian eval: match on all samples, then split old/new."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])
    assert y_pred.size == y_true.size

    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = list(map(list, zip(*ind)))
    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) / y_pred.size

    old_acc = 0
    total_old = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old += sum(w[:, i])
    old_acc /= max(total_old, 1)

    new_acc = 0
    total_new = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new += sum(w[:, i])
    new_acc /= max(total_new, 1)
    return total_acc, old_acc, new_acc


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform(size=224):
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_split_file(split_path):
    if not split_path or not os.path.exists(split_path):
        return None
    splits = np.load(split_path, allow_pickle=True, encoding="latin1")
    if hasattr(splits, "item"):
        splits = splits.item()
    known = list(splits["known_classes"])
    unknown = []
    unknown_groups = splits["unknown_classes"]
    for key in ("Easy", "Medium", "Hard"):
        unknown.extend(list(unknown_groups[key]))
    class_order = known + unknown
    if len(class_order) != N_TOTAL_CLASSES:
        raise ValueError(f"Invalid CUB split file: expected 200 classes, got {len(class_order)}")
    return {old_class: new_class for new_class, old_class in enumerate(class_order)}


def _parse_cub_metadata(root, class_remap=None):
    images_file = os.path.join(root, "images.txt")
    labels_file = os.path.join(root, "image_class_labels.txt")
    split_file = os.path.join(root, "train_test_split.txt")
    images_dir = os.path.join(root, "images")

    with open(images_file, encoding="utf-8") as f:
        img_paths = [
            os.path.join(images_dir, line.strip().split(" ", 1)[1])
            for line in f
        ]
    with open(labels_file, encoding="utf-8") as f:
        raw_labels = np.array([int(line.split()[1]) - 1 for line in f], dtype=np.int64)
    with open(split_file, encoding="utf-8") as f:
        is_train = np.array([int(line.split()[1]) == 1 for line in f], dtype=bool)

    if class_remap is None:
        labels = raw_labels
    else:
        labels = np.array([class_remap[int(y)] for y in raw_labels], dtype=np.int64)
    return img_paths, labels, is_train


def _make_labeled_mask(labels, is_train):
    train_idx = np.where(is_train)[0]
    train_labels = labels[train_idx]
    labeled_mask = np.zeros(len(train_idx), dtype=bool)
    for c in range(N_KNOWN_CLASSES):
        pos = np.where(train_labels == c)[0]
        if len(pos) == 0:
            continue
        # Match CiPR's deterministic "take one from each pair after sorting" mask.
        chosen = pos[::2]
        labeled_mask[chosen] = True
    return labeled_mask


class CUB200MetadataDataset(Dataset):
    def __init__(self, root, mode, transform, split_path=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        class_remap = _load_split_file(split_path)
        if class_remap is None:
            print("  [WARN] CUB split file not found. Falling back to natural labels; known=0..99.")
        else:
            print(f"  Using CiPR CUB class remap from: {split_path}")

        all_paths, all_labels, is_train = _parse_cub_metadata(root, class_remap)
        train_idx = np.where(is_train)[0]
        test_idx = np.where(~is_train)[0]
        labeled_train = _make_labeled_mask(all_labels, is_train)

        if mode == "train":
            selected = train_idx
            self.mask = labeled_train.astype(np.int64)
        elif mode == "test":
            selected = test_idx
            self.mask = np.zeros(len(test_idx), dtype=np.int64)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.img_paths = [all_paths[i] for i in selected]
        self.labels = all_labels[selected].astype(np.int64)
        print(
            f"[CUB200 {mode}] {len(self.labels)} samples | "
            f"Known: {int((self.labels < N_KNOWN_CLASSES).sum())} | "
            f"Unknown: {int((self.labels >= N_KNOWN_CLASSES).sum())} | "
            f"Labeled: {int(self.mask.sum())}"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx]), int(self.mask[idx]), idx


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    all_feats, all_labels, all_masks = [], [], []
    for batch in tqdm(dataloader, desc="Extracting"):
        imgs = batch[0].to(device)
        labels = batch[1]
        masks = batch[2]
        feats = model(imgs)
        if isinstance(feats, tuple):
            feats = feats[1]
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
        all_masks.append(masks.numpy())
    return (
        np.concatenate(all_feats).astype("float32"),
        np.concatenate(all_labels).astype("int64"),
        np.concatenate(all_masks).astype("int64"),
    )


def load_backbone(pretrain, device):
    print(f"Loading ViT-B/16 backbone from: {pretrain}")
    model = vit_base()
    ckpt = torch.load(pretrain, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    backbone_state = {}
    for k, v in ckpt.items():
        if k.startswith("module.backbone."):
            backbone_state[k[len("module.backbone."):]] = v
        elif k.startswith("backbone."):
            backbone_state[k[len("backbone."):]] = v
        elif not k.startswith(("module.head", "head")):
            backbone_state[k.replace("module.", "")] = v

    msg = model.load_state_dict(backbone_state, strict=False)
    print(f"  Loaded keys: {len(backbone_state)} | Missing: {len(msg.missing_keys)} | Unexpected: {len(msg.unexpected_keys)}")
    return model.to(device).eval()


def train_head(train_feat, train_labels, labeled_mask, device):
    print("\n[Phase 2] Training Linear Classifier Head (768 -> 100)...")
    head = nn.Linear(768, N_KNOWN_CLASSES).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    d_l = np.where(labeled_mask)[0]
    x_lab = torch.from_numpy(train_feat[d_l]).float().to(device)
    y_lab = torch.from_numpy(train_labels[d_l]).long().to(device)

    head.train()
    for _ in range(100):
        for i in range(0, len(x_lab), 256):
            loss = criterion(head(x_lab[i:i + 256]), y_lab[i:i + 256])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict_logits(head, feat, device):
    logits = []
    for i in range(0, len(feat), 512):
        batch = torch.from_numpy(feat[i:i + 512]).float().to(device)
        logits.append(head(batch).cpu().numpy())
    return np.concatenate(logits)


def run_protocol(name, feat, labels, labeled_mask, head, args, device, n_train=None, test_labels=None):
    labeled_mask = labeled_mask.astype(bool)
    train_count = n_train if n_train is not None else len(labeled_mask)
    d_l = np.where(labeled_mask)[0]

    if args.react:
        clip_thresh = np.quantile(feat[d_l], args.react_q)
        feat_for_gen = np.clip(feat, a_min=None, a_max=clip_thresh)
        print(f"  [React:{name}] clip_q={args.react_q:.4f}, thresh={clip_thresh:.4f}")
    else:
        feat_for_gen = feat

    logits = predict_logits(head, feat_for_gen, device)
    pseudo_pred = logits.argmax(axis=1)
    if args.pct > 0:
        gen_scores = compute_gen_score(logits, M=args.m, gamma=args.gamma)
        thresh = np.percentile(gen_scores, args.pct)
        confident = gen_scores < thresh
    else:
        confident = np.zeros(len(labels), dtype=bool)

    sl = np.full(len(labels), -101, dtype=np.int64)
    orig_labeled = np.zeros(len(labels), dtype=bool)
    orig_labeled[:len(labeled_mask)] = labeled_mask
    sl[orig_labeled] = labels[orig_labeled]

    if name == "transductive" and n_train is not None and not args.allow_test_pseudo:
        eligible_for_pseudo = np.zeros(len(labels), dtype=bool)
        eligible_for_pseudo[:n_train] = True
    else:
        eligible_for_pseudo = np.ones(len(labels), dtype=bool)

    aug = confident & ~orig_labeled & eligible_for_pseudo
    sl[aug] = pseudo_pred[aug]
    sm = (orig_labeled | aug).astype(np.float32)

    print(f"\n[Phase 3:{name}] GEN Pseudo-labeling (M={args.m}, gamma={args.gamma}, PCT={args.pct})")
    print(f"Anchors: {int(orig_labeled.sum())} Real + {int(aug.sum())} Pseudo = {int(sm.sum())} Total.")
    print(f"[Phase 4:{name}] Running SNC on {len(feat)} samples (K=200)...")
    _, _, req = run_snc(
        data=l2_normalize(feat),
        req_clust=N_TOTAL_CLASSES,
        distance="cosine",
        ensure_early_exit=True,
        verbose=False,
        labeled=sl,
        mask=sm,
    )

    train_labeled_mask = labeled_mask[:train_count]
    unlb_mask = ~train_labeled_mask
    train_labels = labels[:train_count]
    train_pred = req[:train_count]
    trg_unlb = train_labels[unlb_mask]
    pred_unlb = train_pred[unlb_mask]
    old_mask_unlb = trg_unlb < N_KNOWN_CLASSES
    a_u, o_u, n_u = split_cluster_acc_v2(trg_unlb, pred_unlb, old_mask_unlb)
    h_u = 2 * o_u * n_u / max(o_u + n_u, 1e-12)

    print("\n" + "=" * 60)
    print(f" CUB-200 RESULTS - {name.upper()} / UNLABELED TRAIN ")
    print("=" * 60)
    print(f"  All ACC : {a_u:.4f}  ({a_u:.2%})")
    print(f"  Old ACC : {o_u:.4f}  ({o_u:.2%})")
    print(f"  New ACC : {n_u:.4f}  ({n_u:.2%})")
    print(f"  H-score : {h_u:.4f}  ({h_u:.2%})")
    print("=" * 60)
    print("  Reference: final checkpoint; see protocol label above")

    res = {
        "train_all": a_u, "train_old": o_u, "train_new": n_u, "train_h": h_u
    }

    if n_train is not None and test_labels is not None:
        test_pred = req[n_train:]
        old_mask_test = test_labels < N_KNOWN_CLASSES
        a_t, o_t, n_t = split_cluster_acc_v2(test_labels, test_pred, old_mask_test)
        h_t = 2 * o_t * n_t / max(o_t + n_t, 1e-12)
        print("\n" + "=" * 60)
        print(f" CUB-200 RESULTS - {name.upper()} / TEST SET ")
        print("=" * 60)
        print(f"  All ACC : {a_t:.4f}  ({a_t:.2%})")
        print(f"  Old ACC : {o_t:.4f}  ({o_t:.2%})")
        print(f"  New ACC : {n_t:.4f}  ({n_t:.2%})")
        print(f"  H-score : {h_t:.4f}  ({h_t:.2%})")
        print("=" * 60)
        res.update({
            "test_all": a_t, "test_old": o_t, "test_new": n_t, "test_h": h_t
        })
    return res


def main():
    parser = argparse.ArgumentParser(description="GCD_GESNC CUB-200 evaluation")
    parser.add_argument("--data_root", type=str, default=os.path.expanduser("~/cipr_cub200/data/CUB_200_2011"))
    parser.add_argument("--pretrain", type=str, default=os.path.expanduser("~/GCD_GESNC/checkpoints/cub200/final.pth"))
    parser.add_argument("--feat_dir", type=str, default=os.path.expanduser("~/GCD_GESNC/features"))
    parser.add_argument("--split_path", type=str, default=None,
                        help="Path to CiPR data/splits/cub_osr_splits.pkl. Auto-detected when omitted.")
    parser.add_argument("--protocol", choices=("train_only", "transductive", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pct", type=int, default=10)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--react", action="store_true")
    parser.add_argument("--react_q", type=float, default=0.99)
    parser.add_argument("--allow_test_pseudo", action="store_true",
                        help="Allow pseudo-label anchors from test samples in transductive mode.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--output_csv", type=str, default="", help="Path to save results in CSV format")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.split_path is None:
        candidates = [
            os.path.join(os.getcwd(), "data", "splits", "cub_osr_splits.pkl"),
            os.path.expanduser("~/CiPR/data/splits/cub_osr_splits.pkl"),
            os.path.expanduser("~/cipr_cub200/CiPR/data/splits/cub_osr_splits.pkl"),
            os.path.abspath(os.path.join(os.getcwd(), "..", "CiPR", "data", "splits", "cub_osr_splits.pkl")),
        ]
        args.split_path = next((p for p in candidates if os.path.exists(p)), "")

    print("=" * 65)
    print("GCD_GESNC Pipeline on CUB-200-2011")
    print(f"Protocol: {args.protocol} | PCT={args.pct} | M={args.m} | gamma={args.gamma} | React={args.react}")
    print("=" * 65)

    model = load_backbone(args.pretrain, device)

    transform = get_transform()
    train_dataset = CUB200MetadataDataset(args.data_root, mode="train", transform=transform, split_path=args.split_path)
    test_dataset = CUB200MetadataDataset(args.data_root, mode="test", transform=transform, split_path=args.split_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    cache_tag = "cipr" if args.split_path else "natural"
    os.makedirs(args.feat_dir, exist_ok=True)
    train_feat_path = os.path.join(args.feat_dir, f"cub200_{cache_tag}_train_feat.pt")
    test_feat_path = os.path.join(args.feat_dir, f"cub200_{cache_tag}_test_feat.pt")

    print("\n[Phase 1] Extracting/loading CUB features...")
    if not args.refresh_cache and os.path.exists(train_feat_path) and os.path.exists(test_feat_path):
        print("  Loading cached features...")
        train_data = torch.load(train_feat_path, map_location="cpu", weights_only=False)
        test_data = torch.load(test_feat_path, map_location="cpu", weights_only=False)
        train_feat = train_data["features"].numpy().astype("float32")
        train_labels = train_data["labels"].numpy().astype("int64")
        train_mask = train_data["mask"].numpy().astype("int64")
        test_feat = test_data["features"].numpy().astype("float32")
        test_labels = test_data["labels"].numpy().astype("int64")
    else:
        train_feat, train_labels, train_mask = extract_features(model, train_loader, device)
        test_feat, test_labels, _ = extract_features(model, test_loader, device)
        torch.save({"features": torch.tensor(train_feat), "labels": torch.tensor(train_labels), "mask": torch.tensor(train_mask)}, train_feat_path)
        torch.save({"features": torch.tensor(test_feat), "labels": torch.tensor(test_labels)}, test_feat_path)
        print("  Saved features to disk.")

    labeled_mask = train_mask == 1
    print(f"Loaded train={len(train_feat)} ({int(labeled_mask.sum())} labeled anchors), test={len(test_feat)}.")
    head = train_head(train_feat, train_labels, labeled_mask, device)

    train_only_res = None
    if args.protocol in ("train_only", "both"):
        train_only_res = run_protocol(
            "train_only",
            feat=train_feat,
            labels=train_labels,
            labeled_mask=labeled_mask,
            head=head,
            args=args,
            device=device,
        )

    trans_res = None
    if args.protocol in ("transductive", "both"):
        combined_feat = np.concatenate([train_feat, test_feat])
        combined_labels = np.concatenate([train_labels, test_labels])
        combined_labeled = np.concatenate([labeled_mask, np.zeros(len(test_feat), dtype=bool)])
        trans_res = run_protocol(
            "transductive",
            feat=combined_feat,
            labels=combined_labels,
            labeled_mask=combined_labeled,
            head=head,
            args=args,
            device=device,
            n_train=len(train_feat),
            test_labels=test_labels,
        )

    if args.output_csv:
        import csv
        file_exists = os.path.exists(args.output_csv)
        with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["seed", "protocol", "pct", "m", "gamma", "react", "react_q", 
                          "train_all", "train_old", "train_new", "train_h",
                          "test_all", "test_old", "test_new", "test_h"]
                writer.writerow(header)
            
            # Ghi kết quả transductive
            if trans_res:
                writer.writerow([
                    args.seed, "transductive", args.pct, args.m, args.gamma, int(args.react), args.react_q,
                    trans_res.get("train_all", 0.0), trans_res.get("train_old", 0.0), trans_res.get("train_new", 0.0), trans_res.get("train_h", 0.0),
                    trans_res.get("test_all", 0.0), trans_res.get("test_old", 0.0), trans_res.get("test_new", 0.0), trans_res.get("test_h", 0.0)
                ])
            # Ghi kết quả train_only
            if train_only_res:
                writer.writerow([
                    args.seed, "train_only", args.pct, args.m, args.gamma, int(args.react), args.react_q,
                    train_only_res.get("train_all", 0.0), train_only_res.get("train_old", 0.0), train_only_res.get("train_new", 0.0), train_only_res.get("train_h", 0.0),
                    0.0, 0.0, 0.0, 0.0
                ])
        print(f"Results appended to CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
