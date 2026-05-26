import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from src.pipeline.snc_wrapper import run_snc
from src.utils.gen_entropy import compute_gen_score
from src.utils.metrics import _to_numpy, l2_normalize


N_KNOWN_CLASSES = 80
N_TOTAL_CLASSES = 100


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


def train_head(train_feat, train_labels, labeled_mask, device):
    print(f"\n[Phase 1] Training Linear Classifier Head (768 -> {N_KNOWN_CLASSES}) on {device}...")
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


def _print_result(title, y_true, y_pred):
    old_mask = y_true < N_KNOWN_CLASSES
    a, o, n = split_cluster_acc_v2(y_true, y_pred, old_mask)
    h = 2 * o * n / max(o + n, 1e-12)

    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)
    print(f"  All ACC : {a:.4f}  ({a:.2%})")
    print(f"  Old ACC : {o:.4f}  ({o:.2%})")
    print(f"  New ACC : {n:.4f}  ({n:.2%})")
    print(f"  H-score : {h:.4f}  ({h:.2%})")
    print("=" * 60)
    return a, o, n, h


def run_protocol(name, feat, labels, labeled_mask, head, args, device, n_train=None, test_labels=None):
    labeled_mask = labeled_mask.astype(bool)
    train_count = n_train if n_train is not None else len(labeled_mask)
    real_labeled_count = int(labeled_mask[:train_count].sum())

    d_l = np.where(labeled_mask)[0]
    if args.react:
        clip_thresh = np.quantile(feat[d_l], args.react_q)
        feat_for_gen = np.clip(feat, a_min=None, a_max=clip_thresh)
        clipped_ratio = float((feat > clip_thresh).sum()) / float(feat.size)
        print(f"  [React:{name}] clip_q={args.react_q:.4f}, thresh={clip_thresh:.4f}, clipped={clipped_ratio:.4%}")
    else:
        feat_for_gen = feat

    print(f"\n[Phase 2:{name}] GEN score (M={args.m}, gamma={args.gamma}), Top {args.pct}%...")
    logits = predict_logits(head, feat_for_gen, device)
    pseudo_pred = logits.argmax(axis=1)
    if args.pct > 0:
        gen_scores = compute_gen_score(logits, M=args.m, gamma=args.gamma)
        thresh = np.percentile(gen_scores, args.pct)
        confident = gen_scores < thresh
    else:
        print("  PCT=0 -> Pure SNC (real labels only).")
        confident = np.zeros(len(labels), dtype=bool)

    sl = np.full(len(labels), -101, dtype=np.int64)
    orig_labeled = np.zeros(len(labels), dtype=bool)
    orig_labeled[:len(labeled_mask)] = labeled_mask
    sl[orig_labeled] = labels[orig_labeled]

    if name == "transductive" and args.pseudo_scope == "train":
        eligible_for_pseudo = np.zeros(len(labels), dtype=bool)
        eligible_for_pseudo[:train_count] = True
    else:
        eligible_for_pseudo = np.ones(len(labels), dtype=bool)

    aug = confident & ~orig_labeled & eligible_for_pseudo
    sl[aug] = pseudo_pred[aug]
    sm = (orig_labeled | aug).astype(np.float32)

    print(f"Anchors: {real_labeled_count} Real + {int(aug.sum())} Pseudo = {int(sm.sum())} Total.")
    feat_for_snc = feat_for_gen if args.react else feat
    print(f"\n[Phase 3:{name}] Running SNC on {len(feat_for_snc)} samples (K={N_TOTAL_CLASSES})...")
    snc_inputs = dict(
        data=l2_normalize(feat_for_snc),
        req_clust=N_TOTAL_CLASSES,
        distance="cosine",
        ensure_early_exit=True,
        verbose=False,
        labeled=sl,
        mask=sm,
    )
    _, _, req = run_snc(**snc_inputs)

    train_labeled_mask = labeled_mask[:train_count]
    unlb_mask = ~train_labeled_mask
    train_labels = labels[:train_count]
    train_pred = req[:train_count]
    a_u, o_u, n_u, h_u = _print_result(
        f"CIFAR-100 RESULTS - {name.upper()} / UNLABELED TRAIN",
        train_labels[unlb_mask],
        train_pred[unlb_mask],
    )
    res = {
        "train_all": a_u, "train_old": o_u, "train_new": n_u, "train_h": h_u
    }

    if n_train is not None and test_labels is not None:
        a_t, o_t, n_t, h_t = _print_result(
            f"CIFAR-100 RESULTS - {name.upper()} / TEST SET",
            test_labels,
            req[n_train:],
        )
        res.update({
            "test_all": a_t, "test_old": o_t, "test_new": n_t, "test_h": h_t
        })
    return res


def main():
    parser = argparse.ArgumentParser(description="GCD_GESNC CIFAR-100 Evaluation")
    parser.add_argument("--feat_dir", type=str, default=os.path.expanduser("~/GCD_GESNC/features"),
                        help="Directory containing cifar100_train_feat.pt and cifar100_test_feat.pt")
    parser.add_argument("--protocol", choices=("train_only", "transductive", "both"), default="transductive",
                        help="train_only = strict CiPR-style train features only; transductive = train+test clustering.")
    parser.add_argument("--pseudo_scope", choices=("all", "train"), default="all",
                        help="In transductive mode, allow pseudo anchors from all samples (legacy) or train only.")
    parser.add_argument("--pct", type=int, default=10,
                        help="Top PCT%% confident samples used as pseudo-labels (0=pure SNC)")
    parser.add_argument("--m", type=int, default=8, help="GEN parameter M")
    parser.add_argument("--gamma", type=float, default=0.1, help="GEN parameter gamma")
    parser.add_argument("--react", action="store_true",
                        help="GEN+React: clip features at a quantile before computing GEN scores")
    parser.add_argument("--react_q", type=float, default=0.99,
                        help="Quantile threshold for React clipping")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default="", help="Path to save results in CSV format")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("GCD_GESNC Pipeline: GEN-Augmented Semi-Supervised SNC (CIFAR-100)")
    print(f"Protocol: {args.protocol} | PCT={args.pct} | M={args.m} | gamma={args.gamma} | React={args.react}")
    if args.protocol in ("transductive", "both"):
        print(f"Pseudo scope: {args.pseudo_scope}")
    print("=" * 65)

    train_feat_path = os.path.join(args.feat_dir, "cifar100_train_feat.pt")
    test_feat_path = os.path.join(args.feat_dir, "cifar100_test_feat.pt")
    if not os.path.exists(train_feat_path):
        print(f"Error: missing features at {train_feat_path}")
        return
    if args.protocol in ("transductive", "both") and not os.path.exists(test_feat_path):
        print(f"Error: missing test features at {test_feat_path}")
        return

    print("Loading extracted features...")
    train_data = torch.load(train_feat_path, weights_only=False)
    train_feat = _to_numpy(train_data["features"], None).astype("float32")
    train_labels = _to_numpy(train_data["labels"], None).astype("int64")
    train_mask = _to_numpy(train_data["mask"], None).astype("int64")
    labeled_mask = train_mask == 1

    test_feat = None
    test_labels = None
    if args.protocol in ("transductive", "both"):
        test_data = torch.load(test_feat_path, weights_only=False)
        test_feat = _to_numpy(test_data["features"], None).astype("float32")
        test_labels = _to_numpy(test_data["labels"], None).astype("int64")
        print(f"Loaded train={len(train_feat)} ({int(labeled_mask.sum())} labeled), test={len(test_feat)}.")
    else:
        print(f"Loaded train={len(train_feat)} ({int(labeled_mask.sum())} labeled).")

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