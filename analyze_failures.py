import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from src.pipeline.snc_wrapper import run_snc
from src.utils.gen_entropy import compute_gen_score
from src.utils.metrics import _to_numpy, l2_normalize
import torch.nn as nn
import random

N_KNOWN_CLASSES = 80
N_TOTAL_CLASSES = 100


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_head(train_feat, train_labels, labeled_mask, device):
    print(f"  Training linear classifier head for failure analysis...")
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


def get_hungarian_mapping(y_true, y_pred):
    """Sử dụng thuật toán Hungarian để ánh xạ nhãn dự đoán sang nhãn thực tế tương ứng."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = list(map(list, zip(*ind)))
    
    # Ánh xạ nhãn dự đoán sang nhãn thực tế
    pred_to_gt_map = {pred: gt for pred, gt in ind}
    
    # Ánh xạ ngược (từ thực tế sang nhãn dự đoán tốt nhất)
    gt_to_pred_map = {gt: pred for pred, gt in ind}
    
    return pred_to_gt_map, gt_to_pred_map, w


# Danh sách tên 100 lớp CIFAR-100 theo thứ tự index chuẩn của torchvision (alphabetical).
# Known classes = index 0..79, Unknown classes = index 80..99.
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",        # 0..79 = Known
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",           # 80..99 = Unknown
]
assert len(CIFAR100_CLASSES) == 100, f"Expected 100 class names, got {len(CIFAR100_CLASSES)}"


def main():
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = "features"

    train_feat_path = os.path.join(feat_dir, "cifar100_train_feat.pt")
    test_feat_path = os.path.join(feat_dir, "cifar100_test_feat.pt")

    if not os.path.exists(train_feat_path):
        print(f"Error: missing features at {train_feat_path}. Hãy trích xuất đặc trưng trước.")
        return
    if not os.path.exists(test_feat_path):
        print(f"Error: missing test features at {test_feat_path}. Hãy trích xuất đặc trưng trước.")
        return

    os.makedirs("results", exist_ok=True)

    print("Loading features...")
    train_data = torch.load(train_feat_path, weights_only=False)
    train_feat = _to_numpy(train_data["features"], None).astype("float32")
    train_labels = _to_numpy(train_data["labels"], None).astype("int64")
    train_mask = _to_numpy(train_data["mask"], None).astype("int64")
    labeled_mask = train_mask == 1

    test_data = torch.load(test_feat_path, weights_only=False)
    test_feat = _to_numpy(test_data["features"], None).astype("float32")
    test_labels = _to_numpy(test_data["labels"], None).astype("int64")
    
    # 1. Chạy lại cấu hình tốt nhất của GESNC trên seed=0
    # Cấu hình tốt nhất: pct=10, m=8, gamma=0.2, react=True, react_q=0.99
    head = train_head(train_feat, train_labels, labeled_mask, device)
    
    # Kẹp ngưỡng React
    clip_thresh = np.quantile(train_feat[labeled_mask], 0.99)
    combined_feat = np.concatenate([train_feat, test_feat])
    combined_labels = np.concatenate([train_labels, test_labels])
    combined_labeled = np.concatenate([labeled_mask, np.zeros(len(test_feat), dtype=bool)])
    
    feat_for_gen = np.clip(combined_feat, a_min=None, a_max=clip_thresh)
    logits = predict_logits(head, feat_for_gen, device)
    pseudo_pred = logits.argmax(axis=1)
    
    gen_scores = compute_gen_score(logits, M=8, gamma=0.2)
    thresh = np.percentile(gen_scores, 10)
    confident = gen_scores < thresh
    
    sl = np.full(len(combined_labels), -101, dtype=np.int64)
    sl[combined_labeled] = combined_labels[combined_labeled]
    
    # Chỉ cho phép pseudo-label từ train samples (pseudo_scope="train")
    n_train = len(train_feat)
    eligible_for_pseudo = np.zeros(len(combined_labels), dtype=bool)
    eligible_for_pseudo[:n_train] = True
    
    aug = confident & ~combined_labeled & eligible_for_pseudo
    sl[aug] = pseudo_pred[aug]
    sm = (combined_labeled | aug).astype(np.float32)
    
    print("Running SNC to gather cluster assignments...")
    _, _, req = run_snc(
        data=l2_normalize(feat_for_gen),
        req_clust=N_TOTAL_CLASSES,
        distance="cosine",
        ensure_early_exit=True,
        verbose=False,
        labeled=sl,
        mask=sm,
    )
    
    # Chỉ xét phần Unlabeled Train của tập dữ liệu
    train_labeled_mask = labeled_mask
    unlb_mask = ~train_labeled_mask
    
    y_true_unlb = train_labels[unlb_mask]
    y_pred_unlb = req[:n_train][unlb_mask]
    
    # 2. Hungarian Matching để có nhãn dự đoán đã ánh xạ
    pred_to_gt, gt_to_pred, w_matrix = get_hungarian_mapping(train_labels, req[:n_train])
    
    # Mảng chứa nhãn dự đoán đã được ánh xạ về ground truth của phần Unlabeled Train
    y_mapped_pred = np.array([pred_to_gt.get(pred, -1) for pred in y_pred_unlb])
    
    # 3. Phân tích ma trận nhầm lẫn 20x80 (20 Unknown thực tế vs 80 Known dự đoán nhầm)
    # Các lớp Unknown có id thực tế từ 80 đến 99
    # Các lớp Known có id thực tế từ 0 đến 79
    unknown_ids = np.arange(80, 100)
    known_ids = np.arange(0, 80)
    
    # Khởi tạo ma trận nhầm lẫn kích thước 20 x 80
    conf_matrix_20x80 = np.zeros((20, 80))
    
    for i, unk_id in enumerate(unknown_ids):
        # Lấy tất cả mẫu thuộc lớp Unknown này trong tập Unlabeled Train
        idx = np.where(y_true_unlb == unk_id)[0]
        if len(idx) == 0:
            continue
        
        # Nhãn dự đoán của các mẫu này
        preds = y_mapped_pred[idx]
        total_samples = len(idx)
        
        for j, kn_id in enumerate(known_ids):
            # Số lượng mẫu bị nhầm thành lớp Known j
            num_confused = np.sum(preds == kn_id)
            conf_matrix_20x80[i, j] = (num_confused / total_samples) * 100.0  # Tỷ lệ %
            
    # Tìm Top-5 cặp nhầm lẫn nhiều nhất
    confusions = []
    for i in range(20):
        for j in range(80):
            val = conf_matrix_20x80[i, j]
            if val > 0:
                unk_class_name = CIFAR100_CLASSES[80 + i]
                kn_class_name = CIFAR100_CLASSES[j]
                confusions.append((val, unk_class_name, kn_class_name))
                
    confusions.sort(key=lambda x: x[0], reverse=True)
    
    print("\n" + "=" * 60)
    print(" TOP 5 CẶP NHẦM LẪN NGỮ NGHĨA NHIỀU NHẤT (UNKNOWN -> KNOWN) ")
    print("=" * 60)
    for idx, (val, unk, kn) in enumerate(confusions[:5]):
        print(f"  {idx+1}. Lớp Unknown [{unk}] bị nhầm thành lớp Known [{kn}] : {val:.2f}%")
    print("=" * 60)
    
    # 4. Tính toán độ chính xác chi tiết của từng lớp Unknown (Per-class Accuracy)
    # Lớp Unknown được xem là phân cụm đúng nếu nó rơi vào đúng cụm tương ứng của nó (nhãn dự đoán ánh xạ trùng nhãn thực)
    unk_class_accs = []
    for i, unk_id in enumerate(unknown_ids):
        idx = np.where(y_true_unlb == unk_id)[0]
        if len(idx) == 0:
            unk_class_accs.append((0.0, CIFAR100_CLASSES[unk_id]))
            continue
        preds = y_mapped_pred[idx]
        acc = (np.sum(preds == unk_id) / len(idx)) * 100.0
        unk_class_accs.append((acc, CIFAR100_CLASSES[unk_id]))
        
    unk_class_accs.sort(key=lambda x: x[0])  # Sắp xếp từ thấp đến cao (tệ nhất đến tốt nhất)
    
    print("\n" + "=" * 60)
    print(" XẾP HẠNG 20 LỚP UNKNOWN THEO ĐỘ CHÍNH XÁC (TỪ TỆ NHẤT) ")
    print("=" * 60)
    for idx, (acc, name) in enumerate(unk_class_accs):
        print(f"  {idx+1:2d}. Lớp [{name:16s}] - Accuracy: {acc:6.2f}%")
    print("=" * 60)

    # 5. Vẽ đồ thị và trực quan hóa kết quả
    os.makedirs("figures", exist_ok=True)
    
    # Đồ thị 1: Heatmap của ma trận nhầm lẫn 20x80 (Chỉ vẽ các ô có nhầm lẫn đáng kể > 2% để tránh quá tải trực quan)
    plt.figure(figsize=(24, 10))
    # Để đồ thị đẹp, ta chọn 20 lớp Known bị nhầm nhiều nhất để hiển thị trục hoành
    known_confusion_sums = np.sum(conf_matrix_20x80, axis=0)
    top_confused_known_indices = np.argsort(known_confusion_sums)[-25:]  # lấy 25 lớp Known bị nhầm nhiều nhất
    
    reduced_matrix = conf_matrix_20x80[:, top_confused_known_indices]
    reduced_known_names = [CIFAR100_CLASSES[idx] for idx in top_confused_known_indices]
    unknown_names = [CIFAR100_CLASSES[idx] for idx in unknown_ids]
    
    sns.heatmap(reduced_matrix, annot=True, fmt=".1f", cmap="YlOrRd", 
                xticklabels=reduced_known_names, yticklabels=unknown_names, cbar_kws={'label': 'Tỷ lệ nhầm lẫn (%)'})
    plt.title("Ma trận nhầm lẫn chi tiết (Trích lọc 20 lớp Unknown vs 25 lớp Known bị nhầm lẫn nhiều nhất)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Lớp Đã biết bị nhầm lẫn (Known classes)", fontsize=12, labelpad=10)
    plt.ylabel("Lớp Chưa biết thực tế (Unknown classes)", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figures/fig_confusion_matrix_20x80.png", dpi=150)
    plt.close()
    
    # Đồ thị 2: Biểu đồ cột ngang per-class accuracy của 20 lớp Unknown
    plt.figure(figsize=(12, 8))
    acc_vals = [x[0] for x in unk_class_accs]
    class_names = [x[1] for x in unk_class_accs]
    
    # Tạo bảng màu chuyển sắc từ đỏ (thấp) sang xanh (cao)
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(acc_vals)))
    
    bars = plt.barh(class_names, acc_vals, color=colors, edgecolor='grey', height=0.6)
    plt.title("Độ chính xác phân loại chi tiết trên từng lớp Chưa biết (Unknown Classes ACC)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Độ chính xác (%)", fontsize=12, labelpad=10)
    plt.ylabel("Các lớp Chưa biết (Unknown Classes)", fontsize=12, labelpad=10)
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Thêm số liệu hiển thị ở đầu mỗi cột
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                 va='center', ha='left', fontsize=9, color='black', fontweight='semibold')
                 
    plt.tight_layout()
    plt.savefig("figures/fig_per_class_acc.png", dpi=150)
    plt.close()
    
    print("\nĐã vẽ thành công các đồ thị phân tích lỗi:")
    print("  -> figures/fig_confusion_matrix_20x80.png")
    print("  -> figures/fig_per_class_acc.png")
    
    # Lưu báo cáo lỗi dạng văn bản để chèn vào tài liệu
    with open("results/failure_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write("=== BÁO CÁO PHÂN TÍCH LỖI NGỮ NGHĨA TRÊN CÁC LỚP CHƯA BIẾT (UNKNOWN) ===\n\n")
        f.write("1. TOP 5 CẶP LỚP NHẦM LẪN NGỮ NGHĨA NHIỀU NHẤT:\n")
        for idx, (val, unk, kn) in enumerate(confusions[:5]):
            f.write(f"   {idx+1}. Unknown [{unk}] -> Known [{kn}] : {val:.2f}%\n")
        f.write("\n2. XẾP HẠNG ĐỘ CHÍNH XÁC CỦA 20 LỚP UNKNOWN (TỆ NHẤT ĐẾN TỐT NHẤT):\n")
        for idx, (acc, name) in enumerate(unk_class_accs):
            f.write(f"   {idx+1:2d}. Lớp [{name:16s}] - ACC: {acc:6.2f}%\n")
    print("Báo cáo phân tích lỗi chi tiết lưu tại: results/failure_analysis_report.txt")


if __name__ == "__main__":
    main()
