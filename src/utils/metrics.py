import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

def split_cluster_acc(y_true, y_pred, mask):
    """
    Tính độ chính xác gom cụm (Clustering Accuracy) riêng biệt cho All, Old (Known), và New (Unknown).
    Sử dụng thuật toán Hungarian (Kuhn-Munkres) để tìm ra phép gán tối ưu giữa các cụm dự đoán và nhãn thực tế.

    Tham số:
        y_true: Mảng nhãn thực tế (Ground Truth)
        y_pred: Mảng nhãn cụm do mô hình dự đoán (Cluster Assignments)
        mask: Mảng boolean, True đại diện cho các class 'Old' (Đã biết).

    Trả về:
        tuple: (all_acc, old_acc, new_acc)
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    
    # Kích thước ma trận chi phí (D x D) dựa trên số nhãn lớn nhất
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    # Xây dựng ma trận tần suất đếm số lần khớp giữa y_pred và y_true
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Hàm linear_sum_assignment mặc định tìm giá trị cực tiểu (min weight matching)
    # Vì ta muốn tìm khớp cực đại (maximize accuracy), ta truyền vào `-w` (hoặc w.max() - w)
    ind = linear_sum_assignment(w.max() - w)
    
    # 1. Tính độ chính xác tổng thể (All Accuracy)
    sum_all = sum([w[i, j] for i, j in zip(*ind)])
    all_acc = sum_all * 1.0 / y_pred.size

    # 2. Tính độ chính xác cho các class Cũ (Old Accuracy)
    old_weight_sum = 0
    for i, j in zip(*ind):
        # Đếm các mẫu khớp nhau nằm trong vùng 'mask == True'
        old_weight_sum += np.sum((y_pred == i) & (y_true == j) & mask)
    old_acc = old_weight_sum * 1.0 / np.sum(mask) if np.sum(mask) > 0 else 0

    # 3. Tính độ chính xác cho các class Mới (New Accuracy)
    new_weight_sum = 0
    for i, j in zip(*ind):
        # Đếm các mẫu khớp nhau nằm trong vùng 'mask == False'
        new_weight_sum += np.sum((y_pred == i) & (y_true == j) & ~mask)
    new_acc = new_weight_sum * 1.0 / np.sum(~mask) if np.sum(~mask) > 0 else 0

    return all_acc, old_acc, new_acc

def l2_normalize(x):
    """Chuẩn hóa L2 (L2 Normalization) cho ma trận numpy 2D theo chiều ngang (axis=1)"""
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def _to_numpy(t, device):
    """Hàm hỗ trợ an toàn để ép kiểu một tensor PyTorch thành mảng Numpy"""
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()
