import numpy as np

def compute_gen_score(logits, M=16, gamma=0.15):
    """
    Tính điểm Generalized Entropy Score (GES) để phát hiện Out-of-Distribution (OOD).
    Tham khảo: Báo cáo CiPR (Liu et al. 2023)
    
    Tham số:
        logits (np.ndarray): Đầu ra thô của classifier (Linear Head). Kích thước: (N, num_classes)
        M (int): Số lượng class có xác suất cao nhất được xét để tính điểm (Truncation).
        gamma (float): Siêu tham số kiểm soát độ sắc nét của phân phối xác suất.
        
    Trả về:
        np.ndarray: Mảng điểm GEN cho mỗi mẫu. Kích thước: (N,)
        Ghi chú: Điểm GEN thấp -> Mô hình tự tin mẫu đó thuộc nhóm Known.
    """
    # 1. Chuyển đổi logits sang probabilities bằng hàm Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 2. Sắp xếp xác suất giảm dần (Class xác suất cao nhất nằm đầu tiên)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]

    # 3. Truncation: Chỉ lấy M xác suất cao nhất để loại bỏ nhiễu từ các class đuôi dài
    top_m_probs = sorted_probs[:, :M]

    # 4. Công thức tính Generalized Entropy (GES)
    # GES(p) = tổng_m=1^M [ (p_m)^gamma * (1 - p_m)^gamma ]
    # GES là hàm lồi, giúp trừng phạt các dự đoán có phân phối đồng đều (uncertainty).
    ges_scores = np.sum((top_m_probs ** gamma) * ((1.0 - top_m_probs) ** gamma), axis=1)

    return ges_scores
