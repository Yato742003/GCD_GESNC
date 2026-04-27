import numpy as np
from src.snc.clustering import SNC

def run_snc(data, req_clust, distance='cosine', ensure_early_exit=True, verbose=True, labeled=None, mask=None):
    """
    Hàm Wrapper (bọc ngoài) cho thuật toán SNC để chạy chế độ Semi-supervised (Bán giám sát).
    
    Tham số:
        data: Ma trận vector đặc trưng (đã chuẩn hóa L2).
        req_clust: Số lượng cụm mục tiêu (K=100).
        distance: Hàm khoảng cách sử dụng để xây dựng đồ thị K-NN (mặc định: cosine).
        ensure_early_exit: Nếu True, sẽ thoát sớm khỏi các vòng lặp chia cụm không cần thiết, giúp tăng tốc độ.
        labeled: Mảng chứa nhãn (ground truth + pseudo-labels) đóng vai trò làm Neo (Anchors).
        mask: Mảng boolean (1/0) xác định mẫu nào là Neo, mẫu nào là Unlabeled.
        
    Trả về:
        prd: Mảng chứa tất cả các vách ngăn (partitions) được tạo ra trong quá trình gom cụm.
        num_clust: Danh sách số lượng cụm ở mỗi bước chia.
        req_c: Phân hoạch cuối cùng có đúng 'req_clust' cụm.
    """
    # Gọi hàm SNC gốc từ thư viện CiPR với các tham số tương ứng
    # Hàm gốc trả về (c, num_clust, req_c, d_all)
    prd, num_clust, req_c, _ = SNC(
        data=data,
        req_clust=req_clust,
        distance=distance,
        ensure_early_exit=ensure_early_exit,
        verbose=verbose,
        labeled=labeled,
        mask=mask
    )
    
    return prd, num_clust, req_c
