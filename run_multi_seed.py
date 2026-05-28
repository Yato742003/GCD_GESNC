import os
import sys
import subprocess
import csv
import numpy as np
from scipy.stats import ttest_rel

# Danh sách hạt giống ngẫu nhiên cho thực nghiệm
CIFAR100_SEEDS = [0, 1, 2, 42, 123]
CUB200_SEEDS = [0, 1, 2]

# Cấu hình tham số dòng lệnh cho từng tập dữ liệu và phương pháp
CONFIGS = {
    "cifar100": {
        "script": "main_eval.py",
        "seeds": CIFAR100_SEEDS,
        "base_args": ["--protocol", "transductive", "--pseudo_scope", "train"],
        "cipr": ["--pct", "0"],
        "gesnc": ["--pct", "10", "--m", "16", "--gamma", "0.05", "--react", "--react_q", "0.99"]
    },
    "cub200": {
        "script": "eval_cub_gesnc.py",
        "seeds": CUB200_SEEDS,
        "base_args": ["--protocol", "transductive"],
        "cipr": ["--pct", "0"],
        "gesnc": ["--pct", "10", "--m", "8", "--gamma", "0.1"]
    }
}


def read_results_from_csv(csv_path):
    results = []
    if not os.path.exists(csv_path):
        return results
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "seed": int(row["seed"]),
                "protocol": row["protocol"],
                "pct": int(row["pct"]),
                "train_all": float(row["train_all"]),
                "train_old": float(row["train_old"]),
                "train_new": float(row["train_new"]),
                "train_h": float(row["train_h"]),
                "test_all": float(row["test_all"]),
                "test_old": float(row["test_old"]),
                "test_new": float(row["test_new"]),
                "test_h": float(row["test_h"])
            })
    return results


def run_experiment(dataset_name, config):
    print("=" * 70)
    print(f" KHỞI CHẠY THỰC NGHIỆM ĐA HẠT GIỐNG CHO TẬP DỮ LIỆU: {dataset_name.upper()} ")
    print("=" * 70)
    
    script = config["script"]
    seeds = config["seeds"]
    base_args = config["base_args"]
    
    # Tạo thư mục lưu kết quả tạm thời
    os.makedirs("results", exist_ok=True)
    csv_cipr = f"results/{dataset_name}_cipr_temp.csv"
    csv_gesnc = f"results/{dataset_name}_gesnc_temp.csv"
    
    # Xóa file cũ nếu có để tránh cộng dồn kết quả cũ
    if os.path.exists(csv_cipr):
        os.remove(csv_cipr)
    if os.path.exists(csv_gesnc):
        os.remove(csv_gesnc)
        
    # 1. Chạy CiPR Baseline
    print(f"\n>>> [1/2] Bắt đầu chạy CiPR Baseline ({len(seeds)} hạt giống)...")
    for seed in seeds:
        print(f"  --> Đang chạy seed={seed}...")
        cmd = [sys.executable, script] + base_args + config["cipr"] + ["--seed", str(seed), "--output_csv", csv_cipr]
        subprocess.run(cmd, check=True)
        
    # 2. Chạy GESNC Đề xuất
    print(f"\n>>> [2/2] Bắt đầu chạy GESNC Đề xuất ({len(seeds)} hạt giống)...")
    for seed in seeds:
        print(f"  --> Đang chạy seed={seed}...")
        cmd = [sys.executable, script] + base_args + config["gesnc"] + ["--seed", str(seed), "--output_csv", csv_gesnc]
        subprocess.run(cmd, check=True)

    # Đọc kết quả thu được, chỉ lấy protocol=transductive
    res_cipr = [r for r in read_results_from_csv(csv_cipr) if r["protocol"] == "transductive"]
    res_gesnc = [r for r in read_results_from_csv(csv_gesnc) if r["protocol"] == "transductive"]

    if len(res_cipr) != len(seeds) or len(res_gesnc) != len(seeds):
        print(f"[WARNING] Số lượng kết quả không khớp: CiPR={len(res_cipr)}, GESNC={len(res_gesnc)}, seeds={len(seeds)}")

    return res_cipr, res_gesnc


def analyze_and_format(dataset_name, cipr_runs, gesnc_runs):
    print("\n" + "=" * 70)
    print(f" PHÂN TÍCH THỐNG KÊ VÀ KIỂM ĐỊNH T-TEST CẶP: {dataset_name.upper()} ")
    print("=" * 70)
    
    metrics = ["train_all", "train_old", "train_new", "train_h"]
    metric_labels = {
        "train_all": "All ACC",
        "train_old": "Old ACC",
        "train_new": "New ACC",
        "train_h": "H-score"
    }
    
    summary = {}
    
    for m in metrics:
        cipr_vals = np.array([run[m] for run in cipr_runs]) * 100.0  # chuyển sang %
        gesnc_vals = np.array([run[m] for run in gesnc_runs]) * 100.0
        
        # ddof=1 cho sample standard deviation (chuẩn cho báo cáo khoa học mẫu nhỏ)
        cipr_mean, cipr_std = np.mean(cipr_vals), np.std(cipr_vals, ddof=1)
        gesnc_mean, gesnc_std = np.mean(gesnc_vals), np.std(gesnc_vals, ddof=1)
        diff_mean = gesnc_mean - cipr_mean

        # Paired t-test (yêu cầu 2 mảng cùng kích thước)
        if len(cipr_vals) > 1 and len(cipr_vals) == len(gesnc_vals):
            t_stat, p_val = ttest_rel(gesnc_vals, cipr_vals)
        else:
            t_stat, p_val = 0.0, 1.0
            
        summary[m] = {
            "cipr_mean": cipr_mean, "cipr_std": cipr_std,
            "gesnc_mean": gesnc_mean, "gesnc_std": gesnc_std,
            "diff": diff_mean,
            "t_stat": t_stat, "p_value": p_val
        }
        
        sig_text = "Cực kỳ ý nghĩa (p < 0.01)" if p_val < 0.01 else \
                   "Có ý nghĩa (p < 0.05)" if p_val < 0.05 else \
                   "Không có ý nghĩa thống kê (p >= 0.05)"
                   
        print(f"\nChỉ số {metric_labels[m]}:")
        print(f"  CiPR Baseline : {cipr_mean:.2f}% ± {cipr_std:.2f}%")
        print(f"  GESNC Đề xuất : {gesnc_mean:.2f}% ± {gesnc_std:.2f}%")
        print(f"  Độ chênh lệch : {diff_mean:+.2f}%")
        print(f"  Paired t-test : t-value = {t_stat:.4f}, p-value = {p_val:.4f} ({sig_text})")

    # In mã nguồn LaTeX để chèn trực tiếp vào luận văn
    print("\n" + "-" * 70)
    print(f" MÃ NGUỒN BẢNG LATEX CHO LUẬN VĂN ({dataset_name.upper()}) ")
    print("-" * 70)
    
    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{So sánh hiệu năng đa hạt giống ({dataset_name.upper()}) giữa CiPR và GESNC}}
\\label{{tab:multi_seed_{dataset_name}}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Phương pháp}} & \\textbf{{All ACC (\\%)}} & \\textbf{{Old ACC (\\%)}} & \\textbf{{New ACC (\\%)}} & \\textbf{{H-score (\\%)}} \\\\
\\midrule
CiPR Baseline & {summary["train_all"]["cipr_mean"]:.2f} $\\pm$ {summary["train_all"]["cipr_std"]:.2f} 
              & {summary["train_old"]["cipr_mean"]:.2f} $\\pm$ {summary["train_old"]["cipr_std"]:.2f} 
              & {summary["train_new"]["cipr_mean"]:.2f} $\\pm$ {summary["train_new"]["cipr_std"]:.2f} 
              & {summary["train_h"]["cipr_mean"]:.2f} $\\pm$ {summary["train_h"]["cipr_std"]:.2f} \\\\
\\textbf{{GESNC (Đề xuất)}} & \\textbf{{{summary["train_all"]["gesnc_mean"]:.2f} $\\pm$ {summary["train_all"]["gesnc_std"]:.2f}}} 
                          & \\textbf{{{summary["train_old"]["gesnc_mean"]:.2f} $\\pm$ {summary["train_old"]["gesnc_std"]:.2f}}} 
                          & \\textbf{{{summary["train_new"]["gesnc_mean"]:.2f} $\\pm$ {summary["train_new"]["gesnc_std"]:.2f}}} 
                          & \\textbf{{{summary["train_h"]["gesnc_mean"]:.2f} $\\pm$ {summary["train_h"]["gesnc_std"]:.2f}}} \\\\
\\midrule
\\textit{{Cải thiện (\\%)}} & \\textit{{{summary["train_all"]["diff"]:+.2f}\\%}} 
                          & \\textit{{{summary["train_old"]["diff"]:+.2f}\\%}} 
                          & \\textit{{{summary["train_new"]["diff"]:+.2f}\\%}} 
                          & \\textit{{{summary["train_h"]["diff"]:+.2f}\\%}} \\\\
\\textit{{Trị số p (t-test)}} & {summary["train_all"]["p_value"]:.4f} 
                           & {summary["train_old"]["p_value"]:.4f} 
                           & {summary["train_new"]["p_value"]:.4f} 
                           & {summary["train_h"]["p_value"]:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    print(latex_table)
    print("=" * 70)
    
    # Ghi nhận kết quả phân tích tổng hợp vào kết quả chính thức
    final_csv = f"results/{dataset_name}_multi_seed_final.csv"
    with open(final_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "cipr_mean", "cipr_std", "gesnc_mean", "gesnc_std", "diff", "t_stat", "p_value"])
        for m in metrics:
            writer.writerow([
                metric_labels[m], 
                summary[m]["cipr_mean"], summary[m]["cipr_std"],
                summary[m]["gesnc_mean"], summary[m]["gesnc_std"],
                summary[m]["diff"], summary[m]["t_stat"], summary[m]["p_value"]
            ])
    print(f"Báo cáo thống kê chính thức lưu tại: {final_csv}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm đa hạt giống và kiểm định paired t-test")
    parser.add_argument("--dataset", choices=("cifar100", "cub200", "both"), default="both")
    args = parser.parse_args()
    
    datasets_to_run = []
    if args.dataset in ("cifar100", "both"):
        datasets_to_run.append("cifar100")
    if args.dataset in ("cub200", "both"):
        datasets_to_run.append("cub200")
        
    for dataset in datasets_to_run:
        cipr_runs, gesnc_runs = run_experiment(dataset, CONFIGS[dataset])
        analyze_and_format(dataset, cipr_runs, gesnc_runs)


if __name__ == "__main__":
    main()
