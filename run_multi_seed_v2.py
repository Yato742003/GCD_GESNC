"""
run_multi_seed_v2.py — Phiên bản 2: Config tối ưu (M=16, γ=0.05)
=================================================================
Khác biệt so với v1 (run_multi_seed.py):
  - CIFAR-100: M=16, γ=0.05 (thay vì M=8, γ=0.2)
  - CUB-200:   M=16, γ=0.05 (thay vì M=8, γ=0.1)
  - Vẫn dùng --pseudo_scope train (giống v1, fair comparison)
  - Output file: *_v2_final.csv (tránh ghi đè v1)

Chạy: python run_multi_seed_v2.py --dataset both
"""

import os
import sys
import subprocess
import csv
import numpy as np
from scipy.stats import ttest_rel

# Danh sách hạt giống ngẫu nhiên cho thực nghiệm
CIFAR100_SEEDS = [0, 1, 2, 42, 123]
CUB200_SEEDS = [0, 1, 2]

# ═══════════════════════════════════════════════════════════════════
#  CONFIG V2 — Khác v1: M=16, γ=0.05 (vẫn giữ pseudo_scope=train)
# ═══════════════════════════════════════════════════════════════════
CONFIGS = {
    "cifar100": {
        "script": "main_eval.py",
        "seeds": CIFAR100_SEEDS,
        "base_args": ["--protocol", "transductive", "--pseudo_scope", "train"],
        "cipr": ["--pct", "0"],
        "gesnc": ["--pct", "10", "--m", "16", "--gamma", "0.05", "--react", "--react_q", "0.99"]
        #                         ↑ M=16       ↑ γ=0.05  (v1: M=8, γ=0.2)
    },
    "cub200": {
        "script": "eval_cub_gesnc.py",
        "seeds": CUB200_SEEDS,
        "base_args": ["--protocol", "transductive"],
        "cipr": ["--pct", "0"],
        "gesnc": ["--pct", "10", "--m", "16", "--gamma", "0.05"]
        #                         ↑ M=16       ↑ γ=0.05  (v1: M=8, γ=0.1)
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
    print(f" [V2] KHỞI CHẠY THỰC NGHIỆM ĐA HẠT GIỐNG: {dataset_name.upper()} ")
    print(f"  Config: M=16, γ=0.05, pseudo_scope=train")
    print("=" * 70)
    
    script = config["script"]
    seeds = config["seeds"]
    base_args = config["base_args"]
    
    os.makedirs("results", exist_ok=True)
    
    # CiPR baseline (PCT=0) là deterministic → dùng lại kết quả v1
    csv_cipr = f"results/{dataset_name}_cipr_temp.csv"
    if os.path.exists(csv_cipr):
        print(f"\n>>> [SKIP] CiPR baseline: dùng lại kết quả từ {csv_cipr} (deterministic)")
    else:
        print(f"\n[ERROR] Không tìm thấy {csv_cipr}! Chạy run_multi_seed.py trước để có baseline.")
        sys.exit(1)
    
    # Chỉ chạy GESNC V2 (M=16, γ=0.05)
    csv_gesnc = f"results/{dataset_name}_gesnc_v2_temp.csv"
    if os.path.exists(csv_gesnc):
        os.remove(csv_gesnc)
        
    print(f"\n>>> Chạy GESNC V2 — M=16, γ=0.05, PCT=10 ({len(seeds)} seeds)...")
    for seed in seeds:
        print(f"  --> seed={seed}...")
        cmd = [sys.executable, script] + base_args + config["gesnc"] + ["--seed", str(seed), "--output_csv", csv_gesnc]
        subprocess.run(cmd, check=True)

    # Đọc kết quả
    res_cipr = [r for r in read_results_from_csv(csv_cipr) if r["protocol"] == "transductive"]
    res_gesnc = [r for r in read_results_from_csv(csv_gesnc) if r["protocol"] == "transductive"]

    if len(res_gesnc) != len(seeds):
        print(f"[WARNING] Số kết quả GESNC không khớp: {len(res_gesnc)} vs {len(seeds)} seeds")

    return res_cipr, res_gesnc


def analyze_and_format(dataset_name, cipr_runs, gesnc_runs):
    print("\n" + "=" * 70)
    print(f" [V2] PHÂN TÍCH THỐNG KÊ: {dataset_name.upper()} ")
    print(f"  GESNC config: M=16, γ=0.05, PCT=10, pseudo_scope=train")
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
        cipr_vals = np.array([run[m] for run in cipr_runs]) * 100.0
        gesnc_vals = np.array([run[m] for run in gesnc_runs]) * 100.0
        
        cipr_mean, cipr_std = np.mean(cipr_vals), np.std(cipr_vals, ddof=1)
        gesnc_mean, gesnc_std = np.mean(gesnc_vals), np.std(gesnc_vals, ddof=1)
        diff_mean = gesnc_mean - cipr_mean

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
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                   
        print(f"\n  {metric_labels[m]}:")
        print(f"    CiPR  : {cipr_mean:.2f}% ± {cipr_std:.2f}%")
        print(f"    GESNC : {gesnc_mean:.2f}% ± {gesnc_std:.2f}%")
        print(f"    Δ     : {diff_mean:+.2f}%  (t={t_stat:.3f}, p={p_val:.4f} {sig})")

    # So sánh v1 vs v2
    print("\n" + "-" * 70)
    print(" SO SÁNH V1 (M=8, γ=0.2) vs V2 (M=16, γ=0.05)  [cùng scope=train]")
    print("-" * 70)
    v1_csv = f"results/{dataset_name}_multi_seed_final.csv"
    if os.path.exists(v1_csv):
        with open(v1_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                m_label = row["metric"]
                v1_mean = float(row["gesnc_mean"])
                # Tìm metric key tương ứng
                for mk, ml in metric_labels.items():
                    if ml == m_label:
                        v2_mean = summary[mk]["gesnc_mean"]
                        print(f"  {m_label:10s}: v1={v1_mean:.2f}%  →  v2={v2_mean:.2f}%  (Δ={v2_mean - v1_mean:+.2f}%)")
    else:
        print(f"  [INFO] File v1 ({v1_csv}) không tìm thấy — bỏ qua so sánh.")

    # Ghi CSV chính thức v2
    final_csv = f"results/{dataset_name}_multi_seed_v2_final.csv"
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
    print(f"\n✅ Kết quả V2 lưu tại: {final_csv}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-seed V2: M=16, γ=0.05, pseudo_scope=train")
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
