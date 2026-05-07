#!/bin/bash
# ablation_cifar_full.sh — Ablation PCT + M x gamma cho CIFAR-100
# Chạy: nohup bash ablation_cifar_full.sh > ~/cipr_cub200/logs/ablation_cifar.log 2>&1 &

FEAT_DIR=~/GCD_GESNC/features
SCRIPT=~/cipr_cub200/GCD_GESNC/main_eval.py

# ── Part 1: PCT Ablation (M=8, gamma=0.1 fixed) ──────────────────────────────
echo "========================================================"
echo "  CIFAR-100 Ablation Part 1: PCT sweep (M=8, gamma=0.1)"
echo "========================================================"

for PCT in 0 5 10 15 20; do
    echo ""
    echo ">>> PCT=$PCT"
    python $SCRIPT --pct $PCT --m 8 --gamma 0.1 --feat_dir $FEAT_DIR \
    | grep -E "(All ACC|Old ACC|New ACC|H-score|Anchors|Delta|>>>)"
    echo "--------------------------------------------------------"
done

# ── Part 2: M x gamma Grid Search (PCT=10 fixed) ─────────────────────────────
echo ""
echo "========================================================"
echo "  CIFAR-100 Ablation Part 2: M x gamma (PCT=10)"
echo "========================================================"

for M in 4 8 16; do
    for GAMMA in 0.05 0.1 0.2; do
        echo ""
        echo ">>> M=$M | gamma=$GAMMA"
        python $SCRIPT --pct 10 --m $M --gamma $GAMMA --feat_dir $FEAT_DIR \
        | grep -E "(All ACC|Old ACC|New ACC|H-score|Anchors|Delta|>>>)"
        echo "--------------------------------------------------------"
    done
done

echo ""
echo "========================================================"
echo "  Done! CIFAR-100 ablation complete."
echo "========================================================"
