#!/bin/bash
# ablation_m_gamma.sh — Grid search M x gamma cho GEN score
# PCT=15 (best từ ablation PCT), pretrain=epoch60
# Chạy: bash ablation_m_gamma.sh | tee ablation_m_gamma.log

PRETRAIN=~/cipr_cub200/CiPR/checkpoints/run/cipr_cub/00060.pth
PCT=15
SCRIPT=~/cipr_cub200/GCD_GESNC/eval_cub_gesnc.py

echo "========================================================"
echo "  GEN Ablation: M x gamma grid search (PCT=$PCT)"
echo "========================================================"

for M in 4 8 16; do
    for GAMMA in 0.05 0.1 0.2; do
        echo ""
        echo ">>> M=$M | gamma=$GAMMA"
        python $SCRIPT \
            --pct $PCT \
            --m $M \
            --gamma $GAMMA \
            --pretrain $PRETRAIN \
        | grep -E "(All ACC|Old ACC|New ACC|H-score|Anchors|>>>)"
        echo "--------------------------------------------------------"
    done
done

echo ""
echo "========================================================"
echo "  Done! Grid search complete."
echo "========================================================"
