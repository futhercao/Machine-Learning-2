#!/bin/bash
# GPU 1: v1 itransformer_xl (long horizons) + v2 patchtst_strong + v2 itransformer_strong
set -e
cd "$(dirname "$0")"
mkdir -p logs_v2
export CUDA_VISIBLE_DEVICES=1
PY=/opt/conda/bin/python

# itransformer_xl for pl=336, 720
for pl in 336 720; do
    log="logs_v2/itransformer_xl_pl${pl}.log"
    echo "[g1 v1] itransformer_xl pl=$pl"
    $PY train.py --model itransformer_xl --pred_len $pl --device cuda \
        --batch_size 128 --epochs 30 --lr 5e-4 \
        --ckpt_dir checkpoints_v2 > "$log" 2>&1
    tail -1 "$log"
done

V2B="patchtst_strong itransformer_strong"
for pl in 96 192 336 720; do
    for m in $V2B; do
        for s in 2026 42 1234; do
            log="logs_v2/${m}_pl${pl}_s${s}.log"
            echo "[g1 v2] $m pl=$pl s=$s"
            $PY train_v2.py --model $m --pred_len $pl --seed $s \
                --device cuda --batch_size 128 --epochs 40 \
                --ckpt_dir checkpoints_v2 > "$log" 2>&1
            tail -1 "$log"
        done
    done
done

echo "[GPU1 ALL DONE]"
