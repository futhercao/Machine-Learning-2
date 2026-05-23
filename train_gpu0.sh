#!/bin/bash
# GPU 0: v1 models + v2 timemixer + v2 tsmixer
set -e
cd "$(dirname "$0")"
mkdir -p logs_v2
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/bin/python

V1="dlinear dlinear_k51 dlinear_k15 dlinear_mlp nlinear itransformer"
for pl in 96 192 336 720; do
    for m in $V1; do
        log="logs_v2/${m}_pl${pl}.log"
        echo "[g0 v1] $m pl=$pl"
        $PY train.py --model $m --pred_len $pl --device cuda \
            --batch_size 256 --epochs 30 --lr 1e-3 \
            --ckpt_dir checkpoints_v2 > "$log" 2>&1
        tail -1 "$log"
    done
done

V2A="timemixer tsmixer"
for pl in 96 192 336 720; do
    for m in $V2A; do
        for s in 2026 42 1234; do
            log="logs_v2/${m}_pl${pl}_s${s}.log"
            echo "[g0 v2] $m pl=$pl s=$s"
            $PY train_v2.py --model $m --pred_len $pl --seed $s \
                --device cuda --batch_size 256 --epochs 40 \
                --ckpt_dir checkpoints_v2 > "$log" 2>&1
            tail -1 "$log"
        done
    done
done

echo "[GPU0 ALL DONE]"
