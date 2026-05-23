#!/bin/bash
# Train comprehensive ensemble on GPU.
# v1 models (1 seed each, fast) + v2 models (3 seeds each) for all horizons.

set -e
cd "$(dirname "$0")"
mkdir -p logs_v2

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=/opt/conda/bin/python

# ============== V1 models (1 seed, already strong baselines) ==============
V1_MODELS="dlinear dlinear_k51 dlinear_k15 dlinear_mlp nlinear itransformer"

for pl in 96 192 336 720; do
    for m in $V1_MODELS; do
        log="logs_v2/${m}_pl${pl}.log"
        echo "[v1] $m pl=$pl -> $log"
        $PY train.py --model $m --pred_len $pl --device cuda \
            --batch_size 256 --epochs 30 --lr 1e-3 \
            --ckpt_dir checkpoints_v2 \
            > "$log" 2>&1
        tail -3 "$log"
    done
done

# iTransformer-XL only for long horizons
for pl in 336 720; do
    log="logs_v2/itransformer_xl_pl${pl}.log"
    echo "[v1] itransformer_xl pl=$pl -> $log"
    $PY train.py --model itransformer_xl --pred_len $pl --device cuda \
        --batch_size 128 --epochs 30 --lr 5e-4 \
        --ckpt_dir checkpoints_v2 \
        > "$log" 2>&1
    tail -3 "$log"
done

# ============== V2 models with 3 seeds ==============
V2_MODELS="tsmixer timemixer patchtst_strong itransformer_strong"
SEEDS="2026 42 1234"

for pl in 96 192 336 720; do
    for m in $V2_MODELS; do
        for s in $SEEDS; do
            log="logs_v2/${m}_pl${pl}_s${s}.log"
            echo "[v2] $m pl=$pl seed=$s -> $log"
            # use smaller batch for big models on long horizon
            if [ "$m" = "itransformer_strong" ] || [ "$m" = "patchtst_strong" ]; then
                bs=128
            else
                bs=256
            fi
            $PY train_v2.py --model $m --pred_len $pl --seed $s \
                --device cuda --batch_size $bs --epochs 40 \
                --ckpt_dir checkpoints_v2 \
                > "$log" 2>&1
            tail -3 "$log"
        done
    done
done

echo "[ALL DONE]"
