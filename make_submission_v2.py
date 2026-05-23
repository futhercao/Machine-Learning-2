"""
Package SLIM submission: only essential code + ckpts actually used in ensemble_config.json.

Usage:
    python make_submission_v2.py --names "张三-2021001,李四-2021002,王五-2021003"
"""
import argparse
import json
import os
import zipfile
from pathlib import Path


# Only essential files for reproduction + inference
CODE_FILES = [
    'data_loader.py',
    'models.py',          # needed because v1 builder still referenced
    'models_v2.py',
    'train_v2.py',
    'train_gpu0.sh',
    'train_gpu1.sh',
    'predict_v2.py',
    'eval_demo.py',
    'eval_v2.py',
    'ensemble_config.json',
    '设计思路.md',
    '运行说明.md',
]


def disp_to_ckpt(disp, pl):
    """ensemble model name -> ckpt filename"""
    # e.g. 'itransformer_strong_s1234' -> 'itransformer_strong_pl<L>_s1234.pt'
    if '_s' in disp and disp.rsplit('_s', 1)[1].isdigit():
        builder, s = disp.rsplit('_s', 1)
        return f"{builder}_pl{pl}_s{s}.pt"
    return f"{disp}_pl{pl}.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--names', required=True)
    p.add_argument('--results_dir', default='results_v2')
    p.add_argument('--ckpt_dir', default='checkpoints_v2')
    p.add_argument('--config', default='ensemble_config.json')
    p.add_argument('--out_zip', default=None)
    args = p.parse_args()

    members = [s.strip() for s in args.names.split(',') if s.strip()]
    base = '时间序列-' + '-'.join(members)
    if args.out_zip is None:
        args.out_zip = f"{base}.zip"

    proj = Path(__file__).resolve().parent

    # Determine which ckpts are actually used
    with open(proj / args.config) as f:
        cfg = json.load(f)
    used_ckpts = set()
    for k, v in cfg.items():
        if k == 'avg_val_mse': continue
        for disp in v['models']:
            used_ckpts.add(disp_to_ckpt(disp, k))

    pred_files = [proj / args.results_dir / f'pred_{L}.npy' for L in [96, 192, 336, 720]]
    for f in pred_files:
        assert f.exists(), f"missing prediction: {f}"

    with zipfile.ZipFile(args.out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # code
        for name in CODE_FILES:
            src = proj / name
            if src.exists():
                zf.write(src, arcname=f"{base}/{name}")
                print(f"  + {name}")
            else:
                print(f"  [missing] {name}")
        # only used ckpts
        for ckpt_name in sorted(used_ckpts):
            src = proj / args.ckpt_dir / ckpt_name
            if not src.exists():
                print(f"  [WARN] ckpt missing: {ckpt_name}")
                continue
            zf.write(src, arcname=f"{base}/{args.ckpt_dir}/{ckpt_name}")
        print(f"  + {len(used_ckpts)} checkpoints (slim)")
        # outputs
        for f in pred_files:
            zf.write(f, arcname=f"{base}/{f.name}")
            print(f"  + {f.name}")

    print(f"\n[done] {args.out_zip}  ({os.path.getsize(args.out_zip) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
