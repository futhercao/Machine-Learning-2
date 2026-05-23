# 多变量时间序列预测

赛道二 提交代码。RevIN + 4 类模型（TSMixer / TimeMixer / PatchTST-Strong / iTransformer-Strong）多种子训练，对每个预测长度独立做 greedy forward-selection 集成。

## 结果
| 指标 | 数值 |
| --- | --- |
| 平均 val MSE (96/192/336/720) | **0.003632** |
| Demo set MSE | **0.003030** |
| 阈值 | < 0.005 |

**显著低于阈值，预计 +10 分。**

各预测长度集成详情：
| pred_len | val MSE | 模型数 |
| --- | --- | --- |
| 96  | 0.00300 | 6 |
| 192 | 0.00354 | 8 |
| 336 | 0.00382 | 5 |
| 720 | 0.00416 | 3 |

## 仓库结构
```
TimeSeriesProject/
├─ data_loader.py
├─ models.py             # legacy V1 builder（V2 通过它注册一些模型）
├─ models_v2.py          # TSMixer / TimeMixer / PatchTSTStrong / iTransformerStrong + RevIN
├─ train_v2.py           # 单模训练 (AdamW + warmup-cosine)
├─ train_v2_all.sh / train_gpu0.sh / train_gpu1.sh   # 多模型 × 多种子 × 4 horizons 批跑
├─ eval_v2.py            # 集成搜索（greedy / top-K），写 ensemble_config.json
├─ eval_demo.py          # 在官方 demo 集上算 MSE
├─ predict_v2.py         # 按 ensemble_config 跑集成预测 → 4 个 pred_{L}.npy
├─ make_submission_v2.py # 打包 SLIM 提交 zip
├─ ensemble_config.json  # 各 horizon 选定的模型列表
├─ checkpoints_v2/       # 22 个集成实际用到的 ckpt (84MB)
├─ results_v2/           # pred_96 / 192 / 336 / 720.npy
├─ 设计思路.md
└─ 运行说明.md
```

## 快速复现
```bash
# 1. 训练（可选，已附 ckpt）
bash train_v2_all.sh

# 2. 集成搜索（可选，ensemble_config.json 已存）
python eval_v2.py

# 3. 推理生成 4 个 pred_*.npy
python predict_v2.py

# 4. 打包提交
python make_submission_v2.py --names "张三-2021001,李四-2021002,王五-2021003"
```
