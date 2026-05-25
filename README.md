# 多变量时间序列预测（赛道二）

100 维变量，输入过去 96 步，分别预测未来 96 / 192 / 336 / 720 步。RevIN + 4 类互补模型（TSMixer / TimeMixer / PatchTST-Strong / iTransformer-Strong）多种子训练，**对每个预测长度独立做 greedy forward-selection 集成**。

## 结果
| 指标 | 数值 |
| --- | --- |
| 平均 val MSE (96/192/336/720) | **0.003632** |
| Demo set MSE | **0.003030** |

各预测长度集成详情：
| pred_len | val MSE | 模型数 |
| --- | --- | --- |
| 96  | 0.00300 | 6 |
| 192 | 0.00354 | 8 |
| 336 | 0.00382 | 5 |
| 720 | 0.00416 | 3 |

---

## 一、目录结构

```
TimeSeriesProject/
├─ data_loader.py        # TSDataset + 训练/验证 split (无时间泄漏)
├─ models.py             # V1 builder (V2 fallback 时用到)
├─ models_v2.py          # 4 类 V2 模型 + RevIN
├─ train_v2.py           # 单 (model, pred_len, seed) 训练
├─ train_v2_all.sh / train_gpu0.sh / train_gpu1.sh   # 批量训练队列
├─ eval_v2.py            # 集成搜索（greedy + top-K）→ 写 ensemble_config.json
├─ eval_demo.py          # 用官方 demo 集算 MSE
├─ predict_v2.py         # 按 ensemble_config 跑推理 → 4 个 pred_<L>.npy
├─ dump_val.py           # 导出 val 输入与 ground truth（自检用）
├─ make_submission_v2.py # 打包提交 zip
├─ ensemble_config.json  # 各 horizon 选定的模型表
├─ checkpoints_v2/       # 22 个集成实际用到的 ckpt (84MB)
├─ results_v2/           # pred_96.npy / pred_192.npy / pred_336.npy / pred_720.npy
├─ 设计思路.pdf / .tex
└─ 运行说明.md
```

---

## 二、整体流程

```
[官方 train.csv (20000×100)]      [test_x.npy (评测时下发)]
        ↓                                    ↓
   按时间序最后 10% 划 val                (放置好即可)
        ↓                                    ↓
        ↓                       ┌────────────────────────────┐
        ↓                       │ 22 个集成用到的 ckpt        │
   train_v2.py × ~74            │ checkpoints_v2/*.pt        │
   (每 model × seed × pred_len) │ + ensemble_config.json     │
        ↓                       └─────────────┬──────────────┘
   checkpoints_v2/*.pt                        ↓
        ↓                              predict_v2.py
   eval_v2.py (贪心选集成)                    ↓
        ↓                              results_v2/pred_<L>.npy × 4
   ensemble_config.json                       ↓
                                       make_submission_v2.py
                                              ↓
                                  时间序列-<姓名>...zip
```

---

## 三、训练流程（已完成，无需重跑）

如要从零复现：

```bash
# 1) 训练 ~74 个 ckpt（4 model × 3 seed × 4 pred_len, 双卡约 8-10 小时）
bash train_v2_all.sh   # 或 train_gpu0.sh / train_gpu1.sh 分卡并行

# 2) 贪心搜索每个 horizon 的最优集成（输出 ensemble_config.json）
python eval_v2.py
```

训练超参（`train_v2.py` 默认）：AdamW + warmup-cosine（warmup 5%）、MSE loss、batch 64。每个 ckpt 文件名：`<model>_pl<L>_s<seed>.pt`，按 val MSE 选最优。

**git 仓库只附了集成实际用到的 22 个 ckpt**（slim 自 74）。要复现集成搜索过程，需要先重训得到完整 74 个 ckpt（或修改 `eval_v2.py` 让它只考虑已有的 ckpt）。

---

## 四、验证集 & 如何自检

**val 集是怎么来的？**
`data_loader.make_loaders(...)`：按时间顺序取 `train.csv` 的**最后 10%** 作为 val（约 2000 行），输入窗口可以横跨 train/val 边界（但 label 必须严格在 val 区域内），**无时间泄漏**。

注意：与赛道一不同，时间序列的 val 是**固定的末尾切片**，不是随机抽样。

**自检步骤**：

```bash
# A) 导出 val 输入与 ground truth（需要 train.csv）
python dump_val.py --train_csv <PATH_TO>/train/train.csv
# 输出: results_v2/val_truth/x_val_pl<L>.npy + y_val_pl<L>.npy + info.json
#       (每个 horizon 独立, x 形状 (N,96,100), y 形状 (N,L,100))

# B) 用当前集成预测 val 输入
python predict_v2.py --test_path results_v2/val_truth/x_val_pl96.npy --out_dir _val_pred_96
# 重复 4 次 (96/192/336/720), 然后比对 _val_pred_<L>/pred_<L>.npy vs val_truth/y_val_pl<L>.npy

# C) 最直接的方法: 重新跑 eval_v2.py, 它会输出每个 horizon 的 val MSE
python eval_v2.py
# 期望 avg val MSE ≈ 0.003632 (与 ensemble_config.json 里记录的一致)
```

**最快的自检 (无需训练数据)**：
直接看 `ensemble_config.json` 里每个 horizon 的 `val_mse` 字段，以及顶层 `avg_val_mse`。这是训练时记录的真实值，未来推理过程中再做的任何 val 评估应该跟这里一致。

```bash
python -c "import json; d=json.load(open('ensemble_config.json'));
print('avg val MSE:', d['avg_val_mse']);
[print(f'pl={k}: val_mse={v[\"val_mse\"]:.6f}') for k,v in d.items() if k!='avg_val_mse']"
```

也可以用 `eval_demo.py` 在官方 demo set 上验证（如有 demo 的 ground truth）：

```bash
python eval_demo.py --pred_dir results_v2 --gt_dir <PATH_TO_demo_gt>
# 期望: Average MSE ≈ 0.003030
```

---

## 五、测试集到达后的步骤（最常用）

> **假设场景**：评测方给一个 `test_x.npy`，形状 `(N, 96, 100)`，每行是一个长度 96 的输入窗口。需要为每个窗口分别预测未来 96/192/336/720 步。

```bash
# 1) 跑集成推理，生成 4 个 npy
python predict_v2.py --test_path <PATH_TO>/test_x.npy --out_dir results_v2
# -> results_v2/pred_96.npy  shape=(N, 96, 100)
# -> results_v2/pred_192.npy shape=(N,192,100)
# -> results_v2/pred_336.npy shape=(N,336,100)
# -> results_v2/pred_720.npy shape=(N,720,100)

# 2) 打包提交
python make_submission_v2.py --names "姓名1-学号1,姓名2-学号2,姓名3-学号3"
# 生成 时间序列-姓名1-学号1-姓名2-学号2-姓名3-学号3.zip
```

`predict_v2.py` 已实测在 CPU / GPU 均可运行（默认 `--device cuda`，无 GPU 自动 fallback CPU）。

注意：`predict_v2.py` 默认会把输出 `np.clip(pred, 0, 1)`（因为官方 train.csv 是 0-1 区间）。如果测试集不在该区间，加 `--no_clip` 关闭。

---

## 六、输出格式

四个 npy 文件，dtype 都是 float32：

| 文件 | 形状 | 含义 |
| --- | --- | --- |
| `pred_96.npy`  | (N, 96, 100)  | 未来 96 步预测 |
| `pred_192.npy` | (N, 192, 100) | 未来 192 步预测 |
| `pred_336.npy` | (N, 336, 100) | 未来 336 步预测 |
| `pred_720.npy` | (N, 720, 100) | 未来 720 步预测 |

---

## 七、关键设计点
详见 `设计思路.pdf`。简要：
- **RevIN**：每样本独立归一化 + 反归一化，解决非平稳问题
- **4 类互补模型**：MLP（TSMixer / TimeMixer）+ Transformer（PatchTST / iTransformer），覆盖不同建模偏好
- **Per-horizon ensemble**：短 horizon 用简单模型，长 horizon 用 Transformer 压住累计误差；每个 pred_len 各选各的子集
- **贪心选择**：从所有 ckpt 中循环加入"能让 val MSE 下降最多"的一个，直到加不动
