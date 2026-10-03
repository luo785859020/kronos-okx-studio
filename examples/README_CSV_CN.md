# 本地 CSV → Kronos-small → 预测结果

这个入口用于先验证数据、模型和运行环境，不启动 Web UI，也不执行训练。

默认使用仓库内 `tests/data/regression_input.csv`：前 256 行作为历史，后 30 行作为留出区间。模型只能看到历史行情和目标时间戳；留出行情仅用于计算误差。这是历史区间预测验证，不是实时行情预测，也不能用一次窗口的误差说明策略盈利能力。

## 1. 运行环境

按主 README 在仓库根目录创建 `.venv` 并安装依赖后，可以直接调用环境内的 Python，不需要激活，也不会更改系统 Python。以下命令从仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe examples\predict_csv.py --validate-only
.\.venv\Scripts\python.exe examples\predict_csv.py
```

本机使用 Python 3.11、NumPy 1.26.4、Pandas 2.2.2；初始 CPU 验证使用 PyTorch 2.6.0，后续已升级到 PyTorch 2.11.0+cu128 并验证 CPU 与 RTX 5060 Ti GPU 推理。其他环境请先检查 PyTorch 构建与 `torch.cuda.is_available()`，安装方法见主 README。

若在其他机器复现初始 CPU 环境，先准备 Python 3.11，并从项目根目录执行。GPU 安装方案见主 README：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt numpy==1.26.4 pytest==8.3.5 PyYAML==6.0.2
.\.venv\Scripts\python.exe -m pip check
```

首次推理需要联网下载 Small 模型和 Tokenizer。入口固定使用与原有回归测试相同的模型 revision，避免上游权重更新改变结果。下载完成后可增加 `--local-files-only` 使用本地缓存；模型保存在 Hugging Face 默认缓存目录，不提交到 Git。

## 2. CSV 数据要求

必需列：`timestamps, open, high, low, close`。可选列：`volume, amount`。列顺序不限。

```csv
timestamps,open,high,low,close,volume,amount
2024-06-18 11:15:00,11.27,11.28,11.26,11.27,379,427161
2024-06-18 11:20:00,11.27,11.28,11.27,11.27,277,312192
```

- 时间戳应为交易所本地时间、不带时区、严格递增且不重复。不会静默排序或删除数据。
- 行情列必须为有限数值，不接受 NaN 或无穷大。
- 缺少 `volume` 时，核心预测器将 `volume` 和 `amount` 填零；有 `volume` 但缺少 `amount` 时，用成交量乘以 OHLC 均价估算。
- 行情周期和复权口径应统一。不要将多个标的直接拼接成一条时间序列。
- `lookback` 范围是 2–512，`pred-len` 范围是 1–512；后者受当前解码实现的窗口长度约束。
- 格式检查不等于完整的行情质量或交易规则校验。

## 3. 使用自己的 CSV

从项目根目录执行，例如使用仓库附带的港股 CSV：

```powershell
.\.venv\Scripts\python.exe examples\predict_csv.py --csv finetune_csv\data\HK_ali_09988_kline_5min_all.csv --lookback 256 --pred-len 30 --output-dir outputs\csv-hk-small
```

默认留出模式至少需要 `start + lookback + pred-len` 行。`--start` 是从零开始的历史窗口起点，默认 0。只检查格式、不下载权重时加 `--validate-only`。

默认 `--top-k 1 --top-p 1 --sample-count 1 --seed 123 --threads 4`，用于复现最小链路。希望进行随机采样时，可以使用 `--top-k 0 --top-p 0.9 --sample-count 5`。固定 seed 不保证跨硬件、跨依赖版本逐位一致。

## 4. 预测真正未知的未来区间

额外提供一份只有 `timestamps` 列的 CSV，恰好包含 `pred-len` 个未来交易时间戳：

```powershell
.\.venv\Scripts\python.exe examples\predict_csv.py --csv data\history.csv --future-timestamps data\future_times.csv --lookback 256 --pred-len 30 --output-dir outputs\future-small
```

此模式使用行情文件最后 `lookback` 行，所有目标时间戳必须晚于最后一条历史数据，不支持 `--start`。需要调用方依据交易日历处理午休、周末、节假日；脚本不会自行推算交易时段。没有真实未来价格时，不计算预测误差。

## 5. 输出文件

默认目录为项目根下的 `outputs/csv-small`：

- `prediction.csv`：时间戳与六列预测行情。
- `history.csv`：实际输入模型的历史窗口，便于追溯。
- `comparison.csv`：留出区间 OHLC 真实值和预测值。
- `run.json`：数据路径及 SHA-256、模型 revision、环境版本、窗口、采样参数、时间范围，以及留出区间逐列 MAE/RMSE。

重复使用同一输出目录会覆盖上述结果；切换到未来预测模式会移除旧的 `comparison.csv`。保存不同实验时请使用不同的 `--output-dir`。`outputs/`、环境与下载缓存已加入 Git 忽略规则。

输出保留模型原始连续数值，不额外强制 OHLC 关系或成交量/成交额非负。例如输入成交额恒为零时，预测可能有接近零的微小负数；接入业务系统前需要单独设计输出约束。

## 6. 归一化修复与测试

`finetune_csv/finetune_base_model.py` 中的 `CustomKlineDataset` 现在仅用前 `lookback_window` 行计算均值、标准差，然后对整个训练窗口使用相同统计量并裁剪到 `[-clip, clip]`。Tokenizer 与预测器训练共用这个 Dataset，因此两条训练路径都获得修复。

新增测试覆盖格式校验、窗口边界、未来时间戳、结果对齐和未来数据不影响历史归一化。运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_csv_workflow.py -q
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -m pytest tests -q
```

后一条命令还执行原有真实模型回归测试，需要已有权重缓存或网络连接。此阶段不启动正式微调；训练前仍需准备配置中的数据/模型路径，以及独立的时间外测试集。

## 7. 本机验证记录

2026-10-03 已使用 Python 3.11.17 / PyTorch 2.6.0+cpu 完成：

- 24 项新增测试与 4 项原有模型回归测试，合计 28 项全部通过。
- 2,500 行回归 CSV 与 93,912 行港股 CSV 均通过校验，并分别完成 256 行历史 → 30 行预测。
- 结果分别位于 `outputs/csv-small` 与 `outputs/csv-hk-small`。
- 将同一历史窗口与未来时间戳单独保存后，以 `--future-timestamps` 离线复跑；在不提供未来价格的情况下，预测与留出模式完全一致。这个验证结果位于 `outputs/csv-future-small`，并非实时行情预测。
- `pip check` 通过；已安装依赖清单位于 `outputs/environment-cpu.txt`。
