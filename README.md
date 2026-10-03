# Kronos OKX Studio

**本地 CSV 验证 · 欧易 OKX 只读行情 · CPU / NVIDIA GPU 推理 · 中文可视化看板**

Kronos OKX Studio 是基于 [Kronos](https://github.com/shiyu-coder/Kronos) 构建的本地金融时序预测工作台。项目先打通“CSV → Kronos-small → 可追溯预测结果”，再接入 OKX 公开现货 K 线与浏览器交互，适合研究、数据验证和模型实验。

本项目是非官方应用扩展，不是 OKX 官方客户端，也不声称原创训练了 Kronos 基础模型。模型代码、权重名称与原作者署名均保留；上游说明见 [README_UPSTREAM.md](README_UPSTREAM.md)，许可证见 [LICENSE](LICENSE)。

> **只预测，不交易。** OKX 集成不读取 API 密钥、不访问账户、不创建订单。页面中的预测变化率不是实际收益、置信度或买卖建议。当前不提供完整的实盘执行系统，也没有为实时预测建立经交易成本验证的收益承诺。

## 目录

- [功能与边界](#功能与边界)
- [快速开始](#快速开始)
- [CPU 与 GPU 配置](#cpu-与-gpu-配置)
- [本地 CSV 最小链路](#本地-csv-最小链路)
- [OKX 实时预测看板](#okx-实时预测看板)
- [命令行实时预测](#命令行实时预测)
- [数据格式与时间约定](#数据格式与时间约定)
- [输出与记录隔离](#输出与记录隔离)
- [测试与实测结果](#测试与实测结果)
- [CSV 微调与归一化](#csv-微调与归一化)
- [目录结构](#目录结构)
- [常见问题](#常见问题)
- [安全与发布说明](#安全与发布说明)
- [来源、致谢与引用](#来源致谢与引用)

## 功能与边界

| 功能 | 当前实现 |
| --- | --- |
| 本地 CSV | 校验字段、时间顺序、缺失值和有限数值；支持历史留出验证及显式未来时间戳 |
| 默认模型 | 固定版本的 Kronos-small 与 Kronos-Tokenizer-base |
| OKX 行情 | REST 初始化历史，WebSocket 接收现货 K 线，只使用已收盘行情 |
| 币对选择 | BTC、DOGE（狗狗币）、ETH、SOL 固定下拉列表；其他现货币对可手动输入 |
| 推理设备 | CPU 或可用的 NVIDIA CUDA GPU；检测显卡与实际 CUDA 计算能力 |
| 可视化 | 历史 / 预测蜡烛图、价格卡片、连接状态、运行日志、历史记录与跟随最新 |
| 异常处理 | 行情连续性及新鲜度检查、断线重连、旧窗口去重；异常时不伪造新预测 |
| 结果追溯 | 保存输入、预测、模型版本、参数、设备、推理时间和数据哈希 |
| CSV 训练 | 修复归一化统计读取未来目标的问题；保留上游训练入口 |

**明确限制：**

- OKX 看板一次只管理一个预测进程，不是多币种并行交易终端。
- 支持现货，例如 `DOGE-USDT`；不接受 `BTC-USDT-SWAP` 合约币对。
- OKX 看板使用 Kronos-small；上游 CSV 页面和示例保留了其他模型入口。
- 只针对本机访问设计，不应直接部署到公网。
- 模型输出保留原始数值，不保证每根预测 K 线满足业务约束；研究者应独立检查 OHLC、成交量和成交额。

## 快速开始

### 1. 克隆并创建环境

推荐 Python 3.11。本项目的实际验证环境为 Windows、Python 3.11；下列命令在 **PowerShell** 中执行，后续均以仓库根目录为工作目录。

```powershell
git clone https://github.com/luo785859020/kronos-okx-studio.git
cd kronos-okx-studio
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

如果仓库是私有的，克隆前需要通过 GitHub 的正常登录方式获取访问权限。不要把访问令牌直接写进克隆地址或脚本。

### 2. 安装依赖

**只使用 CPU：**

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r webui/requirements.txt
```

**使用 NVIDIA GPU：**先按下一节安装兼容的 CUDA 版 PyTorch，再安装 `webui/requirements.txt`。CUDA 版 PyTorch 同时保留 CPU 推理能力，不需要两套模型权重。

`webui/requirements.txt` 会引入 OKX 与根目录依赖。不需要 Web UI 时，可以只安装 `requirements-okx.txt`；仅运行本地 CSV 入口时安装 `requirements.txt`。

### 3. 先验证 CSV 并缓存权重

```powershell
.\.venv\Scripts\python.exe examples\predict_csv.py --validate-only
.\.venv\Scripts\python.exe examples\predict_csv.py
```

第一条只校验仓库自带数据；第二条执行真实模型预测，首次运行会从 Hugging Face 下载固定版本权重。默认数据无需额外准备。

> 看板启动模型时使用 `--local-files-only`，因此应先完成这一步。网络下载权重和访问 OKX 行情是两条独立链路；“只用本地权重”不代表实时行情可以离线运行。

### 4. 打开看板

```powershell
.\.venv\Scripts\python.exe webui\run.py
```

- **OKX 实时看板：** <http://127.0.0.1:7070/okx>
- **上游 CSV 实验页面：** <http://127.0.0.1:7070/>

不需要自动打开浏览器时，也可运行 `python webui/app.py`。CSV 页面从仓库根目录 `data/` 扫描 CSV / Feather 文件；其中部分功能是上游遗留交互，最小链路验证优先使用 `examples/predict_csv.py`。

Linux 可用 `python3 -m venv .venv` 创建环境，并将示例中的 `.\.venv\Scripts\python.exe` 换为 `.venv/bin/python`。GPU 支持还需依据实际驱动和 PyTorch 构建验证；不要将 Windows 实测结果视为所有系统的兼容性保证。

## CPU 与 GPU 配置

### NVIDIA CUDA 环境

下面是本机 RTX 5060 Ti 已验证的安装组合。安装或替换 PyTorch 前，请先停止预测并退出 Web UI，避免进程仍占用动态库。

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r webui/requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

不同显卡和驱动请参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)，不要把本机组合当作通用要求。

### 页面如何选择

1. 在“推理设备”中选择 CPU 或检测通过的 GPU。
2. GPU 选项显示设备编号、名称和总显存；不可用时禁用并说明原因。
3. 运行中需要先点击“停止预测”，等待设置解锁，再切换设备。
4. GPU 启动出错不会静默回退 CPU，应检查日志或明确改选 CPU。

设备检测会执行小型 CUDA 计算，并缓存到 Web UI 重启。检测通过不代表当前空闲显存足够容纳任意模型和窗口；显存不足仍可能在实际加载或推理时失败。新进程的首次 GPU 推理可能需要明显更长的初始化时间。

## 本地 CSV 最小链路

### 默认留出验证

```powershell
.\.venv\Scripts\python.exe examples\predict_csv.py --lookback 256 --pred-len 30 --device cpu --local-files-only
```

默认读取 `tests/data/regression_input.csv`，前 256 行作为历史，之后 30 行作为留出区间。未来行情只用于计算误差，模型输入包含历史行情与目标时间戳，不包含未来价格。

### 使用自己的文件

```powershell
.\.venv\Scripts\python.exe examples\predict_csv.py --csv data\history.csv --lookback 256 --pred-len 12 --device cuda:0 --local-files-only --output-dir outputs\csv-custom
```

留出模式至少需要 `start + lookback + pred_len` 行。`--start` 指定历史窗口起始行；历史窗口为 2–512 行，预测长度为 1–512 行。

### 只有历史数据，预测未来

准备仅含 `timestamps` 列的 `data/future_times.csv`，每行一个未来时间戳，再运行：

```powershell
.\.venv\Scripts\python.exe examples\predict_csv.py --csv data\history.csv --future-timestamps data\future_times.csv --lookback 256 --pred-len 12 --device cpu --local-files-only --output-dir outputs\csv-future
```

该模式使用行情文件最后 `lookback` 行，未来时间戳数量必须等于 `pred_len`，且全部晚于最后一条历史数据；不支持非零 `--start`。没有真实未来价格时不计算 MAE / RMSE。交易日历、午休、节假日等需要由调用方处理。

详细说明见 [本地 CSV 使用文档](examples/README_CSV_CN.md)。

## OKX 实时预测看板

### 操作步骤

1. 打开 `/okx`，选择现货币对。
2. 狗狗币直接选 **`DOGE-USDT · 狗狗币`**，不需要删除 BTC 后再搜索。
3. 其他现货币对选择“其他币对（手动输入）”，例如 `XRP-USDT`。
4. 选择周期、符合自己服务资格的区域、历史窗口和预测长度。
5. 选择 CPU / GPU，点击“启动实时预测”。
6. 先加载模型并生成当前窗口预测，之后每根新 K 线收盘时自动更新。
7. 点击历史记录可回看；勾选“跟随最新”恢复自动展示最新记录。
8. 不再使用时点击“停止预测”。**关闭浏览器不会停止后台预测进程。**

默认配置：`BTC-USDT`、`5m`、256 根历史、预测 12 根、CPU。可选周期为 `1m / 3m / 5m / 15m / 30m / 1H / 2H / 4H`。

### 图表和状态说明

- 历史 K 线与预测 K 线分别显示，分界线标明预测起点。
- 小于 1 个报价单位的币种会显示更多小数位，避免 DOGE 波动被两位小数掩盖。
- 页面每 3 秒读取本地状态与记录；这不代表每 3 秒重新推理。
- “本地面板已连接”是浏览器能访问本地服务；“行情已连接”是预测进程的行情连接状态。
- 过期数据会标成历史快照，不能当作当前预测。
- “预测变化”仅为预测期末收盘价相对最后实际收盘价的变化，未扣除成本，也不是置信度。

### 区域与密钥

当前公开行情链路**不需要 API Key**，不要为了运行本项目申请交易或提现权限。

`global / us / eea` 分别使用对应区域的公开 REST 与 WebSocket 地址。应根据实际服务资格选择，不要通过更换区域绕过限制；403 / 451 等访问限制不会触发自动跨区域切换。

连接、区域、时间戳和密钥范围说明见 [OKX 使用文档](examples/README_OKX_CN.md)。

## 命令行实时预测

### 单次 DOGE GPU 验证

```powershell
.\.venv\Scripts\python.exe examples\okx_realtime_predict.py --inst-id DOGE-USDT --bar 5m --device cuda:0 --once --local-files-only --output-dir outputs\okx-gpu
```

### 持续 BTC CPU 预测

```powershell
.\.venv\Scripts\python.exe examples\okx_realtime_predict.py --inst-id BTC-USDT --bar 5m --device cpu --local-files-only
```

按 `Ctrl+C` 结束。实验时可使用 `--max-predictions 2` 限制新增预测次数，或用 `--run-seconds 600` 限制流式观察阶段的时长；模型加载等初始化时间不计入这一流式时长。

其他参数包括 `--lookback`、`--pred-len`、`--region`、`--threads`、`--seed` 和 `--output-dir`。命令行不会像 UI 那样自动为不同 CPU/GPU 或窗口参数生成独立根目录；做对比实验时请显式指定不同的 `--output-dir`，避免重复窗口被去重。

## 数据格式与时间约定

建议 CSV 使用以下完整结构：

```csv
timestamps,open,high,low,close,volume,amount
2024-06-18 11:15:00,11.27,11.28,11.26,11.27,379,427161
2024-06-18 11:20:00,11.27,11.28,11.27,11.27,277,312192
```

| 字段 | 含义 |
| --- | --- |
| `timestamps` | 可解析的时间戳，严格递增，不重复、不为空 |
| `open / high / low / close` | 开、高、低、收；必须是有限数值 |
| `volume` | 成交量，建议提供 |
| `amount` | 成交额，建议提供，并确认与报价币种单位一致 |

本地 CSV 推理入口允许缺少 `volume / amount`，但这不等于训练数据可以省略配置所需字段；CSV 训练建议提供完整七列。已有字段中的 NaN、无穷值或无法转换的数值会被拒绝。

**两种入口的时间语义不同，不能混用：**

- 本地 CSV：使用无时区时间戳，由数据提供方确保采用一致的市场本地时间；不会自动转换时区。
- OKX：统一采用 **UTC 的 K 线开盘时间**。CSV 中不带时区标记，但语义仍为 UTC；看板也显示 UTC，不是电脑本地时间。

OKX 仅接收 `confirm=1` 的已收盘 K 线。现货 `vol` 映射为 `volume`，`volCcyQuote` 映射为 `amount`；不得将合约张数当成现货成交量。实时入口另行检查 OHLC 关系、非负量额、时间对齐、连续性与新鲜度。

预测区间第一根 K 线从最后一根历史 K 线结束时开始；推理实际完成时间更晚，不能假设能够按该根开盘价成交。

## 输出与记录隔离

| 入口 / 设备 | 默认结果位置 |
| --- | --- |
| 本地 CSV | `outputs/csv-small/` |
| OKX 命令行 | `outputs/okx/<region>/<inst-id>/<bar>/` |
| OKX UI / CPU | `outputs/okx-ui/l256-p12/<region>/<inst-id>/<bar>/` |
| OKX UI / GPU 0 | `outputs/okx-ui/l256-p12/cuda-0/<region>/<inst-id>/<bar>/` |

`l256-p12` 表示历史 256 根、预测 12 根。不同参数、设备、区域、币对、周期的 UI 记录独立保存；CPU 原记录不会因为切换 GPU 而被移动或覆盖。

每次 OKX 预测使用最后一根历史 K 线的 UTC 时间作为子目录：

```text
20261003T150000Z/
├── history.csv       实际使用的历史输入
├── prediction.csv    原始模型预测
├── signal.json       原始预测变化率，明确标记非订单
└── run.json          模型版本、设备、参数、环境、时间与输入哈希
```

流目录还会保存 `latest.json`、状态、锁文件与日志。缺少完整 `run.json` 的记录不会出现在看板历史列表中。CSV 入口另有 `comparison.csv` 和 MAE / RMSE 信息；重复使用同一 CSV 输出目录会覆盖此前结果，应按实验分目录保存。

### 固定模型版本

| 资源 | 标识 / revision |
| --- | --- |
| 模型 | `NeoQuasar/Kronos-small` |
| 模型 revision | `901c26c1332695a2a8f243eb2f37243a37bea320` |
| Tokenizer | `NeoQuasar/Kronos-Tokenizer-base` |
| Tokenizer revision | `0e0117387f39004a9016484a186a908917e22426` |

权重不包含在 Git 仓库中，首次使用由 Hugging Face 下载。CPU 与 GPU 共用同一份权重缓存；模型名称不随本项目重命名而改变。

## 测试与实测结果

### 运行测试

```powershell
.\.venv\Scripts\python.exe -m pip install pytest PyYAML
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pip check
```

浏览器回归为可选项：

```powershell
.\.venv\Scripts\python.exe -m pip install playwright
.\.venv\Scripts\python.exe -m pytest tests\test_okx_browser.py -q
```

Windows 浏览器测试使用已安装的 Microsoft Edge；其他系统使用 Playwright Chromium，可通过 `python -m playwright install chromium` 安装。缺少可选浏览器工具时相应测试会跳过。模型回归测试需要已缓存权重或可访问 Hugging Face；OKX 单元测试使用模拟数据，不访问账户。

请使用 `pytest tests`，而不是直接收集整个仓库。上游 `finetune/qlib_test.py` 是额外的 Qlib 评估入口，缺少其可选依赖时会导致根目录全量收集失败。

### 已验证环境

截至 **2026-10-04**，本机完整验证为 **112 项测试通过**，包含 4 项真实浏览器回归：

- Windows、Python 3.11、PyTorch `2.11.0+cu128`。
- NVIDIA GeForce RTX 5060 Ti，标称 8GB 显存。
- CSV 预测、真实 DOGE GPU 单次预测、BTC GPU 看板运行已通过。
- DOGE 固定下拉列表、自定义币对、启停、历史回看及设备隔离已验证。
- 桌面及 390px 手机布局正常，所测页面无 JavaScript 错误。

以下是**同一份 DOGE 输入**、256 根历史预测 12 根的本机测试，不是所有设备上的性能保证：

| 设备 | 预热后 3 次推理中位耗时 |
| --- | --- |
| CPU，4 线程 | 0.557 秒 |
| RTX 5060 Ti / `cuda:0` | 0.097 秒 |

GPU 在该测试下约快 **5.7 倍**。计时只包含预测调用，不包括模型加载、网络请求或页面刷新；新进程第一次 GPU 推理约需 16 秒。GPU 加速不代表预测更准确，也不代表更高收益。

## CSV 微调与归一化

正式训练前应先检查数据和归一化，避免只因训练损失下降就误以为模型有效。

本项目已调整 `finetune_csv/finetune_base_model.py` 中的 `CustomKlineDataset`：

1. 仅用历史 `lookback_window` 行计算各特征均值和标准差。
2. 将同一组统计量用于历史与未来目标窗口。
3. 归一化后按配置裁剪，未来目标不参与统计量计算。
4. Tokenizer 与预测器共用这一数据集处理逻辑，并有回归测试覆盖。

该修复不代替按时间划分训练 / 验证 / 测试集，也不能单独证明不存在其他数据泄漏。训练配置中的数据路径、模型路径、窗口及设备仍需自行调整；本项目发布不包含新微调权重。

训练入口与额外依赖参见 [CSV 微调中文文档](finetune_csv/README_CN.md)；Qlib 路线及原版研究示例参见 [上游说明](README_UPSTREAM.md)。不建议在行情最小链路未验证前直接启动正式训练。

## 目录结构

```text
kronos-okx-studio/
├── README.md                     本项目说明
├── README_UPSTREAM.md            保留的上游原版说明
├── LICENSE                       上游 MIT 许可证与署名
├── model/                        Kronos 模型与 Tokenizer
├── integrations/okx_market.py    OKX 公开行情、确认收盘与数据检查
├── examples/predict_csv.py       本地 CSV 最小预测入口
├── examples/okx_realtime_predict.py
│                                只读实时预测进程
├── examples/README_CSV_CN.md     CSV 详细说明
├── examples/README_OKX_CN.md     OKX 详细说明
├── webui/okx_dashboard.py        本地进程控制和记录接口
├── webui/templates/okx.html      中文实时看板
├── webui/static/                图表交互和样式
├── finetune_csv/                 CSV 训练及归一化修复
├── finetune/                     上游 Qlib 训练 / 评估入口
├── tests/                       数据、行情、UI 和模型回归
├── requirements.txt             基础依赖
└── requirements-okx.txt         OKX 入口依赖
```

## 常见问题

### 下拉列表找不到狗狗币

当前版本使用固定选择框，第二项为 `DOGE-USDT · 狗狗币`。如果仍看到只有 BTC 的输入建议，请重启旧 Web UI 服务并按 `Ctrl+F5` 刷新。预测运行中设置会锁定，需要先停止当前预测。

### GPU 显示不可用

先确认启动 UI 的 Python 环境安装了 CUDA 版 PyTorch，而不只是系统装了 CUDA。检查 `torch.cuda.is_available()`；修改环境后重启 Web UI。页面有检测原因，日志中有模型实际设备。不要把 `nvidia-smi` 能显示显卡当作 PyTorch 已能使用 GPU 的充分条件。

### 首次点击启动没有立即出图

首次需要加载模型，GPU 可能还有初始化耗时。查看运行日志，不要重复创建进程。如果提示本地权重缺失，先执行不带 `--local-files-only` 的 CSV 预测下载权重。

### 本地服务已连接，但行情未连接

这两种状态不同：检查预测进程是否已启动、网络是否可达、服务区域是否正确。未启动或已停止时“行情未连接”是正常状态；网络异常时可能重连或暂停。不要把历史快照误当成新行情。

### 更换设备后历史记录不见了

记录按设备及参数分目录保存，并非被删除。切回原设备、币对、周期和窗口即可查看对应记录。

### 可以自动买卖或连接账户吗

当前不能。公开行情预测不需要密钥，项目没有交易执行、账户持仓或完整风控闭环。原始预测变化率也不能替代独立的样本外评估和成本验证。

### 7070 端口已被占用

先确认是否已有本项目 Web UI 在运行，避免启动第二个服务。正常结束终端服务使用 `Ctrl+C`；退出前先在页面停止预测。不要为了方便将监听地址改为公网地址。

## 安全与发布说明

- OKX 控制接口仅接受本机回环访问，并检查请求来源与控制令牌；子进程不通过 shell 拼接任意命令。
- 子进程环境会剔除约定的 OKX 密钥变量；项目不要求填写真实密钥。
- `.venv/`、`.python/`、缓存、模型权重、`outputs/` 和本机数据目录不应上传。上游已提供的示例数据及素材保留在代码仓库中。
- `.gitignore` 是辅助措施，不能替代提交前审查；不要提交真实密钥、账户信息或未经授权的数据。
- 仓库默认作为私有个人项目发布。若未来改成公开仓库，应重新审查历史提交、数据分发权限、截图与运行日志。
- Flask 调试模式和开发重载器默认关闭；整个工作台仍是本地研究应用，不是经过公网安全加固的托管服务。

## 来源、致谢与引用

- 上游模型项目：[shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)。
- 模型与 Tokenizer：[NeoQuasar / Hugging Face](https://huggingface.co/NeoQuasar)。
- 原始论文：[Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739)。
- 本仓库主要增加本地验证入口、OKX 只读集成、中文看板、CPU/GPU 选择、运行记录与相关测试，并修复 CSV 归一化的未来信息泄漏。

研究使用请保留上游引用：

```bibtex
@misc{shi2025kronos,
  title={Kronos: A Foundation Model for the Language of Financial Markets},
  author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
  year={2025},
  eprint={2508.02739},
  archivePrefix={arXiv},
  primaryClass={q-fin.ST}
}
```

代码遵循 [MIT License](LICENSE)，保留原作者的版权声明。模型权重及第三方数据的使用条件应分别查阅对应资源；本项目改名不改变其来源或权利归属。
