# OKX 实时只读预测与 API 密钥说明

## 范围

当前实现只接收公开行情并预测：不读取 API Key，不查询账户，不创建订单，不接模拟盘或实盘交易。公开 K 线不需要身份验证，因此现在不必申请密钥。

默认配置：`BTC-USDT` 现货、`5m`、256 根已收盘历史、预测未来 12 根、Kronos-small、CPU。看板支持选择 CPU 或可用的 NVIDIA CUDA GPU；只有安装了兼容显卡的 CUDA 版 PyTorch，才能使用 `--device cuda:0`。

本入口统一使用 **UTC、K 线开盘时间**。CSV 中的 `timestamps` 不带时区，但含义始终是 UTC，不是电脑本地时间。预测第一根 K 线的开盘时间等于最后一根历史 K 线的结束时间；推理完成晚于这个时间，不能据此假定能按该根开盘价成交。

## 安装与运行

在 `D:\Kronos` 下安装额外依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-okx.txt
```

只运行一次，获取最新已收盘行情并预测：

```powershell
& D:\Kronos\.venv\Scripts\python.exe D:\Kronos\examples\okx_realtime_predict.py --once --local-files-only
```

持续接收行情，每根新收盘 K 线生成一次预测，按 Ctrl+C 停止：

```powershell
& D:\Kronos\.venv\Scripts\python.exe D:\Kronos\examples\okx_realtime_predict.py --local-files-only
```

有界验证示例：最多运行约 330 秒或保存 2 次新预测，先达到的条件生效。时间限制从模型加载后开始，退出时可能额外等待接收线程关闭：

```powershell
.\.venv\Scripts\python.exe examples\okx_realtime_predict.py --local-files-only --max-predictions 2 --run-seconds 330 --output-dir outputs\okx-check
```

`--local-files-only` 仅限制模型权重加载，行情仍需要联网。首次未缓存模型时，去掉该参数允许下载。

可用选项包括 `--inst-id ETH-USDT`、`--bar 1m`、`--lookback 512`、`--pred-len 12`、`--region global|us|eea`。只支持当前可交易的现货，不支持 `BTC-USDT-SWAP` 合约；切换参数做实验时，应使用不同的输出目录。

## 区域与连接

必须根据你有资格使用的 OKX 站点选择 `--region`，不能通过换域名绕过地区限制。程序遇到 403/451 不自动切换地区。对应地址为：

| region | REST | K 线 WebSocket |
|---|---|---|
| global | `https://openapi.okx.com` | `wss://ws.okx.com/ws/v5/business` |
| us | `https://us.okx.com` | `wss://wsus.okx.com/ws/v5/business` |
| eea | `https://eea.okx.com` | `wss://wseea.okx.com/ws/v5/business` |

已实测 global；其他区域需在符合服务条件的对应环境中验证。代码使用默认 443 端口。OKX 在 2026-09-30 发布公告，将于 2026-10-31 停用 WebSocket 8443 端口，不应再增加 `:8443`。

## 数据处理与保护

- REST 初始化、WebSocket 接收在接收线程执行；模型在主线程加载一次并重复使用。
- 只接收 `confirm=1`；未收盘数据不进入历史窗口。
- 现货 `vol` 映射为 `volume`，`volCcyQuote` 映射为 `amount`。不要把合约张数按现货数量处理。
- 检查 OHLC 合理性、有限数值、非负成交量、时间对齐、连续性和新鲜度。
- 心跳、重连指数退避；连接恢复和约每分钟检查时通过 REST 补齐最新窗口。
- 发现缺口先补数，无法补齐则暂停预测。断线期间不补做已经过时的预测，只恢复最新窗口。
- 相同收盘时间只保存一次；通过每次运行的 `run.json` 完成标记实现重启去重。
- 每个输出目录有进程锁，避免两个进程竞争写入。同一收盘时间更换模型参数时请使用新的输出目录。
- 最多排队一个历史窗口，模型落后时用较新的窗口替换旧任务；不会无限积压后再发布过时信号。
- 行情接收、推理异常均写入滚动日志；鉴权、签名和交易代码没有实现，也不会自动执行交易。

## 输出

默认输出：`outputs/okx/global/BTC-USDT/5m/`。

- `latest.json`：最新预测目录和未经交易成本调整的预测变化率。
- `realtime.log`：连接、补数、预测、异常及停止记录。
- `YYYYMMDDTHHMMSSZ/history.csv`：这一时点实际输入的 256 根已收盘 K 线。
- `YYYYMMDDTHHMMSSZ/prediction.csv`：未来 12 根原始模型输出。
- `YYYYMMDDTHHMMSSZ/signal.json`：期末预测收盘价与历史收盘价的相对变化率，明确标记为 `forecast_only_not_an_order`。
- `YYYYMMDDTHHMMSSZ/run.json`：模型版本、数据哈希、交易所时间、推理起止时间、参数与环境版本。

目录名是最后一根历史 K 线的 **UTC 开盘时间**。例如 `20261003T125500Z` 表示北京时间 2026-10-03 20:55 开盘、21:00 收盘的 5 分钟 K 线。

`signal.json` 不是买卖指令，也不是实现收益。没有手续费、滑点、成交和持仓记账；不输出策略收益指标。输出不强制满足价格/量额的全部市场约束，不能直接当作可成交报价。

## 密钥怎么申请

### 当前公开行情：不用申请

这个程序不需要任何密钥。不要为了运行它开通 Trade 或 Withdraw 权限，也不要把密钥提交到聊天、源码或日志。

### 后续模拟盘：创建模拟盘专用密钥

按 OKX 官方 API FAQ 与模拟盘指南（核对日期：2026-10-03）：

1. 登录你有资格使用的 OKX 官方站点，在 **交易 → 模拟交易** 中进入模拟环境。
2. 打开 **个人中心/头像 → 模拟交易 API → 创建 API Key**。不同地区或版本的菜单名称可能略有差异，应确认页面明确写着模拟交易。
3. 名称可以填 `kronos-demo`，用途选自用 API；设置独立的 Passphrase，并按页面要求完成安全验证。
4. 仅查模拟账户时使用 Read；后续模拟下单才增加 Trade。不启用 Withdraw。按当前页面逐项核对权限，如果出现 Transfer、Loan、Earn 等额外选项，不启用本任务用不到的权限；不要把 Trade 当作无风险权限。
5. 有固定出口公网 IP 时配置 IP 白名单。不要填写 `127.0.0.1` 或内网 IP 代替出口地址，也不要为图方便开放任意来源。
6. 安全保存 **API Key、Secret Key、Passphrase**。Passphrase 是创建时自设的 API 口令，不是登录密码；遗忘后需重新创建密钥。

后续模拟盘代码还必须使用对应区域与模拟盘 API Key，并在请求头设置 `x-simulated-trading: 1`；仅把普通实盘密钥改名为 demo 并不能切换环境。

当前没有模拟盘下单代码，也没有读取这些环境变量。将来对接时只从本地秘密存储或进程环境读取，不把真实值写入示例文件。

### 普通账户密钥入口

网页版：**头像 → API 和连接（API and connections）→ 创建 API Key**。

App：**功能菜单 → API → 创建 API Key**。

普通账户密钥不是本阶段所需的模拟盘密钥。账户资格、支持产品和可见入口以所在站点为准；不要为本项目提前开通实盘交易权限。

官方 FAQ 提到部分地区创建 API Key 有账户资产条件；如果页面出现限制，以对应地区当前要求为准，不应把它理解成所有地区或模拟盘都必须充值。本阶段公开行情不需要密钥。

### 官方资料

- API FAQ（创建步骤、权限和模拟盘说明）：https://www.okx.com/en-us/help/api-faq
- Global API 文档：https://www.okx.com/docs-v5/
- US API 文档：https://app.okx.com/docs-v5/en/
- EEA API 文档：https://my.okx.com/docs-v5/zh/
- WebSocket 端口公告：https://www.okx.com/en-us/help/okx-websocket-port-8443-discontinuation-announcement

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_okx_market.py -q
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -m pytest tests -q
```

OKX 单元测试使用模拟行情，不访问网络或账户；真实网络验证需要单独执行入口。仅安装基础依赖而没有 `websocket-client` 时，OKX 可选测试会跳过。

2026-10-03 已完成真实 REST、443 WebSocket 与 Small 模型验证，并跨过北京时间 21:00 的 5 分钟收盘自动生成新预测。验证输出位于 `outputs/okx-stream-validation/global/BTC-USDT/5m/`。测试进程已按运行上限退出，没有后台交易服务。

本次完整测试结果：59 项通过，其中 31 项为新增 OKX 测试；`pip check` 和静态语法检查通过。

## 可视化看板

仓库的 Flask Web UI 已增加 OKX 实时只读页面。启动 UI：

```powershell
.\.venv\Scripts\python.exe webui\run.py
```

浏览器打开 `http://127.0.0.1:7070/okx`。页面可以启动/停止一个本地预测进程，显示连接状态，绘制历史与预测 K 线，查看记录和日志。页面关闭不会停止进程，必须点击“停止预测”。

UI 默认使用 CPU，可在“推理设备”中选择检测通过的 GPU；CPU/GPU 共用本地已缓存的 Kronos-small 权重，请先用命令行完成一次模型下载和验证。UI 的结果目录是 `outputs/okx-ui/`，与命令行入口的 `outputs/okx/` 分开。UI 绑定本机回环地址，控制接口不允许远程访问；当前仍不需要 API Key。

页面每 3 秒自动刷新。点击历史记录会暂停“跟随最新”，勾选后恢复；停止预测后，已保存的图表与记录仍可查看。不同参数分别保存在 `l256-p12` 等子目录中，避免不同窗口相互覆盖。

狗狗币：在“现货币对”下拉列表中直接选择 `DOGE-USDT · 狗狗币`，再点击“启动实时预测”。列表固定展示 BTC、DOGE、ETH、SOL，不会因为当前选中 BTC 就隐藏其他币对。其他现货币对可选“其他币对（手动输入）”后填写，例如 `XRP-USDT`。如果正在预测 BTC 等其他币对，先点击“停止预测”，等待设置解锁后再切换；看板一次只管理一个币对。DOGE 复用已缓存的 Kronos-small，不需要单独下载模型、训练或申请 API 密钥。低价币的价格卡片、图表坐标与悬停价格会保留更多小数位；CPU 记录保存在独立的 `outputs/okx-ui/l256-p12/global/DOGE-USDT/5m/` 目录。

也可以在仓库根目录独立执行一次狗狗币预测，不影响看板当前运行的币对：

```powershell
.\.venv\Scripts\python.exe examples\okx_realtime_predict.py --inst-id DOGE-USDT --bar 5m --once --local-files-only
```

已验证：84 项测试通过，真实浏览器下的启停、跨收盘自动刷新、历史回看和跟随最新正常；桌面与 390 像素手机布局无横向溢出。

### CPU / GPU 推理选择

- 默认选中 CPU。GPU 选项会显示设备编号、显卡名称和总显存；未通过 CUDA 小型计算检查的设备不能启动，并显示原因。
- 先停止当前预测，再切换推理设备。切换 GPU 不需要重新下载模型；程序不会在 GPU 出错时静默回退 CPU。
- CPU 的旧记录路径不变；`cuda:0` 的记录单独保存在 `outputs/okx-ui/l256-p12/cuda-0/global/DOGE-USDT/5m/` 等目录。显卡总显存不等于当前可用显存；加载模型时显存不足仍可能失败，应查看运行日志或缩短窗口。
- 如果提示“CPU 版 PyTorch”，仅选择 GPU 或安装系统 CUDA 不能改变当前 PyTorch 构建。需要先停止预测并退出 Web UI，再为启动 UI 的同一个虚拟环境安装兼容的 CUDA 版 PyTorch。安装步骤见 `webui/README.md` 的设备说明。重新启动 UI 后会重新检测。
- 2026-10-03 已将本机环境从 `torch 2.6.0+cpu` 升级到 `torch 2.11.0+cu128`，RTX 5060 Ti 8GB 的 CUDA 计算检查通过。真实 DOGE 单次 GPU 预测、BTC 看板 GPU 连续预测及 105 项回归测试均已通过，CPU 选项仍可使用。
- 同一 DOGE 输入（256 根历史、预测 12 根），模型预热后三次推理的中位耗时：CPU 四线程约 0.557 秒，GPU 约 0.097 秒，约快 5.7 倍；不包含权重加载和行情网络耗时。新进程第一次 GPU 推理约需 16 秒，不应将初始化耗时当作稳定推理速度。
- 升级前后的依赖清单、设备验证和测速报告保存在 `outputs/environment-cuda-setup/`；原 CPU 预测记录没有移动或删除。刷新已打开的旧看板页面后即可看到新设备选项。

安装并验证 CUDA 环境后，也可单独运行一次 GPU 预测；使用独立输出目录，避免与 CPU 命令行记录重叠：

```powershell
.\.venv\Scripts\python.exe examples\okx_realtime_predict.py --inst-id DOGE-USDT --device cuda:0 --once --local-files-only --output-dir outputs/okx-gpu
```
