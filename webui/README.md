# Kronos OKX Studio — Web UI

Local Web UI for Kronos OKX Studio, an unofficial application built on the upstream Kronos financial prediction model. See the repository's main README for setup, attribution, and the read-only operating scope.

## ✨ Features

- **Multi-format data support**: Supports CSV, Feather and other financial data formats
- **Smart time window**: Fixed 400+120 data point time window slider selection
- **Real model prediction**: Integrated real Kronos model, supports multiple model sizes
- **Prediction quality control**: Adjustable temperature, nucleus sampling, sample count and other parameters
- **Multi-device support**: Supports CPU, CUDA, MPS and other computing devices
- **Comparison analysis**: Detailed comparison between prediction results and actual data
- **K-line chart display**: Professional financial K-line chart display

## 🚀 Quick Start

### Method 1: Start with Python script
```bash
cd webui
python run.py
```

### Method 2: Start with Shell script
```bash
cd webui
chmod +x start.sh
./start.sh
```

### Method 3: Start Flask application directly
```bash
cd webui
python app.py
```

After successful startup, visit http://127.0.0.1:7070/okx for the OKX real-time dashboard, or http://127.0.0.1:7070/ for the local CSV dashboard.

## 📡 OKX Real-time Dashboard

The `/okx` page controls the read-only OKX public-market integration added in this repository. It displays the latest confirmed candles, forecast candles, saved prediction records, process logs, connection state, and a raw forecast return indicator.

The page starts one local `okx_realtime_predict.py` child process at a time. It uses public OKX spot candles only; it does not read API keys, access an account, submit orders, or provide a trading recommendation. The control endpoints accept requests only from `127.0.0.1`/`::1` and require the page's short-lived control token.

Before starting the UI, install the root and OKX dependencies into the active environment:

```powershell
python -m pip install -r requirements-okx.txt
python -m pip install -r webui/requirements.txt
```

The UI defaults to CPU and lets the user select an available NVIDIA CUDA GPU under **推理设备**. Models always load with `--local-files-only`. Download and validate Kronos-small once from the repository's root environment before clicking Start. The process output is stored under `outputs/okx-ui/`; the UI uses a separate output root from the command-line OKX runner.

The local `/api/okx/devices` endpoint reports the active interpreter's PyTorch build and detected devices. Each GPU must pass a small CUDA calculation before it is selectable; CPU-only builds, missing drivers, and incompatible CUDA kernels show a disabled GPU option with an explanation. Stop the current prediction before changing devices. GPU launch failures do not silently fall back to CPU. CUDA detection is cached until the Web UI is restarted; the small check is not a guarantee that the full model will fit in free GPU memory.

To enable CUDA on a supported Windows/NVIDIA environment, first stop all predictions and exit the Web UI. Install a CUDA build of PyTorch into the **same virtual environment** that runs the UI, using the official PyTorch installation selector. For a compatible CUDA 12.8 setup, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
.\.venv\Scripts\python.exe webui\run.py
```

This changes the environment, not just a UI setting; do not run the installation while its Python processes are using PyTorch. A CUDA build also supports CPU inference. Re-run the repository tests after changing PyTorch. Setup reference: https://pytorch.org/get-started/locally/ .

The dashboard defaults to `BTC-USDT`, `5m`, 256 history rows, and 12 forecast rows. Select the OKX regional endpoint that you are eligible to use; the application never switches regions to bypass an access restriction. All displayed timestamps are UTC.

The instrument dropdown always lists `BTC-USDT`, `DOGE-USDT` (Dogecoin), `ETH-USDT`, and `SOL-USDT`; the current choice does not filter out other instruments. Stop the current forecast, select `DOGE-USDT · 狗狗币`, then start again. For another spot instrument, select **其他币对（手动输入）** and enter its ID, such as `XRP-USDT`. DOGE reuses Kronos-small without a separate model download or training run, and keeps its records separate from BTC. Price cards and chart labels retain extra decimal places for prices below one quote-currency unit. The dashboard manages only one instrument at a time.

The October 4, 2026 selector fix passes 112 tests in this environment, including four browser regressions covering direct DOGE selection, custom-input validation, launch arguments, and active-instrument restoration. `tests/test_okx_browser.py` uses Playwright with Edge on Windows or Chromium elsewhere and skips when the optional browser tooling is unavailable.

Closing the browser does not stop the child process. Use **停止预测** before closing the page. The UI is intended for local observation and development, not public deployment. It binds to `127.0.0.1` with Flask debug mode and the development reloader disabled.

The chart library is served locally from the installed Plotly package, without a CDN dependency. Status and records refresh every three seconds. Selecting a saved record disables follow-latest mode until it is checked again. The dashboard labels stale data as a historical snapshot rather than presenting it as a current forecast.

Outputs are isolated by history/prediction settings, device, region, instrument, and bar. Existing CPU records stay at `outputs/okx-ui/l256-p12/global/BTC-USDT/5m/`; CUDA device 0 uses `outputs/okx-ui/l256-p12/cuda-0/global/BTC-USDT/5m/`. Switching devices therefore displays a separate set of records without overwriting CPU results for the same candle. Each run stores the original input, predictions, device, signal, and metadata; files without a completed `run.json` are hidden. The HTTP endpoints never accept filesystem paths or arbitrary process arguments.

Validation on October 3, 2026: the Windows Python 3.11 environment now uses `torch 2.11.0+cu128` on an NVIDIA GeForce RTX 5060 Ti (8 GB). CUDA execution, real DOGE one-shot GPU inference, and a live BTC dashboard forecast on `cuda:0` succeeded. `python -m pytest tests -q` passes 105 tests, including 45 dashboard tests; `pip check` passes. Device selection, unavailable-GPU rejection, per-device records, and mocked GPU/CPU process controls have been tested. The dashboard has also been checked in a headless browser with live OKX data for start/stop, candle-close updates, historical browsing, follow-latest behavior, desktop layout, and a 390-pixel mobile layout.

For one saved DOGE input with 256 history rows and 12 forecast rows, the median of three warmed-up prediction calls was 0.557 seconds on CPU (4 threads) and 0.097 seconds on this GPU, about 5.7 times faster. These measurements exclude model loading and network calls; they are not an end-to-end latency guarantee. The first GPU prediction in a new process took about 16 seconds, so initial loading/warm-up can be noticeably slower. The environment snapshots, device check, and benchmark are stored locally under `outputs/environment-cuda-setup/`.

## 📋 Usage Steps

1. **Load data**: Select financial data file from data directory
2. **Load model**: Select Kronos model and computing device
3. **Set parameters**: Adjust prediction quality parameters
4. **Select time window**: Use slider to select 400+120 data point time range
5. **Start prediction**: Click prediction button to generate results
6. **View results**: View prediction results in charts and tables

## 🔧 Prediction Quality Parameters

### Temperature (T)
- **Range**: 0.1 - 2.0
- **Effect**: Controls prediction randomness
- **Recommendation**: 1.2-1.5 for better prediction quality

### Nucleus Sampling (top_p)
- **Range**: 0.1 - 1.0
- **Effect**: Controls prediction diversity
- **Recommendation**: 0.95-1.0 to consider more possibilities

### Sample Count
- **Range**: 1 - 5
- **Effect**: Generate multiple prediction samples
- **Recommendation**: 2-3 samples to improve quality

## 📊 Supported Data Formats

### Required Columns
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price

### Optional Columns
- `volume`: Trading volume
- `amount`: Trading amount (not used for prediction)
- `timestamps`/`timestamp`/`date`: Timestamp

## 🤖 Model Support

- **Kronos-mini**: 4.1M parameters, lightweight fast prediction
- **Kronos-small**: 24.7M parameters, balanced performance and speed
- **Kronos-base**: 102.3M parameters, high quality prediction

## 🖥️ GPU Acceleration Support

- **CPU**: General computing, best compatibility
- **CUDA**: NVIDIA GPU acceleration, best performance
- **MPS**: Apple Silicon GPU acceleration, recommended for Mac users

## ⚠️ Notes

- `amount` column is not used for prediction, only for display
- Time window is fixed at 400+120=520 data points
- Ensure data file contains sufficient historical data
- First model loading may require download, please be patient

## 🔍 Comparison Analysis

The system automatically provides comparison analysis between prediction results and actual data, including:
- Price difference statistics
- Error analysis
- Prediction quality assessment

## 🛠️ Technical Architecture

- **Backend**: Flask + Python
- **Frontend**: HTML + CSS + JavaScript
- **Charts**: Plotly.js
- **Data processing**: Pandas + NumPy
- **Model**: Hugging Face Transformers

## 📝 Troubleshooting

### Common Issues
1. **Port occupied**: Modify port number in app.py
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Model loading failed**: Check network connection and model ID
4. **Data format error**: Ensure data column names and format are correct

### Log Viewing
Detailed runtime information will be displayed in the console at startup, including model status and error messages.

## 📄 License

This project follows the license terms of the original Kronos project.

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve this Web UI!

## 📞 Support

If you have questions, please check:
1. Project documentation
2. GitHub Issues
3. Console error messages
