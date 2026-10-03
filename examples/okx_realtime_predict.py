import argparse
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import queue
import random
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch
from filelock import FileLock, Timeout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.okx_market import (
    BAR_MILLISECONDS, ENDPOINTS, FEATURE_COLUMNS, MarketDataError,
    OKXMarketClient, merge_candles, validate_history,
)
from examples.predict_csv import MODEL_ID, MODEL_REVISION, TOKENIZER_ID, TOKENIZER_REVISION
from model import Kronos, KronosPredictor, KronosTokenizer


LOGGER = logging.getLogger(__name__)


def future_timestamps(history, interval_ms, pred_len):
    return pd.Series(pd.date_range(
        start=history["timestamps"].iloc[-1] + pd.Timedelta(milliseconds=interval_ms),
        periods=pred_len, freq=pd.Timedelta(milliseconds=interval_ms),
    ), name="timestamps")


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class RuntimeStatus:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.lock = threading.Lock()

    def update(self, **values):
        with self.lock:
            self.data.update(values)
            self.data["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
            atomic_json(self.path, self.data)


def enqueue_latest(events, snapshot):
    try:
        events.put_nowait(snapshot)
    except queue.Full:
        try:
            events.get_nowait()
        except queue.Empty:
            pass
        events.put_nowait(snapshot)
        LOGGER.warning("Prediction is lagging; replacing queued history with the latest closed window")


def market_worker(client, lookback, events, stop_event, status=None):
    history = None
    last_emitted = None
    def connection_state(state):
        if status is not None:
            status.update(connection=state, heartbeat_at=pd.Timestamp.now(tz="UTC").isoformat())

    stream = client.stream(stop_event, on_state=connection_state)
    try:
        for candles in stream:
            if stop_event.is_set():
                break
            try:
                if candles is None or history is None:
                    history = client.fetch_history(lookback)
                else:
                    updated = merge_candles(history, candles, lookback)
                    if updated is history:
                        continue
                    history = updated
                server_ms = client.server_time_ms()
                try:
                    validate_history(history, lookback, client.interval_ms, server_ms)
                except MarketDataError:
                    LOGGER.warning("Refreshing REST history after a gap or stale candle")
                    history = client.fetch_history(lookback)
                    server_ms = client.server_time_ms()
                    validate_history(history, lookback, client.interval_ms, server_ms)
                last_open = history["timestamps"].iloc[-1]
                if last_emitted is None or last_open > last_emitted:
                    enqueue_latest(events, (history.copy(), server_ms, time.monotonic()))
                    last_emitted = last_open
                    if status is not None:
                        status.update(last_candle=str(last_open), last_market_at=pd.Timestamp.now(tz="UTC").isoformat(), data_error=None)
            except (MarketDataError, ValueError) as error:
                LOGGER.warning("Market data rejected; no forecast issued: %s", error)
                if status is not None:
                    status.update(data_error=str(error))
    except Exception as error:
        enqueue_latest(events, error)
    finally:
        stream.close()
        client.close()


def forecast_window(predictor, history, args, stream_dir, server_ms):
    run_id = history["timestamps"].iloc[-1].strftime("%Y%m%dT%H%M%SZ")
    run_dir = stream_dir / run_id
    if (run_dir / "run.json").exists():
        LOGGER.info("Skipping already saved candle %s", run_id)
        return False
    validate_history(history, args.lookback, BAR_MILLISECONDS[args.bar], server_ms)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timestamps = future_timestamps(history, BAR_MILLISECONDS[args.bar], args.pred_len)
    started = pd.Timestamp.now(tz="UTC")
    predictions = predictor.predict(
        df=history[FEATURE_COLUMNS], x_timestamp=history["timestamps"],
        y_timestamp=timestamps, pred_len=args.pred_len,
        T=1.0, top_k=1, top_p=1.0, sample_count=1, verbose=False,
    )
    if list(predictions.columns) != FEATURE_COLUMNS or predictions.shape != (args.pred_len, 6):
        raise ValueError("Model returned an unexpected prediction shape or columns.")
    if not np.isfinite(predictions.to_numpy()).all():
        raise ValueError("Model returned nonfinite predictions.")
    if not pd.DatetimeIndex(predictions.index).equals(pd.DatetimeIndex(timestamps)):
        raise ValueError("Prediction timestamps do not match the requested horizon.")
    last_close = float(history["close"].iloc[-1])
    signal = {
        "kind": "forecast_only_not_an_order",
        "last_actual_close": last_close,
        "predicted_final_close": float(predictions["close"].iloc[-1]),
        "predicted_return": float(predictions["close"].iloc[-1] / last_close - 1),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(run_dir / "history.csv", index=False)
    predictions.to_csv(run_dir / "prediction.csv", index_label="timestamps")
    report = {
        "mode": "okx_public_read_only", "inst_id": args.inst_id, "inst_type": "SPOT",
        "region": args.region, "bar": args.bar, "timestamp_convention": "UTC candle opening time; CSV is timezone-naive UTC",
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer": TOKENIZER_ID, "tokenizer_revision": TOKENIZER_REVISION,
        "device": args.device, "seed": args.seed, "threads": args.threads,
        "lookback": args.lookback, "pred_len": args.pred_len,
        "temperature": 1.0, "top_k": 1, "top_p": 1.0, "sample_count": 1,
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "history_start": str(history["timestamps"].iloc[0]),
        "history_end": str(history["timestamps"].iloc[-1]),
        "candle_closed_at": str(timestamps.iloc[0]),
        "exchange_time": str(pd.to_datetime(server_ms, unit="ms", utc=True)),
        "prediction_start": str(timestamps.iloc[0]), "prediction_end": str(timestamps.iloc[-1]),
        "inference_started_at": started.isoformat(),
        "inference_completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "history_sha256": hashlib.sha256((run_dir / "history.csv").read_bytes()).hexdigest(),
        "signal": signal,
    }
    atomic_json(run_dir / "signal.json", signal)
    atomic_json(run_dir / "run.json", report)
    atomic_json(stream_dir / "latest.json", {"run_dir": str(run_dir.resolve()), "run_id": run_id, **signal})
    LOGGER.info("Saved %s: forecast return %+.4f%% (not a trade instruction)", run_dir, signal["predicted_return"] * 100)
    return True


def build_parser():
    parser = argparse.ArgumentParser(description="Read-only OKX spot candles -> Kronos-small. No keys or trading endpoints.")
    parser.add_argument("--inst-id", default="BTC-USDT")
    parser.add_argument("--bar", choices=BAR_MILLISECONDS, default="5m")
    parser.add_argument("--region", choices=ENDPOINTS, default="global", help="Use your eligible regional OKX service; never switch regions to bypass restrictions.")
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/okx")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--once", action="store_true", help="Fetch the latest closed REST window, forecast once, then exit.")
    parser.add_argument("--max-predictions", type=int, default=0, help="Stop after this many newly saved forecasts; 0 means unlimited.")
    parser.add_argument("--run-seconds", type=int, default=0, help="Bound the streaming observation period; 0 means unlimited.")
    parser.add_argument("--stop-file", type=Path, help="Stop streaming when this local control file appears.")
    return parser


def run(args, stream_dir, status=None):
    client = OKXMarketClient(args.inst_id, args.bar, args.region)
    events = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    worker = None
    try:
        if status is not None:
            status.update(phase="loading", connection="disconnected", data_error=None, device=args.device)
        client.validate_instrument()
        client.fetch_history(args.lookback)
        torch.set_num_threads(args.threads)
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=args.local_files_only).eval()
        model = Kronos.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=args.local_files_only).eval()
        predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=512)
        LOGGER.info("Model loaded on %s (PyTorch %s)", predictor.device, torch.__version__)
        if args.once:
            history = client.fetch_history(args.lookback)
            forecast_window(predictor, history, args, stream_dir, client.server_time_ms())
            return 0
        worker = threading.Thread(target=market_worker, args=(client, args.lookback, events, stop_event, status), daemon=True)
        worker.start()
        if status is not None:
            status.update(phase="running")
        started = time.monotonic()
        completed = 0
        while not args.run_seconds or time.monotonic() - started < args.run_seconds:
            if args.stop_file is not None and args.stop_file.exists():
                LOGGER.info("Stop requested by local controller")
                break
            try:
                snapshot = events.get(timeout=1)
            except queue.Empty:
                continue
            if isinstance(snapshot, Exception):
                raise snapshot
            history, server_ms, observed_at = snapshot
            server_ms += int((time.monotonic() - observed_at) * 1000)
            try:
                if status is not None:
                    status.update(phase="predicting")
                saved = forecast_window(predictor, history, args, stream_dir, server_ms)
                completed += int(saved)
                if status is not None:
                    status.update(phase="running", forecasts_this_session=completed)
            except MarketDataError as error:
                LOGGER.warning("Skipped queued stale window: %s", error)
            if args.max_predictions and completed >= args.max_predictions:
                break
        LOGGER.info("Streaming observation ended; saved %s new forecasts", completed)
        return 0
    finally:
        stop_event.set()
        if worker is not None:
            worker.join(timeout=12)
        else:
            client.close()
        if status is not None:
            status.update(phase="stopped", connection="disconnected")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 2 <= args.lookback <= 512 or not 1 <= args.pred_len <= 512:
        parser.error("lookback must be 2..512 and pred-len must be 1..512.")
    if args.threads < 1 or args.max_predictions < 0 or args.run_seconds < 0:
        parser.error("threads must be positive; run-seconds and max-predictions must be nonnegative.")
    try:
        validation_client = OKXMarketClient(args.inst_id, args.bar, args.region)
        validation_client.close()
    except MarketDataError as error:
        parser.error(str(error))
    stream_dir = args.output_dir / args.region / args.inst_id / args.bar
    stream_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), RotatingFileHandler(stream_dir / "realtime.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"),
    ])
    LOGGER.info("READ-ONLY mode: no API keys, account requests, or orders; timestamps use UTC")
    status = None
    try:
        with FileLock(stream_dir / "runner.lock", timeout=0):
            status = RuntimeStatus(stream_dir / "status.json")
            return run(args, stream_dir, status)
    except Timeout:
        LOGGER.error("Another runner is already using %s", stream_dir)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user")
        return 0
    except Exception:
        if status is not None:
            status.update(phase="error", connection="disconnected")
        LOGGER.exception("Forecast runner stopped without issuing any orders")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
