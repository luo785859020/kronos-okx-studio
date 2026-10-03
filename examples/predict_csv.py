import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer


PRICE_COLUMNS = ["open", "high", "low", "close"]
FEATURE_COLUMNS = PRICE_COLUMNS + ["volume", "amount"]
MODEL_ID = "NeoQuasar/Kronos-small"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
MAX_CONTEXT = 512


def read_timestamps(frame):
    if "timestamps" not in frame.columns:
        raise ValueError("CSV must contain a timestamps column.")
    timestamps = pd.to_datetime(frame["timestamps"], errors="raise")
    if timestamps.empty or timestamps.isna().any():
        raise ValueError("Timestamps must be nonempty and contain no missing values.")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and sorted in increasing order.")
    if timestamps.dt.tz is not None:
        raise ValueError("Use timezone-naive timestamps in the exchange's local time.")
    return timestamps.reset_index(drop=True)


def load_csv(path):
    frame = pd.read_csv(path)
    frame["timestamps"] = read_timestamps(frame)
    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing price columns: {missing}")
    columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Price, volume and amount columns must contain finite values.")
    return frame[["timestamps"] + columns]


def prepare_inputs(frame, lookback, pred_len, start=0, future_timestamps=None):
    if not 2 <= lookback <= MAX_CONTEXT:
        raise ValueError(f"lookback must be between 2 and {MAX_CONTEXT}.")
    if not 1 <= pred_len <= MAX_CONTEXT:
        raise ValueError(f"pred_len must be between 1 and {MAX_CONTEXT}.")
    if start < 0:
        raise ValueError("start must be nonnegative.")
    if future_timestamps is not None:
        if start != 0:
            raise ValueError("start is only supported in holdout mode.")
        if len(frame) < lookback:
            raise ValueError(f"Need at least {lookback} history rows, got {len(frame)}.")
        history = frame.iloc[-lookback:].reset_index(drop=True)
        timestamps = read_timestamps(pd.DataFrame({"timestamps": future_timestamps}))
        if len(timestamps) != pred_len:
            raise ValueError("Future timestamp count must equal pred_len.")
        if timestamps.iloc[0] <= history["timestamps"].iloc[-1]:
            raise ValueError("Future timestamps must follow the historical window.")
        return history, timestamps, None
    required = start + lookback + pred_len
    if len(frame) < required:
        raise ValueError(f"Holdout mode needs at least {required} rows, got {len(frame)}.")
    history = frame.iloc[start:start + lookback].reset_index(drop=True)
    actual = frame.iloc[start + lookback:required].reset_index(drop=True)
    return history, actual["timestamps"], actual


def save_results(output_dir, history, predictions, actual, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "prediction.csv", index_label="timestamps")
    history.to_csv(output_dir / "history.csv", index=False)
    if actual is not None:
        comparison = actual.set_index("timestamps")[PRICE_COLUMNS].add_suffix("_actual")
        comparison = comparison.join(predictions[PRICE_COLUMNS].add_suffix("_predicted"))
        comparison.to_csv(output_dir / "comparison.csv", index_label="timestamps")
        metadata["metrics"] = {}
        for column in PRICE_COLUMNS:
            difference = predictions[column].to_numpy() - actual[column].to_numpy()
            metadata["metrics"][column] = {
                "mae": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(difference ** 2))),
            }
    else:
        (output_dir / "comparison.csv").unlink(missing_ok=True)
    with (output_dir / "run.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)


def build_parser():
    parser = argparse.ArgumentParser(description="Run Kronos-small on a local CSV.")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "tests/data/regression_input.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/csv-small")
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--pred-len", type=int, default=30)
    parser.add_argument("--start", type=int, default=0, help="Zero-based history start row in holdout mode.")
    parser.add_argument("--future-timestamps", type=Path, help="CSV with explicit future timestamps; uses the latest history rows.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true", help="Use already cached model weights without downloads.")
    parser.add_argument("--validate-only", action="store_true", help="Check data without loading model weights.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1 or args.sample_count < 1:
        parser.error("threads and sample-count must be positive.")
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature must be positive and finite.")
    if args.top_k < 0 or not 0 < args.top_p <= 1:
        parser.error("top-k must be nonnegative and top-p must be in (0, 1].")
    try:
        output_files = {args.output_dir.resolve() / name for name in ("prediction.csv", "history.csv", "comparison.csv", "run.json")}
        input_files = [args.csv] + ([args.future_timestamps] if args.future_timestamps else [])
        if any(path.resolve() in output_files for path in input_files):
            raise ValueError("Output files must not overwrite an input CSV.")
        frame = load_csv(args.csv)
        future = None
        if args.future_timestamps is not None:
            future = read_timestamps(pd.read_csv(args.future_timestamps))
        history, timestamps, actual = prepare_inputs(frame, args.lookback, args.pred_len, args.start, future)
    except (OSError, ValueError, AttributeError) as error:
        parser.error(str(error))

    print(f"CSV validated: {len(frame)} rows; columns={list(frame.columns)}", flush=True)
    print(f"History: {history['timestamps'].iloc[0]} -> {history['timestamps'].iloc[-1]}", flush=True)
    print(f"Prediction: {timestamps.iloc[0]} -> {timestamps.iloc[-1]} ({args.pred_len} rows)", flush=True)
    if args.validate_only:
        return 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    print(f"Loading {MODEL_ID} on {args.device}...", flush=True)
    tokenizer = KronosTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=args.local_files_only,
    ).eval()
    model = Kronos.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=args.local_files_only,
    ).eval()
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=MAX_CONTEXT)
    predictions = predictor.predict(
        df=history.drop(columns="timestamps"),
        x_timestamp=history["timestamps"],
        y_timestamp=timestamps,
        pred_len=args.pred_len,
        T=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_count=args.sample_count,
        verbose=True,
    )
    if predictions.shape != (args.pred_len, len(FEATURE_COLUMNS)) or not np.isfinite(predictions.to_numpy()).all():
        raise RuntimeError("Model returned an invalid prediction shape or nonfinite values.")
    metadata = {
        "mode": "holdout" if actual is not None else "forecast",
        "csv": str(args.csv.resolve()),
        "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "csv_rows": len(frame),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "device": str(predictor.device),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lookback": args.lookback,
        "pred_len": args.pred_len,
        "start": args.start if actual is not None else len(frame) - args.lookback,
        "seed": args.seed,
        "threads": args.threads,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "sample_count": args.sample_count,
        "history_start": str(history["timestamps"].iloc[0]),
        "history_end": str(history["timestamps"].iloc[-1]),
        "prediction_start": str(timestamps.iloc[0]),
        "prediction_end": str(timestamps.iloc[-1]),
    }
    save_results(args.output_dir, history, predictions, actual, metadata)
    print(predictions.head().to_string())
    print(f"Saved results to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
