import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.predict_csv import FEATURE_COLUMNS, load_csv, main, prepare_inputs, save_results


@pytest.fixture
def market_data():
    frame = pd.DataFrame({"timestamps": pd.date_range("2024-01-02 09:30", periods=12, freq="5min")})
    for column_index, column in enumerate(FEATURE_COLUMNS):
        frame[column] = np.arange(12, dtype=np.float32) + 10 + column_index
    return frame


def test_csv_optional_volume_and_amount(tmp_path, market_data):
    path = tmp_path / "prices.csv"
    market_data.drop(columns=["volume", "amount"]).to_csv(path, index=False)
    loaded = load_csv(path)
    assert list(loaded.columns) == ["timestamps", "open", "high", "low", "close"]
    history, timestamps, actual = prepare_inputs(loaded, lookback=4, pred_len=2, start=1)
    pd.testing.assert_frame_equal(history, loaded.iloc[1:5].reset_index(drop=True))
    pd.testing.assert_series_equal(timestamps, loaded["timestamps"].iloc[5:7].reset_index(drop=True))
    assert len(actual) == 2


@pytest.mark.parametrize("problem", ["missing_price", "missing_time", "duplicate", "unsorted", "nan", "infinity"])
def test_invalid_csv_rejected(tmp_path, market_data, problem):
    if problem == "missing_price":
        market_data = market_data.drop(columns="close")
    elif problem == "missing_time":
        market_data = market_data.drop(columns="timestamps")
    elif problem == "duplicate":
        market_data.loc[1, "timestamps"] = market_data.loc[0, "timestamps"]
    elif problem == "unsorted":
        market_data = market_data.iloc[::-1]
    elif problem == "nan":
        market_data.loc[0, "close"] = np.nan
    else:
        market_data.loc[0, "volume"] = np.inf
    path = tmp_path / "invalid.csv"
    market_data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_csv(path)


@pytest.mark.parametrize("lookback,pred_len,start", [(1, 2, 0), (513, 2, 0), (4, 0, 0), (4, 513, 0), (4, 2, -1), (10, 3, 0)])
def test_invalid_windows_rejected(market_data, lookback, pred_len, start):
    with pytest.raises(ValueError):
        prepare_inputs(market_data, lookback, pred_len, start)


def test_forecast_uses_latest_history_and_explicit_timestamps(market_data):
    future = pd.Series(pd.to_datetime(["2024-01-03 09:30", "2024-01-03 09:35"]), name="timestamps")
    history, timestamps, actual = prepare_inputs(market_data, 4, 2, future_timestamps=future)
    pd.testing.assert_frame_equal(history, market_data.tail(4).reset_index(drop=True))
    pd.testing.assert_series_equal(timestamps, future)
    assert actual is None


def test_overlapping_future_timestamps_rejected(market_data):
    with pytest.raises(ValueError, match="must follow"):
        prepare_inputs(market_data, 4, 2, future_timestamps=market_data["timestamps"].tail(2))


def test_future_timestamp_count_must_match(market_data):
    future = pd.Series(pd.to_datetime(["2024-01-03 09:30"]))
    with pytest.raises(ValueError, match="count must equal"):
        prepare_inputs(market_data, 4, 2, future_timestamps=future)


def test_forecast_rejects_holdout_offset(market_data):
    future = pd.Series(pd.to_datetime(["2024-01-03 09:30", "2024-01-03 09:35"]))
    with pytest.raises(ValueError, match="only supported in holdout"):
        prepare_inputs(market_data, 4, 2, start=1, future_timestamps=future)


def test_validation_does_not_download_models(tmp_path, market_data, monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("Validation must not load models")

    monkeypatch.setattr("examples.predict_csv.KronosTokenizer.from_pretrained", unexpected_download)
    path = tmp_path / "data.csv"
    market_data.to_csv(path, index=False)
    assert main(["--csv", str(path), "--lookback", "4", "--pred-len", "2", "--validate-only"]) == 0


def test_saved_predictions_align_with_actuals(tmp_path, market_data):
    history, timestamps, actual = prepare_inputs(market_data, 4, 2)
    predictions = actual.set_index("timestamps")[FEATURE_COLUMNS] + 1.0
    save_results(tmp_path, history, predictions, actual, {"mode": "holdout"})
    comparison = pd.read_csv(tmp_path / "comparison.csv", parse_dates=["timestamps"])
    pd.testing.assert_series_equal(comparison["timestamps"], timestamps)
    report = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert report["metrics"]["close"] == {"mae": 1.0, "rmse": 1.0}
    assert len(pd.read_csv(tmp_path / "prediction.csv")) == 2


def test_forecast_removes_stale_comparison(tmp_path, market_data):
    history, _, actual = prepare_inputs(market_data, 4, 2)
    predictions = actual.set_index("timestamps")[FEATURE_COLUMNS]
    save_results(tmp_path, history, predictions, actual, {"mode": "holdout"})
    save_results(tmp_path, history, predictions, None, {"mode": "forecast"})
    assert not (tmp_path / "comparison.csv").exists()
    report = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert "metrics" not in report


def test_input_csv_cannot_be_overwritten(tmp_path, market_data):
    path = tmp_path / "prediction.csv"
    market_data.to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["--csv", str(path), "--output-dir", str(tmp_path), "--validate-only"])
    assert error.value.code == 2
    assert path.read_bytes() == before


@pytest.fixture
def csv_dataset_class(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "finetune_csv"))
    from finetune_base_model import CustomKlineDataset

    return CustomKlineDataset


def make_dataset(dataset_class, path):
    return dataset_class(
        str(path), data_type="val", lookback_window=4, predict_window=2,
        train_ratio=0.0, val_ratio=1.0, test_ratio=0.0,
    )


def test_csv_normalization_uses_only_history(tmp_path, market_data, csv_dataset_class):
    path = tmp_path / "training.csv"
    market_data.to_csv(path, index=False)
    dataset = make_dataset(csv_dataset_class, path)
    normalized, timestamps = dataset[0]
    values = market_data[FEATURE_COLUMNS].iloc[:7].to_numpy(dtype=np.float32)
    past_values = values[:4]
    expected = np.clip((values - past_values.mean(axis=0)) / (past_values.std(axis=0) + 1e-5), -5, 5)
    np.testing.assert_allclose(normalized.numpy(), expected)
    assert timestamps.shape == (7, 5)


def test_future_changes_do_not_affect_normalized_history(tmp_path, market_data, csv_dataset_class):
    path = tmp_path / "training.csv"
    market_data.to_csv(path, index=False)
    original, _ = make_dataset(csv_dataset_class, path)[0]
    market_data.loc[4:6, FEATURE_COLUMNS] += 1000.0
    market_data.to_csv(path, index=False)
    changed, _ = make_dataset(csv_dataset_class, path)[0]
    np.testing.assert_array_equal(original[:4].numpy(), changed[:4].numpy())


def test_constant_history_remains_finite(tmp_path, market_data, csv_dataset_class):
    market_data.loc[:3, FEATURE_COLUMNS] = 10.0
    path = tmp_path / "constant.csv"
    market_data.to_csv(path, index=False)
    normalized, _ = make_dataset(csv_dataset_class, path)[0]
    assert np.isfinite(normalized.numpy()).all()
    np.testing.assert_array_equal(normalized[:4].numpy(), np.zeros((4, 6)))
    assert np.max(np.abs(normalized.numpy())) <= 5
