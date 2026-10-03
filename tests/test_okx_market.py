import json
import queue
import threading
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

websocket = pytest.importorskip("websocket", reason="Install requirements-okx.txt to test the optional OKX integration.")

from integrations.okx_market import (
    BAR_MILLISECONDS, FEATURE_COLUMNS, MarketDataError, OKXMarketClient,
    merge_candles, parse_candle, validate_history,
)
from examples.okx_realtime_predict import (
    build_parser, enqueue_latest, forecast_window, future_timestamps, market_worker,
)


INTERVAL = BAR_MILLISECONDS["5m"]
BASE_TIME = int(pd.Timestamp("2024-01-01").value // 1_000_000)


def candle_row(index=0, confirm="1"):
    return [str(BASE_TIME + index * INTERVAL), "100", "102", "99", "101", "2", "222", "202", confirm]


def history_frame(count=4, start=0):
    return pd.DataFrame([parse_candle(candle_row(index), INTERVAL) for index in range(start, start + count)])


def test_spot_fields_map_base_volume_and_quote_amount():
    candle = parse_candle(candle_row(), INTERVAL)
    assert candle["timestamps"] == pd.Timestamp("2024-01-01")
    assert candle["volume"] == 2
    assert candle["amount"] == 202
    assert candle["timestamps"].tz is None


def test_doge_candles_preserve_small_prices_and_volume_units():
    client = OKXMarketClient(inst_id="DOGE-USDT")
    try:
        message = {"arg": {"channel": "candle5m", "instId": "DOGE-USDT"}, "data": [[str(BASE_TIME), "0.102346", "0.102350", "0.102340", "0.102347", "1234", "1234", "126.296198", "1"]]}
        candles = client.parse_message(json.dumps(message))
        assert len(candles) == 1
        assert candles[0]["close"] == pytest.approx(0.102347)
        assert candles[0]["volume"] == 1234
        assert candles[0]["amount"] == pytest.approx(126.296198)
    finally:
        client.close()


def test_unfinished_candle_is_ignored():
    assert parse_candle(candle_row(confirm="0"), INTERVAL) is None


@pytest.mark.parametrize("index,value", [(1, "nan"), (2, "90"), (3, "103"), (4, "0"), (5, "-1"), (7, "inf"), (8, "2"), (0, "1")])
def test_invalid_candles_rejected(index, value):
    row = candle_row()
    row[index] = value
    with pytest.raises(MarketDataError):
        parse_candle(row, INTERVAL)


def test_wrong_row_length_rejected():
    with pytest.raises(MarketDataError):
        parse_candle(candle_row()[:-1], INTERVAL)


def test_duplicate_and_older_messages_do_not_change_history():
    history = history_frame()
    assert merge_candles(history, [parse_candle(candle_row(2), INTERVAL)], 4) is history
    merged = merge_candles(history, [parse_candle(candle_row(4), INTERVAL)] * 2, 4)
    pd.testing.assert_frame_equal(merged, history_frame(start=1))


@pytest.mark.parametrize("server_delta", [-1, INTERVAL])
def test_future_or_stale_history_rejected(server_delta):
    with pytest.raises(MarketDataError):
        validate_history(history_frame(), 4, INTERVAL, BASE_TIME + 4 * INTERVAL + server_delta)


def test_missing_history_candle_rejected():
    history = history_frame(5).drop(index=2).reset_index(drop=True)
    with pytest.raises(MarketDataError, match="missing"):
        validate_history(history, 4, INTERVAL, BASE_TIME + 5 * INTERVAL)


def test_valid_closed_history():
    validate_history(history_frame(), 4, INTERVAL, BASE_TIME + 4 * INTERVAL + 10_000)


def test_only_spot_instrument_names_supported():
    with pytest.raises(MarketDataError, match="spot"):
        OKXMarketClient("BTC-USDT-SWAP")


def test_websocket_message_filtering():
    client = OKXMarketClient()
    assert client.parse_message("pong") == []
    assert client.parse_message(json.dumps({"event": "subscribe"})) == []
    message = {"arg": {"channel": "candle5m", "instId": "BTC-USDT"}, "data": [candle_row(), candle_row(1, "0")]}
    assert len(client.parse_message(json.dumps(message))) == 1
    message["arg"]["instId"] = "ETH-USDT"
    assert client.parse_message(json.dumps(message)) == []
    client.close()


@pytest.mark.parametrize("message", ["not-json", "[]", '{"event":"error","code":"60012","msg":"bad request"}'])
def test_websocket_errors_are_not_silently_accepted(message):
    client = OKXMarketClient()
    with pytest.raises(MarketDataError):
        client.parse_message(message)
    client.close()


def test_rest_history_paginates_and_drops_unconfirmed(monkeypatch):
    client = OKXMarketClient()
    rows = [candle_row(index) for index in range(512)] + [candle_row(512, "0")]
    calls = []

    def get_page(path, params):
        calls.append(params.copy())
        selected = [row for row in rows if "after" not in params or int(row[0]) < int(params["after"])]
        return selected[::-1][:params["limit"]]

    monkeypatch.setattr(client, "_get", get_page)
    monkeypatch.setattr(client, "server_time_ms", lambda: BASE_TIME + 512 * INTERVAL + 1000)
    history = client.fetch_history(512)
    assert len(calls) == 2
    assert "after" in calls[1]
    pd.testing.assert_frame_equal(history, history_frame(512))
    client.close()


def test_http_failure_retries_without_any_authentication(monkeypatch):
    unavailable = Mock(status_code=500)
    success = Mock(status_code=200)
    success.json.return_value = {"code": "0", "data": [{"ts": str(BASE_TIME)}]}
    session = Mock()
    session.get.side_effect = [unavailable, success]
    monkeypatch.setattr("integrations.okx_market.time.sleep", lambda delay: None)
    client = OKXMarketClient(session=session)
    assert client.server_time_ms() == BASE_TIME
    assert session.get.call_count == 2
    for call in session.get.call_args_list:
        assert "headers" not in call.kwargs
        assert call.args[0].endswith("/api/v5/public/time")


def test_access_denied_does_not_retry_or_switch_regions():
    session = Mock()
    session.get.return_value.status_code = 451
    client = OKXMarketClient(region="us", session=session)
    with pytest.raises(MarketDataError, match="denied"):
        client.server_time_ms()
    assert session.get.call_count == 1
    assert session.get.call_args.args[0].startswith("https://us.okx.com/")


def test_live_instrument_validation_rejects_suspended_spot(monkeypatch):
    client = OKXMarketClient()
    monkeypatch.setattr(client, "_get", lambda *args: [{"instId": "BTC-USDT", "instType": "SPOT", "state": "suspend"}])
    with pytest.raises(MarketDataError, match="not a live spot"):
        client.validate_instrument()
    client.close()


def test_websocket_heartbeat_and_reconnect(monkeypatch):
    first = Mock()
    first.recv.side_effect = [websocket.WebSocketTimeoutException(), websocket.WebSocketTimeoutException()]
    second = Mock()
    connect = Mock(side_effect=[first, second])
    monkeypatch.setattr("integrations.okx_market.websocket.create_connection", connect)
    stop_event = Mock()
    stop_event.is_set.return_value = False
    client = OKXMarketClient()
    stream = client.stream(stop_event)
    assert next(stream) is None
    assert next(stream) is None
    assert connect.call_count == 2
    assert "ping" in [call.args[0] for call in first.send.call_args_list]
    first.close.assert_called_once()
    stop_event.wait.assert_called_once_with(1)
    stream.close()
    second.close.assert_called_once()
    client.close()


def test_queue_keeps_only_latest_snapshot():
    events = queue.Queue(maxsize=1)
    enqueue_latest(events, "old")
    enqueue_latest(events, "new")
    assert events.get_nowait() == "new"


def test_worker_repairs_gap_and_ignores_reconnect_duplicates():
    client = Mock(interval_ms=INTERVAL)
    client.stream.return_value = (event for event in [None, None, [parse_candle(candle_row(5), INTERVAL)]])
    client.fetch_history.side_effect = [history_frame(), history_frame(), history_frame(start=2)]
    client.server_time_ms.side_effect = [BASE_TIME + 4 * INTERVAL + 1000, BASE_TIME + 4 * INTERVAL + 1000, BASE_TIME + 6 * INTERVAL + 1000, BASE_TIME + 6 * INTERVAL + 1000]
    events = queue.Queue()
    market_worker(client, 4, events, threading.Event())
    snapshots = [events.get_nowait() for _ in range(events.qsize())]
    assert len(snapshots) == 2
    pd.testing.assert_frame_equal(snapshots[-1][0], history_frame(start=2))
    client.close.assert_called_once()


def test_crypto_future_times_continue_across_weekends():
    history = history_frame()
    history["timestamps"] += pd.Timedelta(days=4, hours=23, minutes=40)
    future = future_timestamps(history, INTERVAL, 3)
    assert future.iloc[0] == pd.Timestamp("2024-01-06 00:00:00")
    assert (future.diff().dropna() == pd.Timedelta(minutes=5)).all()


def test_forecast_persistence_and_restart_deduplication(tmp_path):
    args = build_parser().parse_args(["--lookback", "4", "--pred-len", "2"])
    history = history_frame()
    timestamps = future_timestamps(history, INTERVAL, 2)
    predictor = Mock()
    predictor.predict.return_value = pd.DataFrame(np.full((2, 6), 102.0), columns=FEATURE_COLUMNS, index=timestamps)
    server_ms = BASE_TIME + 4 * INTERVAL + 1000
    assert forecast_window(predictor, history, args, tmp_path, server_ms)
    assert not forecast_window(predictor, history, args, tmp_path, server_ms)
    predictor.predict.assert_called_once()
    run_dir = tmp_path / "20240101T001500Z"
    report = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert report["mode"] == "okx_public_read_only"
    assert report["signal"]["kind"] == "forecast_only_not_an_order"
    assert (run_dir / "prediction.csv").exists()
    assert (tmp_path / "latest.json").exists()
    assert predictor.predict.call_args.kwargs["x_timestamp"].iloc[-1] < timestamps.iloc[0]


def test_nonfinite_model_output_is_not_saved(tmp_path):
    args = build_parser().parse_args(["--lookback", "4", "--pred-len", "2"])
    history = history_frame()
    predictor = Mock()
    predictor.predict.return_value = pd.DataFrame(np.nan, index=future_timestamps(history, INTERVAL, 2), columns=FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="nonfinite"):
        forecast_window(predictor, history, args, tmp_path, BASE_TIME + 4 * INTERVAL)
    assert not list(tmp_path.iterdir())
