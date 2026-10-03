import json
import logging
import re
import threading
import time

import numpy as np
import pandas as pd
import requests
import websocket


LOGGER = logging.getLogger(__name__)
ENDPOINTS = {
    "global": ("https://openapi.okx.com", "wss://ws.okx.com/ws/v5/business"),
    "us": ("https://us.okx.com", "wss://wsus.okx.com/ws/v5/business"),
    "eea": ("https://eea.okx.com", "wss://wseea.okx.com/ws/v5/business"),
}
BAR_MILLISECONDS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1H": 3_600_000, "2H": 7_200_000, "4H": 14_400_000,
}
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


class MarketDataError(ValueError):
    pass


def parse_candle(row, interval_ms):
    if not isinstance(row, (list, tuple)) or len(row) != 9 or row[8] not in ("0", "1"):
        raise MarketDataError("Expected an OKX nine-field candle with confirm=0 or 1.")
    if row[8] != "1":
        return None
    try:
        timestamp_ms = int(row[0])
        values = np.asarray([row[index] for index in (1, 2, 3, 4, 5, 7)], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise MarketDataError("Candle contains invalid numeric fields.") from error
    if timestamp_ms < 0 or timestamp_ms % interval_ms:
        raise MarketDataError("Candle timestamp is not aligned to the requested interval.")
    if not np.isfinite(values).all() or (values[:4] <= 0).any() or (values[4:] < 0).any():
        raise MarketDataError("Candle contains nonfinite, nonpositive price, or negative volume values.")
    open_price, high_price, low_price, close_price = values[:4]
    if high_price < max(open_price, low_price, close_price) or low_price > min(open_price, close_price):
        raise MarketDataError("Candle violates OHLC price ordering.")
    return {"timestamps": pd.to_datetime(timestamp_ms, unit="ms"), **dict(zip(FEATURE_COLUMNS, values))}


def merge_candles(history, candles, lookback):
    if not candles:
        return history
    additions = pd.DataFrame(candles)
    if history is not None and not history.empty:
        additions = additions[additions["timestamps"] > history["timestamps"].iloc[-1]]
        if additions.empty:
            return history
        additions = pd.concat([history, additions], ignore_index=True)
    return additions.drop_duplicates("timestamps").sort_values("timestamps").tail(lookback).reset_index(drop=True)


def validate_history(history, lookback, interval_ms, server_time_ms):
    if history is None or len(history) != lookback:
        raise MarketDataError(f"Need {lookback} confirmed candles before forecasting.")
    times = history["timestamps"]
    if not times.diff().iloc[1:].eq(pd.Timedelta(milliseconds=interval_ms)).all():
        raise MarketDataError("History contains a missing, duplicated, or out-of-order candle.")
    close_time_ms = int(times.iloc[-1].value // 1_000_000) + interval_ms
    if server_time_ms < close_time_ms:
        raise MarketDataError("Latest candle has not closed according to OKX server time.")
    if server_time_ms - close_time_ms >= interval_ms:
        raise MarketDataError("Latest confirmed candle is stale; prediction is paused.")


class OKXMarketClient:
    def __init__(self, inst_id="BTC-USDT", bar="5m", region="global", session=None):
        if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", inst_id):
            raise MarketDataError("Only spot instrument IDs such as BTC-USDT are supported.")
        if bar not in BAR_MILLISECONDS or region not in ENDPOINTS:
            raise MarketDataError("Unsupported bar or region.")
        self.inst_id = inst_id
        self.bar = bar
        self.interval_ms = BAR_MILLISECONDS[bar]
        self.rest_url, self.ws_url = ENDPOINTS[region]
        self.session = session if session is not None else requests.Session()

    def close(self):
        self.session.close()

    def _get(self, path, params=None):
        for attempt in range(3):
            try:
                response = self.session.get(self.rest_url + path, params=params, timeout=(10, 20))
                if response.status_code in (403, 451):
                    raise MarketDataError("OKX denied access. Check account region and service eligibility; no region fallback is performed.")
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.RequestException(f"Temporary HTTP status {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise MarketDataError("Invalid OKX response object.")
                if payload.get("code") in ("50011", "50040"):
                    raise requests.RequestException("OKX public rate limit reached")
                if payload.get("code") != "0":
                    raise MarketDataError(f"OKX API error {payload.get('code')}: {payload.get('msg')}")
                if not isinstance(payload.get("data"), list):
                    raise MarketDataError("OKX response is missing its data array.")
                return payload["data"]
            except requests.RequestException as error:
                if attempt == 2:
                    raise MarketDataError(f"Public REST request failed: {error}") from error
                time.sleep(2 ** attempt)
        raise MarketDataError("Public REST retries exhausted.")

    def validate_instrument(self):
        instruments = self._get("/api/v5/public/instruments", {"instType": "SPOT", "instId": self.inst_id})
        if not any(item.get("instId") == self.inst_id and item.get("instType") == "SPOT" and item.get("state") == "live" for item in instruments):
            raise MarketDataError(f"{self.inst_id} is not a live spot instrument on this regional endpoint.")

    def server_time_ms(self):
        data = self._get("/api/v5/public/time")
        if not data or "ts" not in data[0]:
            raise MarketDataError("OKX server time is missing.")
        return int(data[0]["ts"])

    def fetch_history(self, lookback):
        if not 2 <= lookback <= 512:
            raise MarketDataError("lookback must be between 2 and 512.")
        candles = {}
        cursor = None
        for _ in range(4):
            params = {"instId": self.inst_id, "bar": self.bar, "limit": min(lookback + 1, 300)}
            if cursor is not None:
                params["after"] = str(cursor)
            rows = self._get("/api/v5/market/candles", params)
            if not rows:
                break
            for row in rows:
                candle = parse_candle(row, self.interval_ms)
                if candle is not None:
                    candles.setdefault(candle["timestamps"], candle)
            if len(candles) >= lookback:
                break
            next_cursor = min(int(row[0]) for row in rows)
            if cursor is not None and next_cursor >= cursor:
                break
            cursor = next_cursor
        history = merge_candles(None, list(candles.values()), lookback)
        validate_history(history, lookback, self.interval_ms, self.server_time_ms())
        return history

    def parse_message(self, message):
        if message == "pong":
            return []
        try:
            payload = json.loads(message)
        except (ValueError, TypeError) as error:
            raise MarketDataError("Invalid WebSocket JSON.") from error
        if not isinstance(payload, dict):
            raise MarketDataError("Invalid WebSocket object.")
        if payload.get("event") == "error":
            raise MarketDataError(f"OKX subscription error {payload.get('code')}: {payload.get('msg')}")
        if payload.get("event") == "notice":
            raise websocket.WebSocketException("OKX requested a connection restart.")
        expected = {"channel": "candle" + self.bar, "instId": self.inst_id}
        argument = payload.get("arg", {})
        if not isinstance(argument, dict) or any(argument.get(key) != value for key, value in expected.items()) or "data" not in payload:
            return []
        if not isinstance(payload["data"], list):
            raise MarketDataError("Invalid WebSocket candle array.")
        candles = [parse_candle(row, self.interval_ms) for row in payload["data"]]
        return [candle for candle in candles if candle is not None]

    def stream(self, stop_event=None, on_state=None):
        stop_event = stop_event if stop_event is not None else threading.Event()
        retry_delay = 1
        while not stop_event.is_set():
            connection = None
            try:
                if on_state is not None:
                    on_state("connecting")
                connection = websocket.create_connection(self.ws_url, timeout=10)
                connection.send(json.dumps({"op": "subscribe", "args": [{"channel": "candle" + self.bar, "instId": self.inst_id}]}))
                LOGGER.info("Connected to %s; refreshing confirmed REST history", self.ws_url)
                yield None
                awaiting_pong = False
                last_refresh = time.monotonic()
                last_heartbeat = 0
                while not stop_event.is_set():
                    try:
                        message = connection.recv()
                    except websocket.WebSocketTimeoutException:
                        if awaiting_pong:
                            raise websocket.WebSocketConnectionClosedException("Heartbeat timed out")
                        connection.send("ping")
                        awaiting_pong = True
                        continue
                    if not message:
                        raise websocket.WebSocketConnectionClosedException("Empty WebSocket response")
                    awaiting_pong = False
                    candles = self.parse_message(message)
                    if on_state is not None and time.monotonic() - last_heartbeat >= 10:
                        on_state("connected")
                        last_heartbeat = time.monotonic()
                    if candles:
                        retry_delay = 1
                        yield candles
                    if time.monotonic() - last_refresh >= 60:
                        yield None
                        last_refresh = time.monotonic()
            except websocket.WebSocketBadStatusException as error:
                if error.status_code in (403, 451):
                    raise MarketDataError("WebSocket access denied; check the eligible regional service.") from error
                LOGGER.warning("WebSocket handshake failed: %s; retrying in %ss", error, retry_delay)
            except (websocket.WebSocketException, OSError) as error:
                LOGGER.warning("WebSocket disconnected: %s; reconnecting in %ss", error, retry_delay)
            finally:
                if connection is not None:
                    connection.close()
            if on_state is not None and not stop_event.is_set():
                on_state("reconnecting")
            stop_event.wait(retry_delay)
            retry_delay = min(retry_delay * 2, 30)
