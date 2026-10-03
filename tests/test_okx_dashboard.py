import json
import os
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

pytest.importorskip("flask", reason="Install webui/requirements.txt to test the dashboard.")
pytest.importorskip("plotly")
pytest.importorskip("websocket")

from flask import Flask
from filelock import FileLock
from webui.okx_dashboard import DEFAULT_CONFIG, OKXController, create_blueprint, detect_devices, parse_config
from examples.okx_realtime_predict import RuntimeStatus


@pytest.fixture(autouse=True)
def clear_device_detection_cache():
    detect_devices.cache_clear()
    yield
    detect_devices.cache_clear()


@pytest.fixture
def controller(tmp_path):
    process = Mock(pid=12345)
    process.poll.return_value = None
    factory = Mock(return_value=process)
    manager = OKXController(tmp_path / "predictions", process_factory=factory)
    yield manager
    process.poll.return_value = 0
    manager.stop(wait=True)


@pytest.fixture
def client(controller):
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parents[1] / "webui/templates"))
    app.register_blueprint(create_blueprint(controller))
    app.config["TESTING"] = True
    return app.test_client()


def seed_record(controller, run_id="20261003T125500Z", config=None):
    config = config or DEFAULT_CONFIG
    directory = controller.directory(config) / run_id
    directory.mkdir(parents=True)
    rows = pd.DataFrame({"timestamps": pd.date_range("2026-10-03 12:50", periods=2, freq="5min"), "open": [100, 101], "high": [102, 103], "low": [99, 100], "close": [101, 102], "volume": [2, 3], "amount": [202, 306]})
    rows.to_csv(directory / "history.csv", index=False)
    rows["timestamps"] += pd.Timedelta(minutes=10)
    rows.to_csv(directory / "prediction.csv", index=False)
    report = {"history_end": "2026-10-03 12:55:00", "inference_completed_at": "2026-10-03T13:00:02Z", "signal": {"last_actual_close": 101, "predicted_final_close": 102, "predicted_return": 0.01}}
    (directory / "run.json").write_text(json.dumps(report), encoding="utf-8")
    return directory


@pytest.mark.parametrize("values", [{"inst_id": "../BTC"}, {"inst_id": "BTC-USDT-SWAP"}, {"bar": []}, {"region": "other"}, {"lookback": True}, {"lookback": "1.5"}, {"lookback": 1}, {"pred_len": 513}, {"device": "gpu"}, {"device": "cuda"}, {"device": "cuda:-1"}, {"device": "cuda:00"}, {"device": "cuda:100"}, {"device": "../../"}, {"device": []}, {"device": None}, {"output_dir": "C:/"}, None])
def test_invalid_start_configuration(values):
    with pytest.raises(ValueError):
        parse_config(values)


def test_public_page_and_offline_chart_library(client):
    page = client.get("/okx")
    assert page.status_code == 200
    assert "API 密钥" in page.get_data(as_text=True)
    assert '<select id="inst-preset"' in page.get_data(as_text=True)
    assert '<option value="DOGE-USDT">DOGE-USDT · 狗狗币</option>' in page.get_data(as_text=True)
    assert '<option value="custom">其他币对（手动输入）</option>' in page.get_data(as_text=True)
    assert '<div id="custom-inst-field" hidden>' in page.get_data(as_text=True)
    assert '<datalist' not in page.get_data(as_text=True)
    assert 'id="device" name="device"' in page.get_data(as_text=True)
    assert b"cdn.plot.ly" not in page.data
    assert page.headers["Cache-Control"] == "no-store"
    chart = client.get("/okx/plotly.js")
    assert chart.status_code == 200
    assert chart.mimetype == "application/javascript"
    assert len(chart.data) > 100_000


def test_empty_dashboard_has_no_fabricated_records(client):
    assert client.get("/api/okx/status").json["phase"] == "stopped"
    assert client.get("/api/okx/records").json == {"records": []}
    assert client.get("/api/okx/result/latest").status_code == 404


def test_control_requires_token_and_same_origin(client, controller):
    assert client.post("/api/okx/start", json={}).status_code == 403
    assert client.post("/api/okx/stop", headers={"X-OKX-Control": "wrong"}).status_code == 403
    assert client.post("/api/okx/start", json={}, headers={"X-OKX-Control": controller.csrf_token, "Origin": "https://other.example"}).status_code == 403
    controller.process_factory.assert_not_called()


def test_dashboard_is_local_only_and_rejects_rebinding(client):
    assert client.get("/okx", environ_overrides={"REMOTE_ADDR": "192.168.1.2"}).status_code == 403
    assert client.get("/api/okx/status", base_url="http://other.example").status_code == 403
    assert client.get("/api/okx/devices", environ_overrides={"REMOTE_ADDR": "192.168.1.2"}).status_code == 403


@pytest.mark.parametrize("inst_id", ["BTC-USDT", "DOGE-USDT", "XRP-USDT"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
def test_start_uses_selected_device_fixed_script_and_no_secrets(client, controller, monkeypatch, inst_id, device):
    monkeypatch.setenv("OKX_API_KEY", "do-not-inherit")
    monkeypatch.setattr("webui.okx_dashboard.detect_devices", lambda: {"devices": [{"id": device, "available": True}]})
    headers = {"X-OKX-Control": controller.csrf_token}
    response = client.post("/api/okx/start", json={"inst_id": inst_id, "device": device}, headers=headers)
    assert response.status_code == 200
    assert response.json["phase"] == "starting"
    call = controller.process_factory.call_args
    command = call.args[0]
    assert command[2].endswith("okx_realtime_predict.py")
    assert command[command.index("--device") + 1] == device
    assert response.json["active_config"]["device"] == device
    assert command[command.index("--inst-id") + 1] == inst_id
    assert "--stop-file" in command and "--local-files-only" in command
    assert call.kwargs["shell"] is False
    assert "OKX_API_KEY" not in call.kwargs["env"]
    assert client.post("/api/okx/start", json={}, headers=headers).status_code == 409


def test_device_api_explains_cpu_only_pytorch(client, monkeypatch):
    monkeypatch.setattr("webui.okx_dashboard.torch.version.cuda", None)
    available = Mock(side_effect=AssertionError("CPU build must not initialize CUDA"))
    monkeypatch.setattr("webui.okx_dashboard.torch.cuda.is_available", available)
    response = client.get("/api/okx/devices")
    assert response.status_code == 200
    assert response.json["devices"][0]["id"] == "cpu"
    assert response.json["devices"][0]["available"] is True
    assert response.json["devices"][1]["available"] is False
    assert "CPU 版 PyTorch" in response.json["devices"][1]["reason"]
    assert response.headers["Cache-Control"] == "no-store"
    available.assert_not_called()


def test_device_detection_without_available_driver(monkeypatch):
    monkeypatch.setattr("webui.okx_dashboard.torch.version.cuda", "12.8")
    monkeypatch.setattr("webui.okx_dashboard.torch.cuda.is_available", lambda: False)
    devices = detect_devices()["devices"]
    assert devices[0]["available"] is True
    assert devices[1]["available"] is False
    assert "驱动" in devices[1]["reason"]


def test_cuda_detection_probes_each_device_and_caches(monkeypatch):
    monkeypatch.setattr("webui.okx_dashboard.torch.version.cuda", "12.8")
    monkeypatch.setattr("webui.okx_dashboard.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("webui.okx_dashboard.torch.cuda.device_count", lambda: 2)
    properties = Mock(name="properties")
    properties.name = "Test GPU"
    properties.total_memory = 8 * 1024 ** 3
    monkeypatch.setattr("webui.okx_dashboard.torch.cuda.get_device_properties", lambda index: properties)
    probe = MagicMock()
    allocator = Mock(side_effect=[probe, RuntimeError("no kernel image available")])
    monkeypatch.setattr("webui.okx_dashboard.torch.ones", allocator)
    devices = detect_devices()["devices"]
    assert devices[1]["id"] == "cuda:0" and devices[1]["available"] is True
    assert "8.0 GB" in devices[1]["label"]
    assert devices[2]["id"] == "cuda:1" and devices[2]["available"] is False
    assert "no kernel image" in devices[2]["reason"]
    assert allocator.call_args_list[0].kwargs["device"] == "cuda:0"
    assert allocator.call_args_list[1].kwargs["device"] == "cuda:1"
    detect_devices()
    assert allocator.call_count == 2


def test_cuda_detection_failure_preserves_cpu(monkeypatch):
    monkeypatch.setattr("webui.okx_dashboard.torch.version.cuda", "12.8")
    monkeypatch.setattr("webui.okx_dashboard.torch.cuda.is_available", Mock(side_effect=RuntimeError("driver failure")))
    devices = detect_devices()["devices"]
    assert devices[0]["available"] is True
    assert devices[1]["available"] is False
    assert "driver failure" in devices[1]["reason"]


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
def test_unavailable_gpu_is_rejected_before_spawning(client, controller, monkeypatch, device):
    monkeypatch.setattr("webui.okx_dashboard.detect_devices", lambda: {"devices": [{"id": "cuda:0", "available": False, "reason": "GPU 不可用"}]})
    response = client.post("/api/okx/start", json={"device": device}, headers={"X-OKX-Control": controller.csrf_token})
    assert response.status_code == 400
    assert "GPU" in response.json["error"]
    controller.process_factory.assert_not_called()
    assert not controller.output_root.exists()


def test_gpu_records_are_isolated_without_moving_legacy_cpu_records(client, controller):
    cpu_directory = seed_record(controller)
    config = parse_config({"device": "cuda:0"})
    assert cpu_directory.parent == controller.output_root / "l256-p12/global/BTC-USDT/5m"
    assert controller.directory(config) == controller.output_root / "l256-p12/cuda-0/global/BTC-USDT/5m"
    assert client.get("/api/okx/records?device=cuda:0").json == {"records": []}
    gpu_directory = seed_record(controller, config=config)
    report = json.loads((gpu_directory / "run.json").read_text(encoding="utf-8"))
    report["device"] = "cuda:0"
    (gpu_directory / "run.json").write_text(json.dumps(report), encoding="utf-8")
    assert client.get("/api/okx/result/latest?device=cuda:0").json["report"]["device"] == "cuda:0"
    assert "device" not in client.get("/api/okx/result/latest").json["report"]


def test_invalid_api_payload_does_not_spawn(client, controller):
    headers = {"X-OKX-Control": controller.csrf_token}
    assert client.post("/api/okx/start", json={"bar": []}, headers=headers).status_code == 400
    assert client.post("/api/okx/start", data="", headers=headers).status_code == 400
    assert client.get("/api/okx/status?region=../").status_code == 400
    controller.process_factory.assert_not_called()


def test_stop_requests_graceful_exit_and_preserves_records(controller):
    directory = seed_record(controller)
    controller.start({})
    controller.process.wait.side_effect = lambda timeout: setattr(controller.process.poll, "return_value", 0)
    controller.stop(wait=True)
    assert (directory.parent / "stop.request").exists()
    assert (directory / "run.json").exists()
    controller.process.terminate.assert_not_called()
    assert controller.status(DEFAULT_CONFIG)["phase"] == "stopped"


def test_stop_terminates_unresponsive_owned_process(controller, monkeypatch):
    kill_tree = Mock()
    monkeypatch.setattr("webui.okx_dashboard.subprocess.run", kill_tree)
    controller.start({})
    controller.process.wait.side_effect = [subprocess.TimeoutExpired("child", 18), 0]
    controller.stop(wait=True)
    if os.name == "nt":
        assert kill_tree.call_args.args[0] == ["taskkill", "/PID", "12345", "/T", "/F"]
    else:
        controller.process.terminate.assert_called_once()
    controller.process.kill.assert_not_called()


def test_active_directory_lock_prevents_second_runner(controller):
    directory = controller.directory(DEFAULT_CONFIG)
    directory.mkdir(parents=True)
    with FileLock(directory / "runner.lock"):
        with pytest.raises(RuntimeError, match="进程"):
            controller.start({})
    controller.process_factory.assert_not_called()


def test_status_reflects_heartbeat_and_process_failure(controller):
    controller.start({})
    RuntimeStatus(controller.directory(DEFAULT_CONFIG) / "status.json").update(phase="running", connection="connected", forecasts_this_session=2)
    status = controller.status(DEFAULT_CONFIG)
    assert status["connection"] == "connected"
    assert status["runtime"]["forecasts_this_session"] == 2
    controller.process.poll.return_value = 1
    status = controller.status(DEFAULT_CONFIG)
    assert status["phase"] == "error"
    assert status["connection"] == "disconnected"


def test_records_return_utc_times_and_latest_result(client, controller):
    seed_record(controller)
    result = client.get("/api/okx/result/latest")
    assert result.status_code == 200
    assert result.json["history"][0]["timestamps"].endswith("Z")
    assert len(result.json["prediction"]) == 2
    records = client.get("/api/okx/records").json["records"]
    assert records[0]["id"] == "20261003T125500Z"
    assert client.get("/api/okx/result/not-a-date").status_code == 400


def test_incomplete_run_is_hidden_and_config_isolated(controller):
    seed_record(controller)
    partial = controller.directory(DEFAULT_CONFIG) / "20261003T130000Z"
    partial.mkdir()
    (partial / "prediction.csv").write_text("partial", encoding="utf-8")
    assert len(controller.records(DEFAULT_CONFIG)) == 1
    other = parse_config({"pred_len": 24})
    assert controller.records(other) == []


def test_doge_records_keep_fractional_prices_and_are_isolated(client, controller):
    seed_record(controller)
    config = parse_config({"inst_id": "DOGE-USDT"})
    assert config["inst_id"] == "DOGE-USDT"
    assert client.get("/api/okx/result/latest?inst_id=DOGE-USDT").status_code == 404
    directory = seed_record(controller, config=config)
    for name in ("history", "prediction"):
        frame = pd.read_csv(directory / f"{name}.csv")
        frame[["open", "high", "low", "close", "amount"]] *= 0.00123456
        frame.to_csv(directory / f"{name}.csv", index=False)
    result = client.get("/api/okx/result/latest?inst_id=DOGE-USDT")
    assert result.status_code == 200
    assert result.json["history"][0]["close"] == pytest.approx(0.12469056)
    assert result.json["prediction"][1]["close"] == pytest.approx(0.12592512)
    assert client.get("/api/okx/result/latest").json["history"][0]["close"] == 101
    assert len(client.get("/api/okx/records?inst_id=DOGE-USDT").json["records"]) == 1


def test_result_rejects_nonfinite_values(client, controller):
    directory = seed_record(controller)
    frame = pd.read_csv(directory / "prediction.csv")
    frame.loc[0, "close"] = float("nan")
    frame.to_csv(directory / "prediction.csv", index=False)
    assert client.get("/api/okx/result/latest").status_code == 400


def test_runtime_status_merges_updates(tmp_path):
    status = RuntimeStatus(tmp_path / "status.json")
    status.update(phase="loading")
    status.update(connection="connected")
    result = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert result["phase"] == "loading"
    assert result["connection"] == "connected"
    assert "updated_at" in result
