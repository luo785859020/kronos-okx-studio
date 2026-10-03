import atexit
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import torch
from filelock import FileLock, Timeout
from flask import Blueprint, Response, abort, jsonify, render_template, request
from plotly.offline import get_plotlyjs

from integrations.okx_market import BAR_MILLISECONDS, ENDPOINTS, FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = {"inst_id": "BTC-USDT", "bar": "5m", "region": "global", "lookback": 256, "pred_len": 12, "device": "cpu"}
RUN_ID = re.compile(r"\d{8}T\d{6}Z")


def parse_config(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULT_CONFIG):
        raise ValueError("只允许设置币对、周期、区域、历史窗口、预测长度和推理设备。")
    config = {**DEFAULT_CONFIG, **values}
    if not isinstance(config["inst_id"], str) or not re.fullmatch(r"[A-Z0-9]{1,20}-[A-Z0-9]{1,20}", config["inst_id"]):
        raise ValueError("请输入现货币对，例如 BTC-USDT；不支持合约。")
    if not isinstance(config["bar"], str) or not isinstance(config["region"], str) or config["bar"] not in BAR_MILLISECONDS or config["region"] not in ENDPOINTS:
        raise ValueError("不支持的周期或区域。")
    if not isinstance(config["device"], str) or not re.fullmatch(r"cpu|cuda:(?:0|[1-9]\d?)", config["device"]):
        raise ValueError("推理设备必须是 cpu 或 cuda:0 等 CUDA 设备编号。")
    for field, minimum in (("lookback", 2), ("pred_len", 1)):
        value = config[field]
        if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
            raise ValueError("窗口长度必须为整数。")
        config[field] = int(value)
        if not minimum <= config[field] <= 512:
            raise ValueError(f"{field} 必须在 {minimum} 到 512 之间。")
    return config


@lru_cache(maxsize=1)
def detect_devices():
    devices = [{"id": "cpu", "label": "CPU · 处理器", "available": True, "reason": ""}]
    try:
        if torch.version.cuda is None:
            reason = "当前环境是 CPU 版 PyTorch；使用 GPU 需安装支持显卡的 CUDA 版 PyTorch，并重启 Web UI。"
        elif not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            reason = "当前 PyTorch 无法使用 CUDA；请检查 NVIDIA 显卡、驱动及 CUDA 版 PyTorch，之后重启 Web UI。"
        else:
            reason = ""
            for index in range(torch.cuda.device_count()):
                device_id = f"cuda:{index}"
                device = {"id": device_id, "label": f"GPU · {device_id}", "available": False, "reason": ""}
                try:
                    properties = torch.cuda.get_device_properties(index)
                    device["label"] = f"GPU {index} · {properties.name} · {properties.total_memory / 1024 ** 3:.1f} GB"
                    with torch.inference_mode():
                        probe = torch.ones((2, 2), device=device_id)
                        (probe @ probe).sum().item()
                        del probe
                    device["available"] = True
                except Exception as error:
                    device["reason"] = f"CUDA 计算检查失败：{error}；请检查显卡架构支持、驱动和可用显存。"
                devices.append(device)
        if reason:
            devices.append({"id": "cuda:0", "label": "GPU · NVIDIA CUDA（不可用）", "available": False, "reason": reason})
    except Exception as error:
        devices.append({"id": "cuda:0", "label": "GPU · NVIDIA CUDA（检测失败）", "available": False, "reason": f"CUDA 检测失败：{error}；CPU 仍可使用。"})
    return {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda, "devices": devices}


def read_json(path):
    if path.stat().st_size > 131_072:
        raise ValueError("记录文件过大。")
    return json.loads(path.read_text(encoding="utf-8"))


class OKXController:
    def __init__(self, output_root=None, process_factory=None):
        self.output_root = Path(output_root or PROJECT_ROOT / "outputs/okx-ui").resolve()
        self.process_factory = process_factory or subprocess.Popen
        self.process = None
        self.config = None
        self.started_at = None
        self.stopping = False
        self.stop_requested = False
        self.lock = threading.RLock()
        self.log_handle = None
        self.csrf_token = secrets.token_urlsafe(32)

    def output_base(self, config):
        base = self.output_root / f"l{config['lookback']}-p{config['pred_len']}"
        return base if config["device"] == "cpu" else base / config["device"].replace(":", "-")

    def directory(self, config):
        directory = (self.output_base(config) / config["region"] / config["inst_id"] / config["bar"]).resolve()
        if not directory.is_relative_to(self.output_root):
            raise ValueError("输出路径超出工作目录。")
        return directory

    def _close_log(self):
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def start(self, values):
        config = parse_config(values)
        with self.lock:
            if self.stopping or self.process is not None and self.process.poll() is None:
                raise RuntimeError("已有预测进程运行或正在停止，请先等待结束。")
            if config["device"] != "cpu":
                device = next((item for item in detect_devices()["devices"] if item["id"] == config["device"]), None)
                if device is None or not device["available"]:
                    reason = device["reason"] if device else "未检测到此 GPU；请刷新页面并选择可用设备。"
                    raise ValueError(reason)
            self._close_log()
            directory = self.directory(config)
            directory.mkdir(parents=True, exist_ok=True)
            try:
                with FileLock(directory / "runner.lock", timeout=0):
                    pass
            except Timeout as error:
                raise RuntimeError("此输出目录正被其他预测进程使用。") from error
            stop_file = directory / "stop.request"
            stop_file.unlink(missing_ok=True)
            status_file = directory / "status.json"
            status_file.unlink(missing_ok=True)
            command = [
                sys.executable, "-u", str(PROJECT_ROOT / "examples/okx_realtime_predict.py"),
                "--inst-id", config["inst_id"], "--bar", config["bar"], "--region", config["region"],
                "--lookback", str(config["lookback"]), "--pred-len", str(config["pred_len"]),
                "--device", config["device"], "--threads", "4", "--local-files-only",
                "--output-dir", str(self.output_base(config)), "--stop-file", str(stop_file),
            ]
            environment = {key: value for key, value in os.environ.items() if key not in {"OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE"}}
            environment["PYTHONIOENCODING"] = "utf-8"
            self.log_handle = (directory / "process.log").open("wb")
            try:
                self.process = self.process_factory(
                    command, cwd=str(PROJECT_ROOT), env=environment, stdin=subprocess.DEVNULL,
                    stdout=self.log_handle, stderr=subprocess.STDOUT, shell=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except Exception:
                self._close_log()
                raise
            self.config = config
            self.started_at = pd.Timestamp.now(tz="UTC").isoformat()
            self.stop_requested = False
            return self.status(config)

    def _finish_stop(self, process):
        try:
            try:
                process.wait(timeout=18)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW, timeout=10, check=False,
                    )
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            with self.lock:
                self.stopping = False
                self._close_log()

    def stop(self, wait=False):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._close_log()
                return
            if self.stopping:
                return
            self.stopping = True
            self.stop_requested = True
            (self.directory(self.config) / "stop.request").write_text("stop", encoding="utf-8")
            process = self.process
        if wait:
            self._finish_stop(process)
        else:
            threading.Thread(target=self._finish_stop, args=(process,), daemon=True).start()

    def logs(self, config):
        path = self.directory(config) / "process.log"
        if not path.exists():
            return []
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 12_000))
            return stream.read().decode("utf-8", errors="replace").splitlines()[-35:]

    def status(self, config):
        with self.lock:
            exit_code = self.process.poll() if self.process is not None else None
            running = self.process is not None and exit_code is None
            selected_active = self.config == config
            runtime = {}
            try:
                runtime = read_json(self.directory(config) / "status.json")
            except (OSError, ValueError):
                pass
            if not running:
                phase = "error" if selected_active and exit_code not in (None, 0) and not self.stop_requested else "stopped"
                self._close_log()
            elif self.stopping:
                phase = "stopping"
            elif not selected_active:
                phase = "other_running"
            else:
                phase = runtime.get("phase", "starting")
            return {
                "running": running, "phase": phase, "active_config": self.config,
                "connection": runtime.get("connection", "disconnected") if running and selected_active else "disconnected",
                "runtime": runtime, "started_at": self.started_at, "exit_code": exit_code,
                "pid": self.process.pid if running else None, "logs": self.logs(config),
                "server_time": pd.Timestamp.now(tz="UTC").isoformat(), "read_only": True,
            }

    def records(self, config):
        directory = self.directory(config)
        items = []
        for path in sorted(directory.glob("*/run.json"), reverse=True):
            if not RUN_ID.fullmatch(path.parent.name) or not path.resolve().is_relative_to(directory):
                continue
            try:
                report = read_json(path)
                items.append({
                    "id": path.parent.name, "history_end": report["history_end"],
                    "completed_at": report["inference_completed_at"],
                    "predicted_return": report["signal"]["predicted_return"],
                })
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if len(items) == 100:
                break
        return items

    def result(self, config, run_id):
        if run_id == "latest":
            records = self.records(config)
            if not records:
                raise FileNotFoundError("尚无预测记录，请先启动预测。")
            run_id = records[0]["id"]
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("无效的预测记录 ID。")
        base = self.directory(config)
        directory = (base / run_id).resolve()
        if not directory.is_relative_to(base):
            raise ValueError("记录路径超出工作目录。")
        report_path = (directory / "run.json").resolve()
        if not report_path.is_relative_to(directory):
            raise ValueError("记录路径超出工作目录。")
        report = read_json(report_path)
        result = {"id": run_id, "report": report}
        for name in ("history", "prediction"):
            path = (directory / f"{name}.csv").resolve()
            if not path.is_relative_to(directory) or path.stat().st_size > 1_000_000:
                raise ValueError("无效的行情文件。")
            frame = pd.read_csv(path, parse_dates=["timestamps"])
            if len(frame) > 512 or not np.isfinite(frame[FEATURE_COLUMNS].to_numpy()).all():
                raise ValueError("行情文件包含无效数值或超出长度限制。")
            frame["timestamps"] = frame["timestamps"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            result[name] = frame[["timestamps"] + FEATURE_COLUMNS].to_dict(orient="records")
        return result


def create_blueprint(controller):
    blueprint = Blueprint("okx_dashboard", __name__)

    @blueprint.before_request
    def local_only():
        if request.remote_addr not in ("127.0.0.1", "::1") or urlsplit(request.host_url).hostname not in ("localhost", "127.0.0.1", "::1"):
            abort(403)
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
                abort(403)
            if not secrets.compare_digest(request.headers.get("X-OKX-Control", ""), controller.csrf_token):
                abort(403)

    @blueprint.after_request
    def disable_control_cache(response):
        if request.path != "/okx/plotly.js":
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @blueprint.errorhandler(ValueError)
    def invalid_request(error):
        return jsonify(error=str(error)), 400

    @blueprint.errorhandler(FileNotFoundError)
    def missing_result(error):
        return jsonify(error="预测记录不存在，或尚未生成。"), 404

    @blueprint.route("/okx")
    def dashboard():
        return render_template("okx.html", control_token=controller.csrf_token)

    @blueprint.route("/okx/plotly.js")
    def plotly_asset():
        response = Response(get_plotlyjs(), mimetype="application/javascript")
        response.headers["Cache-Control"] = "private, max-age=86400"
        return response

    @blueprint.route("/api/okx/status")
    def status():
        config = parse_config(request.args.to_dict())
        return jsonify(controller.status(config))

    @blueprint.route("/api/okx/devices")
    def devices():
        return jsonify(detect_devices())

    @blueprint.route("/api/okx/start", methods=["POST"])
    def start():
        try:
            return jsonify(controller.start(request.get_json(silent=True)))
        except RuntimeError as error:
            return jsonify(error=str(error)), 409

    @blueprint.route("/api/okx/stop", methods=["POST"])
    def stop():
        controller.stop()
        return jsonify(success=True)

    @blueprint.route("/api/okx/records")
    def records():
        config = parse_config(request.args.to_dict())
        return jsonify(records=controller.records(config))

    @blueprint.route("/api/okx/result/<run_id>")
    def result(run_id):
        config = parse_config(request.args.to_dict())
        return jsonify(controller.result(config, run_id))

    return blueprint


def register_okx(app):
    controller = OKXController()
    app.extensions["okx_controller"] = controller
    app.register_blueprint(create_blueprint(controller))
    atexit.register(controller.stop, wait=True)
