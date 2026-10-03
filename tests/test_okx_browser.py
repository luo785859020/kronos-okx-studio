import os
from pathlib import Path
import threading
from unittest.mock import Mock

import pytest

playwright_api = pytest.importorskip("playwright.sync_api", reason="Install Playwright and a browser to run UI regression tests.")
pytest.importorskip("flask")
pytest.importorskip("plotly")
pytest.importorskip("websocket")

from flask import Flask
from werkzeug.serving import make_server

from webui.okx_dashboard import OKXController, create_blueprint


@pytest.fixture
def dashboard_page(tmp_path, monkeypatch):
    def process_factory(*args, **kwargs):
        process = Mock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = lambda timeout: setattr(process.poll, "return_value", 0)
        return process

    factory = Mock(side_effect=process_factory)
    controller = OKXController(tmp_path / "records", process_factory=factory)
    monkeypatch.setattr("webui.okx_dashboard.detect_devices", lambda: {"torch_version": "test", "devices": [{"id": "cpu", "label": "CPU", "available": True, "reason": ""}]})
    root = Path(__file__).resolve().parents[1] / "webui"
    app = Flask(__name__, template_folder=str(root / "templates"), static_folder=str(root / "static"), static_url_path="/static")
    app.register_blueprint(create_blueprint(controller))
    server = make_server("127.0.0.1", 0, app, threaded=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(channel="msedge" if os.name == "nt" else None, headless=True)
            except playwright_api.Error as error:
                pytest.skip(f"Browser unavailable: {error}")
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/okx")
                page.wait_for_function("!document.getElementById('start-btn').disabled")
                yield page, controller
                assert not errors, errors
            finally:
                browser.close()
    finally:
        controller.stop(wait=True)
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_all_presets_remain_selectable_from_btc_and_doge_starts(dashboard_page):
    page, controller = dashboard_page
    assert page.locator("#inst-preset").evaluate("select => select.tagName") == "SELECT"
    assert page.locator("#inst-preset").input_value() == "BTC-USDT"
    expected = ["BTC-USDT", "DOGE-USDT", "ETH-USDT", "SOL-USDT", "custom"]
    assert page.locator("#inst-preset option").evaluate_all("options => options.map(option => option.value)") == expected
    assert "狗狗币" in page.locator('#inst-preset option[value="DOGE-USDT"]').inner_text()
    for inst_id in expected[:-1]:
        page.select_option("#inst-preset", inst_id)
        assert page.locator("#inst-id").input_value() == inst_id
        assert page.locator("#custom-inst-field").is_hidden()
        assert page.locator("#settings-form").evaluate("form => form.checkValidity()")
        assert page.locator("#inst-preset option").count() == 5
    page.select_option("#inst-preset", "DOGE-USDT")
    page.locator("#start-btn").click()
    page.wait_for_function("document.getElementById('settings-fields').disabled && !document.getElementById('stop-btn').disabled")
    assert controller.config["inst_id"] == "DOGE-USDT"
    command = controller.process_factory.call_args.args[0]
    assert command[command.index("--inst-id") + 1] == "DOGE-USDT"


def test_custom_input_validation_and_return_to_preset(dashboard_page):
    page, controller = dashboard_page
    page.select_option("#inst-preset", "custom")
    assert page.locator("#custom-inst-field").is_visible()
    assert not page.locator("#settings-form").evaluate("form => form.checkValidity()")
    page.locator("#start-btn").click()
    controller.process_factory.assert_not_called()
    page.locator("#inst-id").fill("BTC-USDT-SWAP")
    assert not page.locator("#settings-form").evaluate("form => form.checkValidity()")
    page.select_option("#inst-preset", "DOGE-USDT")
    assert page.locator("#custom-inst-field").is_hidden()
    assert page.locator("#settings-form").evaluate("form => form.checkValidity()")
    page.select_option("#inst-preset", "custom")
    page.locator("#inst-id").fill("xrp-usdt")
    page.locator("#inst-id").press("Tab")
    assert page.locator("#inst-id").input_value() == "XRP-USDT"
    page.locator("#start-btn").click()
    page.wait_for_function("document.getElementById('settings-fields').disabled && !document.getElementById('stop-btn').disabled")
    assert controller.config["inst_id"] == "XRP-USDT"


@pytest.mark.parametrize("inst_id, preset", [("DOGE-USDT", "DOGE-USDT"), ("XRP-USDT", "custom")])
def test_running_instrument_restores_correct_selector_on_reload(dashboard_page, inst_id, preset):
    page, controller = dashboard_page
    controller.start({"inst_id": inst_id})
    page.reload()
    page.wait_for_function("document.getElementById('settings-fields').disabled && !document.getElementById('stop-btn').disabled")
    assert page.locator("#inst-preset").input_value() == preset
    assert page.locator("#inst-id").input_value() == inst_id
    assert page.locator("#custom-inst-field").is_visible() == (preset == "custom")
    page.locator("#stop-btn").click()
    page.wait_for_function("!document.getElementById('settings-fields').disabled")
    page.select_option("#inst-preset", "DOGE-USDT")
    assert page.locator("#inst-id").input_value() == "DOGE-USDT"
