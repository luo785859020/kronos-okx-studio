(() => {
    "use strict";
    const element = id => document.getElementById(id);
    const form = element("settings-form");
    const token = document.querySelector('meta[name="okx-control"]').content;
    const state = {busy: false, polling: false, generation: 0, selectedId: null, renderedKey: null, records: [], running: false, stopping: false, result: null, devices: null};
    const phaseLabels = {starting: "正在启动", loading: "加载模型", running: "等待新收盘", predicting: "正在预测", stopped: "已停止", stopping: "正在停止", error: "进程异常", other_running: "其他配置运行中"};
    const connectionLabels = {connecting: "连接中", connected: "已连接", reconnecting: "重连中", disconnected: "未连接"};
    const intervals = {"1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000, "30m": 1800000, "1H": 3600000, "2H": 7200000, "4H": 14400000};

    function config() {
        return {inst_id: element("inst-id").value.trim().toUpperCase(), bar: element("bar").value, region: element("region").value, lookback: Number(element("lookback").value), pred_len: Number(element("pred-len").value), device: element("device").value};
    }

    function setInstrument(value) {
        const normalized = value.trim().toUpperCase();
        const preset = element("inst-preset");
        const known = [...preset.options].some(option => option.value === normalized);
        element("inst-id").value = normalized;
        preset.value = known ? normalized : "custom";
        element("custom-inst-field").hidden = known;
    }

    function query(values) { return new URLSearchParams(values).toString(); }
    function utc(value) {
        if (!value) return null;
        const text = value.replace(" ", "T");
        const date = new Date(/[zZ]$|[+-]\d\d:\d\d$/.test(text) ? text : `${text}Z`);
        return Number.isNaN(date.getTime()) ? null : date;
    }
    function displayTime(value) { const date = utc(value); return date ? date.toISOString().replace("T", " ").slice(0, 19) : "—"; }
    function priceDecimals(value, minimum = 4) {
        const magnitude = Math.abs(value);
        return Number.isFinite(magnitude) && magnitude > 0 && magnitude < 1 ? Math.min(12, Math.max(minimum, 5 - Math.floor(Math.log10(magnitude)))) : minimum;
    }
    function number(value) { return Number.isFinite(value) ? value.toLocaleString("en-US", {maximumFractionDigits: priceDecimals(value)}) : "—"; }
    function percent(value) { return Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${(value * 100).toFixed(3)}%` : "—"; }
    function message(text, error = false) { element("message").textContent = text; element("message").className = error ? "error" : ""; element("message").hidden = !text; }

    async function api(path, options = {}) {
        const response = await fetch(path, {cache: "no-store", ...options, headers: {"Content-Type": "application/json", "X-OKX-Control": token, ...options.headers}, signal: AbortSignal.timeout(15000)});
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || `请求失败（${response.status}）`);
        return data;
    }

    function buttons() {
        const available = state.devices?.devices.some(device => device.id === element("device").value && device.available);
        element("start-btn").disabled = state.busy || state.running || state.stopping || !available;
        element("stop-btn").disabled = state.busy || !state.running || state.stopping;
        element("settings-fields").disabled = state.busy || state.running || state.stopping || !state.devices;
    }

    function deviceLabels() {
        const selected = state.devices?.devices.find(device => device.id === element("device").value);
        const gpu = state.devices?.devices.filter(device => device.id !== "cpu") || [];
        const note = selected?.reason || (selected?.id === "cpu" && !gpu.some(device => device.available) ? gpu[0]?.reason : "切换推理设备前请先停止当前预测；不会自动退回 CPU。");
        element("device-note").textContent = state.devices ? `PyTorch ${state.devices.torch_version} · ${note || "请选择可用设备。"}` : "正在检测当前 Python 环境的 GPU 支持…";
        element("model-device").textContent = `${element("device").value === "cpu" ? "CPU" : `GPU · ${element("device").value}`} · 固定版本 · 本地权重缓存`;
        buttons();
    }

    async function loadDevices() {
        const devices = await api("/api/okx/devices");
        const select = element("device");
        const selected = select.value;
        select.replaceChildren();
        for (const device of devices.devices) {
            const option = document.createElement("option");
            option.value = device.id; option.textContent = device.label; option.disabled = !device.available;
            select.append(option);
        }
        select.value = devices.devices.some(device => device.id === selected) ? selected : "cpu";
        state.devices = devices;
        deviceLabels();
    }

    function applyStatus(status) {
        state.running = status.running;
        state.stopping = status.phase === "stopping";
        element("process-state").textContent = phaseLabels[status.phase] || status.phase;
        const heartbeat = utc(status.runtime.heartbeat_at);
        const expired = status.connection === "connected" && (!heartbeat || Date.now() - heartbeat.getTime() > 90000);
        element("connection-state").textContent = expired ? "心跳过期" : connectionLabels[status.connection] || "未知";
        element("connection-state").className = status.connection === "connected" && !expired ? "positive" : "";
        element("session-count").textContent = `${status.runtime.forecasts_this_session || 0} 次`;
        element("runtime-note").textContent = status.runtime.data_error || (status.phase === "error" ? `进程退出（${status.exit_code}），请查看日志。` : "模型仅使用已收盘行情；网络异常时暂停并尝试恢复。");
        const log = element("runtime-log");
        const nextLog = status.logs.length ? status.logs.join("\n") : "等待启动预测进程…";
        if (log.textContent !== nextLog) { log.textContent = nextLog; log.scrollTop = log.scrollHeight; }
        buttons();
    }

    function renderRecords() {
        const body = element("record-body");
        body.replaceChildren();
        element("record-count").textContent = `${state.records.length} 条`;
        if (!state.records.length) {
            const row = body.insertRow(); const cell = row.insertCell(); cell.colSpan = 3; cell.className = "empty-row"; cell.textContent = "暂无记录";
            return;
        }
        for (const record of state.records) {
            const row = body.insertRow();
            if (record.id === state.selectedId) row.classList.add("selected");
            row.insertCell().textContent = displayTime(record.history_end).slice(5, 16);
            const change = row.insertCell(); change.textContent = percent(record.predicted_return); change.className = record.predicted_return >= 0 ? "positive" : "negative";
            const button = document.createElement("button"); button.type = "button"; button.textContent = record.id === state.selectedId ? "查看中" : "查看";
            button.setAttribute("aria-label", `查看 ${displayTime(record.history_end)} 的预测`);
            button.addEventListener("click", async () => {
                element("follow-latest").checked = false;
                state.generation += 1;
                try { await showResult(record.id, config(), state.generation); } catch (error) { message(error.message, true); }
            });
            row.insertCell().append(button);
        }
    }

    function updateBadge() {
        if (!state.result) return;
        const newest = state.records[0]?.id === state.selectedId;
        const closeTime = utc(state.result.report.candle_closed_at);
        const stale = closeTime && Date.now() - closeTime.getTime() >= intervals[config().bar];
        const badge = element("view-badge");
        badge.textContent = !element("follow-latest").checked ? "历史回看" : stale ? "历史快照 · 非实时" : newest && state.running ? "最新预测" : "已保存快照";
        badge.className = `small-tag ${stale ? "stale" : state.running && newest ? "live" : ""}`;
    }

    async function showResult(runId, values, generation) {
        const key = `${query(values)}:${runId}`;
        if (state.renderedKey === key) { updateBadge(); return; }
        const result = await api(`/api/okx/result/${encodeURIComponent(runId)}?${query(values)}`);
        if (generation !== state.generation) return;
        state.result = result;
        state.selectedId = result.id;
        const signal = result.report.signal;
        const priceFormat = `,.${Math.max(priceDecimals(signal.last_actual_close, 2), priceDecimals(signal.predicted_final_close, 2))}f`;
        element("last-close").textContent = number(signal.last_actual_close);
        element("predicted-close").textContent = number(signal.predicted_final_close);
        element("predicted-return").textContent = percent(signal.predicted_return);
        element("predicted-return").className = signal.predicted_return >= 0 ? "positive" : "negative";
        const completed = displayTime(result.report.inference_completed_at);
        element("completed-time").textContent = completed.slice(11);
        element("completed-date").textContent = completed.slice(0, 10);
        element("chart-subtitle").textContent = `历史截止 ${displayTime(result.report.history_end)} UTC · 预测 ${result.prediction.length} 根 · 价格单位 ${values.inst_id.split("-")[1]}`;
        const chart = element("okx-chart");
        element("chart-empty").hidden = true;
        chart.hidden = false;
        if (!window.Plotly) throw new Error("图表组件加载失败，请刷新页面。");
        const trace = (rows, name, forecast) => ({type: "candlestick", name, x: rows.map(row => row.timestamps), open: rows.map(row => row.open), high: rows.map(row => row.high), low: rows.map(row => row.low), close: rows.map(row => row.close), increasing: {line: {color: forecast ? "#8872c9" : "#008c79", width: 1}, fillcolor: forecast ? "#ddd4f6" : "#b0dfd1"}, decreasing: {line: {color: forecast ? "#ac80bd" : "#bd6675", width: 1}, fillcolor: forecast ? "#eedcf2" : "#ecc5cc"}});
        const boundary = result.prediction[0].timestamps;
        await Plotly.react(chart, [trace(result.history, "历史 K 线", false), trace(result.prediction, "预测 K 线", true)], {
            margin: {l: 16, r: 70, t: 18, b: 38}, paper_bgcolor: "#fff", plot_bgcolor: "#fff", showlegend: false,
            font: {family: '"Segoe UI", "Microsoft YaHei", sans-serif', size: 10, color: "#78909c"},
            xaxis: {type: "date", rangeslider: {visible: false}, gridcolor: "#f0f4f6", zeroline: false, tickformat: "%m-%d\n%H:%M", title: {text: "UTC", font: {size: 10}}},
            yaxis: {side: "right", gridcolor: "#edf2f5", zeroline: false, tickformat: priceFormat, hoverformat: priceFormat, automargin: true, fixedrange: false},
            dragmode: "pan", uirevision: query(values),
            shapes: [{type: "rect", xref: "x", yref: "paper", x0: boundary, x1: result.prediction.at(-1).timestamps, y0: 0, y1: 1, fillcolor: "#9276ca0c", line: {width: 0}, layer: "below"}, {type: "line", xref: "x", yref: "paper", x0: boundary, x1: boundary, y0: 0, y1: 1, line: {color: "#b8a6d3", dash: "dot", width: 1}}],
        }, {responsive: true, displaylogo: false, scrollZoom: true, modeBarButtonsToRemove: ["lasso2d", "select2d"]});
        state.renderedKey = key;
        renderRecords();
        updateBadge();
    }

    function resetResult() {
        state.selectedId = null; state.renderedKey = null; state.result = null; state.records = [];
        for (const id of ["last-close", "predicted-close", "predicted-return", "completed-time"]) element(id).textContent = "—";
        element("completed-date").textContent = "尚无预测记录";
        element("view-badge").textContent = "等待数据"; element("view-badge").className = "small-tag";
        element("chart-subtitle").textContent = "只使用已确认收盘行情，未来价格由模型生成。";
        element("chart-empty").hidden = false; element("okx-chart").hidden = true;
        element("follow-latest").checked = true;
        if (window.Plotly) Plotly.purge(element("okx-chart"));
        renderRecords();
    }

    async function poll() {
        if (state.polling || !form.checkValidity()) return;
        state.polling = true;
        const generation = state.generation;
        try {
            if (!state.devices) await loadDevices();
            const values = config();
            const [status, recordData] = await Promise.all([api(`/api/okx/status?${query(values)}`), api(`/api/okx/records?${query(values)}`)]);
            if (generation !== state.generation) return;
            if (status.running && status.active_config && Object.keys(values).some(name => status.active_config[name] !== values[name])) {
                if (![...element("device").options].some(option => option.value === status.active_config.device)) {
                    const option = document.createElement("option"); option.value = status.active_config.device; option.textContent = `GPU · ${status.active_config.device}（运行中）`; option.disabled = true; element("device").append(option);
                }
                for (const [name, value] of Object.entries(status.active_config)) form.elements.namedItem(name).value = value;
                setInstrument(status.active_config.inst_id);
                applyStatus(status);
                deviceLabels();
                state.generation += 1; resetResult();
                return;
            }
            applyStatus(status);
            element("market-label").textContent = `${values.inst_id} · ${values.bar}`;
            element("horizon-label").textContent = `未来 ${values.pred_len} 根 K 线`;
            state.records = recordData.records;
            renderRecords();
            if (state.records.length && (element("follow-latest").checked || !state.selectedId)) await showResult(state.records[0].id, values, generation);
            updateBadge();
            element("poll-dot").className = "dot online";
            element("refresh-label").textContent = "本地面板已连接 · 每 3 秒刷新";
        } catch (error) {
            if (generation === state.generation) {
                element("poll-dot").className = "dot error";
                element("refresh-label").textContent = "刷新失败，正在重试";
                message(error.message, true);
            }
        } finally { state.polling = false; }
    }

    form.addEventListener("submit", async event => {
        event.preventDefault(); if (!form.reportValidity() || element("start-btn").disabled) return;
        state.busy = true; buttons(); message("");
        try { applyStatus(await api("/api/okx/start", {method: "POST", body: JSON.stringify(config())})); message("预测进程已启动。只读取公开行情，不访问账户或下单。"); }
        catch (error) { message(error.message, true); }
        finally { state.busy = false; buttons(); await poll(); }
    });
    element("stop-btn").addEventListener("click", async () => {
        state.busy = true; buttons();
        try { await api("/api/okx/stop", {method: "POST", body: "{}"}); state.stopping = true; message("已请求停止，正在等待进程退出。历史预测记录会保留。"); }
        catch (error) { message(error.message, true); }
        finally { state.busy = false; buttons(); await poll(); }
    });
    element("refresh-btn").addEventListener("click", poll);
    element("follow-latest").addEventListener("change", () => { state.generation += 1; updateBadge(); poll(); });
    element("inst-preset").addEventListener("change", () => {
        if (element("inst-preset").value === "custom") {
            element("custom-inst-field").hidden = false;
            element("inst-id").value = "";
            element("inst-id").focus();
        } else setInstrument(element("inst-preset").value);
    });
    element("inst-id").addEventListener("change", () => { element("inst-id").value = element("inst-id").value.trim().toUpperCase(); });
    element("settings-fields").addEventListener("change", () => { state.generation += 1; resetResult(); deviceLabels(); message(""); poll(); });
    setInstrument(element("inst-id").value);
    buttons();
    poll();
    setInterval(() => { if (!document.hidden) poll(); }, 3000);
    document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
})();
