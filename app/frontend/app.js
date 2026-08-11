/* ============================================================
   MLProcessGuard — dashboard logic
   ============================================================ */
const API = ""; // same-origin; set to a full URL if frontend/backend are split

const state = { faultTypes: [], selectedFault: "none", charts: {} };

const el = (id) => document.getElementById(id);

function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "\u2014";
  return Number(n).toFixed(digits);
}

Chart.defaults.color = "#9aa8b8";
Chart.defaults.font.family = "'IBM Plex Mono', monospace";
Chart.defaults.font.size = 11;
Chart.defaults.borderColor = "#1a2430";

const GRID = { color: "#1a2430" };
const CHART_COLORS = {
  process: "#3fc1b5",
  caution: "#dba94c",
  alarm: "#dd6455",
  info: "#5b8fd6",
  muted: "#6b7787",
};

function baseLineOptions(yTitle) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { labels: { boxWidth: 10, boxHeight: 10, padding: 12 } },
      tooltip: { backgroundColor: "#0e131b", borderColor: "#223040", borderWidth: 1 },
    },
    scales: {
      x: { grid: GRID, ticks: { maxTicksLimit: 8 }, title: { display: true, text: "minutes", color: "#6b7787" } },
      y: { grid: GRID, title: { display: !!yTitle, text: yTitle || "", color: "#6b7787" } },
    },
  };
}

function makeOrUpdateChart(key, canvasId, config) {
  if (state.charts[key]) {
    state.charts[key].data = config.data;
    state.charts[key].options = config.options;
    state.charts[key].update();
    return state.charts[key];
  }
  const ctx = document.getElementById(canvasId).getContext("2d");
  state.charts[key] = new Chart(ctx, config);
  return state.charts[key];
}

// ---------------------------------------------------------------
async function loadFaultTypes() {
  const res = await fetch(`${API}/api/fault-types`);
  state.faultTypes = await res.json();
  const wrap = el("fault-buttons");
  wrap.innerHTML = "";
  state.faultTypes.forEach((f) => {
    const btn = document.createElement("button");
    btn.className = "fault-btn" + (f.name === state.selectedFault ? " selected" : "");
    btn.textContent = f.name === "none" ? "normal operation" : f.name.replace(/_/g, " ");
    btn.title = f.description;
    btn.addEventListener("click", () => {
      state.selectedFault = f.name;
      document.querySelectorAll(".fault-btn").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
    });
    wrap.appendChild(btn);
  });
}

async function loadReport() {
  const strip = el("metric-strip");
  try {
    const res = await fetch(`${API}/api/report`);
    if (!res.ok) throw new Error("no report");
    const m = await res.json();
    const cells = [];
    if (m.fault_classifier && m.fault_classifier.test_macro_f1 !== undefined) {
      cells.push(["Fault classifier", `${fmt(m.fault_classifier.test_macro_f1 * 100, 1)}%`, "test macro-F1"]);
    }
    if (m.anomaly_detector && m.anomaly_detector.test_auc !== undefined) {
      cells.push(["Anomaly detector", fmt(m.anomaly_detector.test_auc, 3), "test ROC-AUC"]);
    }
    if (m.soft_sensor && m.soft_sensor.test_rmse !== undefined) {
      cells.push(["Soft sensor", `${fmt(m.soft_sensor.test_rmse, 4)}`, "test RMSE, mol/L"]);
    }
    if (m.rul_predictor && m.rul_predictor.test_rmse) {
      cells.push(["RUL predictor", `${fmt(m.rul_predictor.test_rmse, 0)} min`, "test RMSE"]);
    }
    if (m.data_generation) {
      cells.push(["Training episodes", `${m.data_generation.n_episodes}`, "simulated, 48h each"]);
    }
    strip.innerHTML = cells
      .map(([label, value, sub]) => `
        <div class="metric-cell">
          <div class="metric-value">${value}</div>
          <div class="metric-label">${label} &middot; ${sub}</div>
        </div>`)
      .join("");
  } catch (e) {
    strip.innerHTML = `
      <div class="metric-cell" style="grid-column: 1 / -1;">
        <div class="metric-label">No training report found yet &mdash; run <code>python run.py --hours 5</code>
        locally, then redeploy with the produced <code>run_output/</code> included.</div>
      </div>`;
  }
}

function setStatus(text, mode) {
  el("sim-status-text").textContent = text;
  const dot = el("sim-status-dot");
  dot.classList.remove("live", "alarm");
  if (mode) dot.classList.add(mode);
}

function setAlarmChip(chipId, valueId, text, level) {
  const chip = el(chipId);
  chip.classList.remove("state-normal", "state-caution", "state-alarm");
  if (level) chip.classList.add(`state-${level}`);
  el(valueId).textContent = text;
}

function setPidTagState(tag, level) {
  const g = document.querySelector(`.pid-tag[data-tag="${tag}"]`);
  if (!g) return;
  g.classList.remove("tag-active", "tag-caution", "tag-alarm");
  if (level === "normal") g.classList.add("tag-active");
  if (level === "caution") g.classList.add("tag-caution");
  if (level === "alarm") g.classList.add("tag-alarm");
}

function renderTimeline(container, windowT, preds) {
  container.innerHTML = "";
  if (!preds || !preds.length) return;
  const colorFor = (label) => (label === "none" ? "#1c2530" : CHART_COLORS.alarm);
  preds.forEach((label) => {
    const seg = document.createElement("div");
    seg.className = "timeline-seg";
    seg.style.background = colorFor(label);
    seg.title = label;
    container.appendChild(seg);
  });
}

async function runScenario() {
  const btn = el("run-btn");
  btn.disabled = true;
  btn.textContent = "Running\u2026";
  setStatus("SIMULATION RUNNING", "live");

  const severityPct = Number(el("severity-slider").value);
  const spec = state.faultTypes.find((f) => f.name === state.selectedFault);
  let severity = null;
  if (spec && spec.name !== "none" && spec.severity_range) {
    const [lo, hi] = spec.severity_range;
    severity = lo + (severityPct / 100) * (hi - lo);
  }

  try {
    const res = await fetch(`${API}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fault_type: state.selectedFault, severity }),
    });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    renderResult(d);
    setStatus(d.fault_type === "none" ? "SYSTEM NORMAL" : "FAULT INJECTED", d.fault_type === "none" ? "live" : "alarm");
  } catch (e) {
    console.error(e);
    setStatus("SIMULATION ERROR", "alarm");
    alert("Simulation failed \u2014 the trained models may not be loaded on this server yet. "
      + "Run run.py to train them, place run_output/models/ next to config.yaml, and restart the API.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run scenario";
  }
}

function renderResult(d) {
  const n = d.t_min.length;
  el("val-T").textContent = fmt(d.T_true[n - 1], 1);
  el("val-Tc").textContent = fmt(d.Tc_cmd[n - 1], 1);
  el("val-Ca").textContent = fmt(d.Ca_true[n - 1], 4);
  el("val-Ca-soft").textContent = d.soft_sensor_pred ? fmt(d.soft_sensor_pred[n - 1], 4) : "\u2014";

  const isFault = d.fault_type !== "none";
  setPidTagState("T", isFault ? "caution" : "normal");
  setPidTagState("Tc", isFault ? "caution" : "normal");
  setPidTagState("Level", "normal");
  setPidTagState("Ca", isFault ? "caution" : "normal");

  // --- classifier / anomaly alarm chips ---
  if (d.classifier_pred && d.classifier_pred.length) {
    const last = d.classifier_pred[d.classifier_pred.length - 1];
    const conf = d.classifier_confidence[d.classifier_confidence.length - 1];
    const level = last === "none" ? "normal" : "alarm";
    setAlarmChip("alarm-classifier", "alarm-classifier-value",
      `${last.replace(/_/g, " ")} (${fmt(conf * 100, 0)}%)`, level);
    if (level === "alarm") setPidTagState("Tc", "alarm");
  } else {
    setAlarmChip("alarm-classifier", "alarm-classifier-value", "model not loaded", null);
  }

  if (d.anomaly_score && d.anomaly_score.length) {
    const last = d.anomaly_score[d.anomaly_score.length - 1];
    const level = last > 0.15 ? "alarm" : last > 0.05 ? "caution" : "normal";
    setAlarmChip("alarm-anomaly", "alarm-anomaly-value", fmt(last, 4), level);
  } else {
    setAlarmChip("alarm-anomaly", "alarm-anomaly-value", "model not loaded", null);
  }

  if (d.rul_pred_min && d.rul_pred_min.length && isFault) {
    const last = d.rul_pred_min[d.rul_pred_min.length - 1];
    const level = last < 120 ? "alarm" : last < 480 ? "caution" : "normal";
    setAlarmChip("alarm-rul", "alarm-rul-value", `${fmt(last, 0)} min`, level);
  } else {
    setAlarmChip("alarm-rul", "alarm-rul-value", isFault ? "n/a for this fault" : "no fault active", null);
  }

  el("alarm-latency").textContent = `${fmt(d.compute_ms, 0)} ms`;

  // --- temperature chart ---
  makeOrUpdateChart("temp", "chart-temp", {
    type: "line",
    data: {
      labels: d.t_min,
      datasets: [
        { label: "T (true)", data: d.T_true, borderColor: CHART_COLORS.process, pointRadius: 0, borderWidth: 1.6 },
        { label: "T setpoint", data: d.T_setpoint, borderColor: CHART_COLORS.muted, pointRadius: 0, borderWidth: 1, borderDash: [3, 3] },
      ],
    },
    options: baseLineOptions("K"),
  });

  // --- concentration chart ---
  const analyzerPoints = d.t_min.map((t, i) => (d.Ca_analyzer[i] === null ? null : { x: t, y: d.Ca_analyzer[i] }))
    .filter(Boolean);

  makeOrUpdateChart("ca", "chart-ca", {
    type: "line",
    data: {
      labels: d.t_min,
      datasets: [
        { label: "Ca true", data: d.Ca_true, borderColor: CHART_COLORS.muted, pointRadius: 0, borderWidth: 1 },
        { label: "Soft sensor est.", data: d.soft_sensor_pred || [], borderColor: CHART_COLORS.info, pointRadius: 0, borderWidth: 1.6 },
        { label: "Analyzer sample", data: analyzerPoints, borderColor: CHART_COLORS.caution,
          backgroundColor: CHART_COLORS.caution, showLine: false, pointRadius: 3, type: "scatter" },
      ],
    },
    options: baseLineOptions("mol/L"),
  });

  // --- anomaly score chart ---
  makeOrUpdateChart("anomaly", "chart-anomaly", {
    type: "line",
    data: {
      labels: d.window_t_min,
      datasets: [
        { label: "anomaly score", data: d.anomaly_score || [], borderColor: CHART_COLORS.alarm,
          pointRadius: 0, borderWidth: 1.6, fill: true, backgroundColor: "rgba(221,100,85,0.08)" },
      ],
    },
    options: { ...baseLineOptions("score"), plugins: { legend: { display: false } } },
  });

  // --- RUL chart ---
  makeOrUpdateChart("rul", "chart-rul", {
    type: "line",
    data: {
      labels: d.t_min,
      datasets: [
        { label: "est. RUL", data: d.rul_pred_min || [], borderColor: CHART_COLORS.caution,
          pointRadius: 0, borderWidth: 1.6 },
      ],
    },
    options: { ...baseLineOptions("minutes"), plugins: { legend: { display: false } } },
  });

  renderTimeline(el("chart-timeline"), d.window_t_min, d.classifier_pred);
}

// ---------------------------------------------------------------
async function init() {
  const repoLinks = document.querySelectorAll("#repo-link, #footer-repo-link");
  // Placeholder — update to your actual repo URL before deploying.
  repoLinks.forEach((a) => (a.href = "https://github.com/YOUR-USERNAME/mlprocessguard"));

  el("severity-slider").addEventListener("input", (e) => {
    el("severity-value").textContent = `${e.target.value}%`;
  });
  el("run-btn").addEventListener("click", runScenario);

  await Promise.all([loadFaultTypes(), loadReport()]);
  setStatus("SYSTEM IDLE", null);
}

init();
