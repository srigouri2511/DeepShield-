const API = `${window.location.origin}/api`;

const state = {
  aiAvailable: false,
  model: "Not configured",
  historyCount: 0,
};

function escapeHTML(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function requestJSON(url, options = {}) {
  return fetch(url, options).then(async response => {
    const payload = await response.json().catch(() => ({ error: "Invalid server response" }));
    if (!response.ok) {
      throw new Error(payload.error || "Request failed");
    }
    return payload;
  });
}

function isAICopilotEnabled() {
  const toggle = document.getElementById("ai-toggle");
  return Boolean(state.aiAvailable && toggle && toggle.checked);
}

function fillField(id, value) {
  const field = document.getElementById(id);
  if (!field) return;
  field.value = value;
  field.focus();
}

function showTab(name, event) {
  document.querySelectorAll(".tab").forEach(tab => tab.classList.remove("active"));
  document.querySelectorAll(".tab-button").forEach(button => button.classList.remove("active"));
  const target = document.getElementById(`tab-${name}`);
  if (target) {
    target.classList.add("active");
  }
  const tabButton = document.querySelector(`.tab-button[data-tab="${name}"]`);
  if (tabButton) {
    tabButton.classList.add("active");
  } else if (event?.currentTarget) {
    event.currentTarget.classList.add("active");
  }
  if (name === "history") {
    loadHistory();
  }
}

function updateStatusView(status) {
  state.aiAvailable = Boolean(status.ai_enabled);
  state.model = status.model || "Not configured";
  state.historyCount = Number(status.history_count || 0);

  const toggle = document.getElementById("ai-toggle");
  const modeText = state.aiAvailable ? "Hybrid AI mode available" : "Heuristic-only mode";
  const statusText = state.aiAvailable
    ? `OpenAI model ready: ${state.model}`
    : "OpenAI key not detected. Local heuristics are still fully usable.";
  const hintText = state.aiAvailable
    ? "AI Copilot will enrich summaries and recommendations."
    : "Set OPENAI_API_KEY and restart the server to enable it.";

  if (toggle) {
    toggle.checked = state.aiAvailable;
    toggle.disabled = !state.aiAvailable;
  }

  document.getElementById("status-mode").textContent = modeText;
  document.getElementById("status-model").textContent = statusText;
  document.getElementById("status-banner").textContent = statusText;
  document.getElementById("hero-mode").textContent = state.aiAvailable ? "Hybrid" : "Heuristic";
  document.getElementById("hero-model").textContent = state.aiAvailable ? state.model : "Not configured";
  document.getElementById("history-count").textContent = String(state.historyCount);
  document.getElementById("ai-hint").textContent = hintText;
}

async function loadStatus() {
  try {
    const status = await requestJSON(`${API}/status`);
    updateStatusView(status);
  } catch (error) {
    updateStatusView({ ai_enabled: false, model: "Unavailable", history_count: 0 });
    document.getElementById("status-banner").textContent = error.message;
  }
}

function buildEvidence(data) {
  const evidence = data.evidence?.length ? data.evidence : data.flags || [];
  if (!evidence.length) {
    return "";
  }

  return `
    <div class="reasoning-block">
      <h4 class="block-title">Key evidence</h4>
      <div class="evidence-list">
        ${evidence.map(item => `<span class="evidence-item">${escapeHTML(item)}</span>`).join("")}
      </div>
    </div>
  `;
}

function buildActions(data) {
  const actions = data.recommended_actions || [];
  if (!actions.length) {
    return "";
  }

  return `
    <div class="action-block">
      <h4 class="block-title">Recommended actions</h4>
      <ul class="action-list">
        ${actions.map(item => `<li>${escapeHTML(item)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderResult(containerId, data) {
  const container = document.getElementById(containerId);
  const heuristic = data.heuristic || {
    label: data.label,
    confidence: data.confidence,
    risk_score: data.risk_score,
    flags: data.flags || [],
  };
  const ai = data.ai || { available: state.aiAvailable, used: false, model: state.model };
  const analysisMode = data.analysis_mode || "heuristic";
  const aiBadge = ai.used ? `<span class="badge ai">AI ${escapeHTML(ai.model || state.model)}</span>` : "";
  const evidenceMarkup = buildEvidence(data);
  const actionsMarkup = buildActions(data);

  container.classList.remove("hidden");
  container.innerHTML = `
    <div class="result-header">
      <div>
        <div class="result-chip-row">
          <span class="badge ${escapeHTML(data.label || "clean")}">${escapeHTML((data.label || "unknown").replace(/_/g, " "))}</span>
          <span class="badge ${escapeHTML(analysisMode)}">${escapeHTML(analysisMode)}</span>
          ${aiBadge}
        </div>
        <h4 class="result-title">${escapeHTML(data.summary || "Analysis complete")}</h4>
      </div>
      <div class="mode-chip mono">${escapeHTML(data.task || "analysis")}</div>
    </div>

    <p class="result-summary">${escapeHTML(data.reasoning || "No reasoning provided.")}</p>

    <div class="score-grid">
      <div class="score-card">
        <span>Final confidence</span>
        <strong>${escapeHTML(formatPercent(data.confidence))}</strong>
      </div>
      <div class="score-card">
        <span>Risk score</span>
        <strong>${escapeHTML(formatPercent(data.risk_score ?? data.similarity ?? 0))}</strong>
      </div>
      <div class="score-card">
        <span>Heuristic confidence</span>
        <strong>${escapeHTML(formatPercent(heuristic.confidence ?? data.confidence))}</strong>
      </div>
      <div class="score-card">
        <span>AI usage</span>
        <strong>${ai.used ? "Used" : ai.available ? "Available" : "Unavailable"}</strong>
      </div>
    </div>

    ${evidenceMarkup}
    ${actionsMarkup}

    <div class="raw-block">
      <h4 class="block-title">Raw response</h4>
      <pre>${escapeHTML(JSON.stringify(data, null, 2))}</pre>
    </div>
  `;
}

function renderError(containerId, message) {
  renderResult(containerId, {
    label: "error",
    task: "request",
    confidence: 0,
    risk_score: 0,
    summary: "The request could not be completed.",
    reasoning: message,
    recommended_actions: ["Review the input and try again."],
    analysis_mode: "heuristic",
    ai: { available: state.aiAvailable, used: false, model: state.model },
  });
}

async function detectURL() {
  const input = document.getElementById("url-input").value.trim();
  if (!input) return;

  try {
    const data = await requestJSON(`${API}/detect/url/reputation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: input, use_ai: isAICopilotEnabled() }),
    });
    renderResult("url-result", data);
    loadStatus();
  } catch (error) {
    renderError("url-result", error.message);
  }
}

async function detectEmail() {
  const input = document.getElementById("email-input").value.trim();
  if (!input) return;

  try {
    const data = await requestJSON(`${API}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: "phishing_triage", input, use_ai: isAICopilotEnabled() }),
    });
    renderResult("email-result", data);
    loadStatus();
  } catch (error) {
    renderError("email-result", error.message);
  }
}

async function detectImage() {
  const input = document.getElementById("image-path").value.trim();
  if (!input) return;

  try {
    const data = await requestJSON(`${API}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: "deepfake_detection", input, use_ai: isAICopilotEnabled() }),
    });
    renderResult("image-result", data);
    loadStatus();
  } catch (error) {
    renderError("image-result", error.message);
  }
}

async function analyzeHeaders() {
  const headers = document.getElementById("headers-input").value.trim();
  if (!headers) return;

  try {
    const data = await requestJSON(`${API}/detect/email/headers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ headers, use_ai: isAICopilotEnabled() }),
    });
    renderResult("headers-result", data);
    loadStatus();
  } catch (error) {
    renderError("headers-result", error.message);
  }
}

async function verifyIdentity() {
  const imageA = document.getElementById("img-a").files[0];
  const imageB = document.getElementById("img-b").files[0];
  const context = document.getElementById("verify-context").value.trim();

  if (!imageA || !imageB) {
    renderError("verify-result", "Select both images before running a comparison.");
    return;
  }

  const form = new FormData();
  form.append("image_a", imageA);
  form.append("image_b", imageB);
  form.append("context", context);

  try {
    const data = await requestJSON(`${API}/verify`, {
      method: "POST",
      body: form,
    });
    renderResult("verify-result", data);
    loadStatus();
  } catch (error) {
    renderError("verify-result", error.message);
  }
}

function renderHistory(items) {
  const el = document.getElementById("history-list");
  if (!items.length) {
    el.innerHTML = `<div class="empty-state">No scans yet. Run an analysis to build the session log.</div>`;
    return;
  }

  el.innerHTML = items.map(item => `
    <article class="history-item">
      <div>
        <h4>${escapeHTML(item.type || "analysis")}</h4>
        <div class="history-meta">${escapeHTML(item.summary || item.input || "")}</div>
        <div class="history-meta mono">mode: ${escapeHTML(item.analysis_mode || "heuristic")} | confidence: ${escapeHTML(formatPercent(item.confidence || 0))}</div>
      </div>
      <span class="badge ${escapeHTML(item.label || "clean")}">${escapeHTML((item.label || "unknown").replace(/_/g, " "))}</span>
    </article>
  `).join("");
}

async function loadHistory() {
  try {
    const items = await requestJSON(`${API}/history`);
    renderHistory(items);
    state.historyCount = items.length;
    document.getElementById("history-count").textContent = String(items.length);
  } catch (error) {
    document.getElementById("history-list").innerHTML = `<div class="empty-state">${escapeHTML(error.message)}</div>`;
  }
}

async function clearHistory() {
  try {
    await requestJSON(`${API}/history`, { method: "DELETE" });
    await loadHistory();
    await loadStatus();
  } catch (error) {
    document.getElementById("history-list").innerHTML = `<div class="empty-state">${escapeHTML(error.message)}</div>`;
  }
}

window.addEventListener("keydown", event => {
  if (event.key !== "Enter" || event.shiftKey) return;
  const active = document.querySelector(".tab.active")?.id;
  if (active === "tab-url") detectURL();
  else if (active === "tab-email") detectEmail();
  else if (active === "tab-image") detectImage();
});

window.addEventListener("load", async () => {
  await loadStatus();
  await loadHistory();
});
