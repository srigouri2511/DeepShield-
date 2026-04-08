const API = `${window.location.origin}/api`;

async function requestJSON(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({ error: "Invalid server response" }));
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

function showTab(name, event) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`tab-${name}`).classList.add("active");
  if (event?.currentTarget) {
    event.currentTarget.classList.add("active");
  }
  if (name === "history") loadHistory();
}

function confidenceColor(score) {
  if (score >= 0.7) return "#ff4d4f";
  if (score >= 0.4) return "#faad14";
  return "#52c41a";
}

function renderError(containerId, message) {
  const el = document.getElementById(containerId);
  el.classList.remove("hidden");
  el.innerHTML = `
    <div class="verdict">
      <span class="badge suspicious">error</span>
      <span style="color:var(--muted);font-size:0.9rem">${message}</span>
    </div>
  `;
}

function renderResult(containerId, data) {
  const el = document.getElementById(containerId);
  el.classList.remove("hidden");

  const label = data.label || data.decision || (data.match === true ? "same_person" : data.match === false ? "different_person" : "unknown");
  const confidence = data.confidence ?? 0;
  const gaugeValue = data.risk_score ?? data.similarity ?? confidence;
  const flags = data.flags || data.details?.flags || [];

  el.innerHTML = `
    <div class="verdict">
      <span class="badge ${label}">${label.replace("_", " ")}</span>
      <span style="color:var(--muted);font-size:0.9rem">${(confidence * 100).toFixed(1)}% confidence</span>
    </div>
    <div class="confidence-bar-wrap">
      <div class="confidence-label">Risk Score</div>
      <div class="confidence-bar">
        <div class="confidence-fill" style="width:${gaugeValue * 100}%;background:${confidenceColor(gaugeValue)}"></div>
      </div>
    </div>
    ${flags.length ? `<div class="flags">${flags.map(f => `<span class="flag-tag">⚠ ${f}</span>`).join("")}</div>` : ""}
    <details style="margin-top:1rem">
      <summary style="cursor:pointer;color:var(--muted);font-size:0.85rem">Raw JSON</summary>
      <pre style="margin-top:0.5rem;font-size:0.8rem;color:var(--muted);overflow:auto">${JSON.stringify(data, null, 2)}</pre>
    </details>
  `;
}

async function detectURL() {
  const input = document.getElementById("url-input").value.trim();
  if (!input) return;
  try {
    const data = await requestJSON(`${API}/detect/url/reputation`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: input })
    });
    renderResult("url-result", data);
  } catch (error) {
    renderError("url-result", error.message);
  }
}

async function detectEmail() {
  const input = document.getElementById("email-input").value.trim();
  if (!input) return;
  try {
    const data = await requestJSON(`${API}/detect`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: "phishing_triage", input })
    });
    renderResult("email-result", data);
  } catch (error) {
    renderError("email-result", error.message);
  }
}

async function detectImage() {
  const input = document.getElementById("image-path").value.trim();
  if (!input) return;
  try {
    const data = await requestJSON(`${API}/detect`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: "deepfake_detection", input })
    });
    renderResult("image-result", data);
  } catch (error) {
    renderError("image-result", error.message);
  }
}

async function analyzeHeaders() {
  const headers = document.getElementById("headers-input").value.trim();
  if (!headers) return;
  try {
    const data = await requestJSON(`${API}/detect/email/headers`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ headers })
    });
    renderResult("headers-result", data);
  } catch (error) {
    renderError("headers-result", error.message);
  }
}

async function verifyIdentity() {
  const a = document.getElementById("img-a").files[0];
  const b = document.getElementById("img-b").files[0];
  if (!a || !b) return alert("Please select both images.");
  const form = new FormData();
  form.append("image_a", a);
  form.append("image_b", b);
  try {
    const data = await requestJSON(`${API}/verify`, { method: "POST", body: form });
    renderResult("verify-result", data);
  } catch (error) {
    renderError("verify-result", error.message);
  }
}

async function loadHistory() {
  try {
    const items = await requestJSON(`${API}/history`);
  const el = document.getElementById("history-list");
  if (!items.length) { el.innerHTML = `<p style="color:var(--muted)">No history yet.</p>`; return; }
  el.innerHTML = items.map(item => `
    <div class="history-item">
      <div>
        <div class="history-input">${item.input || "—"}</div>
        <div class="history-meta">Type: ${item.type} &nbsp;|&nbsp; Label: ${item.label || item.label || "—"}</div>
      </div>
      <span class="badge ${item.label || 'clean'}">${item.label || "—"}</span>
    </div>
  `).join("");
  } catch (error) {
    const el = document.getElementById("history-list");
    el.innerHTML = `<p style="color:var(--muted)">${error.message}</p>`;
  }
}

async function clearHistory() {
  await fetch(`${API}/history`, { method: "DELETE" });
  loadHistory();
}

// Enter key support
document.addEventListener("keydown", e => {
  if (e.key !== "Enter") return;
  const active = document.querySelector(".tab.active")?.id;
  if (active === "tab-url") detectURL();
  else if (active === "tab-email") detectEmail();
  else if (active === "tab-image") detectImage();
});
