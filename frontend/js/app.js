import { api } from "./api.js";
import {
  confidenceLabel,
  describeError,
  escapeHtml,
  highlightRange,
  placeholderRange,
  relativeTime,
  safeUrl,
  statusMeta,
  validateQuestion,
} from "./format.js";

// Question templates only: the bracketed part is selected for the user to replace.
const TEMPLATES = [
  "What is [condition]?",
  "What are the common symptoms of [condition]?",
  "What causes [condition]?",
  "How is [condition] diagnosed?",
  "What are the treatment options for [condition]?",
  "How can [condition] be prevented?",
  "What is the difference between [term A] and [term B]?",
];

const ICONS = {
  check: '<path d="m5 12 5 5 9-10"/>',
  alert: '<path d="M12 3 2 20h20zM12 10v4M12 17h.01"/>',
  book: '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2zM4 5v16"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
  plug: '<path d="M9 2v6M15 2v6M6 8h12v4a6 6 0 0 1-12 0zM12 18v4"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
};
const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] ?? ICONS.info}</svg>`;

const $ = (sel) => document.querySelector(sel);
const els = {
  form: $("#askForm"),
  question: $("#question"),
  charCount: $("#charCount"),
  error: $("#questionError"),
  submit: $("#submitBtn"),
  clear: $("#clearBtn"),
  templates: $("#templates"),
  results: $("#results"),
  history: $("#historyList"),
  clearHistory: $("#clearHistory"),
  status: $("#systemStatus"),
  sysInfo: $("#sysInfo"),
  theme: $("#themeToggle"),
};

const limits = { min: 3, max: 500 };
let busy = false;

// ---------------------------------------------------------------- theme
function storageGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function storageSet(key, value) {
  try { localStorage.setItem(key, value); } catch { /* storage unavailable */ }
}
function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
}
applyTheme(storageGet("medquery-theme"));
els.theme.addEventListener("click", () => {
  const current = document.documentElement.dataset.theme
    ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  storageSet("medquery-theme", next);
});

// ---------------------------------------------------------------- input
function updateCount() {
  const n = els.question.value.length;
  els.charCount.textContent = `${n} / ${limits.max}`;
  els.charCount.classList.toggle("over", n > limits.max * 0.9);
}
function showFieldError(msg) {
  els.error.textContent = msg ?? "";
  els.error.hidden = !msg;
  els.question.setAttribute("aria-invalid", msg ? "true" : "false");
}

els.question.addEventListener("input", () => { updateCount(); showFieldError(null); });
els.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); els.form.requestSubmit(); }
});
els.clear.addEventListener("click", () => {
  els.question.value = ""; updateCount(); showFieldError(null); els.question.focus();
});

for (const t of TEMPLATES) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.textContent = t;
  b.addEventListener("click", () => {
    els.question.value = t;
    updateCount();
    els.question.focus();
    const r = placeholderRange(t);
    if (r) els.question.setSelectionRange(r[0], r[1]);
  });
  els.templates.append(b);
}

// ---------------------------------------------------------------- submit
function setBusy(state) {
  busy = state;
  els.submit.disabled = state;
  els.submit.classList.toggle("loading", state);
  els.submit.querySelector(".btn-label").textContent = state ? "Analysing…" : "Ask MedQuery";
  els.results.setAttribute("aria-busy", String(state));
}

function renderLoading(question) {
  els.results.innerHTML = `
    <div class="card loading-card">
      <p class="result-question">${escapeHtml(question)}</p>
      <ol class="progress-steps">
        <li class="active">Preprocessing question</li>
        <li>Encoding with BERT</li>
        <li>Retrieving evidence</li>
        <li>Extracting answer</li>
      </ol>
      <p class="hint">The first query can take longer while models load.</p>
    </div>`;
  const items = [...els.results.querySelectorAll(".progress-steps li")];
  let i = 0;
  const timer = setInterval(() => {
    if (!els.results.contains(items[0])) return clearInterval(timer);
    items[i]?.classList.replace("active", "done");
    i = Math.min(i + 1, items.length - 1);
    items[i]?.classList.add("active");
  }, 900);
}

function renderError(message, question) {
  els.results.innerHTML = `
    <div class="card error-card" role="alert">
      ${icon("alert")}
      <div>
        <h2>Something went wrong</h2>
        <p>${escapeHtml(message)}</p>
        <button type="button" class="btn btn-secondary" id="retryBtn">Retry</button>
      </div>
    </div>`;
  $("#retryBtn").addEventListener("click", () => ask(question));
}

async function ask(question) {
  if (busy) return;
  setBusy(true);
  renderLoading(question);
  try {
    const result = await api.ask(question);
    renderResult(result);
    refreshHistory();
  } catch (err) {
    renderError(describeError(err), question);
  } finally {
    setBusy(false);
  }
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const v = validateQuestion(els.question.value, limits);
  if (!v.ok) { showFieldError(v.message); els.question.focus(); return; }
  ask(v.value);
});

// ---------------------------------------------------------------- results
function renderResult(r) {
  const node = $("#resultTemplate").content.firstElementChild.cloneNode(true);
  const meta = statusMeta(r.status);

  const banner = node.querySelector(".status-banner");
  banner.className = `status-banner tone-${meta.tone}`;
  banner.innerHTML = `${icon(meta.icon)}<span>${escapeHtml(meta.label)}</span>`;
  node.querySelector(".result-question").textContent = r.question;

  node.querySelector(".notices").innerHTML = (r.notices ?? [])
    .map((n) => `<div class="notice notice-${escapeHtml(n.level)}">${icon(n.level === "info" ? "info" : "alert")}<p>${escapeHtml(n.message)}</p></div>`)
    .join("");

  if (r.answer) {
    const block = node.querySelector(".answer-block");
    block.hidden = false;
    block.querySelector(".answer-text").textContent = r.answer.text;
    const c = confidenceLabel(r.answer.confidence);
    block.querySelector(".confidence-label").textContent = `Model confidence: ${c.label} (${c.pct}%)`;
    const meter = block.querySelector(".meter");
    meter.setAttribute("aria-valuenow", String(c.pct));
    meter.setAttribute("aria-label", "Answer confidence");
    meter.dataset.level = c.label.toLowerCase();
    meter.querySelector("span").style.width = `${c.pct}%`;
  }

  if (r.evidence?.length) {
    const block = node.querySelector(".evidence-block");
    block.hidden = false;
    block.querySelector(".evidence-list").innerHTML = r.evidence.map((ev, i) => {
      const score = typeof ev.relevance_score === "number" ? `<span class="score">relevance ${ev.relevance_score.toFixed(2)}</span>` : "";
      const used = ev.used_for_answer ? '<span class="tag">answer source</span>' : "";
      return `<li class="evidence${ev.used_for_answer ? " used" : ""}">
        <div class="evidence-head"><span class="ev-num">[${i + 1}]</span><span class="ev-title">${escapeHtml(ev.source_title || ev.source_id)}</span>${used}${score}</div>
        <p>${highlightRange(ev.text, ev.highlight_start, ev.highlight_end)}</p>
      </li>`;
    }).join("");
  }

  if (r.sources?.length) {
    const block = node.querySelector(".sources-block");
    block.hidden = false;
    block.querySelector(".sources-list").innerHTML = r.sources.map((s) => {
      const url = safeUrl(s.url);
      const title = escapeHtml(s.title || s.source_id);
      return `<li>${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${title}</a>` : title}<code>${escapeHtml(s.source_id)}</code></li>`;
    }).join("");
  }

  const a = r.analysis ?? {};
  const bert = a.bert ?? {};
  const pieces = (bert.wordpieces ?? []).map((w) => `<span class="wp">${escapeHtml(w)}</span>`).join("");
  node.querySelector(".analysis-grid").innerHTML = `
    <div><h4>Question type</h4><p>${escapeHtml(a.question_type ?? "—")}</p></div>
    <div><h4>Keywords</h4><p>${(a.keywords ?? []).map((k) => `<span class="kw">${escapeHtml(k)}</span>`).join("") || "—"}</p></div>
    <div><h4>Retrieval</h4><p>${escapeHtml(r.retrieval?.status ?? "—")} · adapter <code>${escapeHtml(r.retrieval?.adapter ?? "—")}</code></p></div>
    <div><h4>BERT encoder</h4><p>${bert.available ? `${bert.token_count} tokens → ${bert.embedding_dim}-d embedding${bert.truncated ? " (truncated)" : ""}` : "unavailable"}</p></div>
    ${pieces ? `<div class="wide"><h4>WordPiece tokens</h4><p class="wps">${pieces}</p></div>` : ""}`;

  const t = r.timings_ms ?? {};
  node.querySelector(".timing").textContent = t.total != null ? `Processed in ${Math.round(t.total)} ms` : "";
  node.querySelector(".disclaimer-inline").textContent = r.disclaimer ?? "";

  els.results.replaceChildren(node);
}

// ---------------------------------------------------------------- history
async function refreshHistory() {
  let items;
  try { items = await api.history(); } catch { return; }
  if (!items.length) {
    els.history.innerHTML = '<li class="history-empty">No questions yet.</li>';
    return;
  }
  els.history.innerHTML = items.map((h) => `
    <li><button type="button" class="history-item" data-id="${escapeHtml(h.id)}">
      <span class="h-q">${escapeHtml(h.question)}</span>
      <span class="h-meta"><span class="h-dot tone-${statusMeta(h.status).tone}"></span>${escapeHtml(statusMeta(h.status).label)} · ${escapeHtml(relativeTime(h.created_at))}</span>
    </button></li>`).join("");
}

els.history.addEventListener("click", async (e) => {
  const btn = e.target.closest(".history-item");
  if (!btn || busy) return;
  try {
    const r = await api.getQuery(btn.dataset.id);
    els.question.value = r.question;
    updateCount();
    renderResult(r);
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    renderError(describeError(err), btn.querySelector(".h-q").textContent);
  }
});

els.clearHistory.addEventListener("click", async () => {
  try { await api.clearHistory(); } catch { /* ignore */ }
  refreshHistory();
});

// ---------------------------------------------------------------- health
async function refreshHealth() {
  const text = els.status.querySelector(".status-text");
  try {
    const h = await api.health();
    limits.min = h.limits?.min_question_chars ?? limits.min;
    limits.max = h.limits?.max_question_chars ?? limits.max;
    els.question.maxLength = limits.max;
    updateCount();
    const kb = h.components.knowledge_base;
    els.status.dataset.state = kb.configured ? "ok" : "warn";
    text.textContent = kb.configured ? "Knowledge base connected" : "Knowledge base not connected";
    const enc = h.components.query_encoder;
    const gen = h.components.answer_generator;
    const dd = els.sysInfo.querySelectorAll("dd");
    dd[0].textContent = `${enc.model} (${enc.state.replace("_", " ")})`;
    dd[1].textContent = kb.configured ? `Adapter: ${kb.adapter}` : "Not connected";
    dd[2].textContent = gen.model ? `${gen.model} (${gen.state.replace("_", " ")})` : gen.mode.replace("_", " ");
  } catch {
    els.status.dataset.state = "down";
    text.textContent = "Server offline";
  }
}

updateCount();
refreshHealth();
refreshHistory();
