// Pure helpers (no DOM access) so they can be unit-tested with `node --test`.

export const STATUS_META = {
  ANSWERED: { tone: "success", label: "Answer found in evidence", icon: "check" },
  LOW_CONFIDENCE: { tone: "warning", label: "Low-confidence answer", icon: "alert" },
  EVIDENCE_ONLY: { tone: "info", label: "Evidence retrieved", icon: "book" },
  NO_ANSWER_IN_EVIDENCE: { tone: "info", label: "No direct answer in evidence", icon: "book" },
  NO_EVIDENCE: { tone: "neutral", label: "No relevant evidence found", icon: "search" },
  KB_NOT_CONFIGURED: { tone: "neutral", label: "Knowledge base not connected", icon: "plug" },
  RETRIEVAL_FAILED: { tone: "danger", label: "Retrieval failed", icon: "alert" },
};

export function statusMeta(status) {
  return STATUS_META[status] ?? { tone: "neutral", label: String(status ?? "Unknown"), icon: "info" };
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/** Escape `text` and wrap the [start, end) range in <mark>. Invalid ranges are ignored. */
export function highlightRange(text, start, end) {
  const s = String(text ?? "");
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > s.length || start >= end) {
    return escapeHtml(s);
  }
  return escapeHtml(s.slice(0, start)) + "<mark>" + escapeHtml(s.slice(start, end)) + "</mark>" + escapeHtml(s.slice(end));
}

export function confidenceLabel(score) {
  if (typeof score !== "number" || Number.isNaN(score)) return { pct: 0, label: "Unknown" };
  const pct = Math.round(Math.min(Math.max(score, 0), 1) * 100);
  const label = score >= 0.6 ? "High" : score >= 0.3 ? "Moderate" : "Low";
  return { pct, label };
}

/** Accept only http(s) URLs for source links; anything else is rendered as plain text. */
export function safeUrl(url) {
  try {
    const u = new URL(String(url));
    return u.protocol === "http:" || u.protocol === "https:" ? u.href : null;
  } catch {
    return null;
  }
}

export function validateQuestion(text, { min = 3, max = 500 } = {}) {
  const clean = String(text ?? "").replace(/\s+/g, " ").trim();
  if (clean.length < min) return { ok: false, message: `Please enter at least ${min} characters.` };
  if (clean.length > max) return { ok: false, message: `Please keep your question under ${max} characters.` };
  if (!/\p{L}/u.test(clean)) return { ok: false, message: "Please enter a question in words." };
  return { ok: true, value: clean };
}

export function relativeTime(iso, now = Date.now()) {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const sec = Math.max(0, Math.round((now - t) / 1000));
  if (sec < 60) return "just now";
  if (sec < 3600) return `${Math.floor(sec / 60)} min ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)} h ago`;
  return new Date(t).toLocaleDateString();
}

/** Find the first [placeholder] in a template so the UI can pre-select it. */
export function placeholderRange(template) {
  const m = /\[[^\]]+\]/.exec(template);
  return m ? [m.index, m.index + m[0].length] : null;
}

/** Normalise an API error (network, HTTP or envelope) into a user-facing message. */
export function describeError(err) {
  if (!err) return "Unknown error.";
  if (err.name === "AbortError") return "The request took too long. Please try again.";
  if (err.status === 413) return "Your question is too long.";
  if (err.status === 422 && err.message) return err.message;
  if (err.status >= 500) return "The server ran into a problem. Please retry in a moment.";
  if (err.status === 0 || err instanceof TypeError) return "Cannot reach the MedQuery server. Is the backend running?";
  return err.message || "Request failed.";
}
