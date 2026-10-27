// Thin client for the MedQuery REST API. All responses use the envelope
// { success: true, data } or { success: false, error: { code, message } }.

const API_BASE = window.MEDQUERY_API_BASE ?? "/api";

export class ApiError extends Error {
  constructor(message, status = 0, code = "network_error") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function request(path, { method = "GET", body, timeoutMs = 120000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(API_BASE + path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError("Cannot reach the MedQuery server.", 0);
  } finally {
    clearTimeout(timer);
  }
  let payload = null;
  try {
    payload = await res.json();
  } catch {
    /* non-JSON response */
  }
  if (!res.ok || !payload?.success) {
    const e = payload?.error ?? {};
    throw new ApiError(e.message ?? `Request failed (${res.status}).`, res.status, e.code ?? "http_error");
  }
  return payload.data;
}

export const api = {
  health: () => request("/health", { timeoutMs: 10000 }),
  ask: (question) => request("/query", { method: "POST", body: { question } }),
  getQuery: (id) => request(`/query/${encodeURIComponent(id)}`),
  history: () => request("/history"),
  clearHistory: () => request("/history", { method: "DELETE" }),
};
