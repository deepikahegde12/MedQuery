// Interaction tests: loads the real index.html + app.js into jsdom with a stubbed
// fetch. Response bodies are neutral TEST-ONLY placeholders, not medical content.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { after, before, test } from "node:test";

import { JSDOM } from "jsdom";

const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
const dom = new JSDOM(html, { url: "http://localhost:8000/", pretendToBeVisual: true });
const { window } = dom;

const calls = [];
let queryHandler = () => notConfiguredResult("q");
const history = [];

function json(status, body) {
  return Promise.resolve({ ok: status < 400, status, json: async () => body });
}

function notConfiguredResult(question) {
  return {
    id: "a1", question, status: "KB_NOT_CONFIGURED", answer: null, evidence: [], sources: [],
    retrieval: { status: "NOT_CONFIGURED", adapter: "not_configured", passages_returned: 0 },
    analysis: { question_type: "general", keywords: ["placeholder"], bert: { available: false } },
    notices: [{ level: "warning", code: "kb_not_configured", message: "Evidence retrieval is unavailable." }],
    timings_ms: { total: 12 }, created_at: new Date().toISOString(), disclaimer: "Educational prototype.",
  };
}

function answeredResult(question) {
  return {
    ...notConfiguredResult(question), id: "b2", status: "ANSWERED",
    answer: { text: "ALPHA0", confidence: 0.82, method: "extractive", evidence_index: 0, verified: false },
    evidence: [{ text: "Placeholder <b>passage</b> with ALPHA0 inside.", source_id: "test-doc-0",
                 source_title: "Test Document", source_url: "javascript:alert(1)", relevance_score: 0.9,
                 used_for_answer: true, highlight_start: 32, highlight_end: 38, metadata: {} }],
    sources: [{ source_id: "test-doc-0", title: "Test Document", url: "javascript:alert(1)" }],
    retrieval: { status: "OK", adapter: "test", passages_returned: 1 },
    notices: [],
  };
}

window.fetch = (url, opts = {}) => {
  const path = new URL(url, "http://localhost:8000").pathname;
  calls.push({ path, method: opts.method ?? "GET", body: opts.body });
  if (path === "/api/health") {
    return json(200, { success: true, data: {
      components: {
        query_encoder: { model: "bert-base-uncased", state: "not_loaded" },
        knowledge_base: { configured: false, adapter: "not_configured" },
        answer_generator: { mode: "extractive", model: "qa-model", state: "not_loaded" },
      },
      limits: { min_question_chars: 3, max_question_chars: 500 },
    } });
  }
  if (path === "/api/history" && opts.method === "DELETE") { history.length = 0; return json(200, { success: true, data: { deleted: 1 } }); }
  if (path === "/api/history") return json(200, { success: true, data: history });
  if (path === "/api/query") {
    const q = JSON.parse(opts.body).question;
    const res = queryHandler(q);
    if (res.__error) return json(res.__error, { success: false, error: { code: "x", message: "boom" } });
    history.unshift({ id: res.id, question: q, status: res.status, created_at: res.created_at });
    return json(200, { success: true, data: res });
  }
  if (path.startsWith("/api/query/")) return json(200, { success: true, data: answeredResult("from history") });
  return json(404, { success: false, error: { code: "not_found", message: "nf" } });
};
window.matchMedia = () => ({ matches: false, addEventListener() {} });
window.HTMLElement.prototype.scrollIntoView = () => {};

const $ = (s) => window.document.querySelector(s);
const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));
async function until(fn, ms = 2000) {
  const end = Date.now() + ms;
  while (Date.now() < end) { if (fn()) return; await tick(); }
  throw new Error("condition not met");
}

before(async () => {
  for (const k of ["window", "document", "localStorage", "matchMedia", "navigator", "HTMLElement"]) {
    Object.defineProperty(globalThis, k, { value: window[k], configurable: true, writable: true });
  }
  globalThis.fetch = window.fetch;
  await import("../js/app.js");
  await until(() => $("#systemStatus").dataset.state);
});
after(() => window.close());

test("health check drives the status pill and system info", () => {
  assert.equal($("#systemStatus").dataset.state, "warn");
  assert.equal($("#systemStatus .status-text").textContent, "Knowledge base not connected");
  assert.match($("#sysInfo").textContent, /bert-base-uncased/);
});

test("templates are rendered and select their placeholder", () => {
  const chips = window.document.querySelectorAll(".chip");
  assert.ok(chips.length >= 5);
  chips[0].click();
  const t = $("#question");
  assert.equal(t.value.slice(t.selectionStart, t.selectionEnd), "[condition]");
});

test("client-side validation blocks a too-short question", async () => {
  const before = calls.length;
  $("#question").value = "a";
  $("#askForm").requestSubmit();
  await tick();
  assert.equal($("#questionError").hidden, false);
  assert.equal(calls.filter((c) => c.path === "/api/query").length, calls.slice(0, before).filter((c) => c.path === "/api/query").length);
});

test("not-configured result shows status and notice, no answer or evidence", async () => {
  queryHandler = notConfiguredResult;
  $("#question").value = "What is the placeholder?";
  $("#askForm").requestSubmit();
  await until(() => $(".result-card"));
  assert.equal($(".status-banner").textContent, "Knowledge base not connected");
  assert.match($(".notices").textContent, /Evidence retrieval is unavailable/);
  assert.equal($(".answer-block").hidden, true);
  assert.equal($(".evidence-block").hidden, true);
  assert.equal(JSON.parse(calls.findLast((c) => c.path === "/api/query").body).question, "What is the placeholder?");
  await until(() => $(".history-item"));
});

test("answered result renders answer, escaped highlighted evidence and safe sources", async () => {
  queryHandler = answeredResult;
  $("#question").value = "What is the answer token?";
  $("#askForm").requestSubmit();
  await until(() => $(".answer-block") && !$(".answer-block").hidden);
  assert.equal($(".answer-text").textContent, "ALPHA0");
  assert.match($(".confidence-label").textContent, /High \(82%\)/);
  assert.equal($(".evidence mark").textContent, "ALPHA0");
  assert.equal($(".evidence p b"), null, "evidence HTML must be escaped");
  assert.equal($(".sources-list a"), null, "javascript: URLs must not become links");
});

test("server error shows retry, and retry recovers", async () => {
  queryHandler = () => ({ __error: 500 });
  $("#question").value = "What is the retry case?";
  $("#askForm").requestSubmit();
  await until(() => $("#retryBtn"));
  assert.match($(".error-card").textContent, /server ran into a problem/);
  queryHandler = notConfiguredResult;
  $("#retryBtn").click();
  await until(() => $(".result-card"));
});

test("history items reload a stored result; clear empties the list", async () => {
  await until(() => $(".history-item"));
  $(".history-item").click();
  await until(() => $(".result-question")?.textContent === "from history");
  $("#clearHistory").click();
  await until(() => $(".history-empty"));
});
