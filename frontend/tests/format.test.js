import assert from "node:assert/strict";
import { test } from "node:test";

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
} from "../js/format.js";

test("escapeHtml neutralises markup", () => {
  assert.equal(escapeHtml(`<img src=x onerror="a('b')">&`), "&lt;img src=x onerror=&quot;a(&#39;b&#39;)&quot;&gt;&amp;");
  assert.equal(escapeHtml(null), "");
});

test("highlightRange marks a valid span and escapes everything", () => {
  assert.equal(highlightRange("a <b> c", 2, 5), "a <mark>&lt;b&gt;</mark> c");
});

test("highlightRange ignores invalid ranges", () => {
  for (const [s, e] of [[null, null], [3, 2], [-1, 2], [0, 99], [1.5, 3]]) {
    assert.equal(highlightRange("<x>", s, e), "&lt;x&gt;");
  }
});

test("statusMeta maps every backend status and tolerates unknown ones", () => {
  for (const s of ["ANSWERED", "LOW_CONFIDENCE", "EVIDENCE_ONLY", "NO_ANSWER_IN_EVIDENCE", "NO_EVIDENCE", "KB_NOT_CONFIGURED", "RETRIEVAL_FAILED"]) {
    assert.ok(statusMeta(s).label);
  }
  assert.equal(statusMeta("KB_NOT_CONFIGURED").tone, "neutral");
  assert.equal(statusMeta("WHATEVER").label, "WHATEVER");
});

test("confidenceLabel buckets and clamps", () => {
  assert.deepEqual(confidenceLabel(0.92), { pct: 92, label: "High" });
  assert.deepEqual(confidenceLabel(0.4), { pct: 40, label: "Moderate" });
  assert.deepEqual(confidenceLabel(0.1), { pct: 10, label: "Low" });
  assert.equal(confidenceLabel(1.7).pct, 100);
  assert.equal(confidenceLabel(undefined).label, "Unknown");
});

test("safeUrl only allows http(s)", () => {
  assert.equal(safeUrl("https://example.test/a"), "https://example.test/a");
  assert.equal(safeUrl("javascript:alert(1)"), null);
  assert.equal(safeUrl("not a url"), null);
  assert.equal(safeUrl(undefined), null);
});

test("validateQuestion mirrors backend rules", () => {
  assert.deepEqual(validateQuestion("  what   is x  "), { ok: true, value: "what is x" });
  assert.equal(validateQuestion("a").ok, false);
  assert.equal(validateQuestion("x".repeat(501)).ok, false);
  assert.equal(validateQuestion("123 ??").ok, false);
  assert.equal(validateQuestion("ok?", { min: 2, max: 10 }).ok, true);
});

test("placeholderRange finds the first bracketed slot", () => {
  assert.deepEqual(placeholderRange("What is [condition]?"), [8, 19]);
  assert.equal(placeholderRange("No slot"), null);
});

test("relativeTime formats recent times", () => {
  const now = Date.parse("2026-01-01T12:00:00Z");
  assert.equal(relativeTime("2026-01-01T11:59:30Z", now), "just now");
  assert.equal(relativeTime("2026-01-01T11:45:00Z", now), "15 min ago");
  assert.equal(relativeTime("2026-01-01T09:00:00Z", now), "3 h ago");
  assert.equal(relativeTime("garbage", now), "");
});

test("describeError gives user-facing messages", () => {
  assert.match(describeError({ status: 0 }), /Cannot reach/);
  assert.match(describeError({ status: 503 }), /server ran into a problem/);
  assert.equal(describeError({ status: 422, message: "Too short." }), "Too short.");
  assert.match(describeError({ name: "AbortError" }), /too long/);
  assert.match(describeError({ status: 413 }), /too long/);
});
