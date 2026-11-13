// Run with: npm test
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  describeRound,
  followAnalysis,
  formatDuration,
  MAX_CONSECUTIVE_POLL_FAILURES,
  stepTitle,
  summarizeProgress,
} from "./progress.js";

const noSleep = async () => {};
const REPORT = { case_id: "c1", diagnoses: [{ name: "x" }], confidence: 0.7 };

// ---- describing -------------------------------------------------------------------

test("formatDuration is readable at each scale and null when unknown", () => {
  assert.equal(formatDuration(1300), "1.3 s");
  assert.equal(formatDuration(42000), "42 s");
  assert.equal(formatDuration(125000), "2 min 5 s");
  assert.equal(formatDuration(null), null);
});

test("stepTitle falls back to the raw name for an unknown step", () => {
  assert.equal(stepTitle("rag"), "Offline retrieval");
  assert.equal(stepTitle("something_new"), "something_new");
});

test("describeRound covers running, done (with strategy) and failed rounds", () => {
  assert.equal(describeRound({ iteration: 1, max: 3, status: "running" }), "Round 1 of 3: refining…");
  assert.equal(describeRound({ iteration: 2, max: 3, status: "done", confidence: 0.474, strategy: "expand" }), "Round 2 of 3: confidence 47% (expand)");
  assert.equal(describeRound({ iteration: 3, max: 3, status: "failed" }), "Round 3 of 3: failed, continuing");
});

test("summarizeProgress describes each phase from the status body", () => {
  assert.equal(summarizeProgress(null).phase, "submitting");
  assert.match(summarizeProgress({ status: "queued", queue_position: 1 }).message, /Waiting for a free worker…/);
  assert.match(summarizeProgress({ status: "queued", queue_position: 3 }).message, /2 cases ahead/);
  assert.match(summarizeProgress({ status: "queued", queue_position: 2 }).message, /1 case ahead/);
  assert.equal(summarizeProgress({ status: "running", steps: [{ name: "rag", status: "running" }] }).message, "Offline retrieval…");
  assert.equal(summarizeProgress({ status: "running", steps: [] }).message, "Working…");
});

// ---- following an analysis ---------------------------------------------------------

test("followAnalysis returns the report once the server says completed, reporting each status", async () => {
  const seen = [];
  const statuses = [{ status: "queued" }, { status: "running", steps: [] }, { status: "completed", report: REPORT }];
  const report = await followAnalysis({
    start: async () => ({ case_id: "c1", status: "queued" }),
    fetchStatus: async () => statuses.shift(),
    onProgress: (s) => seen.push(s.status),
    sleep: noSleep,
  });
  assert.equal(report, REPORT);
  assert.deepEqual(seen, ["queued", "running", "completed"]);
});

test("followAnalysis accepts a backend that ignored async mode and returned the report directly", async () => {
  let polled = false;
  const report = await followAnalysis({ start: async () => REPORT, fetchStatus: async () => (polled = true), sleep: noSleep });
  assert.equal(report, REPORT);
  assert.equal(polled, false);
});

test("followAnalysis throws the server's message when the analysis failed", async () => {
  await assert.rejects(
    followAnalysis({
      start: async () => ({ case_id: "c1" }),
      fetchStatus: async () => ({ status: "failed", error: "The analysis failed on the server. Please try again." }),
      sleep: noSleep,
      makeError: (m) => Object.assign(new Error(m), { kind: "server" }),
    }),
    (err) => err.kind === "server" && /failed on the server/.test(err.message)
  );
});

test("followAnalysis survives a transient poll failure and a recovered run completes", async () => {
  let calls = 0;
  const report = await followAnalysis({
    start: async () => ({ case_id: "c1" }),
    fetchStatus: async () => {
      calls += 1;
      if (calls <= 2) throw new Error("network blip");
      return { status: "completed", report: REPORT };
    },
    sleep: noSleep,
  });
  assert.equal(report, REPORT);
  assert.equal(calls, 3);
});

test("followAnalysis gives up after too many consecutive poll failures and rethrows the last", async () => {
  let calls = 0;
  await assert.rejects(
    followAnalysis({
      start: async () => ({ case_id: "c1" }),
      fetchStatus: async () => {
        calls += 1;
        throw new Error(`down ${calls}`);
      },
      sleep: noSleep,
    }),
    /down 5/
  );
  assert.equal(calls, MAX_CONSECUTIVE_POLL_FAILURES);
});

test("a successful poll resets the failure count", async () => {
  const script = [new Error("a"), new Error("b"), new Error("c"), new Error("d"), { status: "running" }, new Error("e"), new Error("f"), { status: "completed", report: REPORT }];
  const report = await followAnalysis({
    start: async () => ({ case_id: "c1" }),
    fetchStatus: async () => {
      const next = script.shift();
      if (next instanceof Error) throw next;
      return next;
    },
    sleep: noSleep,
  });
  assert.equal(report, REPORT);
});

test("followAnalysis stops quietly (returns null) once the user has abandoned the run", async () => {
  let polls = 0;
  let stop = false;
  const result = await followAnalysis({
    start: async () => ({ case_id: "c1" }),
    fetchStatus: async () => {
      polls += 1;
      if (polls === 2) stop = true;
      return { status: "running" };
    },
    shouldStop: () => stop,
    sleep: noSleep,
  });
  assert.equal(result, null);
  assert.equal(polls, 2);
});

test("an error from start (for example a 422) propagates untouched", async () => {
  await assert.rejects(followAnalysis({ start: async () => { throw new Error("422 age required"); }, fetchStatus: async () => ({}), sleep: noSleep }), /422/);
});
