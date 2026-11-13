// Live-progress helpers for the async /diagnose flow (backend/services/jobs.py).
//
// The server reports real step events: clinical -> imaging -> rag -> evidence -> fusion, plus an
// optimizer step that only exists when the first answer was below the confidence threshold.
// Nothing here animates on a timer; it only describes what the server says.

export const POLL_INTERVAL_MS = 1200;
export const MAX_CONSECUTIVE_POLL_FAILURES = 5;

export const STEP_TITLES = {
  clinical: "Clinical extraction",
  imaging: "Imaging read",
  rag: "Offline retrieval",
  evidence: "Live literature",
  fusion: "Fusion and confidence",
  optimizer: "Adaptive refinement",
};

export const STEP_HINTS = {
  clinical: "Structuring symptoms, history and labs",
  imaging: "BiomedCLIP zero-shot findings",
  rag: "BioBERT search, rerank, clinician precedents",
  evidence: "PubMed · Europe PMC · ClinicalTrials.gov · MedlinePlus",
  fusion: "Ranked differential with cited evidence",
  optimizer: "Refines the query and retries while confidence is below 65%",
};

export const stepTitle = (name) => STEP_TITLES[name] ?? name;

/** "1.3 s" / "42 s" / "2 min 5 s"; null when unknown. */
export function formatDuration(ms) {
  if (ms == null) return null;
  const s = ms / 1000;
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 60) return `${Math.round(s)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

/** One line for an optimizer round, e.g. "Round 2 of 3: confidence 47% (expand)". */
export function describeRound(round) {
  const base = `Round ${round.iteration}${round.max ? ` of ${round.max}` : ""}`;
  if (round.status === "running") return `${base}: refining…`;
  if (round.status === "failed") return `${base}: failed, continuing`;
  const conf = round.confidence != null ? `confidence ${Math.round(round.confidence * 100)}%` : "done";
  return `${base}: ${conf}${round.strategy ? ` (${round.strategy})` : ""}`;
}

/** What the processing screen should say, from a /diagnose/{id}/status body (or null before it exists). */
export function summarizeProgress(status) {
  if (!status) return { phase: "submitting", message: "Sending the case to the server…" };
  if (status.status === "queued") {
    const n = status.queue_position;
    return {
      phase: "queued",
      message: n > 1 ? `Waiting for a free worker (${n - 1} case${n - 1 === 1 ? "" : "s"} ahead)…` : "Waiting for a free worker…",
    };
  }
  if (status.status === "running") {
    const running = (status.steps ?? []).find((s) => s.status === "running");
    return { phase: "running", message: running ? `${stepTitle(running.name)}…` : "Working…" };
  }
  return { phase: status.status, message: "" };
}

const defaultSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Start an analysis and follow it to the end.
 *
 *   start()        -> the response of POST /diagnose?async=true: {case_id, status: "queued", ...}, or, from a
 *                     backend that predates async mode and ignored the flag, the finished report itself.
 *   fetchStatus(id)-> a /diagnose/{id}/status body.
 *   onProgress(s)  -> called with every status body received.
 *   shouldStop()   -> true once the user has abandoned this run; the loop then returns null.
 *
 * Returns the FinalReport. Throws what `start` throws; throws makeError(message) when the server reports the
 * analysis failed; tolerates up to MAX_CONSECUTIVE_POLL_FAILURES consecutive poll errors (a blip must not
 * discard a run that is still going) and then rethrows the last one.
 */
export async function followAnalysis({
  start,
  fetchStatus,
  onProgress = () => {},
  shouldStop = () => false,
  sleep = defaultSleep,
  makeError = (message) => new Error(message),
  intervalMs = POLL_INTERVAL_MS,
}) {
  const accepted = await start();
  if (accepted && Array.isArray(accepted.diagnoses)) return accepted; // old server: already a report
  const id = accepted.case_id;

  let failures = 0;
  for (;;) {
    if (shouldStop()) return null;
    let status;
    try {
      status = await fetchStatus(id);
      failures = 0;
    } catch (err) {
      if (++failures >= MAX_CONSECUTIVE_POLL_FAILURES) throw err;
      await sleep(intervalMs);
      continue;
    }
    if (shouldStop()) return null;
    onProgress(status);
    if (status.status === "completed") return status.report;
    if (status.status === "failed") throw makeError(status.error || "The analysis failed on the server.");
    await sleep(intervalMs);
  }
}
