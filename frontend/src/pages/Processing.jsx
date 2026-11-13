import { useEffect, useState } from "react";
import { CheckIcon, Spinner } from "../components/icons.jsx";

const STAGES = [
  { id: "clinical", title: "Clinical extraction", detail: "Structuring symptoms, history and labs" },
  { id: "imaging", title: "Imaging read", detail: "BiomedCLIP zero-shot findings", needsImage: true },
  { id: "rag", title: "Offline retrieval", detail: "BioBERT search, cross-encoder rerank, clinician precedents" },
  { id: "live", title: "Live literature", detail: "PubMed · Europe PMC · ClinicalTrials.gov · MedlinePlus" },
  { id: "fusion", title: "Fusion and confidence", detail: "Ranked differential with cited evidence" },
];

const STAGE_DURATION_MS = 1400;
const MAX_OPTIMIZER_ITERATIONS = 3;

function formatElapsed(totalSeconds) {
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/**
 * Phase 9's /diagnose is fully synchronous -- there is no real
 * progress signal from the server while the request is in flight, so
 * the stage list is a purely client-side illustrative animation: it steps
 * through the named stages on a timer, and if the wait continues past
 * all of them, cycles an "adaptive optimizer" indicator so the screen
 * doesn't look stalled. It does not reflect what the backend is
 * actually doing at any given moment. The elapsed clock, by contrast, is
 * real. `hadImage` only decides whether the imaging stage is shown as
 * skipped.
 */
export default function Processing({ hadImage = true }) {
  const stages = STAGES.filter((stage) => hadImage || !stage.needsImage);
  const [activeStageIndex, setActiveStageIndex] = useState(0);
  const [optimizerIteration, setOptimizerIteration] = useState(null);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setActiveStageIndex((current) => {
        const next = current + 1;
        if (next < stages.length) {
          return next;
        }
        setOptimizerIteration((iter) => (iter === null || iter >= MAX_OPTIMIZER_ITERATIONS ? 1 : iter + 1));
        return current;
      });
    }, STAGE_DURATION_MS);
    return () => clearInterval(timer);
  }, [stages.length]);

  useEffect(() => {
    const clock = setInterval(() => setElapsed((seconds) => seconds + 1), 1000);
    return () => clearInterval(clock);
  }, []);

  return (
    <div className="mx-auto max-w-xl animate-rise">
      <div className="card card-pad">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="eyebrow">Analysis in progress</p>
            <h2 className="mt-1 text-xl font-semibold tracking-tight text-slate-900">Analyzing case</h2>
          </div>
          <div className="text-right">
            <p className="font-mono text-lg font-semibold tabular-nums text-slate-900" aria-label="Elapsed time">
              {formatElapsed(elapsed)}
            </p>
            <p className="text-[11px] text-slate-500">elapsed</p>
          </div>
        </div>

        <p className="mt-2 text-sm text-slate-600">
          Each stage runs against real evidence, so this can take a minute or more. Cases that need refinement
          take longer.
        </p>

        <ol className="mt-6 space-y-1" aria-live="polite">
          {STAGES.map((stage) => {
            if (stage.needsImage && !hadImage) {
              return (
                <li key={stage.id} className="flex items-start gap-3 px-3 py-2.5 text-slate-400">
                  <span className="mt-0.5 h-5 w-5 flex-none rounded-full border-2 border-dashed border-slate-200" />
                  <div>
                    <p className="text-sm font-medium">{stage.title}</p>
                    <p className="text-xs">Skipped: no image was attached</p>
                  </div>
                </li>
              );
            }
            const index = stages.indexOf(stage);
            const isDone = index < activeStageIndex;
            const isActive = index === activeStageIndex;
            return (
              <li
                key={stage.id}
                className={`flex items-start gap-3 rounded-lg px-3 py-2.5 transition ${
                  isActive ? "bg-brand-50 ring-1 ring-inset ring-brand-100" : ""
                }`}
              >
                <StageIcon done={isDone} active={isActive} />
                <div>
                  <p className={`text-sm font-medium ${isDone || isActive ? "text-slate-900" : "text-slate-400"}`}>
                    {stage.title}
                  </p>
                  <p className={`text-xs ${isActive ? "text-slate-600" : "text-slate-400"}`}>{stage.detail}</p>
                </div>
              </li>
            );
          })}
        </ol>

        {optimizerIteration !== null && (
          <div className="mt-5 flex items-center gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
            <Spinner />
            <div>
              <p className="text-sm font-medium text-amber-900">Adaptive optimizer</p>
              <p className="text-xs text-amber-800">
                Still working; if confidence is under 65% the query is refined and retrieval repeated (up to{" "}
                {MAX_OPTIMIZER_ITERATIONS} times).
              </p>
            </div>
          </div>
        )}

        <p className="mt-5 border-t border-slate-100 pt-4 text-xs text-slate-500">
          The stage list is illustrative: the server returns one result at the end rather than reporting
          progress.
        </p>
      </div>
    </div>
  );
}

function StageIcon({ done, active }) {
  if (done) {
    return (
      <span className="mt-0.5 flex h-5 w-5 flex-none items-center justify-center rounded-full bg-emerald-600 text-white">
        <CheckIcon className="h-3 w-3" strokeWidth="2.5" />
      </span>
    );
  }
  if (active) {
    return <Spinner className="mt-0.5 h-5 w-5" />;
  }
  return <span className="mt-0.5 h-5 w-5 flex-none rounded-full border-2 border-slate-200" />;
}
