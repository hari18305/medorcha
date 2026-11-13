import { useEffect, useState } from "react";
import { AlertIcon, CheckIcon, Spinner } from "../components/icons.jsx";
import { describeRound, formatDuration, STEP_HINTS, stepTitle, summarizeProgress } from "../utils/progress.js";

function formatElapsed(totalSeconds) {
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/**
 * Live progress of one analysis. `progress` is the latest GET /diagnose/{id}/status body (or null while
 * the case is still being sent): the steps, their results and timings, and the optimizer's refinement
 * rounds are what the server's pipeline actually reported, not an animation. The elapsed clock is the
 * browser's own.
 */
export default function Processing({ progress = null }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const clock = setInterval(() => setElapsed((seconds) => seconds + 1), 1000);
    return () => clearInterval(clock);
  }, []);

  const summary = summarizeProgress(progress);
  const steps = progress?.steps ?? [];

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

        <p className="mt-2 flex items-center gap-2 text-sm text-slate-600" role="status" data-testid="progress-message">
          {summary.phase !== "completed" && <Spinner className="h-3.5 w-3.5" />}
          {summary.message}
        </p>

        {steps.length > 0 && (
          <ol className="mt-5 space-y-1" aria-live="polite" data-testid="progress-steps">
            {steps.map((step) => (
              <StepRow key={step.name} step={step} />
            ))}
          </ol>
        )}

        <p className="mt-5 border-t border-slate-100 pt-4 text-xs text-slate-500">
          Live progress from the server. Cases that need refinement take longer; you can leave this page open.
        </p>
      </div>
    </div>
  );
}

function StepRow({ step }) {
  const running = step.status === "running";
  const muted = step.status === "pending" || step.status === "skipped";
  return (
    <li
      data-step={step.name}
      data-status={step.status}
      className={`flex items-start gap-3 rounded-lg px-3 py-2.5 transition ${
        running ? "bg-brand-50 ring-1 ring-inset ring-brand-100" : ""
      }`}
    >
      <StepIcon status={step.status} />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-3">
          <p className={`text-sm font-medium ${muted ? "text-slate-400" : "text-slate-900"}`}>{stepTitle(step.name)}</p>
          {step.duration_ms != null && step.status !== "skipped" && (
            <span className="flex-none font-mono text-xs tabular-nums text-slate-400">{formatDuration(step.duration_ms)}</span>
          )}
        </div>
        <p className={`text-xs ${step.status === "failed" ? "text-amber-800" : muted ? "text-slate-400" : "text-slate-600"}`}>
          {step.status === "skipped"
            ? `Skipped: ${step.detail ?? "not needed"}`
            : step.detail || (step.status === "pending" || running ? STEP_HINTS[step.name] : "")}
        </p>
        {step.iterations?.length > 0 && (
          <ul className="mt-1.5 space-y-0.5" data-testid="optimizer-rounds">
            {step.iterations.map((round) => (
              <li key={round.iteration} className="flex items-center gap-1.5 text-xs text-slate-600">
                {round.status === "running" ? <Spinner className="h-3 w-3" /> : <span className="h-1.5 w-1.5 rounded-full bg-slate-300" />}
                {describeRound(round)}
              </li>
            ))}
          </ul>
        )}
      </div>
    </li>
  );
}

function StepIcon({ status }) {
  if (status === "done") {
    return (
      <span className="mt-0.5 flex h-5 w-5 flex-none items-center justify-center rounded-full bg-emerald-600 text-white">
        <CheckIcon className="h-3 w-3" strokeWidth="2.5" />
      </span>
    );
  }
  if (status === "failed") {
    return (
      <span className="mt-0.5 flex h-5 w-5 flex-none items-center justify-center rounded-full bg-amber-500 text-white">
        <AlertIcon className="h-3 w-3" strokeWidth="2.5" />
      </span>
    );
  }
  if (status === "running") return <Spinner className="mt-0.5 h-5 w-5" />;
  if (status === "skipped") return <span className="mt-0.5 h-5 w-5 flex-none rounded-full border-2 border-dashed border-slate-200" />;
  return <span className="mt-0.5 h-5 w-5 flex-none rounded-full border-2 border-slate-200" />;
}
