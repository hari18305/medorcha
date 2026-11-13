// Confidence tiers: green >= THRESHOLD (matches backend/services/confidence.py's
// THRESHOLD=0.65), amber for a "borderline, worth a second look" middle
// band, red below that. Mirrors the backend's threshold rather than
// inventing a separate one client-side.
export const THRESHOLD = 0.65;
const AMBER_FLOOR = 0.4;

function tierFor(confidence) {
  if (confidence >= THRESHOLD) {
    return {
      label: "high",
      bar: "bg-emerald-600",
      text: "text-emerald-700",
      chip: "bg-emerald-50 text-emerald-800 ring-emerald-600/20",
    };
  }
  if (confidence >= AMBER_FLOOR) {
    return {
      label: "moderate",
      bar: "bg-amber-500",
      text: "text-amber-700",
      chip: "bg-amber-50 text-amber-800 ring-amber-600/20",
    };
  }
  return {
    label: "low",
    bar: "bg-rose-600",
    text: "text-rose-700",
    chip: "bg-rose-50 text-rose-800 ring-rose-600/20",
  };
}

/**
 * confidence: a 0-1 float.
 * size: "sm" (thin bar, for per-diagnosis rows) or "md" (default, for
 * the overall-confidence summary).
 */
export default function ConfidenceBar({ confidence, size = "sm", label }) {
  const clamped = Math.max(0, Math.min(1, confidence ?? 0));
  const percent = Math.round(clamped * 100);
  const tier = tierFor(clamped);
  const height = size === "sm" ? "h-1.5" : "h-2.5";

  return (
    <div className="w-full">
      <div className="flex items-center justify-between gap-2">
        {label && <span className="text-xs font-medium text-slate-600">{label}</span>}
        <span className={`ml-auto text-sm font-semibold tabular-nums ${tier.text}`}>{percent}%</span>
      </div>
      <div
        className={`mt-1 w-full overflow-hidden rounded-full bg-slate-100 ${height}`}
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-label={label || "Confidence"}
      >
        <div className={`h-full rounded-full ${tier.bar} transition-all duration-500`} style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

/**
 * The headline confidence readout: a large figure, a bar with the review
 * threshold marked on it, and a plain statement of what the figure means.
 * `reviewRequired` comes from the report (it can be true even above the
 * threshold, e.g. when the optimizer could not run), so it is shown as given
 * rather than recomputed here.
 */
export function ConfidenceGauge({ confidence, reviewRequired }) {
  const clamped = Math.max(0, Math.min(1, confidence ?? 0));
  const percent = Math.round(clamped * 100);
  const tier = tierFor(clamped);
  const thresholdPercent = Math.round(THRESHOLD * 100);

  return (
    <div data-testid="confidence-gauge">
      <div className="flex items-end justify-between gap-3">
        <div>
          <p className="eyebrow">Overall confidence</p>
          <p className={`mt-1 text-5xl font-semibold leading-none tracking-tight tabular-nums ${tier.text}`}>
            {percent}
            <span className="text-2xl font-medium">%</span>
          </p>
        </div>
        <span
          className={`mb-1 inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${
            reviewRequired ? "bg-amber-50 text-amber-900 ring-amber-600/25" : "bg-emerald-50 text-emerald-800 ring-emerald-600/20"
          }`}
        >
          {reviewRequired ? "Review required" : "Above threshold"}
        </span>
      </div>

      <div className="relative mt-5">
        <div
          className="h-3 w-full overflow-hidden rounded-full bg-slate-100"
          role="meter"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          aria-label="Overall confidence"
        >
          <div className={`h-full rounded-full ${tier.bar} transition-all duration-700`} style={{ width: `${percent}%` }} />
        </div>
        <div
          className="pointer-events-none absolute -top-1.5 bottom-[-6px] w-0.5 rounded bg-ink-800"
          style={{ left: `${thresholdPercent}%` }}
          aria-hidden="true"
        />
        <div
          className="absolute top-[22px] -translate-x-1/2 whitespace-nowrap text-[11px] font-medium text-slate-600"
          style={{ left: `${thresholdPercent}%` }}
        >
          Review threshold {thresholdPercent}%
        </div>
      </div>

      <p className="mt-9 text-xs leading-relaxed text-slate-500">
        A heuristic blend of retrieval strength, live-evidence level and agreement between sources. It is not a
        probability that the diagnosis is correct.
      </p>
    </div>
  );
}

export { tierFor };
