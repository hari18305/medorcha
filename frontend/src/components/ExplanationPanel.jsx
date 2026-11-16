import { AlertIcon, CheckIcon } from "./icons.jsx";
import { findingGroups, hasExplanation } from "../utils/plainLanguage.js";

/**
 * The clinician-facing explanation of a report (FinalReport.explanation, built by
 * backend/services/explanation.py). Three independent pieces so the report can place them where they read
 * best: a summary at the top, a "findings considered" card (including normal findings and pertinent
 * negatives) and an evidence-strength card beside the confidence gauge.
 *
 * Each renders nothing when the report has no explanation (reports stored before it existed), so the
 * rest of the report is unaffected.
 */


function Bullets({ items, className = "" }) {
  return (
    <ul className={`space-y-1.5 text-sm leading-relaxed text-slate-700 ${className}`}>
      {items.map((item, index) => (
        <li key={index} className="flex items-start gap-2">
          <span className="mt-2 h-1.5 w-1.5 flex-none rounded-full bg-slate-300" />
          <span>{item}</span>
        </li>
      ))}
    </ul>
  );
}

function Block({ title, children }) {
  return (
    <div className="mt-4 border-t border-slate-100 pt-3">
      <p className="eyebrow">{title}</p>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}

const PART_COLORS = { reference: "bg-brand-600", literature: "bg-sky-500", agreement: "bg-violet-500" };
const HATCH = { backgroundImage: "repeating-linear-gradient(135deg, #e2e8f0 0 4px, #f8fafc 4px 8px)" };

/** The evidence score as a bar of 100 points: each source's slot is filled by what it earned, hatched for what it lost. */
function ScoreBar({ score }) {
  return (
    <div className="relative mt-5 pb-6" data-testid="score-bar">
      <div
        className="flex h-4 w-full overflow-hidden rounded-full bg-slate-100 ring-1 ring-inset ring-slate-200"
        role="img"
        aria-label={`${score.final} of 100 evidence points. ${score.parts.map((p) => `${p.label}: ${p.earned} of ${p.max}`).join("; ")}.`}
      >
        {score.parts.map((part) => (
          <div key={part.key} data-bar-part={part.key} className="relative h-full border-r border-white last:border-r-0" style={{ width: `${part.max}%` }}>
            <div className={`absolute inset-y-0 left-0 ${PART_COLORS[part.key]}`} style={{ width: `${part.max ? (part.earned / part.max) * 100 : 0}%` }} />
            <div className="absolute inset-y-0 right-0" style={{ ...HATCH, width: `${part.max ? (part.lost / part.max) * 100 : 0}%` }} />
          </div>
        ))}
      </div>
      {score.cap && score.raw > score.final && (
        <>
          <div
            data-testid="score-cap-overlay"
            title={`Removed by a safety rule: ${score.raw - score.final} points`}
            className="pointer-events-none absolute top-0 h-4 rounded-r-full bg-rose-500/50 ring-1 ring-inset ring-rose-600/60"
            style={{ left: `${score.final}%`, width: `${score.raw - score.final}%` }}
          />
          <p className="absolute bottom-0 right-0 text-[11px] font-medium text-rose-700">Red: removed by a safety rule</p>
        </>
      )}
      <div className="pointer-events-none absolute -top-1.5 h-[22px] w-0.5 rounded bg-ink-800" style={{ left: `${score.review_line}%` }} aria-hidden="true" />
      <div className="absolute bottom-0 -translate-x-1/2 whitespace-nowrap text-[11px] font-medium text-slate-600" style={{ left: `${score.review_line}%` }}>
        Review line {score.review_line}
      </div>
    </div>
  );
}

function PartRow({ part }) {
  return (
    <li data-score-part={part.key} className="py-3 first:pt-0 last:pb-0">
      <div className="flex items-center justify-between gap-3">
        <p className="flex items-center gap-2 text-sm font-medium text-slate-900">
          <span className={`h-2.5 w-2.5 flex-none rounded-sm ${PART_COLORS[part.key]}`} aria-hidden="true" />
          {part.label}
        </p>
        <p className="flex items-baseline gap-2 tabular-nums">
          <span className="text-sm font-semibold text-slate-900">{part.earned}</span>
          <span className="text-xs text-slate-500">of {part.max} points</span>
          {part.lost > 0 && (
            <span data-lost className="rounded-full bg-rose-50 px-1.5 py-0.5 text-[11px] font-semibold text-rose-800 ring-1 ring-inset ring-rose-600/20">
              −{part.lost}
            </span>
          )}
        </p>
      </div>
      <p className="mt-1 pl-[18px] text-sm leading-relaxed text-slate-600">{part.lost > 0 ? part.why_lost : part.detail}</p>
      {(part.lost > 0 || part.detail) && (
        <details className="mt-1 pl-[18px] text-xs text-slate-500">
          <summary className="cursor-pointer select-none font-medium text-brand-700 hover:underline">
            {part.lost > 0 ? "What was found, and how to raise it" : "What was found"}
          </summary>
          <p className="mt-1 leading-relaxed text-slate-600">{part.detail}</p>
          {part.how_to_raise && (
            <p className="mt-1 leading-relaxed text-slate-700">
              <span className="font-semibold">To raise it:</span> {part.how_to_raise}
            </p>
          )}
        </details>
      )}
    </li>
  );
}

/** "What does 80% mean?": the number, what it is not, and where the missing points went. */
function ScoreLedger({ score, reviewRequired }) {
  const lostTotal = 100 - score.final;
  return (
    <div className="mt-5 border-t border-slate-100 pt-5" data-testid="score-ledger">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="eyebrow">Evidence score</p>
          <p className="mt-1 flex items-baseline gap-1.5 leading-none">
            <span className="text-5xl font-semibold tracking-tight tabular-nums text-slate-900" data-testid="score-final">{score.final}</span>
            <span className="text-xl font-medium text-slate-400">/ 100</span>
          </p>
        </div>
        <span
          className={`mb-1 inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${
            reviewRequired ? "bg-amber-50 text-amber-900 ring-amber-600/25" : "bg-emerald-50 text-emerald-800 ring-emerald-600/20"
          }`}
        >
          {reviewRequired ? "Review required" : "Above the review line"}
        </span>
      </div>

      <ScoreBar score={score} />

      <ul className="divide-y divide-slate-100" data-testid="score-parts">
        {score.parts.map((part) => (
          <PartRow key={part.key} part={part} />
        ))}
      </ul>

      {score.cap && (
        <div data-testid="score-cap" className="mt-3 flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50/70 px-3 py-2.5 text-sm text-rose-900">
          <AlertIcon className="mt-0.5 h-4 w-4 flex-none text-rose-700" />
          <span>
            <span className="font-semibold">Safety rule: −{score.cap.points_removed} points.</span> {score.cap.reason}
          </span>
        </div>
      )}

      <dl className="mt-4 space-y-3 rounded-lg bg-slate-50 px-4 py-3 text-sm leading-relaxed" data-testid="score-meaning">
        <div>
          <dt className="font-semibold text-slate-900">What {score.final} means</dt>
          <dd className="text-slate-700">{score.meaning.is}</dd>
        </div>
        <div>
          <dt className="font-semibold text-slate-900">What it does not mean</dt>
          <dd className="text-slate-700">{score.meaning.is_not}</dd>
        </div>
        <div>
          <dt className="font-semibold text-slate-900">What about the other {lostTotal}?</dt>
          <dd className="text-slate-700">{score.meaning.remainder}</dd>
        </div>
        <div>
          <dt className="font-semibold text-slate-900">The review line</dt>
          <dd className="text-slate-700">{score.meaning.review_line}</dd>
        </div>
      </dl>
    </div>
  );
}

/**
 * The top card of the report: what the system concluded, how far to trust it (the evidence score taken
 * apart, so "80%" is never an unexplained number), why review is needed and what would help.
 */
export function ResultOverview({ explanation, reviewRequired }) {
  if (!hasExplanation(explanation)) return null;
  const { headline, review_reasons: reasons = [], cautions = [], next_steps: steps = [], score } = explanation;
  return (
    <section className="card card-pad" data-testid="explanation-summary">
      <p className="eyebrow">Result</p>
      <p className="mt-1 text-lg font-medium leading-snug text-slate-900">{headline}</p>
      {score && <ScoreLedger score={score} reviewRequired={reviewRequired} />}
      {reasons.length > 0 && (
        <Block title="Why review is needed">
          <Bullets items={reasons} />
        </Block>
      )}
      {cautions.length > 0 && (
        <Block title="Worth knowing">
          <Bullets items={cautions} />
        </Block>
      )}
      {steps.length > 0 && (
        <Block title="What would help">
          <Bullets items={steps} />
        </Block>
      )}
    </section>
  );
}

/** What the case actually showed: reported findings, normal findings and pertinent negatives, history, labs. */
export function FindingsConsidered({ explanation }) {
  if (!hasExplanation(explanation)) return null;
  const groups = findingGroups(explanation);
  return (
    <section className="card card-pad" data-testid="findings-considered">
      <h3 className="text-sm font-semibold text-slate-900">Findings considered</h3>
      {groups.length === 0 ? (
        <p className="mt-2 text-sm italic text-slate-500">
          No specific findings could be extracted from the information provided.
        </p>
      ) : (
        <div className="mt-2 grid gap-x-6 gap-y-1 sm:grid-cols-2">
          {groups.map((group) => (
            <div key={group.id} data-findings-group={group.id} className="mt-2">
              <p className="eyebrow">{group.title}</p>
              <ul className="mt-1.5 flex flex-wrap gap-1.5">
                {group.items.map((item, index) => (
                  <li
                    key={index}
                    className={`rounded-md px-2.5 py-1 text-sm ring-1 ring-inset ${
                      group.id === "normal"
                        ? "bg-emerald-50 text-emerald-900 ring-emerald-600/15"
                        : "bg-slate-50 text-slate-800 ring-slate-200"
                    }`}
                  >
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
      <p className="mt-3 text-xs text-slate-500">
        {groups.some((g) => g.id === "normal")
          ? "Taken from the text, report and laboratory entries you provided. Normal findings and stated negatives are shown because they count against some diagnoses."
          : "Taken from the text, report and laboratory entries you provided. No normal findings or pertinent negatives were stated; adding them (for example “no fever”) can sharpen the differential."}
      </p>
    </section>
  );
}

/** One diagnosis's "why this ranking": findings that fit and findings against / not reported. */
export function WhyThisRanking({ diagnosis }) {
  const fits = diagnosis?.findings_for ?? [];
  const against = diagnosis?.findings_against ?? [];
  if (fits.length === 0 && against.length === 0) return null;
  return (
    <div className="mt-4 grid gap-3 border-t border-slate-100 pt-3 sm:grid-cols-2" data-testid="why-this-ranking">
      <div>
        <p className="eyebrow">Fits this case</p>
        {fits.length > 0 ? (
          <ul className="mt-1.5 space-y-1">
            {fits.map((item, index) => (
              <li key={index} className="flex items-start gap-2 text-sm text-slate-700">
                <CheckIcon className="mt-0.5 h-3.5 w-3.5 flex-none text-emerald-600" strokeWidth="2.5" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-1.5 text-sm italic text-slate-400">Nothing specific noted.</p>
        )}
      </div>
      <div>
        <p className="eyebrow">Against, or not reported</p>
        {against.length > 0 ? (
          <ul className="mt-1.5 space-y-1">
            {against.map((item, index) => (
              <li key={index} className="flex items-start gap-2 text-sm text-slate-700">
                <AlertIcon className="mt-0.5 h-3.5 w-3.5 flex-none text-amber-600" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-1.5 text-sm italic text-slate-400">Nothing specific noted.</p>
        )}
      </div>
    </div>
  );
}
