import { useEffect, useRef, useState } from "react";
import ConfidenceBar, { ConfidenceGauge } from "./ConfidenceBar.jsx";
import ImagingSection from "./ImagingSection.jsx";
import SourceBadge, { classifyEvidenceSource } from "./SourceBadge.jsx";
import { AlertIcon, BookIcon, ExternalIcon, ShieldIcon } from "./icons.jsx";

/**
 * The data-display half of a FinalReport -- confidence, conflicts, imaging
 * findings, differential diagnoses, and supporting evidence. Shared between
 * pages/Report.jsx and pages/Review.jsx so the review screen shows
 * exactly the same underlying data the clinician already saw on the
 * report, not a re-derived summary.
 *
 * Layout: a main column (review notice, conflicts, imaging, differential,
 * evidence) and a sticky sidebar (confidence gauge, then `aside`, then the
 * pipeline-run facts). Report.jsx puts the patient summary in `aside`;
 * Review.jsx puts the decision form there.
 *
 * The "Review Case" entry point itself lives in Report.jsx's header
 * (always visible, on any report -- reviewing isn't restricted to
 * low-confidence cases); this component just surfaces an urgent notice
 * when report.review_required is true, without its own separate button.
 *
 * `imageSubmitted`: whether an image was attached on the submission
 * screen (only known client-side) -- see ImagingSection.
 */

// A fusion conflict that describes the model's own direct read of the
// image disagreeing with BiomedCLIP's pre-filter label is a distinct,
// newer conflict type than a disagreement between text evidence sources
// (Phase 5), so it is tagged separately. The fusion prompt asks for a
// "short, human-readable description" with no fixed wording, hence the
// keyword match rather than an exact string.
const IMAGE_CONFLICT_RE = /biomedclip|pre-?filter|visual read|direct(ly)? (read|visual)|from the image|in the image/i;

export function isImageConflict(text) {
  return IMAGE_CONFLICT_RE.test(text);
}

// The backend's input-sufficiency gate (backend/services/confidence.py,
// INSUFFICIENT_INPUT_NOTE) puts its explanation at the front of
// report.conflicts. It is not a disagreement between evidence sources, so it
// is shown with the review notice instead of under "Conflicting evidence".
// A second note (NONSPECIFIC_INPUT_NOTE) covers a complaint made only of general
// symptoms; it is shown the same way.
const INPUT_NOTE_PREFIXES = ["Insufficient clinical input:", "Non-specific clinical input:"];

export function isInsufficientInputNote(text) {
  return typeof text === "string" && INPUT_NOTE_PREFIXES.some((prefix) => text.startsWith(prefix));
}

// The fusion model cites evidence as [RAG1], [LIVE2], ... (fusion_agent.py
// numbers the offline items and the live items separately, in list order),
// and FinalReport.evidence preserves that order. Reproducing the numbering
// here is what lets a citation link to the item it refers to. If a report
// cites an ID that is not in the evidence list (a model slip: the optimizer now returns the evidence of the round that
// produced the final differential), the chip is shown but not linked.
const CITATION_RE = /\[((?:RAG|LIVE|MEM|VIS)\d+)\]/g;

export function assignEvidenceIds(evidence) {
  let rag = 0;
  let live = 0;
  return evidence.map((item) => (item.type === "live" ? { ...item, eid: `LIVE${++live}` } : { ...item, eid: `RAG${++rag}` }));
}

export function splitCitations(input) {
  const text = String(input ?? "");
  const parts = [];
  let last = 0;
  for (const match of text.matchAll(CITATION_RE)) {
    if (match.index > last) parts.push({ kind: "text", value: text.slice(last, match.index) });
    parts.push({ kind: "cite", value: match[1] });
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", value: text.slice(last) });
  return parts;
}

const FILTERS = [
  { id: "all", label: "All" },
  { id: "offline-kb", label: "Offline KB" },
  { id: "live", label: "Live" },
  { id: "precedent", label: "Precedents" },
];

function filterOf(item) {
  const kind = classifyEvidenceSource(item);
  return kind === "clinician-verified" || kind === "visual-precedent" ? "precedent" : kind;
}

export default function ReportSummary({ report, imageSubmitted = false, aside = null }) {
  const diagnoses = [...(report.diagnoses ?? [])].sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0));
  const evidence = assignEvidenceIds(report.evidence ?? []);
  const knownIds = new Set(evidence.map((item) => item.eid));
  const allConflicts = report.conflicts ?? [];
  const inputNote = allConflicts.find(isInsufficientInputNote);
  const conflicts = allConflicts.filter((c) => !isInsufficientInputNote(c));
  const iterations = report.iteration_count ?? 0;

  const [filter, setFilter] = useState("all");
  const [highlightId, setHighlightId] = useState(null);
  const highlightTimer = useRef(null);
  useEffect(() => () => clearTimeout(highlightTimer.current), []);

  function jumpTo(eid) {
    setFilter("all");
    setHighlightId(eid);
    clearTimeout(highlightTimer.current);
    highlightTimer.current = setTimeout(() => setHighlightId(null), 1600);
    // wait a frame so a filtered-out item is back in the DOM
    requestAnimationFrame(() => {
      document.getElementById(`ev-${eid}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  const counts = {
    all: evidence.length,
    "offline-kb": evidence.filter((e) => filterOf(e) === "offline-kb").length,
    live: evidence.filter((e) => filterOf(e) === "live").length,
    precedent: evidence.filter((e) => filterOf(e) === "precedent").length,
  };
  const visibleEvidence = filter === "all" ? evidence : evidence.filter((e) => filterOf(e) === filter);

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
      <div className="min-w-0 space-y-6">
        {/* Small screens have no sidebar column, so lead with the headline confidence. */}
        <section className="card card-pad lg:hidden">
          <ConfidenceGauge confidence={report.confidence} reviewRequired={Boolean(report.review_required)} />
        </section>

        {report.review_required && (
          <div
            role="alert"
            data-testid="review-notice"
            className="flex items-start gap-3 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3.5"
          >
            <AlertIcon className="mt-0.5 h-5 w-5 text-amber-700" />
            <div>
              <p className="text-sm font-semibold text-amber-950">Clinician review required</p>
              <p className="mt-0.5 text-sm text-amber-900">
                This result is below the review threshold, or could not be fully checked. Do not act on it
                until a clinician has reviewed it.
              </p>
              {inputNote && (
                <p data-testid="insufficient-input-note" className="mt-2 text-sm text-amber-900">
                  {inputNote}
                </p>
              )}
            </div>
          </div>
        )}

        {conflicts.length > 0 && (
          <section role="alert" className="rounded-xl border border-rose-200 bg-rose-50/70 p-4">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-rose-900">
              <AlertIcon className="h-4 w-4" /> Conflicting evidence
            </h3>
            <ul className="mt-2 space-y-2 text-sm text-rose-900">
              {conflicts.map((conflict, index) => (
                <li key={index} className="flex items-start gap-2">
                  {isImageConflict(conflict) ? (
                    <span
                      data-conflict-kind="image"
                      className="mt-0.5 flex-none rounded-full bg-fuchsia-100 px-2 py-0.5 text-xs font-medium text-fuchsia-800 ring-1 ring-inset ring-fuchsia-600/20"
                    >
                      Image vs. BiomedCLIP
                    </span>
                  ) : (
                    <span className="mt-2 h-1.5 w-1.5 flex-none rounded-full bg-rose-400" />
                  )}
                  <span>{conflict}</span>
                </li>
              ))}
            </ul>
          </section>
        )}

        <ImagingSection imaging={report.imaging} imageSubmitted={imageSubmitted} />

        <section>
          <div className="flex items-baseline justify-between">
            <h3 className="text-base font-semibold text-slate-900">Differential diagnosis</h3>
            <p className="text-xs text-slate-500">Ranked by the model; citations link to the evidence below</p>
          </div>
          <div className="mt-3 space-y-3">
            {diagnoses.length === 0 && (
              <div className="card card-pad text-sm text-slate-600">No diagnoses were produced for this case.</div>
            )}
            {diagnoses.map((diagnosis, index) => (
              <article
                key={index}
                className={`card card-pad ${index === 0 ? "ring-1 ring-brand-200" : ""}`}
                data-rank={diagnosis.rank ?? index + 1}
              >
                <div className="flex items-start gap-3">
                  <span
                    className={`flex h-7 w-7 flex-none items-center justify-center rounded-full text-sm font-semibold ${
                      index === 0 ? "bg-brand-700 text-white" : "bg-slate-100 text-slate-600"
                    }`}
                  >
                    {diagnosis.rank ?? index + 1}
                  </span>
                  <div className="min-w-0 flex-1">
                    <h4 className="text-base font-semibold leading-snug text-slate-900">{diagnosis.name}</h4>
                    <div className="mt-2 max-w-sm">
                      <ConfidenceBar confidence={diagnosis.confidence} size="sm" label="Model-stated confidence" />
                    </div>
                  </div>
                </div>

                {diagnosis.supporting_evidence?.length > 0 ? (
                  <ul className="mt-4 space-y-2 border-t border-slate-100 pt-3">
                    {diagnosis.supporting_evidence.map((item, i) => (
                      <li key={i} className="text-sm leading-relaxed text-slate-700">
                        <CitedText text={item} knownIds={knownIds} onJump={jumpTo} />
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-4 border-t border-slate-100 pt-3 text-sm italic text-slate-400">
                    No cited supporting evidence.
                  </p>
                )}
              </article>
            ))}
          </div>
        </section>

        <section>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="text-base font-semibold text-slate-900">Supporting evidence</h3>
            {evidence.length > 0 && (
              <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Filter evidence by source">
                {FILTERS.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    role="tab"
                    aria-selected={filter === f.id}
                    onClick={() => setFilter(f.id)}
                    disabled={counts[f.id] === 0 && f.id !== "all"}
                    className={`rounded-full px-3 py-1 text-xs font-medium transition disabled:opacity-40 ${
                      filter === f.id
                        ? "bg-ink-800 text-white"
                        : "bg-white text-slate-600 ring-1 ring-inset ring-slate-300 hover:ring-slate-400"
                    }`}
                  >
                    {f.label} <span className="tabular-nums opacity-70">{counts[f.id]}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {report.live_evidence_available === false && (
            <p
              role="status"
              className="mt-3 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5 text-sm text-amber-900"
            >
              <AlertIcon className="mt-0.5 h-4 w-4 text-amber-700" />
              <span>
                Live literature evidence was unavailable for this case — every live source failed to
                respond. Only the offline knowledge base (and any clinician precedents) are shown, so this
                differential has not been checked against current literature.
              </span>
            </p>
          )}

          <ul className="mt-3 space-y-2.5">
            {visibleEvidence.map((item) => (
              <EvidenceCard key={item.eid} item={item} highlighted={highlightId === item.eid} />
            ))}
            {evidence.length === 0 && (
              <li className="card card-pad text-sm text-slate-600">No supporting evidence was retrieved for this case.</li>
            )}
          </ul>
        </section>
      </div>

      <aside className="min-w-0 space-y-4 lg:sticky lg:top-6">
        <section className="card card-pad hidden lg:block">
          <ConfidenceGauge confidence={report.confidence} reviewRequired={Boolean(report.review_required)} />
        </section>

        {aside}

        <section className="card card-pad">
          <div className="flex items-center gap-2">
            <ShieldIcon className="h-4 w-4 text-brand-700" />
            <h3 className="text-sm font-semibold text-slate-900">Pipeline run</h3>
          </div>
          <dl className="mt-3 divide-y divide-slate-100 text-sm">
            <RunFact
              label="Refinement rounds"
              value={
                iterations > 0
                  ? `${iterations} of 3`
                  : inputNote
                    ? "Skipped (no clinical input)"
                    : "None needed"
              }
            />
            <RunFact
              label="Live literature"
              value={report.live_evidence_available === false ? "Unavailable" : "Checked"}
              tone={report.live_evidence_available === false ? "warn" : undefined}
            />
            <RunFact label="Offline evidence" value={`${counts["offline-kb"] + counts.precedent} passages`} />
            <RunFact label="Live evidence" value={`${counts.live} items`} />
            <RunFact
              label="Imaging"
              value={
                !report.imaging
                  ? imageSubmitted
                    ? "No findings"
                    : "None submitted"
                  : report.imaging.unsupported_modality
                    ? "Not assessed"
                    : report.imaging.inconclusive
                      ? "Inconclusive"
                      : report.imaging.modality
              }
            />
          </dl>
        </section>
      </aside>
    </div>
  );
}

function RunFact({ label, value, tone }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2">
      <dt className="text-slate-500">{label}</dt>
      <dd className={`text-right font-medium ${tone === "warn" ? "text-amber-800" : "text-slate-900"}`}>{value}</dd>
    </div>
  );
}

/** Text with [RAG1]/[LIVE2] markers rendered as chips that jump to that evidence item. */
function CitedText({ text, knownIds, onJump }) {
  return (
    <>
      {splitCitations(text).map((part, index) => {
        if (part.kind === "text") return <span key={index}>{part.value}</span>;
        const linked = knownIds.has(part.value);
        const cls =
          "mx-0.5 inline-flex items-center rounded-md px-1.5 py-0.5 align-baseline font-mono text-[11px] font-semibold ring-1 ring-inset";
        return linked ? (
          <button
            key={index}
            type="button"
            data-citation={part.value}
            onClick={() => onJump(part.value)}
            title={`Show ${part.value} in the evidence list`}
            className={`${cls} bg-brand-50 text-brand-800 ring-brand-200 transition hover:bg-brand-100`}
          >
            {part.value}
          </button>
        ) : (
          <span
            key={index}
            data-citation={part.value}
            title="This item is not in the evidence list returned with this report"
            className={`${cls} bg-slate-100 text-slate-500 ring-slate-200`}
          >
            {part.value}
          </span>
        );
      })}
    </>
  );
}

function EvidenceCard({ item, highlighted }) {
  const [expanded, setExpanded] = useState(false);
  const isLive = item.type === "live";
  const body = isLive ? item.summary : item.text;
  const long = (body ?? "").length > 260;

  return (
    <li
      id={`ev-${item.eid}`}
      data-evidence-id={item.eid}
      className={`card p-4 text-sm transition ${highlighted ? "animate-flash border-brand-400" : ""}`}
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex-none rounded-md bg-ink-800 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-white">
          {item.eid}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1">
            {isLive ? (
              <a
                href={item.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-start gap-1 font-medium leading-snug text-brand-800 hover:underline"
              >
                {item.title}
                <ExternalIcon className="mt-0.5 h-3.5 w-3.5" />
              </a>
            ) : (
              <span className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500">
                <BookIcon className="h-3.5 w-3.5" />
                {item.source}
              </span>
            )}
            <SourceBadge item={item} />
          </div>

          <p className={`mt-1.5 leading-relaxed text-slate-700 ${long && !expanded ? "line-clamp-3" : ""}`}>{body}</p>
          {long && (
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className="mt-1 text-xs font-medium text-brand-700 hover:underline"
            >
              {expanded ? "Show less" : "Show more"}
            </button>
          )}

          <p className="mt-2 text-xs text-slate-500">
            {isLive ? (
              <>
                {item.source}
                {item.publication_date ? ` · ${item.publication_date}` : ""} · evidence level{" "}
                <span className="font-medium text-slate-700">{item.evidence_level}</span>
              </>
            ) : (
              <>
                relevance <span className="font-medium tabular-nums text-slate-700">{Math.round((item.score ?? 0) * 100)}%</span>
              </>
            )}
          </p>
        </div>
      </div>
    </li>
  );
}
