/**
 * Source-type badge for one evidence item (Phase 10: offline KB / live /
 * clinician-verified text precedent / visual precedent).
 *
 * FinalReport.evidence is a flat list of dicts the orchestrator tags with
 * `type: "rag" | "live"`. Phase 13's case memory does NOT get its own type:
 * run_rag() folds those entries into the RAG list and tags the `source`
 * string with a prefix instead ("clinician_verified:<reviewer>" for the
 * text collection, "visual_precedent:<reviewer>" for the image collection),
 * so the badge has to look at the prefix as well as `type`.
 */
export function classifyEvidenceSource(item) {
  const source = typeof item?.source === "string" ? item.source : "";
  if (source.startsWith("clinician_verified:")) return "clinician-verified";
  if (source.startsWith("visual_precedent:")) return "visual-precedent";
  if (item?.type === "live") return "live";
  return "offline-kb";
}

const BADGE_STYLES = {
  "offline-kb": {
    label: "Offline KB",
    hint: "Retrieved from the curated offline knowledge base",
    className: "bg-slate-100 text-slate-700 ring-slate-500/20",
  },
  live: {
    label: "Live evidence",
    hint: "Fetched live from a public literature / trial source",
    className: "bg-sky-50 text-sky-700 ring-sky-600/20",
  },
  "clinician-verified": {
    label: "Clinician-verified precedent",
    hint: "A past clinician override on a similar-sounding case — a precedent, not guaranteed correct",
    className: "bg-violet-50 text-violet-700 ring-violet-600/20",
  },
  "visual-precedent": {
    label: "Visual precedent",
    hint: "A past clinician override on a similar-looking image — a precedent, not guaranteed correct",
    className: "bg-fuchsia-50 text-fuchsia-700 ring-fuchsia-600/20",
  },
};

export default function SourceBadge({ item }) {
  const kind = classifyEvidenceSource(item);
  const { label, hint, className } = BADGE_STYLES[kind];
  return (
    <span
      data-source-kind={kind}
      title={hint}
      className={`inline-flex flex-none items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${className}`}
    >
      {label}
    </span>
  );
}
