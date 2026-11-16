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
    label: "Reference library",
    hint: "From the curated reference library (MedlinePlus and PubMed texts held on this system)",
    className: "bg-slate-100 text-slate-700 ring-slate-500/20",
  },
  live: {
    label: "Recent research",
    hint: "Fetched for this case from PubMed, Europe PMC, ClinicalTrials.gov or MedlinePlus",
    className: "bg-sky-50 text-sky-800 ring-sky-600/20",
  },
  "clinician-verified": {
    label: "Past clinician decision",
    hint: "A clinician overrode the diagnosis on a similar case. A precedent to weigh, not guaranteed correct",
    className: "bg-violet-50 text-violet-800 ring-violet-600/20",
  },
  "visual-precedent": {
    label: "Past decision, similar image",
    hint: "A clinician overrode the diagnosis on a similar-looking image. A precedent to weigh, not guaranteed correct",
    className: "bg-fuchsia-50 text-fuchsia-800 ring-fuchsia-600/20",
  },
};

export default function SourceBadge({ item }) {
  const kind = classifyEvidenceSource(item);
  const { label, hint, className } = BADGE_STYLES[kind];
  return (
    <span
      data-source-kind={kind}
      title={hint}
      className={`inline-flex flex-none items-center rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${className}`}
    >
      {label}
    </span>
  );
}
