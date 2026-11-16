// Wording helpers: the interface speaks to clinicians, so internal names (RAG, LIVE, BiomedCLIP ...)
// are translated here, in one place, and nowhere else. The data keys and ids that tests and the
// backend rely on (RAG1, LIVE2, data-citation ...) are NOT changed; only what the reader sees.

const REFERENCE_NAMES = { RAG: "Library", LIVE: "Research", MEM: "Past case", VIS: "Past case" };

/** "RAG1" -> "Library 1", "LIVE2" -> "Research 2", "MEM3" / "VIS1" -> "Past case n". Unknown ids pass through. */
export function evidenceLabel(eid) {
  const match = /^(RAG|LIVE|MEM|VIS)(\d+)$/.exec(eid ?? "");
  if (!match) return eid ?? "";
  return `${REFERENCE_NAMES[match[1]]} ${match[2]}`;
}

/**
 * How many sources the system cited for a diagnosis. This is a count, not a quality grade, so it is worded
 * as a count: the system's own per-diagnosis figure is not a probability and is not shown as a bar.
 */
export function citationSupport(diagnosis) {
  const n = (diagnosis?.supporting_evidence ?? []).length;
  if (n === 0) return { level: "none", label: "No source cited" };
  return { level: n >= 2 ? "several" : "one", label: `${n} cited source${n === 1 ? "" : "s"}` };
}

// What an evidence level means in clinical terms. The level is assigned from the type of publication, so it
// describes the kind of source, not how well it fits this patient.
const LEVEL_WORDS = {
  high: "guideline, systematic review or trial",
  moderate: "review, observational study or registered trial",
  low: "other report or overview",
};

export function evidenceLevelWords(level) {
  return LEVEL_WORDS[level] ?? String(level ?? "unknown");
}

// How a signal level from the backend explanation is shown: a word and a colour family.
export const SIGNAL_LEVELS = {
  strong: { label: "Strong", tone: "good" },
  consistent: { label: "Consistent", tone: "good" },
  moderate: { label: "Moderate", tone: "mid" },
  limited: { label: "Limited", tone: "mid" },
  disagreement: { label: "Disagreement", tone: "bad" },
  none: { label: "None found", tone: "bad" },
  unavailable: { label: "Unavailable", tone: "bad" },
};

export function signalLevel(level) {
  return SIGNAL_LEVELS[level] ?? { label: String(level ?? "Unknown"), tone: "mid" };
}

/** True when the explanation carries anything worth drawing (older stored reports have none). */
export function hasExplanation(explanation) {
  return Boolean(explanation && typeof explanation === "object" && explanation.headline);
}

/** The findings groups that actually have content, in reading order, with the imaging normal finding merged in. */
export function findingGroups(explanation) {
  const f = explanation?.findings ?? {};
  const imagingNormal = explanation?.imaging?.normal ? [explanation.imaging.statement] : [];
  const groups = [
    { id: "present", title: "Symptoms and signs reported", items: f.present ?? [] },
    { id: "normal", title: "Normal findings and pertinent negatives", items: [...(f.negatives ?? []), ...imagingNormal] },
    { id: "history", title: "Past history", items: f.history ?? [] },
    { id: "labs", title: "Laboratory values", items: f.labs ?? [] },
    { id: "report", title: "Findings stated in the report", items: f.report_findings ?? [] },
  ];
  return groups.filter((g) => g.items.length > 0);
}

/**
 * Plain-text summary of a report for pasting into notes. Mirrors what the screen says, including the
 * safety wording, so a copied result never loses the "research prototype" and "review required" context.
 */
export function buildSummaryText(report, submitted = {}) {
  const ex = report?.explanation;
  const score = ex?.score;
  const who = [submitted?.sex, submitted?.age ? `${submitted.age} years` : null].filter(Boolean).join(", ");
  const lines = [
    "MedOrchestrate decision-support summary",
    "Research prototype: supports, never replaces, clinical judgement. Not a medical device.",
    "",
    `Case: ${report?.case_id ?? "unknown"}`,
  ];
  if (who) lines.push(`Patient: ${who}`);
  lines.push(
    report?.review_required
      ? "Review: REQUIRED. A clinician must check this result before it is used."
      : "Review: not flagged by the system; clinical judgement still applies."
  );
  if (score) {
    lines.push(
      `Evidence score: ${score.final} of 100 (review line ${score.review_line}). ` +
        "This is how well the evidence supports the result, not the chance it is correct."
    );
  } else if (typeof report?.confidence === "number") {
    lines.push(`Evidence confidence: ${Math.round(report.confidence * 100)}% (not a probability).`);
  }
  if (ex?.headline) lines.push("", ex.headline);

  const diagnoses = [...(report?.diagnoses ?? [])].sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0));
  if (diagnoses.length) {
    lines.push("", "Differential diagnosis:");
    diagnoses.forEach((d, i) => {
      lines.push(`${i + 1}. ${d.name} (${citationSupport(d).label})`);
      if (d.findings_for?.length) lines.push(`   Fits this case: ${d.findings_for.join("; ")}`);
      if (d.findings_against?.length) lines.push(`   Against, or not reported: ${d.findings_against.join("; ")}`);
    });
  }
  const section = (title, items) => {
    if (items?.length) lines.push("", `${title}:`, ...items.map((item) => `- ${item}`));
  };
  section("Why review is needed", ex?.review_reasons);
  section("Worth knowing", ex?.cautions);
  section("What would help", ex?.next_steps);
  const normals = findingGroups(ex).find((g) => g.id === "normal");
  section("Normal findings and pertinent negatives", normals?.items);
  return lines.join("\n");
}
