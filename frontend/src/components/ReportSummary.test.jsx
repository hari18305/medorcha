import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import ReportSummary, { assignEvidenceIds, isImageConflict, isInsufficientInputNote, splitCitations } from "./ReportSummary.jsx";

const rag = (text, score = 0.8, source = "MedlinePlus: Hypertension") => ({ type: "rag", text, score, source });
const live = (title, extra = {}) => ({
  type: "live", title, summary: `${title} abstract`, source: "PubMed", url: "https://example.org/x", evidence_level: "high", ...extra,
});

const REPORT = {
  case_id: "C1",
  confidence: 0.72,
  review_required: false,
  iteration_count: 0,
  live_evidence_available: true,
  conflicts: [],
  imaging: null,
  diagnoses: [
    { name: "Tension headache", confidence: 0.4, rank: 2, supporting_evidence: ["Common cause [RAG2]."] },
    { name: "Hypertensive emergency", confidence: 0.65, rank: 1, supporting_evidence: ["Severe headache [RAG1] and trials [LIVE1], see [LIVE9]."] },
  ],
  evidence: [rag("Severe hypertension can cause headache."), rag("Tension headache is common.", 0.6), live("A trial of blood pressure control")],
};

const renderReport = (over = {}, props = {}) => render(<ReportSummary report={{ ...REPORT, ...over }} {...props} />);

describe("ReportSummary: pure helpers", () => {
  it("numbers offline and live evidence separately, in list order", () => {
    const ids = assignEvidenceIds([rag("a"), live("b"), rag("c"), live("d")]).map((e) => e.eid);
    expect(ids).toEqual(["RAG1", "LIVE1", "RAG2", "LIVE2"]);
  });

  it("splits citation markers out of text", () => {
    expect(splitCitations("x [RAG1] y [LIVE2][MEM3] z")).toEqual([
      { kind: "text", value: "x " }, { kind: "cite", value: "RAG1" }, { kind: "text", value: " y " },
      { kind: "cite", value: "LIVE2" }, { kind: "cite", value: "MEM3" }, { kind: "text", value: " z" },
    ]);
    expect(splitCitations(null)).toEqual([]);
  });

  it("recognises the two kinds of note it must keep out of the conflicts list", () => {
    expect(isInsufficientInputNote("Insufficient clinical input: nothing to reason from")).toBe(true);
    expect(isInsufficientInputNote("Non-specific clinical input: only general symptoms")).toBe(true);
    expect(isInsufficientInputNote("Live evidence disagrees with the knowledge base")).toBe(false);
    expect(isImageConflict("BiomedCLIP pre-filter suggests infarct, but the text says otherwise")).toBe(true);
    expect(isImageConflict("Two sources disagree on dosing")).toBe(false);
  });
});

describe("ReportSummary: differential", () => {
  it("orders diagnoses by rank, not by list position", () => {
    renderReport();
    const names = screen.getAllByRole("heading", { level: 4 }).map((h) => h.textContent);
    expect(names).toEqual(["Hypertensive emergency", "Tension headache"]);
  });

  it("says so when there are no diagnoses", () => {
    renderReport({ diagnoses: [] });
    expect(screen.getByText(/No diagnoses were produced/)).toBeInTheDocument();
  });

  it("links a citation to its evidence item and shows an unknown citation as inert", async () => {
    renderReport();
    const chip = screen.getByRole("button", { name: "Library 1" });
    await userEvent.click(chip);
    await waitFor(() => expect(Element.prototype.scrollIntoView).toHaveBeenCalled());
    expect(document.getElementById("ev-RAG1")).toBeInTheDocument();

    // [LIVE9] is not in the evidence list: rendered, but not a button
    const unknown = document.querySelector('[data-citation="LIVE9"]');
    expect(unknown.tagName).toBe("SPAN");
    expect(screen.queryByRole("button", { name: "Research 9" })).not.toBeInTheDocument();
  });
});

describe("ReportSummary: review notice and input notes", () => {
  it("shows the review notice, and the headline badge, only when review is required", () => {
    const { unmount } = renderReport();
    expect(screen.queryByTestId("review-notice")).not.toBeInTheDocument();
    expect(screen.getAllByText("Above threshold").length).toBeGreaterThan(0);
    unmount();

    renderReport({ review_required: true });
    expect(screen.getByTestId("review-notice")).toHaveTextContent("Clinician review required");
    expect(screen.getAllByText("Review required").length).toBeGreaterThan(0);
  });

  it("moves the input-sufficiency note out of 'Conflicting evidence' into the review notice", () => {
    const note = "Insufficient clinical input: no symptoms, laboratory values or report findings were provided.";
    renderReport({ review_required: true, conflicts: [note, "PubMed and the KB disagree on first-line therapy"] });
    expect(within(screen.getByTestId("insufficient-input-note")).getByText(note)).toBeInTheDocument();
    const conflicts = screen.getByRole("heading", { name: /Conflicting evidence/ }).closest("section");
    expect(within(conflicts).getByText("PubMed and the KB disagree on first-line therapy")).toBeInTheDocument();
    expect(within(conflicts).queryByText(note)).not.toBeInTheDocument();
  });

  it("treats the non-specific-input note the same way", () => {
    const note = "Non-specific clinical input: the only symptoms reported are general ones.";
    renderReport({ review_required: true, conflicts: [note] });
    expect(screen.getByTestId("insufficient-input-note")).toHaveTextContent(note);
    expect(screen.queryByRole("heading", { name: /Conflicting evidence/ })).not.toBeInTheDocument();
  });

  it("reports skipped refinement when the case had no clinical input", () => {
    renderReport({ review_required: true, conflicts: ["Insufficient clinical input: nothing given."] });
    expect(screen.getByText("Skipped (no clinical input)")).toBeInTheDocument();
  });

  it("tags an image-versus-text conflict in plain words", () => {
    renderReport({ conflicts: ["BiomedCLIP pre-filter suggests hemorrhage, the text does not"] });
    expect(screen.getByText("Image vs. text")).toBeInTheDocument();
  });

  it("shows the plain-language reasoning-failure sentence as given", () => {
    const sentence = "The reasoning step could not be completed (the language-model provider's request limit was reached), so no diagnosis is shown. A clinician must review this case; retrying later may help.";
    renderReport({ diagnoses: [], conflicts: [sentence], review_required: true, confidence: 0 });
    expect(screen.getByText(sentence)).toBeInTheDocument();
    expect(screen.queryByText(/RESOURCE_EXHAUSTED/)).not.toBeInTheDocument();
  });
});

describe("ReportSummary: evidence list", () => {
  it("filters evidence by source and keeps empty categories disabled", async () => {
    renderReport();
    expect(screen.getByRole("tab", { name: /All\s*3/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Past decisions\s*0/ })).toBeDisabled();

    await userEvent.click(screen.getByRole("tab", { name: /Recent research\s*1/ }));
    expect(document.querySelectorAll("[data-evidence-id]")).toHaveLength(1);
    expect(document.getElementById("ev-LIVE1")).toBeInTheDocument();
  });

  it("classifies clinician precedents from the source prefix", () => {
    renderReport({ evidence: [rag("old case", 0.7, "clinician_verified:Dr. Lee"), rag("old image", 0.9, "visual_precedent:Dr. Lee")] });
    expect(screen.getByRole("tab", { name: /Past decisions\s*2/ })).toBeEnabled();
    expect(screen.getByText("Past clinician decision")).toBeInTheDocument();
    expect(screen.getByText("Past decision, similar image")).toBeInTheDocument();
  });

  it("warns that live literature was unavailable instead of implying it was checked", () => {
    renderReport({ live_evidence_available: false });
    expect(screen.getByText(/Live literature evidence was unavailable/)).toBeInTheDocument();
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
  });

  it("says when no evidence was retrieved", () => {
    renderReport({ evidence: [] });
    expect(screen.getByText(/No supporting evidence was retrieved/)).toBeInTheDocument();
  });

  it("collapses long evidence text behind Show more", async () => {
    renderReport({ evidence: [rag("x".repeat(400))] });
    await userEvent.click(screen.getByRole("button", { name: "Show more" }));
    expect(screen.getByRole("button", { name: "Show less" })).toBeInTheDocument();
  });
});

describe("ReportSummary: imaging", () => {
  it("shows nothing about imaging when no image was submitted", () => {
    renderReport();
    expect(document.querySelector('[data-section="imaging"]')).toBeNull();
    expect(screen.getByText("None submitted")).toBeInTheDocument();
  });

  it("shows assessed findings with the zero-shot caveat", () => {
    renderReport({ imaging: { modality: "chest X-ray", findings: ["pneumothorax"], confidence: 0.9 } }, { imageSubmitted: true });
    expect(document.querySelector('[data-imaging-state="assessed"]')).toBeInTheDocument();
    expect(screen.getByText("pneumothorax")).toBeInTheDocument();
    expect(screen.getByText(/not a diagnostic read/)).toBeInTheDocument();
  });

  it("shows an unsupported modality as not assessed, never as a finding", () => {
    renderReport({ imaging: { modality: "ultrasound", findings: ["no findings model available"], confidence: 0.5, unsupported_modality: true } }, { imageSubmitted: true });
    expect(document.querySelector('[data-imaging-state="not-assessed"]')).toBeInTheDocument();
    expect(screen.getByText("Not assessed")).toBeInTheDocument();
    expect(screen.queryByText("no findings model available")).not.toBeInTheDocument();
  });

  it("shows an indecisive read as inconclusive: says so, shows the score, and does not present the labels as findings", () => {
    renderReport({ imaging: { modality: "MRI brain scan", findings: ["edema", "cerebral atrophy"], confidence: 0.34, inconclusive: true } }, { imageSubmitted: true });
    const section = document.querySelector('[data-imaging-state="inconclusive"]');
    expect(section).toBeInTheDocument();
    expect(section).toHaveTextContent("Inconclusive read");
    expect(section).toHaveTextContent("34%");
    expect(section).toHaveTextContent("not used in the evidence search");
    expect(document.querySelector('[data-imaging-state="assessed"]')).toBeNull();
    expect(screen.getByText("Inconclusive")).toBeInTheDocument(); // the run-summary line
  });

  it("shows a read the second image check contradicted as contested, not as a finding", () => {
    renderReport({ imaging: { modality: "CT scan", findings: ["sinus abnormality"], confidence: 0.89, contested: true, cross_check_label: "normal head CT, no acute findings" } }, { imageSubmitted: true });
    const section = document.querySelector('[data-imaging-state="contested"]');
    expect(section).toBeInTheDocument();
    expect(section).toHaveTextContent("Contested read");
    expect(section).toHaveTextContent("normal head CT, no acute findings");
    expect(section).toHaveTextContent("Neither was used in the evidence search");
    expect(document.querySelector('[data-imaging-state="assessed"]')).toBeNull();
    expect(screen.getByText("Contested")).toBeInTheDocument(); // the run-summary line
  });

  it("explains an image that produced no findings", () => {
    renderReport({ imaging: null }, { imageSubmitted: true });
    expect(screen.getByText(/no imaging findings were recorded/)).toBeInTheDocument();
    expect(screen.getByText("No findings")).toBeInTheDocument();
  });
});

const EXPLANATION = {
  headline: "Leading consideration: Hypertensive emergency, supported by 2 cited sources. This result needs clinician review before it is relied on.",
  review_required: true,
  review_reasons: ["The combined evidence score (58%) is below the 65% review line mainly because the literature support was limited or missing."],
  cautions: ["The search was refined 2 times to try to strengthen the evidence."],
  next_steps: ["Check the flagged disagreements against the source documents."],
  score: {
    final: 58, raw: 58, review_line: 65, above_review_line: false, cap: null,
    parts: [
      { key: "reference", label: "Reference library", max: 40, earned: 28, lost: 12, level: "moderate", detail: "The curated reference texts partly match this presentation.", why_lost: "The reference texts matched this presentation only partly.", how_to_raise: "Add specific symptoms, examination findings or laboratory values so the library can match more precisely." },
      { key: "literature", label: "Published research", max: 30, earned: 0, lost: 30, level: "none", detail: "The search found no matching publications or trials.", why_lost: "No matching publications or trials were found.", how_to_raise: "More specific history and findings help the search reach stronger studies." },
      { key: "agreement", label: "Agreement between sources", max: 30, earned: 30, lost: 0, level: "consistent", detail: "No contradictions were flagged between the evidence sources.", why_lost: null, how_to_raise: null },
    ],
    meaning: {
      is: "58 out of 100 evidence points: how well the evidence the system found supports this result.",
      is_not: "It is not the chance that the diagnosis is correct. A correct diagnosis can score low when the evidence found is thin.",
      remainder: "The other 42 points were not earned: evidence the system did not find, could not confirm, or that was capped by a safety rule. They are not a 42% chance of error.",
      review_line: "The review line is 65 points. Below it, a clinician must check the result before it is used.",
    },
  },
  findings: { present: ["severe headache"], negatives: ["No fever"], history: ["hypertension"], labs: ["BP: 190/120 mmHg"], report_findings: [] },
  imaging: { status: "normal", normal: true, statement: "Chest X-ray: no acute abnormality identified.", caveat: "An automated screening aid, not a radiologist's report." },
};

describe("ReportSummary: clinician explanation", () => {
  it("shows the result, why review is needed, cautions and what would help", () => {
    renderReport({ explanation: EXPLANATION, review_required: true });
    const summary = screen.getByTestId("explanation-summary");
    expect(summary).toHaveTextContent("Leading consideration: Hypertensive emergency");
    expect(summary).toHaveTextContent("Why review is needed");
    expect(summary).toHaveTextContent("58%");
    expect(summary).toHaveTextContent("Worth knowing");
    expect(summary).toHaveTextContent("What would help");
  });

  it("lists normal findings and pertinent negatives together, apart from reported symptoms", () => {
    renderReport({ explanation: EXPLANATION });
    const card = screen.getByTestId("findings-considered");
    expect(within(card.querySelector('[data-findings-group="present"]')).getByText("severe headache")).toBeInTheDocument();
    const normal = card.querySelector('[data-findings-group="normal"]');
    expect(within(normal).getByText("No fever")).toBeInTheDocument();
    expect(within(normal).getByText("Chest X-ray: no acute abnormality identified.")).toBeInTheDocument();
    expect(card.querySelector('[data-findings-group="labs"]')).toHaveTextContent("BP: 190/120 mmHg");
  });

  it("takes the score apart: points earned and lost per source, adding up to the headline number", () => {
    renderReport({ explanation: EXPLANATION, review_required: true });
    const ledger = screen.getByTestId("score-ledger");
    expect(screen.getByTestId("score-final")).toHaveTextContent("58");
    const row = (key) => ledger.querySelector(`[data-score-part="${key}"]`);
    expect(row("reference")).toHaveTextContent("28");
    expect(row("reference")).toHaveTextContent("of 40 points");
    expect(row("reference").querySelector("[data-lost]")).toHaveTextContent("−12");
    expect(row("literature").querySelector("[data-lost]")).toHaveTextContent("−30");
    expect(row("agreement").querySelector("[data-lost]")).toBeNull(); // nothing lost, nothing flagged
    // a lost part says why; the way to raise it is one click away
    expect(row("literature")).toHaveTextContent("No matching publications or trials were found.");
    expect(row("literature")).toHaveTextContent("To raise it:");
    // the lost points across the parts are exactly the missing points of the headline
    const lost = [...ledger.querySelectorAll("[data-lost]")].map((n) => Number(n.textContent.replace("−", "")));
    expect(lost.reduce((a, b) => a + b, 0)).toBe(100 - 58);
  });

  it("says what the number means, what it does not mean and what the missing points are", () => {
    renderReport({ explanation: EXPLANATION, review_required: true });
    const meaning = screen.getByTestId("score-meaning");
    expect(meaning).toHaveTextContent("What 58 means");
    expect(meaning).toHaveTextContent("It is not the chance that the diagnosis is correct");
    expect(meaning).toHaveTextContent("What about the other 42?");
    expect(meaning).toHaveTextContent("not a 42% chance of error");
    expect(meaning).toHaveTextContent("65 points");
  });

  it("draws the score as a bar with the review line marked, and describes it for screen readers", () => {
    renderReport({ explanation: EXPLANATION });
    const bar = screen.getByTestId("score-bar");
    expect(bar).toHaveTextContent("Review line 65");
    expect(bar.querySelector('[role="img"]')).toHaveAttribute("aria-label", expect.stringContaining("58 of 100 evidence points"));
    expect(bar.querySelectorAll("[data-bar-part]")).toHaveLength(3);
  });

  it("shows a safety cap as points removed, with its reason", () => {
    const capped = { ...EXPLANATION, score: { ...EXPLANATION.score, final: 50, raw: 93, cap: { points_removed: 43, reason: "Capped at 50: no presenting symptom could be identified." } } };
    renderReport({ explanation: capped, review_required: true });
    expect(screen.getByTestId("score-cap")).toHaveTextContent("Safety rule: −43 points");
    expect(screen.getByTestId("score-cap")).toHaveTextContent("Capped at 50");
    // the bar marks the removed points so it agrees with the headline number (50), not with the parts (93)
    expect(screen.getByTestId("score-cap-overlay").style.left).toBe("50%");
    expect(screen.getByTestId("score-cap-overlay").style.width).toBe("43%");
    renderReport({ explanation: EXPLANATION });
    expect(screen.queryAllByTestId("score-cap")).toHaveLength(1); // only the capped render above has one
  });

  it("flags review in the status chip from the report, not from the number", () => {
    const { unmount } = renderReport({ explanation: EXPLANATION, review_required: true });
    expect(screen.getByTestId("status-chip")).toHaveTextContent("Review required");
    unmount();
    renderReport({ explanation: EXPLANATION, review_required: false });
    expect(screen.getByTestId("status-chip")).toHaveTextContent("Above the review line");
  });

  it("puts the differential straight after the result, before the supporting detail", () => {
    renderReport({ explanation: EXPLANATION });
    const order = ["explanation-summary", "findings-considered"].map((id) => screen.getByTestId(id));
    const firstDiagnosis = document.querySelector("[data-rank]");
    // result card, then the differential, then the findings card
    expect(order[0].compareDocumentPosition(firstDiagnosis) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(firstDiagnosis.compareDocumentPosition(order[1]) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("previews a cited source on hover so the proof is next to the claim", () => {
    renderReport();
    const chip = screen.getByRole("button", { name: "Library 1" });
    expect(chip.getAttribute("title")).toContain("Severe hypertension can cause headache.");
    expect(chip.getAttribute("title")).toContain("MedlinePlus: Hypertension");
  });

  it("shows why a diagnosis was ranked, only when the system gave findings", () => {
    const diagnoses = [
      { name: "Hypertensive emergency", confidence: 0.65, rank: 1, supporting_evidence: ["[RAG1] a"], findings_for: ["severe headache"], findings_against: ["no visual symptoms reported"] },
      { name: "Tension headache", confidence: 0.4, rank: 2, supporting_evidence: [] },
    ];
    renderReport({ diagnoses });
    const why = screen.getAllByTestId("why-this-ranking");
    expect(why).toHaveLength(1); // the second diagnosis has no explanation lists
    expect(why[0]).toHaveTextContent("Fits this case");
    expect(why[0]).toHaveTextContent("severe headache");
    expect(why[0]).toHaveTextContent("Against, or not reported");
    expect(why[0]).toHaveTextContent("no visual symptoms reported");
  });

  it("states how many sources were cited instead of showing the model's own figure as a bar", () => {
    renderReport({
      diagnoses: [
        { name: "A", confidence: 0.9, rank: 1, supporting_evidence: ["[RAG1] x", "[LIVE1] y"] },
        { name: "B", confidence: 0.5, rank: 2, supporting_evidence: ["[RAG2] z"] },
        { name: "C", confidence: 0.3, rank: 3, supporting_evidence: [] },
      ],
    });
    expect(screen.getByText("2 cited sources")).toBeInTheDocument();
    expect(screen.getByText("1 cited source")).toBeInTheDocument();
    expect(screen.getByText("No source cited")).toBeInTheDocument();
    expect(screen.queryByText(/Model-stated confidence/)).not.toBeInTheDocument();
  });

  it("still renders a report stored before explanations existed", () => {
    renderReport({ explanation: undefined });
    expect(screen.queryByTestId("explanation-summary")).not.toBeInTheDocument();
    expect(screen.queryByTestId("findings-considered")).not.toBeInTheDocument();
    expect(screen.queryByTestId("score-ledger")).not.toBeInTheDocument();
    expect(screen.getByTestId("confidence-gauge")).toBeInTheDocument(); // the plain gauge stands in
    expect(screen.getAllByRole("heading", { level: 4 })).toHaveLength(2);
  });

  it("keeps internal names out of the reader-facing text", () => {
    renderReport({ explanation: EXPLANATION });
    const text = document.body.textContent;
    for (const jargon of ["BiomedCLIP", "zero-shot", "RAG1", "LIVE1", "Offline KB"]) {
      expect(text).not.toContain(jargon);
    }
  });
});

const LEADING = { name: "Hypertensive emergency", cited_sources: 2 };

describe("ReportSummary: reading order and progressive disclosure", () => {
  it("leads with the leading diagnosis and the review status, ahead of any score detail", () => {
    renderReport({
      explanation: { ...EXPLANATION, leading: LEADING, status_line: "This result needs clinician review before it is relied on." },
      review_required: true,
    });
    const card = screen.getByTestId("explanation-summary");
    expect(screen.getByTestId("leading-diagnosis")).toHaveTextContent("Hypertensive emergency");
    expect(card).toHaveTextContent("Supported by 2 cited sources");
    expect(card).toHaveTextContent("This result needs clinician review before it is relied on.");
    // the headline comes before the score, which comes before the reasons
    const pos = (el) => [...card.querySelectorAll("*")].indexOf(el);
    expect(pos(screen.getByTestId("leading-diagnosis"))).toBeLessThan(pos(screen.getByTestId("score-ledger")));
  });

  it("keeps the score breakdown one click away rather than open by default", () => {
    renderReport({ explanation: { ...EXPLANATION, leading: LEADING }, review_required: true });
    const details = screen.getByTestId("score-details");
    expect(details.open).toBe(false);
    expect(details).toHaveTextContent("How was this score made, and what do the other 42 points mean?");
    // the number, the bar and the not-a-probability line are visible without opening it
    expect(screen.getByTestId("score-final")).toHaveTextContent("58");
    expect(screen.getByTestId("score-bar")).toBeInTheDocument();
    expect(screen.getByTestId("score-ledger")).toHaveTextContent("not the chance that the diagnosis is correct");
  });

  it("opens the first diagnosis and leaves the others collapsed", () => {
    renderReport();
    const cards = [...document.querySelectorAll("details[data-rank]")];
    expect(cards.map((d) => d.getAttribute("data-rank"))).toEqual(["1", "2"]);
    expect(cards.map((d) => d.open)).toEqual([true, false]);
  });

  it("collapses the evidence list until it is wanted, and a citation click opens it", async () => {
    renderReport();
    const evidence = screen.getByTestId("evidence-section").querySelector("details");
    expect(evidence.open).toBe(false);
    expect(evidence).toHaveTextContent("Supporting evidence");
    await userEvent.click(screen.getByRole("button", { name: "Library 1" }));
    await waitFor(() => expect(evidence.open).toBe(true));
  });

  it("keeps the unavailable-literature warning visible even though the evidence list is collapsed", () => {
    renderReport({ live_evidence_available: false });
    const warning = screen.getByText(/Live literature evidence was unavailable/);
    expect(warning.closest("details")).toBeNull();
  });

  it("opens everything for printing and restores the on-screen state afterwards", () => {
    renderReport();
    const all = () => [...document.querySelectorAll("details")];
    const before = all().map((d) => d.open);
    expect(before.some((open) => !open)).toBe(true);

    window.dispatchEvent(new Event("beforeprint"));
    expect(all().every((d) => d.open)).toBe(true);

    window.dispatchEvent(new Event("afterprint"));
    expect(all().map((d) => d.open)).toEqual(before);
  });
});
