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
    const chip = screen.getByRole("button", { name: "RAG1" });
    await userEvent.click(chip);
    await waitFor(() => expect(Element.prototype.scrollIntoView).toHaveBeenCalled());
    expect(document.getElementById("ev-RAG1")).toBeInTheDocument();

    // [LIVE9] is not in the evidence list: rendered, but not a button
    const unknown = document.querySelector('[data-citation="LIVE9"]');
    expect(unknown.tagName).toBe("SPAN");
    expect(screen.queryByRole("button", { name: "LIVE9" })).not.toBeInTheDocument();
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

  it("tags an image-versus-BiomedCLIP conflict", () => {
    renderReport({ conflicts: ["BiomedCLIP pre-filter suggests hemorrhage, the text does not"] });
    expect(screen.getByText("Image vs. BiomedCLIP")).toBeInTheDocument();
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
    expect(screen.getByRole("tab", { name: /Precedents\s*0/ })).toBeDisabled();

    await userEvent.click(screen.getByRole("tab", { name: /Live\s*1/ }));
    expect(document.querySelectorAll("[data-evidence-id]")).toHaveLength(1);
    expect(document.getElementById("ev-LIVE1")).toBeInTheDocument();
  });

  it("classifies clinician precedents from the source prefix", () => {
    renderReport({ evidence: [rag("old case", 0.7, "clinician_verified:Dr. Lee"), rag("old image", 0.9, "visual_precedent:Dr. Lee")] });
    expect(screen.getByRole("tab", { name: /Precedents\s*2/ })).toBeEnabled();
    expect(screen.getByText("Clinician-verified precedent")).toBeInTheDocument();
    expect(screen.getByText("Visual precedent")).toBeInTheDocument();
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

  it("explains an image that produced no findings", () => {
    renderReport({ imaging: null }, { imageSubmitted: true });
    expect(screen.getByText(/no imaging findings were recorded/)).toBeInTheDocument();
    expect(screen.getByText("No findings")).toBeInTheDocument();
  });
});
