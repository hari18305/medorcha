import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import Report from "./Report.jsx";

const REPORT = {
  case_id: "C-9",
  confidence: 0.5,
  review_required: true,
  iteration_count: 0,
  live_evidence_available: true,
  conflicts: [],
  imaging: null,
  evidence: [],
  diagnoses: [{ name: "Vestibular migraine", rank: 1, confidence: 0.5, supporting_evidence: ["[RAG1] x"] }],
  explanation: {
    headline: "Leading consideration: Vestibular migraine, supported by 1 cited source. This result needs clinician review before it is relied on.",
    leading: { name: "Vestibular migraine", cited_sources: 1 },
    status_line: "This result needs clinician review before it is relied on.",
    review_required: true,
    review_reasons: ["The combined evidence score (50%) is below the 65% review line."],
    cautions: [],
    next_steps: ["Add laboratory results."],
    score: { final: 50, raw: 50, review_line: 65, above_review_line: false, cap: null, parts: [], meaning: { is: "", is_not: "", remainder: "", review_line: "" } },
    findings: { present: [], negatives: [], history: [], labs: [], report_findings: [] },
    imaging: null,
  },
};
const SUBMITTED = { age: "34", sex: "female", clinicalText: "dizziness", history: [], labs: {}, hadImage: false };

const renderReport = () => render(<Report report={REPORT} submittedSummary={SUBMITTED} onNewCase={() => {}} onReviewCase={() => {}} />);

afterEach(() => {
  vi.useRealTimers();
});

describe("Report: taking the result with you", () => {
  it("copies a plain-text summary that keeps the prototype and review warnings", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    renderReport();

    await userEvent.click(screen.getByTestId("copy-summary"));

    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    const text = writeText.mock.calls[0][0];
    expect(text).toContain("Research prototype: supports, never replaces, clinical judgement. Not a medical device.");
    expect(text).toContain("Patient: female, 34 years");
    expect(text).toContain("Review: REQUIRED");
    expect(text).toContain("1. Vestibular migraine (1 cited source)");
    expect(await screen.findByText("Copied")).toBeInTheDocument();
  });

  it("says so when the clipboard is not available instead of pretending it worked", async () => {
    Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
    document.execCommand = vi.fn(() => false); // jsdom has no execCommand; the fallback route fails too
    renderReport();

    await userEvent.click(screen.getByTestId("copy-summary"));

    expect(await screen.findByText("Copy failed")).toBeInTheDocument();
  });

  it("opens the browser's print dialog for Print / save PDF", async () => {
    window.print = vi.fn();
    renderReport();
    await userEvent.click(screen.getByTestId("print-report"));
    expect(window.print).toHaveBeenCalledTimes(1);
  });

  it("marks the action buttons so they are left out of the printout", () => {
    renderReport();
    expect(screen.getByTestId("print-report").closest(".no-print")).not.toBeNull();
    expect(document.body.textContent).toContain("Printed"); // the print-only header line is in the page
  });
});
