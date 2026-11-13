import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, submitReview } from "../services/api.js";
import Review from "./Review.jsx";

vi.mock("../services/api.js", async (importOriginal) => ({ ...(await importOriginal()), submitReview: vi.fn() }));

const REPORT = {
  case_id: "case-77",
  confidence: 0.4,
  review_required: true,
  iteration_count: 3,
  live_evidence_available: true,
  conflicts: [],
  imaging: null,
  evidence: [],
  diagnoses: [
    { name: "Anemia", confidence: 0.65, rank: 1, supporting_evidence: [] },
    { name: "Orthostatic hypotension", confidence: 0.5, rank: 2, supporting_evidence: [] },
  ],
};

const setup = (props = {}) => {
  const handlers = { onBack: vi.fn(), onReviewed: vi.fn() };
  render(<Review report={REPORT} submittedSummary={null} {...handlers} {...props} />);
  return handlers;
};
const choose = (name) => userEvent.click(screen.getByRole("radio", { name: new RegExp(name) }));
const submit = () => userEvent.click(screen.getByRole("button", { name: /Submit review/ }));
// the page also carries the report's own "review required" alert, so scope to the decision panel
const panelAlert = () => within(screen.getByTestId("decision-panel")).getByRole("alert");
const type = (label, text) => userEvent.type(screen.getByLabelText(label), text);

beforeEach(() => {
  submitReview.mockReset();
  submitReview.mockImplementation(async (id, review) => ({ case_id: id, ...review, status: "reviewed" }));
});

describe("Review: validation", () => {
  it("requires a reviewer name", async () => {
    setup();
    await submit();
    expect(panelAlert()).toHaveTextContent("Reviewer name is required.");
    expect(submitReview).not.toHaveBeenCalled();
  });

  it("requires the replacement diagnosis when overriding", async () => {
    setup();
    await choose("Override");
    await type("Reviewer name", "Dr. Rao");
    await submit();
    expect(panelAlert()).toHaveTextContent(/diagnosis you're overriding to/);
    expect(submitReview).not.toHaveBeenCalled();
  });

  it("requires a note when annotating", async () => {
    setup();
    await choose("Annotate");
    await type("Reviewer name", "Dr. Rao");
    await submit();
    expect(panelAlert()).toHaveTextContent("An annotation note is required.");
    expect(submitReview).not.toHaveBeenCalled();
  });
});

describe("Review: what is sent", () => {
  it("accepts with no annotation and no rating by default", async () => {
    setup();
    await type("Reviewer name", "  Dr. Rao ");
    await submit();
    expect(submitReview).toHaveBeenCalledWith("case-77", { decision: "accept", annotation: null, reviewer: "Dr. Rao", rating: null });
  });

  it("writes an override as 'Override diagnosis: <name>', using a suggested diagnosis chip", async () => {
    setup();
    await choose("Override");
    await userEvent.click(screen.getByRole("button", { name: "Orthostatic hypotension" }));
    expect(screen.getByLabelText("Replacement diagnosis")).toHaveValue("Orthostatic hypotension");
    await type("Reviewer name", "Dr. Rao");
    await submit();
    expect(submitReview).toHaveBeenCalledWith("case-77", {
      decision: "override", annotation: "Override diagnosis: Orthostatic hypotension", reviewer: "Dr. Rao", rating: null,
    });
  });

  it("accepts a typed replacement diagnosis", async () => {
    setup();
    await choose("Override");
    await type("Replacement diagnosis", "Vitamin B12 deficiency");
    await type("Reviewer name", "Dr. Rao");
    await submit();
    expect(submitReview.mock.calls[0][1].annotation).toBe("Override diagnosis: Vitamin B12 deficiency");
  });

  it("sends the note for an annotation", async () => {
    setup();
    await choose("Annotate");
    await type("Note", "Recommend follow-up MRI.");
    await type("Reviewer name", "Dr. Rao");
    await submit();
    expect(submitReview.mock.calls[0][1]).toMatchObject({ decision: "annotate", annotation: "Recommend follow-up MRI." });
  });

  it("sends a 1-5 rating, and clicking the same star again clears it", async () => {
    setup();
    await userEvent.click(screen.getByRole("radio", { name: "4 stars" }));
    expect(screen.getByRole("radio", { name: "4 stars" })).toHaveAttribute("aria-checked", "true");
    await userEvent.click(screen.getByRole("radio", { name: "4 stars" }));
    expect(screen.getByRole("radio", { name: "4 stars" })).toHaveAttribute("aria-checked", "false");
    await userEvent.click(screen.getByRole("radio", { name: "2 stars" }));
    await type("Reviewer name", "Dr. Rao");
    await submit();
    expect(submitReview.mock.calls[0][1].rating).toBe(2);
  });
});

describe("Review: outcomes", () => {
  it("shows the closed-case confirmation and returns to the report", async () => {
    const { onReviewed } = setup();
    await userEvent.click(screen.getByRole("radio", { name: "5 stars" }));
    await type("Reviewer name", "Dr. Rao");
    await submit();
    expect(await screen.findByText("Case closed")).toBeInTheDocument();
    expect(screen.getByText(/Dr\. Rao recorded/)).toBeInTheDocument();
    expect(screen.getByText("Review confidence: 5 / 5")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Back to report/ }));
    expect(onReviewed).toHaveBeenCalledTimes(1);
  });

  it("keeps the form and shows the failure when the server rejects the review", async () => {
    submitReview.mockRejectedValueOnce(new ApiError("Case already reviewed", { kind: "validation", status: 422 }));
    setup();
    await type("Reviewer name", "Dr. Rao");
    await submit();
    expect(await within(screen.getByTestId("decision-panel")).findByRole("alert")).toHaveTextContent("Case already reviewed");
    expect(screen.getByLabelText("Reviewer name")).toHaveValue("Dr. Rao");
    expect(screen.queryByText("Case closed")).not.toBeInTheDocument();
  });

  it("the back button calls onBack", async () => {
    const { onBack } = setup();
    await userEvent.click(screen.getByRole("button", { name: /Back to report/ }));
    expect(onBack).toHaveBeenCalledTimes(1);
  });
});
