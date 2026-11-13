import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";
import { ApiError, getDiagnosisStatus, startDiagnosis } from "./services/api.js";

vi.mock("./services/api.js", async (importOriginal) => ({
  ...(await importOriginal()),
  startDiagnosis: vi.fn(),
  getDiagnosisStatus: vi.fn(),
  submitReview: vi.fn(),
}));

const REPORT = {
  case_id: "c-1",
  confidence: 0.72,
  review_required: false,
  iteration_count: 0,
  live_evidence_available: true,
  conflicts: [],
  imaging: null,
  evidence: [{ type: "rag", text: "Severe hypertension can cause headache.", score: 0.8, source: "MedlinePlus" }],
  diagnoses: [{ name: "Hypertensive emergency", confidence: 0.65, rank: 1, supporting_evidence: ["Severe headache [RAG1]."] }],
};
const NEEDS_REVIEW = { ...REPORT, confidence: 0.4, review_required: true, iteration_count: 3 };

async function fillAndSubmit() {
  await userEvent.type(screen.getByLabelText("Age"), "52");
  await userEvent.selectOptions(screen.getByLabelText("Sex"), "female");
  await userEvent.type(screen.getByLabelText("Presenting complaint and history"), "headache and dizziness");
  await userEvent.type(screen.getByLabelText("Past medical history"), "hypertension{Enter}");
  await userEvent.click(screen.getByRole("button", { name: /Analyze case/ }));
}

beforeEach(() => {
  startDiagnosis.mockReset();
  getDiagnosisStatus.mockReset();
});

describe("App: the case-to-report flow", () => {
  it("starts on the case form", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "Describe the patient" })).toBeInTheDocument();
  });

  it("goes from the form to the report and remembers what was submitted", async () => {
    startDiagnosis.mockResolvedValue(REPORT); // a server that answers with the finished report
    render(<App />);
    await fillAndSubmit();

    expect(await screen.findByRole("heading", { name: "Hypertensive emergency" })).toBeInTheDocument();
    expect(window.location.hash).toBe("#/report");
    expect(screen.getByText("Patient summary")).toBeInTheDocument();
    expect(screen.getByText("headache and dizziness")).toBeInTheDocument();
    expect(document.querySelector('[data-summary="history"]')).toHaveTextContent("hypertension");

    const [formData] = startDiagnosis.mock.calls[0];
    expect(formData.get("age")).toBe("52");
    expect(JSON.parse(formData.get("history"))).toEqual(["hypertension"]);
  });

  it("sends a case that needs review straight to the review screen", async () => {
    startDiagnosis.mockResolvedValue(NEEDS_REVIEW);
    render(<App />);
    await fillAndSubmit();

    expect(await screen.findByRole("heading", { name: "Review case" })).toBeInTheDocument();
    expect(window.location.hash).toBe("#/review");
    expect(screen.getByTestId("decision-panel")).toBeInTheDocument();
  });

  it("follows a queued analysis, showing the server's steps, then shows the report", async () => {
    startDiagnosis.mockResolvedValue({ case_id: "c-1", status: "queued" });
    getDiagnosisStatus
      .mockResolvedValueOnce({
        status: "running",
        steps: [{ name: "clinical", status: "done", detail: "2 symptom(s)", duration_ms: 1500 }, { name: "rag", status: "running" }],
      })
      .mockResolvedValue({ status: "completed", steps: [], report: REPORT });
    render(<App />);
    await fillAndSubmit();

    expect(await screen.findByTestId("progress-steps")).toBeInTheDocument();
    expect(screen.getByTestId("progress-message")).toHaveTextContent("Offline retrieval…");
    expect(window.location.hash).toBe("#/processing");

    // the next poll is one interval (1.2 s) later
    expect(await screen.findByRole("heading", { name: "Hypertensive emergency" }, { timeout: 4000 })).toBeInTheDocument();
    expect(getDiagnosisStatus).toHaveBeenCalledWith("c-1");
  });
});

describe("App: failures", () => {
  it("returns to the form with everything still filled in, and can retry", async () => {
    startDiagnosis.mockRejectedValueOnce(new ApiError("Could not reach the MedOrchestrate server.", { kind: "network" })).mockResolvedValue(REPORT);
    render(<App />);
    await fillAndSubmit();

    expect(await screen.findByText("Can't reach the server")).toBeInTheDocument();
    expect(screen.getByLabelText("Age")).toHaveValue(52);
    expect(screen.getByLabelText("Presenting complaint and history")).toHaveValue("headache and dizziness");
    expect(screen.getByRole("button", { name: "Remove hypertension" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("heading", { name: "Hypertensive emergency" })).toBeInTheDocument();
    expect(startDiagnosis).toHaveBeenCalledTimes(2);
  });

  it("treats a failed analysis as a server failure: generic message, retry offered, form kept", async () => {
    startDiagnosis.mockResolvedValue({ case_id: "c-1", status: "queued" });
    getDiagnosisStatus.mockResolvedValue({ status: "failed", error: "The analysis could not be completed." });
    render(<App />);
    await fillAndSubmit();

    expect(await screen.findByText("The server ran into a problem")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
    expect(screen.getByLabelText("Age")).toHaveValue(52);
  });

  it("shows a validation failure from the server without a retry", async () => {
    startDiagnosis.mockRejectedValue(new ApiError("age: must be less than or equal to 120", { kind: "validation", status: 422 }));
    render(<App />);
    await fillAndSubmit();

    expect(await screen.findByText("The server rejected this submission")).toBeInTheDocument();
    expect(screen.getByText("age: must be less than or equal to 120")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
  });
});

describe("App: navigation and new cases", () => {
  it("falls back to the form when a report or review is opened with no report in memory", async () => {
    window.location.hash = "#/report";
    render(<App />);
    expect(await screen.findByRole("heading", { name: "Describe the patient" })).toBeInTheDocument();
    await waitFor(() => expect(window.location.hash).toBe("#/"));
  });

  it("falls back to the form for the processing screen when nothing is being analysed", async () => {
    window.location.hash = "#/processing";
    render(<App />);
    expect(await screen.findByRole("heading", { name: "Describe the patient" })).toBeInTheDocument();
  });

  it("starting a new case from the header clears the report and the form", async () => {
    startDiagnosis.mockResolvedValue(REPORT);
    render(<App />);
    await fillAndSubmit();
    await screen.findByRole("heading", { name: "Hypertensive emergency" });

    await userEvent.click(screen.getByRole("button", { name: /start a new case/ }));
    expect(await screen.findByRole("heading", { name: "Describe the patient" })).toBeInTheDocument();
    expect(screen.getByLabelText("Age")).toHaveValue(null);
    expect(screen.getByLabelText("Presenting complaint and history")).toHaveValue("");
    expect(screen.queryByRole("button", { name: "Remove hypertension" })).not.toBeInTheDocument();
  });

  it("opens the review from the report and returns to it", async () => {
    startDiagnosis.mockResolvedValue(REPORT);
    render(<App />);
    await fillAndSubmit();
    await screen.findByRole("heading", { name: "Hypertensive emergency" });

    await userEvent.click(screen.getByRole("button", { name: /Review case/i }));
    expect(await screen.findByRole("heading", { name: "Review case" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Back to report/ }));
    expect(await screen.findByText("Patient summary")).toBeInTheDocument();
  });

  it("always shows that this is a research prototype", () => {
    render(<App />);
    expect(screen.getByText(/Research prototype/)).toBeInTheDocument();
  });
});
