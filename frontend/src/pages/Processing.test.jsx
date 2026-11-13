import { act, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Processing from "./Processing.jsx";

const STATUS = {
  status: "running",
  steps: [
    { name: "clinical", status: "done", detail: "3 symptom(s), 1 history item(s)", duration_ms: 4200 },
    { name: "imaging", status: "skipped", detail: "no image submitted", duration_ms: null },
    { name: "rag", status: "failed", detail: "this step failed; the analysis continued with less", duration_ms: 18500 },
    { name: "evidence", status: "running", detail: null, duration_ms: null },
    { name: "fusion", status: "pending", detail: null, duration_ms: null },
    {
      name: "optimizer", status: "running", detail: null, duration_ms: null,
      iterations: [
        { iteration: 1, max: 3, status: "done", confidence: 0.47, strategy: "expand" },
        { iteration: 2, max: 3, status: "running" },
      ],
    },
  ],
};

const step = (name) => document.querySelector(`[data-step="${name}"]`);

describe("Processing", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("says the case is being sent before the server has answered", () => {
    render(<Processing progress={null} />);
    expect(screen.getByTestId("progress-message")).toHaveTextContent("Sending the case to the server…");
    expect(screen.queryByTestId("progress-steps")).not.toBeInTheDocument();
  });

  it("shows the queue position while waiting for a worker", () => {
    render(<Processing progress={{ status: "queued", queue_position: 3, steps: [] }} />);
    expect(screen.getByTestId("progress-message")).toHaveTextContent("Waiting for a free worker (2 cases ahead)…");
  });

  it("names the running step in the status line", () => {
    render(<Processing progress={STATUS} />);
    expect(screen.getByTestId("progress-message")).toHaveTextContent("Live literature…");
  });

  it("renders every step with the status the server reported", () => {
    render(<Processing progress={STATUS} />);
    const statuses = Object.fromEntries(STATUS.steps.map((s) => [s.name, step(s.name).getAttribute("data-status")]));
    expect(statuses).toEqual({ clinical: "done", imaging: "skipped", rag: "failed", evidence: "running", fusion: "pending", optimizer: "running" });
  });

  it("shows durations for finished steps and the reason for a skipped one", () => {
    render(<Processing progress={STATUS} />);
    expect(within(step("clinical")).getByText("4.2 s")).toBeInTheDocument();
    expect(within(step("rag")).getByText("19 s")).toBeInTheDocument();
    expect(within(step("imaging")).getByText("Skipped: no image submitted")).toBeInTheDocument();
    expect(within(step("imaging")).queryByText(/\ds$/)).not.toBeInTheDocument();
  });

  it("is honest about a failed step", () => {
    render(<Processing progress={STATUS} />);
    expect(within(step("rag")).getByText("this step failed; the analysis continued with less")).toBeInTheDocument();
  });

  it("shows a hint for steps that have not produced a result yet", () => {
    render(<Processing progress={STATUS} />);
    expect(within(step("fusion")).getByText("Ranked differential with cited evidence")).toBeInTheDocument();
  });

  it("lists the optimizer's refinement rounds", () => {
    render(<Processing progress={STATUS} />);
    const rounds = within(screen.getByTestId("optimizer-rounds")).getAllByRole("listitem").map((li) => li.textContent);
    expect(rounds).toEqual(["Round 1 of 3: confidence 47% (expand)", "Round 2 of 3: refining…"]);
  });

  it("runs the elapsed clock from the browser, one second at a time", () => {
    render(<Processing progress={null} />);
    const clock = screen.getByLabelText("Elapsed time");
    expect(clock).toHaveTextContent("0:00");
    act(() => vi.advanceTimersByTime(65_000));
    expect(clock).toHaveTextContent("1:05");
  });
});
