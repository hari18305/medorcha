import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ConfidenceBar, { ConfidenceGauge, THRESHOLD, tierFor } from "./ConfidenceBar.jsx";

describe("confidence tiers", () => {
  it("uses the backend's 0.65 review threshold", () => {
    expect(THRESHOLD).toBe(0.65);
  });

  it("is high at and above the threshold, moderate from 0.4, low below", () => {
    expect(tierFor(0.65).label).toBe("high");
    expect(tierFor(0.6499).label).toBe("moderate");
    expect(tierFor(0.4).label).toBe("moderate");
    expect(tierFor(0.3999).label).toBe("low");
  });
});

describe("ConfidenceBar", () => {
  it("shows a rounded percentage on an accessible meter", () => {
    render(<ConfidenceBar confidence={0.723} label="Model-stated confidence" />);
    const meter = screen.getByRole("meter", { name: "Model-stated confidence" });
    expect(meter).toHaveAttribute("aria-valuenow", "72");
    expect(screen.getByText("72%")).toBeInTheDocument();
  });

  it("clamps out-of-range and missing values instead of drawing nonsense", () => {
    const { rerender } = render(<ConfidenceBar confidence={1.7} />);
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuenow", "100");
    rerender(<ConfidenceBar confidence={-0.2} />);
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuenow", "0");
    rerender(<ConfidenceBar confidence={undefined} />);
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuenow", "0");
  });
});

describe("ConfidenceGauge", () => {
  it("shows the figure, the threshold marker and states it is not a probability", () => {
    render(<ConfidenceGauge confidence={0.48} reviewRequired />);
    expect(screen.getByTestId("confidence-gauge")).toHaveTextContent("48%");
    expect(screen.getByText("Review threshold 65%")).toBeInTheDocument();
    expect(screen.getByText(/not a\s+probability/)).toBeInTheDocument();
  });

  it("shows the report's own review flag, not one recomputed from the number", () => {
    // 0.9 is above the threshold, but the backend can still require review (e.g. the optimizer could not run)
    const { rerender } = render(<ConfidenceGauge confidence={0.9} reviewRequired />);
    expect(screen.getByText("Review required")).toBeInTheDocument();
    rerender(<ConfidenceGauge confidence={0.3} reviewRequired={false} />);
    expect(screen.getByText("Above threshold")).toBeInTheDocument();
  });
});
