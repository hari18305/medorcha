import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../services/api.js";
import ErrorBanner from "./ErrorBanner.jsx";

describe("ErrorBanner", () => {
  it("renders nothing without an error", () => {
    const { container } = render(<ErrorBanner error={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("words each failure kind differently", () => {
    const cases = [
      [new ApiError("Could not reach the server", { kind: "network" }), "Can't reach the server", "Could not reach the server"],
      [new ApiError("age: must be a number", { kind: "validation", status: 422 }), "The server rejected this submission", "age: must be a number"],
      [new ApiError("No such case", { kind: "not_found", status: 404 }), "Not found", "No such case"],
      [new ApiError("boom", { kind: "other", status: 418 }), "Something went wrong", "boom"],
    ];
    for (const [error, title, description] of cases) {
      const { unmount } = render(<ErrorBanner error={error} />);
      expect(screen.getByRole("alert")).toHaveAttribute("data-error-kind", error.kind);
      expect(screen.getByText(title)).toBeInTheDocument();
      expect(screen.getByText(description)).toBeInTheDocument();
      unmount();
    }
  });

  it("does not show a server error's own text (it may carry internals)", () => {
    render(<ErrorBanner error={new ApiError("Traceback: secret detail", { kind: "server", status: 500 })} />);
    expect(screen.getByText("The server ran into a problem")).toBeInTheDocument();
    expect(screen.queryByText(/secret detail/)).not.toBeInTheDocument();
  });

  it("treats a plain string as a form-validation message", () => {
    render(<ErrorBanner error="Age and sex are required." />);
    expect(screen.getByRole("alert")).toHaveAttribute("data-error-kind", "validation");
    expect(screen.getByText("Please check the form")).toBeInTheDocument();
  });

  it("offers Try again only for network and server failures", async () => {
    const onRetry = vi.fn();
    const retryable = [new ApiError("x", { kind: "network" }), new ApiError("x", { kind: "server", status: 503 })];
    for (const error of retryable) {
      const { unmount } = render(<ErrorBanner error={error} onRetry={onRetry} />);
      await userEvent.click(screen.getByRole("button", { name: "Try again" }));
      unmount();
    }
    expect(onRetry).toHaveBeenCalledTimes(2);

    const notRetryable = [new ApiError("x", { kind: "validation", status: 422 }), new ApiError("x", { kind: "not_found", status: 404 }), "form message"];
    for (const error of notRetryable) {
      const { unmount } = render(<ErrorBanner error={error} onRetry={onRetry} />);
      expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
      unmount();
    }
  });
});
