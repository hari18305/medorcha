import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import HistoryInput from "./HistoryInput.jsx";

// The component is controlled; this keeps the parent's state the way CaseSubmission does.
function Harness({ initial = [], onSubmit }) {
  const [items, setItems] = useState(initial);
  const [draft, setDraft] = useState("");
  return (
    <form onSubmit={onSubmit}>
      <HistoryInput items={items} onItemsChange={setItems} draft={draft} onDraftChange={setDraft} />
      <output data-testid="state">{JSON.stringify({ items, draft })}</output>
    </form>
  );
}
const state = () => JSON.parse(screen.getByTestId("state").textContent);

describe("HistoryInput", () => {
  it("commits the draft on Enter without submitting the surrounding form", async () => {
    const onSubmit = vi.fn((e) => e.preventDefault());
    render(<Harness onSubmit={onSubmit} />);
    await userEvent.type(screen.getByLabelText("Past medical history"), "hypertension{Enter}");
    expect(state()).toEqual({ items: ["hypertension"], draft: "" });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("commits on comma and semicolon", async () => {
    render(<Harness />);
    await userEvent.type(screen.getByLabelText("Past medical history"), "diabetes,smoker;");
    expect(state().items).toEqual(["diabetes", "smoker"]);
  });

  it("adds each term of a pasted list", async () => {
    render(<Harness />);
    screen.getByLabelText("Past medical history").focus();
    await userEvent.paste("asthma, copd; gerd");
    expect(state().items).toEqual(["asthma", "copd", "gerd"]);
  });

  it("does not add the same term twice, ignoring case", async () => {
    render(<Harness initial={["Hypertension"]} />);
    await userEvent.type(screen.getByLabelText("Past medical history"), "hypertension{Enter}");
    expect(state().items).toEqual(["Hypertension"]);
  });

  it("Backspace on an empty box removes the last chip", async () => {
    render(<Harness initial={["a", "b"]} />);
    await userEvent.type(screen.getByLabelText("Past medical history"), "{Backspace}");
    expect(state().items).toEqual(["a"]);
  });

  it("removes a chip with its remove button", async () => {
    render(<Harness initial={["a", "b"]} />);
    await userEvent.click(screen.getByRole("button", { name: "Remove a" }));
    expect(state().items).toEqual(["b"]);
  });

  it("quick-add buttons add a term and disappear once it is present", async () => {
    render(<Harness />);
    await userEvent.click(screen.getByRole("button", { name: /^hypertension$/i }));
    expect(state().items).toEqual(["hypertension"]);
    expect(screen.queryByRole("button", { name: /^hypertension$/i })).not.toBeInTheDocument();
  });

  it("commits what is typed when the box loses focus, so it is not silently dropped", async () => {
    render(<Harness />);
    const box = screen.getByLabelText("Past medical history");
    await userEvent.type(box, "on warfarin");
    fireEvent.blur(box);
    expect(state()).toEqual({ items: ["on warfarin"], draft: "" });
  });
});
