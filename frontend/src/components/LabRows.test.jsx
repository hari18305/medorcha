import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { MAX_LAB_ROWS } from "../utils/caseFields.js";
import LabRows, { newLabRow } from "./LabRows.jsx";

function Harness({ initial = [] }) {
  const [rows, setRows] = useState(initial);
  return (
    <>
      <LabRows rows={rows} onRowsChange={setRows} />
      <output data-testid="rows">{JSON.stringify(rows.map(({ name, value }) => ({ name, value })))}</output>
    </>
  );
}
const rows = () => JSON.parse(screen.getByTestId("rows").textContent);

describe("LabRows", () => {
  it("starts empty with a hint and adds a blank row", async () => {
    render(<Harness />);
    expect(screen.getByText(/None added/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Add lab value/ }));
    expect(rows()).toEqual([{ name: "", value: "" }]);
    expect(screen.queryByText(/None added/)).not.toBeInTheDocument();
  });

  it("edits a row's value without touching its name", async () => {
    render(<Harness initial={[newLabRow("hemoglobin", "")]} />);
    await userEvent.type(screen.getByLabelText("Lab 1 value"), "10.2 g/dL");
    expect(rows()).toEqual([{ name: "hemoglobin", value: "10.2 g/dL" }]);
  });

  it("removes the chosen row only", async () => {
    render(<Harness initial={[newLabRow("a", "1"), newLabRow("b", "2"), newLabRow("c", "3")]} />);
    await userEvent.click(screen.getByRole("button", { name: "Remove lab 2" }));
    expect(rows().map((r) => r.name)).toEqual(["a", "c"]);
  });

  it("stops offering new rows at the limit", () => {
    const full = Array.from({ length: MAX_LAB_ROWS }, (_, i) => newLabRow(`lab${i}`, "1"));
    render(<Harness initial={full} />);
    expect(screen.getByRole("button", { name: /Add lab value/ })).toBeDisabled();
  });
});
