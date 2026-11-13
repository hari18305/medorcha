import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../services/api.js";
import CaseSubmission from "./CaseSubmission.jsx";

const setup = (props = {}) => {
  const onSubmit = vi.fn();
  render(<CaseSubmission onSubmit={onSubmit} {...props} />);
  return onSubmit;
};

async function fillPatient(age = "45", sex = "male") {
  await userEvent.type(screen.getByLabelText("Age"), age);
  await userEvent.selectOptions(screen.getByLabelText("Sex"), sex);
}

const submit = () => userEvent.click(screen.getByRole("button", { name: /Analyze case/ }));

describe("CaseSubmission: validation", () => {
  it("requires age and sex", async () => {
    const onSubmit = setup();
    await userEvent.type(screen.getByLabelText("Presenting complaint and history"), "headache");
    await submit();
    expect(screen.getByRole("alert")).toHaveTextContent("Age and sex are required.");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("requires something to analyse", async () => {
    const onSubmit = setup();
    await fillPatient();
    await submit();
    expect(screen.getByRole("alert")).toHaveTextContent(/at least clinical text, a medical report, lab values or an image/i);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("says plainly that past history alone is not enough", async () => {
    const onSubmit = setup();
    await fillPatient();
    await userEvent.type(screen.getByLabelText("Past medical history"), "hypertension{Enter}");
    await submit();
    expect(screen.getByRole("alert")).toHaveTextContent(/Past history alone is not enough/);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("rejects a half-filled lab row instead of dropping it", async () => {
    const onSubmit = setup();
    await fillPatient();
    await userEvent.type(screen.getByLabelText("Presenting complaint and history"), "fatigue");
    await userEvent.click(screen.getByRole("button", { name: /Add lab value/ }));
    await userEvent.type(screen.getByLabelText("Lab 1 name"), "hemoglobin");
    await submit();
    expect(screen.getByRole("alert")).toHaveTextContent(/lab/i);
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

describe("CaseSubmission: what is sent", () => {
  it("sends the typed fields and a plain summary for the report screen", async () => {
    const onSubmit = setup();
    await fillPatient("39", "female");
    await userEvent.type(screen.getByLabelText("Presenting complaint and history"), "  fatigue and dizziness  ");
    await userEvent.type(screen.getByLabelText("Past medical history"), "iron deficiency anemia{Enter}");
    await userEvent.click(screen.getByRole("button", { name: /Add lab value/ }));
    await userEvent.type(screen.getByLabelText("Lab 1 name"), "hemoglobin");
    await userEvent.type(screen.getByLabelText("Lab 1 value"), "10.2 g/dL");
    await submit();

    expect(onSubmit).toHaveBeenCalledTimes(1);
    const [formData, summary] = onSubmit.mock.calls[0];
    expect(formData.get("age")).toBe("39");
    expect(formData.get("sex")).toBe("female");
    expect(formData.get("clinical_text")).toBe("fatigue and dizziness"); // trimmed
    expect(JSON.parse(formData.get("history"))).toEqual(["iron deficiency anemia"]);
    expect(JSON.parse(formData.get("labs"))).toEqual({ hemoglobin: "10.2 g/dL" });
    expect(formData.has("image")).toBe(false);
    expect(formData.has("medical_report")).toBe(false);
    expect(summary).toMatchObject({ age: "39", sex: "female", clinicalText: "fatigue and dizziness", hadImage: false });
    expect(summary.history).toEqual(["iron deficiency anemia"]);
  });

  it("includes history that was typed but never confirmed with Enter", async () => {
    const onSubmit = setup();
    await fillPatient();
    await userEvent.type(screen.getByLabelText("Presenting complaint and history"), "chest pain");
    await userEvent.type(screen.getByLabelText("Past medical history"), "smoker");
    await submit();
    const [formData] = onSubmit.mock.calls[0];
    expect(JSON.parse(formData.get("history"))).toEqual(["smoker"]);
  });

  it("still includes an unconfirmed history draft when the form is submitted without the box losing focus", async () => {
    // Clicking the button blurs the box, which commits the draft; a submit that skips the blur (the
    // keyboard, an autofill) must not lose it either: the form finalises the draft itself.
    const onSubmit = setup();
    await fillPatient();
    await userEvent.type(screen.getByLabelText("Presenting complaint and history"), "chest pain");
    await userEvent.type(screen.getByLabelText("Past medical history"), "smoker");
    fireEvent.submit(screen.getByRole("button", { name: /Analyze case/ }).closest("form"));
    const [formData] = onSubmit.mock.calls[0];
    expect(JSON.parse(formData.get("history"))).toEqual(["smoker"]);
  });

  it("ignores a fully empty lab row", async () => {
    const onSubmit = setup();
    await fillPatient();
    await userEvent.type(screen.getByLabelText("Presenting complaint and history"), "cough");
    await userEvent.click(screen.getByRole("button", { name: /Add lab value/ }));
    await submit();
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0].has("labs")).toBe(false);
  });

  it("attaches a chosen image and shows a preview that can be removed", async () => {
    const onSubmit = setup();
    await fillPatient();
    const file = new File(["x"], "scan.png", { type: "image/png" });
    await userEvent.upload(screen.getByLabelText(/Drop an image here/), file);
    expect(screen.getByAltText("Selected medical image preview")).toBeInTheDocument();
    expect(screen.getByText("scan.png")).toBeInTheDocument();

    await submit(); // an image alone is accepted by the form
    const [formData, summary] = onSubmit.mock.calls[0];
    expect(formData.get("image")).toBe(file);
    expect(summary.hadImage).toBe(true);

    await userEvent.click(screen.getByRole("button", { name: "Remove image" }));
    expect(screen.queryByAltText("Selected medical image preview")).not.toBeInTheDocument();
  });
});

describe("CaseSubmission: sparse-input hint", () => {
  const hint = () => screen.queryByTestId("sparse-input-hint");

  it("appears for an image with no clinical input", async () => {
    setup();
    expect(hint()).not.toBeInTheDocument();
    await userEvent.upload(screen.getByLabelText(/Drop an image here/), new File(["x"], "a.png", { type: "image/png" }));
    expect(hint()).toBeInTheDocument();
  });

  it("appears for history alone and goes away once there is text", async () => {
    setup();
    await userEvent.type(screen.getByLabelText("Past medical history"), "hypertension{Enter}");
    expect(hint()).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("Presenting complaint and history"), "headache");
    expect(hint()).not.toBeInTheDocument();
  });

  it("is not shown when a lab row supplies the clinical input", async () => {
    setup();
    await userEvent.upload(screen.getByLabelText(/Drop an image here/), new File(["x"], "a.png", { type: "image/png" }));
    await userEvent.click(screen.getByRole("button", { name: /Add lab value/ }));
    await userEvent.type(screen.getByLabelText("Lab 1 name"), "hemoglobin");
    await userEvent.type(screen.getByLabelText("Lab 1 value"), "9");
    expect(hint()).not.toBeInTheDocument();
  });
});

describe("CaseSubmission: examples and errors", () => {
  it("fills the whole form from an example, including history and labs", async () => {
    setup();
    await userEvent.click(screen.getByRole("button", { name: "Anemia, with labs" }));
    expect(screen.getByLabelText("Age")).toHaveValue(39);
    expect(screen.getByLabelText("Sex")).toHaveValue("female");
    expect(screen.getByLabelText("Presenting complaint and history")).toHaveValue("fatigue and dizziness, especially on standing, mild headache");
    expect(screen.getByRole("button", { name: "Remove iron deficiency anemia" })).toBeInTheDocument();
    expect(screen.getByLabelText("Lab 1 name")).toHaveValue("hemoglobin");
    expect(screen.getByLabelText("Lab 1 value")).toHaveValue("10.2 g/dL");
  });

  it("shows the server error with a retry for a network failure", async () => {
    const onRetry = vi.fn();
    setup({ error: new ApiError("down", { kind: "network" }), onRetry });
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("puts a client-side form message ahead of the last server error, without a retry", async () => {
    setup({ error: new ApiError("down", { kind: "network" }), onRetry: vi.fn() });
    await submit(); // age and sex missing
    expect(screen.getByRole("alert")).toHaveTextContent("Age and sex are required.");
    expect(screen.queryByRole("button", { name: "Try again" })).not.toBeInTheDocument();
  });

  it("restores a previous submission after a failure so nothing has to be retyped", () => {
    setup({
      initialValues: { age: "52", sex: "female", clinicalText: "headache", medicalReport: "", history: ["hypertension"], labs: { hemoglobin: "10 g/dL" } },
    });
    expect(screen.getByLabelText("Age")).toHaveValue(52);
    expect(screen.getByLabelText("Presenting complaint and history")).toHaveValue("headache");
    expect(screen.getByRole("button", { name: "Remove hypertension" })).toBeInTheDocument();
    expect(screen.getByLabelText("Lab 1 name")).toHaveValue("hemoglobin");
  });
});
