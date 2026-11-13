import { expect, test } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const IMAGE = path.resolve(here, "../../data/images/synthetic_demo_brain_mri_001.png");

async function fillPatient(page, { age = "52", sex = "female" } = {}) {
  await page.getByLabel("Age", { exact: true }).fill(age);
  await page.getByLabel("Sex", { exact: true }).selectOption(sex);
}

const analyze = (page) => page.getByRole("button", { name: /Analyze case/ }).click();

test.beforeEach(async ({ page }) => {
  await page.goto("/");
});

test("the form refuses to send without age and sex, in the browser, with no request made", async ({ page }) => {
  let posted = false;
  page.on("request", (r) => {
    if (r.method() === "POST" && r.url().includes("/diagnose")) posted = true;
  });
  await page.getByLabel("Presenting complaint and history", { exact: true }).fill("headache");
  await analyze(page);
  await expect(page.getByRole("alert")).toContainText("Age and sex are required.");
  expect(posted).toBe(false);
});

test("an ordinary case runs end to end: progress, report, citations that jump, patient summary", async ({ page }) => {
  await fillPatient(page);
  await page.getByLabel("Presenting complaint and history", { exact: true }).fill("headache and dizziness, history of hypertension");
  await page.getByLabel("Past medical history", { exact: true }).fill("hypertension");
  await analyze(page);

  // the progress screen shows the server's own steps
  await expect(page).toHaveURL(/#\/processing/);
  await expect(page.getByTestId("progress-steps")).toBeVisible();

  // then the report
  await expect(page.getByRole("heading", { name: "Hypertension-related headache" })).toBeVisible();
  await expect(page).toHaveURL(/#\/report/);
  await expect(page.getByTestId("confidence-gauge").first()).toContainText("Above threshold");
  await expect(page.getByText("Patient summary")).toBeVisible();
  await expect(page.locator('[data-summary="history"]')).toContainText("hypertension");

  // a citation chip jumps to its evidence item
  const chip = page.locator('button[data-citation="RAG1"]').first();
  await chip.click();
  await expect(page.locator("#ev-RAG1")).toBeInViewport();
  // 3 live items and 5 offline passages came back from the real endpoint
  await expect(page.locator("[data-evidence-id]")).toHaveCount(8);
});

test("a vague complaint is sent straight to review with the non-specific-input note and a capped score", async ({ page }) => {
  await fillPatient(page, { age: "34" });
  await page.getByLabel("Presenting complaint and history", { exact: true }).fill("feeling generally unwell, mild fatigue, occasional lightheadedness");
  await analyze(page);

  await expect(page.getByRole("heading", { name: "Review case" })).toBeVisible();
  await expect(page).toHaveURL(/#\/review/);
  await expect(page.getByTestId("insufficient-input-note")).toContainText("Non-specific clinical input");
  await expect(page.getByTestId("confidence-gauge").first()).toContainText("60%");
  await expect(page.getByText("Clinician review required")).toBeVisible();

  // the clinician closes the case with an annotation
  await page.getByRole("radio", { name: /Annotate/ }).check({ force: true });
  await page.getByLabel("Note", { exact: true }).fill("Needs history and examination first.");
  await page.getByLabel("Reviewer name", { exact: true }).fill("Dr. Test");
  await page.getByRole("radio", { name: "4 stars" }).click();
  await page.getByRole("button", { name: /Submit review/ }).click();
  await expect(page.getByText("Case closed")).toBeVisible();
  await expect(page.getByText("Review confidence: 4 / 5")).toBeVisible();
});

test("an image with no clinical text is held at 50% and says why", async ({ page }) => {
  await fillPatient(page, { age: "60", sex: "male" });
  await page.setInputFiles("#image", IMAGE);
  await expect(page.getByTestId("sparse-input-hint")).toBeVisible();
  await analyze(page);

  await expect(page.getByRole("heading", { name: "Review case" })).toBeVisible();
  await expect(page.getByTestId("insufficient-input-note")).toContainText("Insufficient clinical input");
  await expect(page.getByTestId("confidence-gauge").first()).toContainText("50%");
  await expect(page.getByText("Pneumothorax").first()).toBeVisible(); // the imaging-led differential is still shown
});

test("weak evidence runs the optimizer's refinement rounds on screen and ends in review", async ({ page }) => {
  await fillPatient(page, { age: "70" });
  await page.getByLabel("Presenting complaint and history", { exact: true }).fill("ambiguous headache, dizziness");
  await analyze(page);

  await expect(page).toHaveURL(/#\/processing/);
  // the rounds are the server's real optimizer events, listed as they happen
  await expect(page.getByTestId("optimizer-rounds")).toBeVisible({ timeout: 30_000 });

  await expect(page.getByRole("heading", { name: "Review case" })).toBeVisible({ timeout: 45_000 });
  await expect(page.getByText("Clinician review required")).toBeVisible();
  await expect(page.getByText("3 of 3")).toBeVisible(); // refinement rounds shown in the run summary
});

test("an override is recorded and the case is closed; the report can be reopened from the confirmation", async ({ page }) => {
  await fillPatient(page, { age: "40" });
  await page.getByLabel("Presenting complaint and history", { exact: true }).fill("headache and dizziness");
  await analyze(page);
  await expect(page.getByRole("heading", { name: "Hypertension-related headache" })).toBeVisible();

  await page.getByRole("button", { name: /Review case/i }).click();
  await expect(page.getByRole("heading", { name: "Review case" })).toBeVisible();
  await page.getByRole("radio", { name: /Override/ }).check({ force: true });
  await page.getByRole("button", { name: "Tension-type headache" }).click();
  await page.getByLabel("Reviewer name", { exact: true }).fill("Dr. Test");
  await page.getByRole("button", { name: /Submit review/ }).click();
  await expect(page.getByText("Case closed")).toBeVisible();
  await expect(page.getByText("Override diagnosis: Tension-type headache")).toBeVisible();
  await page.getByRole("button", { name: /Back to report/ }).click();
  await expect(page.getByText("Patient summary")).toBeVisible();
});

test("a reloaded report page falls back to the form (case data lives only in memory)", async ({ page }) => {
  await fillPatient(page, { age: "40" });
  await page.getByLabel("Presenting complaint and history", { exact: true }).fill("headache and dizziness");
  await analyze(page);
  await expect(page.getByRole("heading", { name: "Hypertension-related headache" })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("heading", { name: "Describe the patient" })).toBeVisible();
});
