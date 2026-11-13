// Run with: npm test   (Node's built-in test runner; no dependencies)
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildLabs,
  finalizeHistory,
  labsToRows,
  MAX_HISTORY_ITEMS,
  MAX_LAB_ROWS,
  mergeHistory,
  parseHistoryInput,
} from "./caseFields.js";

test("parseHistoryInput splits on commas, semicolons and newlines and drops blanks", () => {
  assert.deepEqual(parseHistoryInput("hypertension, diabetes; smoker\nCKD"), ["hypertension", "diabetes", "smoker", "CKD"]);
  assert.deepEqual(parseHistoryInput("  ,, ;  \n"), []);
  assert.deepEqual(parseHistoryInput("  atrial   fibrillation "), ["atrial fibrillation"]);
  assert.deepEqual(parseHistoryInput(undefined), []);
});

test("mergeHistory skips case-insensitive duplicates and keeps the first spelling", () => {
  assert.deepEqual(mergeHistory(["Hypertension"], ["hypertension", "Diabetes", "  diabetes "]), ["Hypertension", "Diabetes"]);
  assert.deepEqual(mergeHistory([], []), []);
});

test("mergeHistory does not mutate its input", () => {
  const existing = ["a"];
  mergeHistory(existing, ["b"]);
  assert.deepEqual(existing, ["a"]);
});

test("finalizeHistory commits text still typed in the box (it must not be lost on submit)", () => {
  const { history, error } = finalizeHistory(["hypertension"], "diabetes, smoker");
  assert.deepEqual(history, ["hypertension", "diabetes", "smoker"]);
  assert.equal(error, null);
});

test("finalizeHistory enforces the item count and length bounds", () => {
  const many = Array.from({ length: MAX_HISTORY_ITEMS + 1 }, (_, i) => `item ${i}`);
  assert.match(finalizeHistory(many, "").error, /limited to/);
  assert.match(finalizeHistory([], "x".repeat(121)).error, /longer than/);
  assert.equal(finalizeHistory([], "x".repeat(120)).error, null);
});

test("buildLabs builds a name -> value object, trimming and keeping units in the value", () => {
  const { labs, error } = buildLabs([
    { name: " hemoglobin ", value: "10.2 g/dL" },
    { name: "hematocrit", value: "31%" },
  ]);
  assert.equal(error, null);
  assert.deepEqual(labs, { hemoglobin: "10.2 g/dL", hematocrit: "31%" });
});

test("buildLabs ignores fully empty rows", () => {
  assert.deepEqual(buildLabs([{ name: "", value: "" }, { name: "  ", value: " " }]), { labs: {}, error: null });
  assert.deepEqual(buildLabs([]), { labs: {}, error: null });
});

test("buildLabs rejects a half-filled row, naming what is missing", () => {
  assert.match(buildLabs([{ name: "sodium", value: "" }]).error, /sodium is missing its value/);
  assert.match(buildLabs([{ name: "", value: "140" }]).error, /140 is missing its name/);
});

test("buildLabs rejects a repeated name regardless of case", () => {
  const { error, labs } = buildLabs([{ name: "Sodium", value: "140" }, { name: "sodium", value: "139" }]);
  assert.match(error, /listed twice/);
  assert.deepEqual(labs, {});
});

test("buildLabs enforces length and row-count bounds", () => {
  assert.match(buildLabs([{ name: "x".repeat(61), value: "1" }]).error, /limited to 60/);
  const rows = Array.from({ length: MAX_LAB_ROWS + 1 }, (_, i) => ({ name: `lab${i}`, value: "1" }));
  assert.match(buildLabs(rows).error, /limited to 30 rows/);
});

test("labsToRows round-trips with buildLabs", () => {
  const labs = { hemoglobin: "10.2 g/dL", hematocrit: "31%" };
  assert.deepEqual(buildLabs(labsToRows(labs)).labs, labs);
  assert.deepEqual(labsToRows(undefined), []);
});
