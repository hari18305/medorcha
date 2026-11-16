// Run with: npm test
import assert from "node:assert/strict";
import { test } from "node:test";
import { citationSupport, evidenceLabel, findingGroups, hasExplanation, signalLevel } from "./plainLanguage.js";

test("evidence ids are shown in words, unknown ids pass through unchanged", () => {
  assert.equal(evidenceLabel("RAG1"), "Library 1");
  assert.equal(evidenceLabel("LIVE12"), "Research 12");
  assert.equal(evidenceLabel("MEM3"), "Past case 3");
  assert.equal(evidenceLabel("VIS1"), "Past case 1");
  assert.equal(evidenceLabel("X9"), "X9");
  assert.equal(evidenceLabel(undefined), "");
});

test("citation support is a count, never a quality grade", () => {
  assert.deepEqual(citationSupport({ supporting_evidence: [] }), { level: "none", label: "No source cited" });
  assert.deepEqual(citationSupport({ supporting_evidence: ["[RAG1] a"] }), { level: "one", label: "1 cited source" });
  assert.deepEqual(citationSupport({ supporting_evidence: ["a", "b", "c"] }), { level: "several", label: "3 cited sources" });
  assert.equal(citationSupport({}).level, "none");
});

test("signal levels map to a word and a tone, with a safe fallback", () => {
  assert.deepEqual(signalLevel("strong"), { label: "Strong", tone: "good" });
  assert.deepEqual(signalLevel("unavailable"), { label: "Unavailable", tone: "bad" });
  assert.deepEqual(signalLevel("brand-new"), { label: "brand-new", tone: "mid" });
});

test("older reports without an explanation are detected", () => {
  assert.equal(hasExplanation(null), false);
  assert.equal(hasExplanation({}), false);
  assert.equal(hasExplanation({ headline: "x" }), true);
});

test("finding groups keep only what has content and merge a normal imaging finding into the normals", () => {
  const groups = findingGroups({
    findings: { present: ["chest pain"], negatives: ["No fever"], history: [], labs: [], report_findings: [] },
    imaging: { normal: true, statement: "Chest X-ray: no acute abnormality identified." },
  });
  assert.deepEqual(groups.map((g) => g.id), ["present", "normal"]);
  assert.deepEqual(groups[1].items, ["No fever", "Chest X-ray: no acute abnormality identified."]);
  assert.deepEqual(findingGroups(null), []);
  // an abnormal or unusable image read is NOT listed as a normal finding
  assert.deepEqual(findingGroups({ findings: {}, imaging: { normal: false, statement: "x" } }), []);
});
