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

import { buildSummaryText } from "./plainLanguage.js";

test("the copied summary keeps the prototype and review warnings and lists the differential in rank order", () => {
  const text = buildSummaryText(
    {
      case_id: "C1",
      confidence: 0.5,
      review_required: true,
      diagnoses: [
        { name: "Second", rank: 2, supporting_evidence: [] },
        { name: "First", rank: 1, supporting_evidence: ["[RAG1] a", "[LIVE1] b"], findings_for: ["pleuritic pain"], findings_against: ["no trauma"] },
      ],
      explanation: {
        headline: "Leading consideration: First, supported by 2 cited sources. This result needs clinician review before it is relied on.",
        score: { final: 58, review_line: 65 },
        review_reasons: ["The combined evidence score (58%) is below the 65% review line."],
        cautions: [],
        next_steps: ["Add laboratory results."],
        findings: { negatives: ["No fever"] },
        imaging: { normal: true, statement: "Chest X-ray: no acute abnormality identified." },
      },
    },
    { sex: "female", age: "39" }
  );
  assert.match(text, /Research prototype: supports, never replaces, clinical judgement\. Not a medical device\./);
  assert.match(text, /Patient: female, 39 years/);
  assert.match(text, /Review: REQUIRED/);
  assert.match(text, /Evidence score: 58 of 100 \(review line 65\)\. This is how well the evidence supports the result, not the chance it is correct\./);
  assert.ok(text.indexOf("1. First (2 cited sources)") < text.indexOf("2. Second (No source cited)"));
  assert.match(text, /Fits this case: pleuritic pain/);
  assert.match(text, /Against, or not reported: no trauma/);
  assert.match(text, /Why review is needed:\n- The combined evidence score/);
  assert.match(text, /What would help:\n- Add laboratory results\./);
  assert.match(text, /Normal findings and pertinent negatives:\n- No fever\n- Chest X-ray: no acute abnormality identified\./);
});

test("a report from before explanations existed still produces a safe summary", () => {
  const text = buildSummaryText({ case_id: "C2", confidence: 0.72, review_required: false, diagnoses: [{ name: "X", rank: 1, supporting_evidence: [] }] });
  assert.match(text, /Review: not flagged by the system; clinical judgement still applies\./);
  assert.match(text, /Evidence confidence: 72% \(not a probability\)\./);
  assert.doesNotMatch(text, /Patient:/);
});
