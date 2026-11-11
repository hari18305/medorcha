# Knowledge base (offline RAG corpus)

`guidelines/guidelines.json` (10 passages) and
`symptom_disease/symptom_disease.json` (32 passages) are read by
`backend/rag/ingest.py` and embedded into the `medorchestrate_kb` Qdrant
collection. **All 42 passages are real, citable content — none is
synthetic.**

## Where the text comes from

| Source | Count | What was taken | Licence / status |
|---|---|---|---|
| MedlinePlus Medical Encyclopedia and health topics (U.S. National Library of Medicine) | 29 | Verbatim text of the page summary plus selected sections (causes, symptoms, exams/tests, or why-performed / abnormal results for imaging pages), Symptoms section first, cut only at a sentence boundary to fit ~340 words | U.S. Government work, public domain |
| PubMed abstracts (NCBI E-utilities) | 13 | A short verbatim abstract excerpt, cited by PMID, first author, journal and year | Abstracts remain the publishers' property; excerpted for prototype retrieval with attribution |

Nothing is paraphrased or rewritten. The only edits are mechanical:
markup is removed, bullet lists are joined with `; ` (ending in `.`),
and the stray space left before punctuation by stripped link tags is
removed.

## Per-record fields

`doc_id`, `source` (the human-readable citation shown in reports),
`url`, `title`, `retrieved` (date fetched), `text`.

## Re-verifying or refreshing

```bash
python -m backend.rag.build_knowledge_base            # fetches only doc_ids not already on disk and appends them
python -m backend.rag.build_knowledge_base --refresh DOC_ID [DOC_ID ...]  # re-fetches just these, in place
python -m backend.rag.build_knowledge_base --rebuild  # re-fetches EVERY passage (refreshes text and dates)
python -m backend.rag.ingest                          # rebuilds the Qdrant collection from the JSON
```

The build tool checks every PubMed PMID's title against PubMed before
writing it, and fails rather than writing a record with no abstract. Two
guidelines whose primary PubMed records carry no abstract (ACC/AHA 2017
and ISH 2020 hypertension) are *not* cited as the guideline: the ISH
2020 passage is the abstract of its separate key-messages summary paper
and is cited as exactly that.

## Caveats

- Prototype-scoped: 42 passages across headache, dizziness, blood
  pressure, cerebrovascular, cognitive (dementia / hydrocephalus), common
  chest conditions and common imaging topics — not a
  comprehensive medical corpus, and not curated by a clinician.
- MedlinePlus pages are consumer-health text, not guideline recommendations.
- No WHO / CDC / NICE passages are included (no clean public text
  endpoint; reuse terms vary) — the guidelines folder holds PubMed
  abstracts of published guidelines, scientific statements, consensus
  criteria and one clinical review instead.

## Why there are 42, not the spec's 20-30

The first 30 had no cognitive-impairment content: an LLM-judged run on a real
brain MRI case (memory loss and unsteady gait) found 0 of 5 retrieved passages
relevant. 12 passages were added on 2026-11-12 for conditions in
`data/test_cases/answer_key.json` that had nothing behind them: Alzheimer
disease, dementia, vascular cognitive impairment, normal pressure
hydrocephalus, hydrocephalus, idiopathic intracranial hypertension,
subarachnoid and hemorrhagic stroke, traumatic brain injury, lung cancer,
pulmonary tuberculosis, and chest injuries (rib fracture). (The whole KB was later rebuilt once with a Symptoms-first section order; see
DEMO_READINESS.md, 2026-11-12.)
