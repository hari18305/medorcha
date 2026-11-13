# Knowledge base (offline RAG corpus)

`guidelines/guidelines.json` (11 passages) and
`symptom_disease/symptom_disease.json` (56 passages) are read by
`backend/rag/ingest.py` and embedded into the `medorchestrate_kb` Qdrant
collection. **All 67 passages are real, citable content — none is
synthetic.**

## Where the text comes from

| Source | Count | What was taken | Licence / status |
|---|---|---|---|
| MedlinePlus Medical Encyclopedia and health topics (U.S. National Library of Medicine) | 47 | Verbatim text of the page summary plus selected sections (causes, symptoms, exams/tests, or why-performed / abnormal results for imaging pages), Symptoms section first, cut only at a sentence boundary to fit ~340 words | U.S. Government work, public domain |
| PubMed abstracts (NCBI E-utilities) | 20 | A short verbatim abstract excerpt, cited by PMID, first author, journal and year | Abstracts remain the publishers' property; excerpted for prototype retrieval with attribution |

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

- Prototype-scoped: 67 passages across headache, dizziness, blood
  pressure, cerebrovascular, cognitive (dementia / hydrocephalus), common
  chest conditions, brain and chest imaging findings and a few general
  presentations (chest pain, fainting, fatigue) — not a
  comprehensive medical corpus, and not curated by a clinician.
- MedlinePlus pages are consumer-health text, not guideline recommendations.
- No WHO / CDC / NICE passages are included (no clean public text
  endpoint; reuse terms vary) — the guidelines folder holds PubMed
  abstracts of published guidelines, scientific statements, consensus
  criteria and one clinical review instead.

## Why there are 67, not the spec's 20-30

The first 30 had no cognitive-impairment content: an LLM-judged run on a real
brain MRI case (memory loss and unsteady gait) found 0 of 5 retrieved passages
relevant. 12 passages were added on 2026-11-12 for conditions in
`data/test_cases/answer_key.json` that had nothing behind them: Alzheimer
disease, dementia, vascular cognitive impairment, normal pressure
hydrocephalus, hydrocephalus, idiopathic intracranial hypertension,
subarachnoid and hemorrhagic stroke, traumatic brain injury, lung cancer,
pulmonary tuberculosis, and chest injuries (rib fracture). (The whole KB was later rebuilt once with a Symptoms-first section order; see
DEMO_READINESS.md, 2026-11-12.)

### The 25 passages added on 2026-11-13 (gap analysis)

Two sources of gaps were checked: the diagnoses in `answer_key.json` and in the 31 live reports that had no passage
behind them, and the findings the imaging models can name (chest: pulmonary edema, atelectasis, mediastinal widening;
brain CT/MRI: mass lesion, vascular malformation, midline shift, calcification, edema). Added, all fetched by
`build_knowledge_base.py` and checked the same way as the rest: MedlinePlus topics (heart failure, brain tumors,
meningitis, chest pain, fainting, fatigue, multiple sclerosis, giant cell arteritis, concussion, brain aneurysm,
arteriovenous malformations, aortic aneurysm, long COVID, refractive errors); encyclopedia articles (pulmonary edema,
atelectasis, cluster headache, brain abscess); PubMed abstracts (vertebrobasilar insufficiency, PRES, intracranial
calcification versus hemorrhage on CT, long-COVID headache, musculoskeletal chest pain, massive cerebral infarction, and the 2021
AHA/ACC chest-pain guideline).

**What it cost, measured** (`python -m scripts.eval_retrieval`, `--gap`): on the 14 real agent queries hit@5 stayed 14/14
but MRR@5 fell from 0.863 to 0.764 (hit@1 11 to 9), because new, plausibly related passages (long-COVID headache for a
child with a cold, PRES for a memory-and-gait case) now rank above the one the developer had marked relevant. Those
relevance sets were written for the 42-passage KB and were not changed. At the old top-10 candidate pool the anticoagulated-fall
case lost its subdural-hematoma passage altogether, so the pool is now 20. Both the pool and the score centre
(-12.5 to -10.4, which restores the mean S_rag of the 14 queries to 0.659) were changed for this; see
`backend/rag/reranker.py`. **What it gained:** 21 queries for the new conditions, written for this purpose by the person
who added the passages (so they measure coverage, not independent retrieval quality), reach a relevant passage in the top 5
in 18 of 21 cases, against 3 of 21 on the original 42 passages. Misses: mediastinal widening, vertebrobasilar
insufficiency and musculoskeletal chest pain.

