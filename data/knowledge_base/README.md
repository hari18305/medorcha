# Knowledge base (offline RAG corpus)

`guidelines/guidelines.json` (7 passages) and
`symptom_disease/symptom_disease.json` (23 passages) are read by
`backend/rag/ingest.py` and embedded into the `medorchestrate_kb` Qdrant
collection. **All 30 passages are real, citable content — none is
synthetic.**

## Where the text comes from

| Source | Count | What was taken | Licence / status |
|---|---|---|---|
| MedlinePlus Medical Encyclopedia and health topics (U.S. National Library of Medicine) | 20 | Verbatim text of the page summary plus selected sections (causes, symptoms, exams/tests, or why-performed / abnormal results for imaging pages), cut only at a sentence boundary to fit ~300 words | U.S. Government work, public domain |
| PubMed abstracts (NCBI E-utilities) | 10 | A short verbatim abstract excerpt, cited by PMID, first author, journal and year | Abstracts remain the publishers' property; excerpted for prototype retrieval with attribution |

Nothing is paraphrased or rewritten. The only edits are mechanical:
markup is removed, bullet lists are joined with `; ` (ending in `.`),
and the stray space left before punctuation by stripped link tags is
removed.

## Per-record fields

`doc_id`, `source` (the human-readable citation shown in reports),
`url`, `title`, `retrieved` (date fetched), `text`.

## Re-verifying or refreshing

```bash
python -m backend.rag.build_knowledge_base   # re-fetches from NLM / NCBI and rewrites the JSON
python -m backend.rag.ingest                 # rebuilds the Qdrant collection from it
```

The build tool checks every PubMed PMID's title against PubMed before
writing it, and fails rather than writing a record with no abstract. Two
guidelines whose primary PubMed records carry no abstract (ACC/AHA 2017
and ISH 2020 hypertension) are *not* cited as the guideline: the ISH
2020 passage is the abstract of its separate key-messages summary paper
and is cited as exactly that.

## Caveats

- Prototype-scoped: 30 passages across headache, dizziness, blood
  pressure, cerebrovascular and common imaging topics — not a
  comprehensive medical corpus, and not curated by a clinician.
- MedlinePlus pages are consumer-health text, not guideline recommendations.
- No WHO / CDC / NICE passages are included (no clean public text
  endpoint; reuse terms vary) — the guidelines folder holds PubMed
  abstracts of published guidelines, scientific statements, consensus
  criteria and one clinical review instead.
