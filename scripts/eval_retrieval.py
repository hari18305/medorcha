"""Retrieval regression harness: does the right knowledge-base passage reach the top 5?

Runs the REAL two-stage pipeline (BioBERT dense retrieval -> cross-encoder
rerank) against an in-memory Qdrant built from the committed corpus, so it
never touches the real embedded store, needs no API key and no network.

The queries are the ones the pipeline's own agents built in the live
31-case run (recovered from the backend log), not hand-written paraphrases.
They are NOT hand-corrected: where the clinical agent dropped a detail, the
query is evaluated as it was, because that is what the system really sees.

The 'relevant' sets are developer judgements taken from answer_key.json
(NOT clinician-verified). 14 queries is a regression check, not a benchmark:
a change that moves one query is anecdote, a change that moves several the
same way is a signal.

    python -m scripts.eval_retrieval              # table + summary
    python -m scripts.eval_retrieval --json out.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient  # noqa: E402

from backend.agents.clinical_agent import merge_structured_fields  # noqa: E402
from backend.agents.rag_agent import _build_query  # noqa: E402
from backend.models.schemas import ClinicalFindings, ImagingFindings  # noqa: E402
from backend.rag import ingest, retriever  # noqa: E402
from backend.rag.reranker import rerank  # noqa: E402

def _query(symptoms: str, history=(), imaging=(), case_history=()) -> str:
    """The query the pipeline builds for a case: what the clinical agent extracted
    in the live run (`symptoms`, `history`, `imaging` - recorded verbatim, quirks
    included, e.g. 'history of history of ...'), plus the case's structured
    history merged in by merge_structured_fields, then the real _build_query."""
    findings = ClinicalFindings(
        demographics={}, symptoms=[symptoms] if symptoms else [], history=list(history), labs={},
        imaging_text_findings=[],
    )
    findings = merge_structured_fields(findings, list(case_history), {})
    img = ImagingFindings(modality="x", findings=list(imaging), confidence=0.9) if imaging else None
    return _build_query(findings, img)


# (label, query, relevant doc_ids). Relevant sets are developer judgements from answer_key.json.
QUERIES = [
    ("CASE001 headache+HTN", _query("headache, dizziness", ["hypertension"], case_history=["hypertension"]),
     {"hypertension_overview", "headache_overview", "headache_tension_type", "hypertensive_emergency_review",
      "white_matter_hyperintensities_small_vessel"}),
    ("CASE009 elderly warfarin fall",
     _query("gradual gait unsteadiness, mild headache", ["history of a fall two months ago", "on warfarin"],
            case_history=["hypertension", "atrial fibrillation", "on anticoagulation"]),
     {"chronic_subdural_hematoma"}),
    ("CASE010 child headache+URTI",
     _query("headache, absence of fever", ["recent upper respiratory infection"]),
     {"headache_overview", "headache_tension_type", "sinusitis_headache", "migraine_overview"}),
    ("CASE011 anemia dizzy standing",
     _query("fatigue, dizziness, especially on standing, mild headache", case_history=["iron deficiency anemia"]),
     {"anemia_overview", "orthostatic_hypotension_low_blood_pressure", "orthostatic_hypotension_in_hypertension_aha_2024"}),
    ("CASE014 migraine",
     _query("throbbing unilateral headache, nausea, light sensitivity", ["similar episodes in the past"],
            case_history=["migraine"]),
     {"migraine_overview", "vestibular_migraine_diagnostic_criteria"}),
    ("CASE015 BPPV",
     _query("spinning vertigo triggered by rolling over in bed, lasting under a minute, absence of headache"),
     {"bppv", "bppv_aao_hnsf_guideline_2017", "dizziness_vertigo_overview"}),
    ("CASE018 head injury+anticoag",
     _query("sudden severe headache, vomiting, drowsiness, confusion", ["fall two days ago"],
            ["normal head CT", "no acute findings"], case_history=["atrial fibrillation", "on warfarin"]),
     {"chronic_subdural_hematoma", "subarachnoid_hemorrhage", "traumatic_brain_injury_overview", "hemorrhagic_stroke_overview"}),
    ("CASE020 cough smoker nodule",
     _query("persistent cough, low-grade fever, breathlessness on exertion", ["long-time smoker"],
            ["pulmonary nodule or mass", "pneumothorax"], case_history=["long-time smoker"]),
     {"copd_chest_xray_context", "pneumonia_chest_xray", "pulmonary_tuberculosis", "lung_cancer_overview",
      "pulmonary_nodule_fleischner_2017"}),
    ("CASE022 pneumothorax", _query("pleuritic chest pain, shortness of breath", imaging=["pneumothorax"]),
     {"pneumothorax_collapsed_lung"}),
    ("REAL003 headache+blurred vision", _query("persistent headache, occasional blurred vision"),
     {"headache_overview", "headache_tension_type", "migraine_overview",
      "idiopathic_intracranial_hypertension_consensus_2018", "secondary_headache_red_flags_snnoop10"}),
    ("REAL004 memory+gait+atrophy",
     _query("gradual memory difficulty, unsteady walking", imaging=["cerebral atrophy"], case_history=["hypertension"]),
     {"alzheimer_disease", "dementia_overview", "vascular_contributions_to_cognitive_impairment_aha_2011",
      "idiopathic_nph_guidelines_2021", "hydrocephalus_overview", "white_matter_hyperintensities_small_vessel"}),
    ("REAL005 chest pain, normal CXR",
     _query("left-sided chest pain", imaging=["normal chest X-ray", "no acute findings"]),
     {"chest_injuries_rib_fracture", "chest_xray_overview"}),
    ("REAL006 pneumothorax text+img",
     _query("sudden sharp chest pain, shortness of breath", imaging=["pneumothorax"]),
     {"pneumothorax_collapsed_lung"}),
    ("REAL007 pneumothorax img only", _query("", imaging=["pneumothorax"]), {"pneumothorax_collapsed_lung"}),
]

RERANK_TOP_K = 5


def build_index() -> QdrantClient:
    client = QdrantClient(":memory:")
    ingest.ingest(client)
    retriever._client = client
    return client


def first_rank(results: list[dict], wanted: set[str]) -> int | None:
    for i, r in enumerate(results, 1):
        if r["doc_id"] in wanted:
            return i
    return None


def evaluate(queries=QUERIES, stage1_k: int = 10, pipeline=None) -> dict:
    """`pipeline(query) -> ranked results` lets a caller evaluate a changed
    pipeline; the default is the production one (retrieve stage1_k -> rerank 5)."""
    rows = []
    for label, query, wanted in queries:
        if pipeline is None:
            ranked = rerank(query, retriever.retrieve(query, top_k=stage1_k), top_k=RERANK_TOP_K)
        else:
            ranked = pipeline(query)
        # where the relevant passage sat in the dense stage alone (all docs)
        dense = retriever.retrieve(query, top_k=1000)
        rows.append({
            "label": label,
            "query": query,
            "dense_rank": first_rank(dense, wanted),
            "final_rank": first_rank(ranked, wanted),
            "top3": [r["doc_id"] for r in ranked[:3]],
            "top_score": round(ranked[0]["score"], 3) if ranked else None,
        })
    n = len(rows)
    hit5 = sum(r["final_rank"] is not None for r in rows)
    hit1 = sum(r["final_rank"] == 1 for r in rows)
    mrr = statistics.fmean((1 / r["final_rank"]) if r["final_rank"] else 0 for r in rows)
    return {"rows": rows, "n": n, "hit@5": hit5, "hit@1": hit1, "mrr": round(mrr, 3)}


def print_report(result: dict) -> None:
    print(f"{'query':32} {'dense':>6} {'final':>6}  top-3")
    for r in result["rows"]:
        flag = "" if r["final_rank"] else "   <-- MISS"
        print(f"{r['label']:32} {str(r['dense_rank'] or '-'):>6} {str(r['final_rank'] or '-'):>6}  "
              f"{[d[:20] for d in r['top3']]}{flag}")
    print(f"\nn={result['n']}  hit@5={result['hit@5']}/{result['n']}  hit@1={result['hit@1']}/{result['n']}  "
          f"MRR@5={result['mrr']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-k", type=int, default=10)
    ap.add_argument("--json", help="write the result to this path")
    args = ap.parse_args()
    build_index()
    res = evaluate(stage1_k=args.stage1_k)
    print_report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
