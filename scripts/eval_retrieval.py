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
from backend.agents.rag_agent import RETRIEVE_TOP_K, _build_query  # noqa: E402
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

# --- Re-judged relevance (2026-11-13) -------------------------------------------------------------------------
# The 14 relevant sets above were written for the 42-passage KB. After it grew to 67, new passages (fainting,
# fatigue, concussion, ...) began to outrank the marked one, and it was unclear how much of the MRR drop was that
# stale key. For each query the 20 dense candidates were listed ALPHABETICALLY (no ranks shown) and every one a
# clinician would plausibly want as evidence for that presentation was added, written down BEFORE this was
# re-scored. Original passages are never removed. Judged by one non-clinician (the developer, with an AI
# assistant) on clinical plausibility alone; it is an audit of the key, not a clinical judgement.
REJUDGED_EXTRA = {
    "CASE001 headache+HTN": {"migraine_overview", "vestibular_migraine_diagnostic_criteria", "secondary_headache_red_flags_snnoop10",
                             "orthostatic_hypotension_in_hypertension_aha_2024"},
    "CASE009 elderly warfarin fall": {"subarachnoid_hemorrhage", "idiopathic_nph_guidelines_2021", "secondary_headache_red_flags_snnoop10"},
    "CASE011 anemia dizzy standing": {"fatigue", "fainting", "dizziness_vertigo_overview"},
    "CASE014 migraine": {"headache_overview", "headache_tension_type", "cluster_headache"},
    "CASE015 BPPV": {"menieres_disease", "vestibular_migraine_diagnostic_criteria"},
    "CASE018 head injury+anticoag": {"concussion", "secondary_headache_red_flags_snnoop10"},
    "CASE020 cough smoker nodule": {"pneumothorax_collapsed_lung"},
    "REAL005 chest pain, normal CXR": {"chest_pain", "chest_pain_guideline_2021", "musculoskeletal_chest_pain_prevalence"},
}


def rejudged_queries():
    return [(label, q, wanted | REJUDGED_EXTRA.get(label, set())) for label, q, wanted in QUERIES]


# --- Knowledge-base gap queries (written 2026-11-13, before the 25 gap passages were first retrieved) -------------
# Queries phrased the way the agents build them (symptom phrases, imaging labels), for conditions the original
# 42-passage KB had nothing on. Relevant sets include an existing passage where one is genuinely relevant. CAVEAT: the
# author knew the new passages existed, so this measures COVERAGE (is there anything to retrieve?) more than
# retrieval quality, and it is small. `GAP_NEW` lets the same queries be run as if the new passages did not exist.
GAP_NEW = {
    "pulmonary_edema", "atelectasis", "cluster_headache", "brain_abscess", "heart_failure", "brain_tumors", "meningitis",
    "chest_pain", "fainting", "fatigue", "multiple_sclerosis", "giant_cell_arteritis", "concussion", "brain_aneurysm",
    "arteriovenous_malformations", "aortic_aneurysm", "long_covid", "refractive_errors",
    "vertebrobasilar_insufficiency_review", "pres_review_2025", "intracranial_calcification_vs_hemorrhage_ct",
    "long_covid_headache", "musculoskeletal_chest_pain_prevalence", "massive_cerebral_infarction_review",
    "chest_pain_guideline_2021",
}
GAP_QUERIES = [
    ("G pulmonary edema (text)", _query("shortness of breath lying flat, frothy sputum, ankle swelling", imaging=["pulmonary edema", "cardiomegaly"]),
     {"pulmonary_edema", "heart_failure"}),
    ("G pulmonary edema (image only)", _query("", imaging=["pulmonary edema"]), {"pulmonary_edema", "heart_failure"}),
    ("G atelectasis (image only)", _query("", imaging=["atelectasis"]), {"atelectasis"}),
    ("G mediastinal widening", _query("sudden tearing chest pain radiating to the back", imaging=["hilar or mediastinal widening"]),
     {"aortic_aneurysm"}),
    ("G brain mass (image only)", _query("", imaging=["mass lesion"]), {"brain_tumors"}),
    ("G vascular malformation (image)", _query("", imaging=["vascular malformation"]),
     {"arteriovenous_malformations", "brain_aneurysm", "hemorrhagic_stroke_overview"}),
    ("G midline shift (image)", _query("headache, drowsiness", imaging=["midline shift"]),
     {"massive_cerebral_infarction_review", "traumatic_brain_injury_overview", "chronic_subdural_hematoma"}),
    ("G CT calcification (image)", _query("", imaging=["calcification"]), {"intracranial_calcification_vs_hemorrhage_ct"}),
    ("G cluster headache", _query("severe one-sided eye pain with tearing, attacks at the same time each night"), {"cluster_headache"}),
    ("G temporal arteritis", _query("new headache, jaw pain when chewing, scalp tenderness, visual loss", case_history=["age 72"]),
     {"giant_cell_arteritis"}),
    ("G meningitis", _query("fever, severe headache, stiff neck, sensitivity to light"),
     {"meningitis", "brain_abscess", "secondary_headache_red_flags_snnoop10"}),
    ("G exertional chest pain", _query("chest pressure on exertion radiating to the left arm, nausea"),
     {"chest_pain", "chest_pain_guideline_2021"}),
    ("G fainting", _query("fainted after standing up, lightheaded beforehand"),
     {"fainting", "orthostatic_hypotension_low_blood_pressure"}),
    ("G chronic fatigue", _query("tired for months, unrefreshed sleep, worse after exertion, poor concentration"), {"fatigue", "long_covid"}),
    ("G vertebrobasilar", _query("recurrent vertigo with double vision and drop attacks"),
     {"vertebrobasilar_insufficiency_review", "tia_overview"}),
    ("G hypertensive encephalopathy", _query("headache, seizure, confusion and visual disturbance, blood pressure 220/120"),
     {"pres_review_2025", "hypertensive_emergency_review"}),
    ("G post-COVID headache", _query("headache persisting for weeks after COVID-19 infection"), {"long_covid_headache", "long_covid"}),
    ("G refractive error", _query("blurred distance vision, squinting, headaches after reading"), {"refractive_errors"}),
    ("G multiple sclerosis", _query("relapsing numbness and optic neuritis in a young woman, white matter lesions"), {"multiple_sclerosis"}),
    ("G concussion", _query("blow to the head, brief loss of consciousness, headache, nausea, confusion"),
     {"concussion", "traumatic_brain_injury_overview"}),
    ("G musculoskeletal chest pain", _query("chest wall pain reproduced by pressing on the ribs, normal ECG and chest X-ray"),
     {"musculoskeletal_chest_pain_prevalence", "chest_injuries_rib_fracture"}),
]



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


def _retrieve(query: str, k: int, exclude: frozenset = frozenset()) -> list[dict]:
    """Dense top-k, as if the doc_ids in `exclude` were not in the index (dense order does not depend on
    the other passages, so dropping them from a larger list and taking k is exactly what a smaller index returns)."""
    if not exclude:
        return retriever.retrieve(query, top_k=k)
    return [r for r in retriever.retrieve(query, top_k=1000) if r["doc_id"] not in exclude][:k]


def evaluate(queries=QUERIES, stage1_k: int = RETRIEVE_TOP_K, pipeline=None, exclude: frozenset = frozenset()) -> dict:
    """`pipeline(query) -> ranked results` lets a caller evaluate a changed
    pipeline; the default is the production one (retrieve stage1_k -> rerank 5)."""
    rows = []
    for label, query, wanted in queries:
        if pipeline is None:
            ranked = rerank(query, _retrieve(query, stage1_k, exclude), top_k=RERANK_TOP_K)
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
    ap.add_argument("--stage1-k", type=int, default=RETRIEVE_TOP_K)
    ap.add_argument("--json", help="write the result to this path")
    ap.add_argument("--gap", action="store_true", help="run the knowledge-base gap queries (with and without the new passages)")
    args = ap.parse_args()
    build_index()
    if args.gap:
        full = evaluate(GAP_QUERIES, stage1_k=RETRIEVE_TOP_K)
        # the original system: 42 passages and a top-10 candidate pool
        old = evaluate(GAP_QUERIES, stage1_k=10, exclude=frozenset(GAP_NEW))
        print("=== gap queries, 67-passage KB, top-20 pool ===")
        print_report(full)
        print("\n=== same queries, original 42-passage KB, top-10 pool ===")
        print_report(old)
        if args.json:
            Path(args.json).write_text(json.dumps({"with_new": full, "original": old}, indent=2), encoding="utf-8")
        raise SystemExit(0)
    res = evaluate(stage1_k=args.stage1_k)
    print_report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
