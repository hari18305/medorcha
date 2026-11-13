"""Developer tool: check (and re-fit) the reranker's score calibration.

    python -m backend.rag.calibrate_reranker

The confidence formula uses the reranker's 0-1 score directly as S_rag, so
the score scale (rerank.SCORE_CENTER / SCORE_SCALE, see reranker.py) has to
fit the CURRENT knowledge base and reranker model. Re-run this whenever
either changes. It never edits anything: it prints, for on-topic queries
(the test cases' own text) and off-topic queries (things the knowledge base
has no passages on), the raw cross-encoder logits and the calibrated scores
under the current constants, plus a suggested center.

Read the output as a sanity check, not a validation: it is a few dozen
queries on a 30-passage corpus. A good scale keeps on-topic scores well above
off-topic ones; a scale that squashes everything toward 0 (or 1) is the
failure this tool exists to catch.
"""

import glob
import json
import statistics
from pathlib import Path

from backend.rag.reranker import SCORE_CENTER, SCORE_SCALE, _get_model, relevance_score
from backend.rag.retriever import retrieve

TEST_CASES_GLOB = "data/test_cases/CASE0*.json"
from backend.agents.rag_agent import RETRIEVE_TOP_K  # the pool size production uses
KEEP_TOP_K = 5

# Queries the knowledge base has nothing on: a good S_rag must stay LOW here.
OFF_TOPIC_QUERIES = [
    "abdominal pain, vomiting, diarrhea for two days",
    "rash and joint pain after a tick bite",
    "knee pain after running, swelling",
    "difficulty urinating, lower back pain",
    "toothache and swollen gum",
    "red itchy eyes with discharge",
]


def _on_topic_queries() -> list[str]:
    queries, seen = [], set()
    for path in sorted(glob.glob(TEST_CASES_GLOB)):
        case = json.loads(Path(path).read_text(encoding="utf-8"))
        text = (case.get("clinical_text") or case.get("medical_report") or "").strip()
        if text and text not in seen:
            seen.add(text)
            queries.append(text)
    return queries


def _mean_top_logit(query: str) -> float:
    """Mean raw logit of the top-5 of the top-10 retrieved candidates."""
    candidates = retrieve(query, top_k=RETRIEVE_TOP_K)
    logits = sorted((float(x) for x in _get_model().predict([(query, c["text"]) for c in candidates])), reverse=True)
    return sum(logits[:KEEP_TOP_K]) / KEEP_TOP_K


def main() -> None:
    on = [_mean_top_logit(q) for q in _on_topic_queries()]
    off = [_mean_top_logit(q) for q in OFF_TOPIC_QUERIES]
    print(f"Current calibration: SCORE_CENTER={SCORE_CENTER}  SCORE_SCALE={SCORE_SCALE}\n")
    print(f"{'':<10} {'n':>3} {'raw logit mean (min..max)':>30} {'calibrated S_rag mean (min..max)':>36}")
    for name, logits in (("on-topic", on), ("off-topic", off)):
        scores = [relevance_score(x) for x in logits]
        print(
            f"{name:<10} {len(logits):>3} "
            f"{statistics.mean(logits):>10.1f} ({min(logits):.1f}..{max(logits):.1f})"
            f"{statistics.mean(scores):>20.2f} ({min(scores):.2f}..{max(scores):.2f})"
        )
    midpoint = (statistics.mean(on) + statistics.mean(off)) / 2
    print(f"\nSuggested center (midway between the two raw means): {midpoint:.1f}   (currently {SCORE_CENTER})")
    on_mean = statistics.mean(relevance_score(x) for x in on)
    if on_mean < 0.2:
        print("WARNING: on-topic S_rag is very low -- confidence will rarely reach the threshold. Lower SCORE_CENTER.")
    if min(relevance_score(x) for x in on) <= max(relevance_score(x) for x in off):
        print("Note: some on-topic queries score no higher than the best off-topic one (the knowledge base lacks content for them).")


if __name__ == "__main__":
    main()
