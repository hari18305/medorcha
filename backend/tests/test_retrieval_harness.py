"""The retrieval harness's relevance sets name knowledge-base passages by id; a rename or a removed passage would
silently turn a query into one that can never hit, so check every id exists."""
from backend.agents import rag_agent
from backend.rag import ingest
from scripts import eval_retrieval as E


def _ids():
    return {doc["doc_id"] for doc in ingest.load_corpus()}


def test_every_relevant_id_in_the_harness_exists_in_the_corpus():
    ids = _ids()
    for label, _query, wanted in E.QUERIES + E.GAP_QUERIES:
        missing = wanted - ids
        assert not missing, f"{label}: {sorted(missing)} not in the knowledge base"


def test_the_gap_set_lists_exactly_the_passages_it_calls_new():
    # GAP_NEW drives the "original 42-passage KB" baseline, so every entry must be a real passage
    assert E.GAP_NEW <= _ids()
    assert len(E.GAP_NEW) == 25


def test_the_candidate_pool_is_wider_than_what_is_kept_and_the_harness_uses_it():
    assert rag_agent.RETRIEVE_TOP_K > rag_agent.RERANK_TOP_K
    assert E.evaluate.__defaults__[1] == rag_agent.RETRIEVE_TOP_K  # stage1_k default follows production


def test_excluding_passages_drops_them_from_the_retrieval_order_without_reordering_the_rest(monkeypatch):
    ranked = [{"doc_id": d} for d in ("a", "new1", "b", "new2", "c", "d")]
    monkeypatch.setattr(E.retriever, "retrieve", lambda q, top_k: ranked[:top_k])
    kept = E._retrieve("q", 3, frozenset({"new1", "new2"}))
    assert [r["doc_id"] for r in kept] == ["a", "b", "c"]
    assert [r["doc_id"] for r in E._retrieve("q", 2)] == ["a", "new1"]  # nothing excluded: plain top-k


def test_rejudged_sets_only_add_to_the_original_ones_and_name_real_passages():
    ids = _ids()
    labels = {label for label, _q, _w in E.QUERIES}
    assert set(E.REJUDGED_EXTRA) <= labels  # a typo in a query label would silently drop its additions
    original = {label: wanted for label, _q, wanted in E.QUERIES}
    for label, _q, wanted in E.rejudged_queries():
        assert original[label] <= wanted  # never removes a passage the original key marked relevant
        assert wanted <= ids, f"{label}: {sorted(wanted - ids)}"
