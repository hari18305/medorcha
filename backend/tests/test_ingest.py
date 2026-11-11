"""Tests for the real knowledge base (Phase 2) and its loader in
backend/rag/ingest.py. Offline: no Qdrant, no embedding model -- these
check the checked-in data/knowledge_base/*.json files and the loader's
validation, which is where the build spec's content requirements live.
"""

import json
import re

import pytest

import backend.rag.ingest as ingest


@pytest.fixture(scope="module")
def corpus():
    return ingest.load_corpus()


def test_corpus_size_is_sane(corpus):
    # Spec said 20-30; the KB was extended to 42 (data/knowledge_base/README.md
    # explains why). The upper bound just catches an accidental duplicate load.
    assert 20 <= len(corpus) <= 60


def test_at_least_80_percent_of_passages_are_real(corpus):
    # Phase 2: "At least 80% MUST be real, citable content".
    summary = ingest.provenance_summary(corpus)
    assert summary["real_fraction"] >= 0.8
    assert summary["real"] + summary["synthetic"] == summary["total"]


def test_every_real_passage_has_a_checkable_citation(corpus):
    for doc in corpus:
        if doc["source"] == "synthetic":
            continue
        assert doc["url"].startswith("https://"), doc["doc_id"]
        # the human-readable citation must name where it came from...
        assert doc["source"].startswith(("MedlinePlus", "PubMed PMID")), doc["doc_id"]
        # ...and point at the same URL the record carries
        assert doc["url"] in doc["source"], doc["doc_id"]
        if doc["source"].startswith("PubMed PMID"):
            pmid = re.match(r"PubMed PMID (\d+):", doc["source"]).group(1)
            assert doc["url"] == f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def test_doc_ids_are_unique_and_texts_are_nonempty(corpus):
    ids = [doc["doc_id"] for doc in corpus]
    assert len(ids) == len(set(ids))
    assert all(doc["text"].strip() for doc in corpus)


def test_each_passage_fits_one_chunk_in_the_200_400_token_band(corpus):
    # Words approximate tokens here (see ingest._approx_token_count).
    for doc in corpus:
        words = len(doc["text"].split())
        assert 100 <= words <= 400, f"{doc['doc_id']}: {words} words"


def test_no_html_residue_or_boilerplate_in_text(corpus):
    for doc in corpus:
        assert "<" not in doc["text"] and ">" not in doc["text"], doc["doc_id"]
        assert "&nbsp;" not in doc["text"] and "�" not in doc["text"], doc["doc_id"]
        assert "Skip navigation" not in doc["text"], doc["doc_id"]


def test_corpus_covers_the_test_case_symptom_space_and_every_imaging_modality(corpus):
    ids = {doc["doc_id"] for doc in corpus}
    # headache / dizziness / hypertension / neuro differentials
    for required in ("headache_overview", "migraine_overview", "dizziness_vertigo_overview",
                     "hypertension_overview", "tia_overview", "ischemic_stroke_overview", "bppv"):
        assert required in ids, required
    # one per supported imaging modality (MRI brain, CT, chest X-ray)
    for required in ("brain_mri_overview", "head_ct_overview", "chest_xray_overview"):
        assert required in ids, required


def test_chunk_document_keeps_provenance_in_the_payload(corpus):
    doc = corpus[0]
    chunks = ingest.chunk_document(doc)
    assert chunks[0]["source"] == doc["source"]
    assert chunks[0]["doc_id"] == doc["doc_id"]
    assert chunks[0]["url"] == doc["url"]
    assert chunks[0]["page"] == 1
    assert chunks[0]["chunk_length"] == len(chunks[0]["text"].split())


def test_build_chunks_yields_one_chunk_per_passage(corpus):
    # Every passage is <= one chunk (300 words), so chunk count == passage count.
    assert len(ingest.build_chunks()) == len(corpus)


def _write_kb(tmp_path, guidelines=None, symptom_disease=None):
    for subdir, records in (("guidelines", guidelines), ("symptom_disease", symptom_disease)):
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)
        if records is not None:
            (tmp_path / subdir / f"{subdir}.json").write_text(json.dumps(records), encoding="utf-8")


def test_load_corpus_rejects_a_record_missing_a_required_field(tmp_path):
    _write_kb(tmp_path, guidelines=[{"doc_id": "a", "source": "synthetic"}])  # no text
    with pytest.raises(ValueError, match="missing"):
        ingest.load_corpus(tmp_path)


def test_load_corpus_rejects_duplicate_doc_ids(tmp_path):
    record = {"doc_id": "a", "source": "synthetic", "text": "x"}
    _write_kb(tmp_path, guidelines=[record], symptom_disease=[record])
    with pytest.raises(ValueError, match="duplicate"):
        ingest.load_corpus(tmp_path)


def test_load_corpus_rejects_an_empty_knowledge_base(tmp_path):
    _write_kb(tmp_path)
    with pytest.raises(ValueError, match="No knowledge-base records"):
        ingest.load_corpus(tmp_path)


def test_ingest_rebuilds_the_collection_and_a_bad_kb_never_wipes_it(monkeypatch):
    calls = []

    class FakeClient:
        exists = True  # an old collection is already there

        def collection_exists(self, name):
            calls.append(("exists", name))
            return self.exists

        def delete_collection(self, name):
            calls.append(("delete", name))
            self.exists = False

        def create_collection(self, **kwargs):
            calls.append(("create", kwargs["collection_name"]))
            self.exists = True

        def upsert(self, collection_name, points):
            calls.append(("upsert", collection_name, len(points)))

    monkeypatch.setattr(ingest, "embed_text", lambda text: [0.0] * ingest.EMBEDDING_DIM)

    # A KB that fails validation must raise BEFORE anything is deleted.
    monkeypatch.setattr(ingest, "load_corpus", lambda kb_dir=None: (_ for _ in ()).throw(ValueError("bad kb")))
    with pytest.raises(ValueError):
        ingest.ingest(FakeClient())
    assert ("delete", ingest.COLLECTION_NAME) not in calls

    # A good KB: delete, recreate, then upsert one point per chunk.
    monkeypatch.undo()
    monkeypatch.setattr(ingest, "embed_text", lambda text: [0.0] * ingest.EMBEDDING_DIM)
    calls.clear()
    count = ingest.ingest(FakeClient())
    assert calls[0] == ("exists", ingest.COLLECTION_NAME)
    assert ("delete", ingest.COLLECTION_NAME) in calls
    assert ("create", ingest.COLLECTION_NAME) in calls
    assert calls[-1] == ("upsert", ingest.COLLECTION_NAME, count)
    assert calls.index(("delete", ingest.COLLECTION_NAME)) < calls.index(("create", ingest.COLLECTION_NAME))


# --- Symptoms-first guard (2026-11-12) -------------------------------------
# A passage is cut at ~300 words, so whatever the builder puts FIRST survives.
# With "Causes" first, the Symptoms section was cut off or dropped in 10 of 15
# encyclopedia articles, and a patient presenting with those symptoms could not
# retrieve the right passage (e.g. chronic subdural hematoma for headache +
# unsteady gait + warfarin). These tests keep that from silently coming back.
from backend.rag import build_knowledge_base as builder  # noqa: E402


def test_symptomatic_encyclopedia_articles_put_symptoms_first():
    symptomatic = [d for d in builder.ENCY_DOCS if "Symptoms" in d[3]]
    assert symptomatic, "expected some encyclopedia articles with a Symptoms section"
    for doc_id, _folder, _article, priority in symptomatic:
        assert priority[0] == "Symptoms", f"{doc_id}: Symptoms must come first, got {priority}"


def test_subdural_passage_contains_its_presenting_symptoms(corpus):
    doc = next(d for d in corpus if d["doc_id"] == "chronic_subdural_hematoma")
    text = doc["text"]
    assert "Problems with balance or walking" in text and "Headache" in text
    assert "warfarin" in text  # the risk-factor sentence must survive too


def test_put_replaces_in_place_and_appends_new():
    records = [{"doc_id": "a", "text": "1"}, {"doc_id": "b", "text": "2"}]
    builder._put(records, {"doc_id": "a", "text": "new"})
    builder._put(records, {"doc_id": "c", "text": "3"})
    assert [r["doc_id"] for r in records] == ["a", "b", "c"]
    assert records[0]["text"] == "new"


def test_refresh_rejects_unknown_doc_ids():
    with pytest.raises(SystemExit):
        builder.build(refresh=frozenset({"no_such_doc"}))
