"""Tests for backend.agents.evidence_agent.

All httpx calls are mocked via a fake httpx.AsyncClient (no real
network access, no real Anthropic calls) so the suite is fully offline.
Covers: PubMed success, timeout-with-retry, empty-results; MedlinePlus,
Europe PMC and ClinicalTrials.gov success / empty / failure; the
evidence-level heuristics; abstract summarization (short/long/Claude-
failure fallback); title de-duplication; and gather_live_evidence's
concurrent four-source merge, per-source failure isolation, "unavailable
only if ALL sources failed" signalling, per-source deadline, and
in-memory cache behavior.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import backend.agents.evidence_agent as evidence_agent
from backend.agents.evidence_agent import (
    EvidenceSourceUnavailableError,
    _assess_evidence_level,
    _dedupe_by_title,
    _summarize_abstract,
    gather_live_evidence,
    search_clinicaltrials,
    search_europepmc,
    search_medlineplus_guidelines,
    search_pubmed,
)
from backend.models.schemas import LiveEvidence

PUBMED_ESEARCH_JSON = {"esearchresult": {"idlist": ["111", "222"]}}
PUBMED_ESEARCH_EMPTY_JSON = {"esearchresult": {"idlist": []}}
PUBMED_ESUMMARY_JSON = {
    "result": {
        "uids": ["111", "222"],
        "111": {"title": "A Systematic Review of Headache in Hypertensive Patients", "pubdate": "2024 Jan"},
        "222": {"title": "A Small Case Report on Dizziness", "pubdate": "2023 Jun"},
    }
}
PUBMED_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID Version="1">111</PMID>
<Article><Abstract><AbstractText>This systematic review examines headache in hypertensive patients.</AbstractText></Abstract></Article>
</MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID Version="1">222</PMID>
<Article><Abstract><AbstractText>A brief case report describing a single patient with dizziness.</AbstractText></Abstract></Article>
</MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""

MEDLINEPLUS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nlmSearchResult>
  <list num="1" start="0" per="1">
    <document rank="0" url="https://medlineplus.gov/highbloodpressure.html">
      <content name="title">&lt;span class="qt0"&gt;High Blood Pressure&lt;/span&gt;</content>
      <content name="organizationName">National Library of Medicine</content>
      <content name="FullSummary">&lt;p&gt;Blood pressure guideline overview for patients.&lt;/p&gt;</content>
    </document>
  </list>
</nlmSearchResult>"""


class _FakeAsyncClient:
    """Mimics `async with httpx.AsyncClient() as client: await client.get(...)`."""

    def __init__(self, handler):
        self._handler = handler
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self._handler(url, params)


def _json_response(url: str, payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("GET", url))


def _text_response(url: str, text: str) -> httpx.Response:
    return httpx.Response(200, text=text, request=httpx.Request("GET", url))


def _patch_client(monkeypatch, handler):
    fake_client = _FakeAsyncClient(handler)
    monkeypatch.setattr(evidence_agent.httpx, "AsyncClient", lambda *a, **kw: fake_client)
    return fake_client


# --------------------------------------------------------------------------
# search_pubmed
# --------------------------------------------------------------------------


async def test_search_pubmed_success(monkeypatch):
    def handler(url, params):
        if url == evidence_agent.PUBMED_ESEARCH_URL:
            return _json_response(url, PUBMED_ESEARCH_JSON)
        if url == evidence_agent.PUBMED_ESUMMARY_URL:
            return _json_response(url, PUBMED_ESUMMARY_JSON)
        if url == evidence_agent.PUBMED_EFETCH_URL:
            return _text_response(url, PUBMED_EFETCH_XML)
        raise AssertionError(f"unexpected URL {url}")

    _patch_client(monkeypatch, handler)

    results = await search_pubmed("headache dizziness hypertension", max_results=2)

    assert len(results) == 2
    first = results[0]
    assert first["title"] == "A Systematic Review of Headache in Hypertensive Patients"
    assert first["source"] == "PubMed"
    assert first["url"] == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert "systematic review" in first["summary"].lower()
    assert first["evidence_level"] == "high"  # "systematic review" keyword
    assert first["publication_date"] == "2024 Jan"

    second = results[1]
    assert second["evidence_level"] == "low"  # case report, no keyword match


async def test_search_pubmed_empty_results_short_circuits(monkeypatch):
    def handler(url, params):
        if url == evidence_agent.PUBMED_ESEARCH_URL:
            return _json_response(url, PUBMED_ESEARCH_EMPTY_JSON)
        raise AssertionError("esummary/efetch should not be called when idlist is empty")

    fake_client = _patch_client(monkeypatch, handler)

    results = await search_pubmed("a query with no hits")

    assert results == []
    assert len(fake_client.calls) == 1  # only esearch was called


async def test_search_pubmed_timeout_retries_then_raises_unavailable(monkeypatch):
    call_count = 0

    def handler(url, params):
        nonlocal call_count
        call_count += 1
        raise httpx.TimeoutException("simulated timeout")

    _patch_client(monkeypatch, handler)

    # A failed source RAISES (rather than returning []) so the caller can
    # tell "PubMed is down" apart from "PubMed found nothing".
    with pytest.raises(EvidenceSourceUnavailableError):
        await search_pubmed("headache dizziness hypertension")

    assert call_count == evidence_agent.MAX_ATTEMPTS  # 1 initial + 1 retry, then give up


async def test_search_pubmed_efetch_failure_degrades_gracefully(monkeypatch):
    def handler(url, params):
        if url == evidence_agent.PUBMED_ESEARCH_URL:
            return _json_response(url, PUBMED_ESEARCH_JSON)
        if url == evidence_agent.PUBMED_ESUMMARY_URL:
            return _json_response(url, PUBMED_ESUMMARY_JSON)
        if url == evidence_agent.PUBMED_EFETCH_URL:
            raise httpx.TimeoutException("efetch down")
        raise AssertionError(f"unexpected URL {url}")

    _patch_client(monkeypatch, handler)

    results = await search_pubmed("headache dizziness hypertension", max_results=2)

    # titles still come through even though abstracts failed
    assert len(results) == 2
    assert results[0]["title"] == "A Systematic Review of Headache in Hypertensive Patients"
    assert results[0]["summary"] == "(no abstract available)"


# --------------------------------------------------------------------------
# search_medlineplus_guidelines
# --------------------------------------------------------------------------


async def test_search_medlineplus_success(monkeypatch):
    def handler(url, params):
        assert url == evidence_agent.MEDLINEPLUS_SEARCH_URL
        return _text_response(url, MEDLINEPLUS_XML)

    _patch_client(monkeypatch, handler)

    results = await search_medlineplus_guidelines("hypertension", max_results=3)

    assert len(results) == 1
    assert results[0]["title"] == "High Blood Pressure"
    assert results[0]["source"] == "MedlinePlus"
    assert results[0]["url"] == "https://medlineplus.gov/highbloodpressure.html"
    assert "guideline overview" in results[0]["summary"]
    assert "<p>" not in results[0]["summary"]  # HTML stripped


async def test_search_medlineplus_timeout_raises_unavailable(monkeypatch):
    def handler(url, params):
        raise httpx.TimeoutException("simulated timeout")

    _patch_client(monkeypatch, handler)

    with pytest.raises(EvidenceSourceUnavailableError):
        await search_medlineplus_guidelines("hypertension")


# --------------------------------------------------------------------------
# Evidence-level heuristic
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A systematic review of migraine treatment", "high"),
        ("Results from a randomized controlled trial of triptans", "high"),
        ("A prospective cohort study of dizziness in older adults", "moderate"),
        ("A narrative review of vestibular disorders", "moderate"),
        ("A single case report of an unusual presentation", "low"),
        ("", "low"),
    ],
)
def test_assess_evidence_level(text, expected):
    assert _assess_evidence_level(text) == expected


# --------------------------------------------------------------------------
# Abstract summarization
# --------------------------------------------------------------------------


async def test_summarize_abstract_short_returned_as_is():
    short_abstract = "A short abstract under the threshold."
    result = await _summarize_abstract(short_abstract)
    assert result == short_abstract


async def test_summarize_abstract_empty():
    result = await _summarize_abstract("")
    assert result == "(no abstract available)"


async def test_summarize_abstract_long_uses_claude(monkeypatch):
    long_abstract = "word " * 200  # well over ABSTRACT_SUMMARIZE_THRESHOLD_CHARS

    fake_client = MagicMock()
    fake_block = MagicMock()
    fake_block.text = "Concise Claude-generated summary."
    fake_response = MagicMock()
    fake_response.content = [fake_block]
    fake_client.messages.create = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(evidence_agent, "_get_anthropic_client", lambda: fake_client)

    result = await _summarize_abstract(long_abstract)

    assert result == "Concise Claude-generated summary."
    fake_client.messages.create.assert_awaited_once()


async def test_summarize_abstract_claude_failure_falls_back_to_truncation(monkeypatch):
    long_abstract = "word " * 200

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
    monkeypatch.setattr(evidence_agent, "_get_anthropic_client", lambda: fake_client)

    result = await _summarize_abstract(long_abstract)

    assert result.endswith("...")
    assert len(result) <= 310  # truncated, not the full 1000-char abstract


# --------------------------------------------------------------------------
# search_europepmc
# --------------------------------------------------------------------------

EUROPEPMC_JSON = {
    "hitCount": 3,
    "resultList": {
        "result": [
            {
                "id": "31000001", "source": "MED", "pmid": "31000001",
                "title": "Candesartan for <i>migraine</i> prophylaxis: a randomized trial.",
                "abstractText": "<h4>Background</h4> A <b>randomized</b> placebo-controlled study in adults.",
                "pubYear": "2019", "firstPublicationDate": "2019-04-02",
                "pubTypeList": {"pubType": ["Journal Article", "Randomized Controlled Trial"]},
            },
            {
                "id": "PMC7000002", "source": "PMC",
                "title": "Vertigo in older adults",
                "abstractText": "An overview of causes.", "pubYear": "2021",
                "pubTypeList": {"pubType": ["Journal Article", "Review"]},
            },
            {
                "id": "PPR12345", "source": "PPR",
                "title": "A systematic review preprint on headache",
                "abstractText": "Not yet peer reviewed.", "pubYear": "2024",
            },
        ],
    },
}


async def test_search_europepmc_success_maps_fields_and_levels(monkeypatch):
    seen = {}

    def handler(url, params):
        assert url == evidence_agent.EUROPEPMC_SEARCH_URL
        seen.update(params)
        return _json_response(url, EUROPEPMC_JSON)

    _patch_client(monkeypatch, handler)

    results = await search_europepmc("migraine prophylaxis", max_results=5)

    assert seen["query"] == "migraine prophylaxis" and seen["format"] == "json"
    assert seen["resultType"] == "core"  # core includes the abstract text
    assert [r["source"] for r in results] == ["Europe PMC"] * 3

    first, second, third = results
    assert first["title"] == "Candesartan for migraine prophylaxis: a randomized trial"  # tags + trailing period stripped
    assert first["url"] == "https://europepmc.org/article/MED/31000001"
    assert "<b>" not in first["summary"] and "randomized" in first["summary"]
    assert first["publication_date"] == "2019-04-02"
    assert first["evidence_level"] == "high"  # pubType: Randomized Controlled Trial
    assert second["url"] == "https://europepmc.org/article/PMC/PMC7000002"
    assert second["evidence_level"] == "moderate"  # pubType: Review
    assert second["publication_date"] == "2021"  # falls back to pubYear
    # a preprint is capped at "low" even though its title says "systematic review"
    assert third["evidence_level"] == "low"


async def test_search_europepmc_empty_results_returns_empty_list(monkeypatch):
    _patch_client(monkeypatch, lambda url, params: _json_response(url, {"hitCount": 0, "resultList": {"result": []}}))

    assert await search_europepmc("zzz no such thing") == []


async def test_search_europepmc_503_retries_then_raises_unavailable(monkeypatch):
    calls = 0

    def handler(url, params):
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"errCode": 503}, request=httpx.Request("GET", url))

    _patch_client(monkeypatch, handler)

    with pytest.raises(EvidenceSourceUnavailableError):
        await search_europepmc("headache")
    assert calls == evidence_agent.MAX_ATTEMPTS


async def test_search_europepmc_non_json_body_raises_unavailable(monkeypatch):
    _patch_client(monkeypatch, lambda url, params: _text_response(url, "<html>maintenance page</html>"))

    with pytest.raises(EvidenceSourceUnavailableError):
        await search_europepmc("headache")


# --------------------------------------------------------------------------
# search_clinicaltrials
# --------------------------------------------------------------------------


def _study(nct, title, status, phases, summary, start="2022-01"):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": title},
            "statusModule": {"overallStatus": status, "startDateStruct": {"date": start}},
            "designModule": {"phases": phases},
            "descriptionModule": {"briefSummary": summary},
        }
    }


CLINICALTRIALS_JSON = {
    "studies": [
        _study("NCT00000001", "Valsartan vs combination therapy", "COMPLETED", ["PHASE3"], "Compares two regimens."),
        _study("NCT00000002", "Early dose-finding in migraine", "RECRUITING", ["PHASE1"], "A small safety study."),
        _study("NCT00000003", "Behavioural vertigo programme", "COMPLETED", [], ""),
    ]
}


async def test_search_clinicaltrials_success_maps_fields_and_levels(monkeypatch):
    seen = {}

    def handler(url, params):
        assert url == evidence_agent.CLINICALTRIALS_SEARCH_URL
        seen.update(params)
        return _json_response(url, CLINICALTRIALS_JSON)

    _patch_client(monkeypatch, handler)

    results = await search_clinicaltrials("hypertension headache", max_results=3)

    assert seen["query.term"] == "hypertension headache" and seen["pageSize"] == 3
    first, second, third = results
    assert first["source"] == "ClinicalTrials.gov"
    assert first["url"] == "https://clinicaltrials.gov/study/NCT00000001"
    assert first["title"] == "Valsartan vs combination therapy"
    assert first["summary"].startswith("Status: COMPLETED. Phase: PHASE3.")
    assert "Compares two regimens." in first["summary"]
    assert first["publication_date"] == "2022-01"
    assert first["evidence_level"] == "high"        # completed Phase 3
    assert second["evidence_level"] == "moderate"   # recruiting Phase 1
    assert third["evidence_level"] == "moderate"    # completed but no phase
    assert third["summary"] == "Status: COMPLETED. Phase: not applicable."  # no brief summary


async def test_search_clinicaltrials_empty_results_returns_empty_list(monkeypatch):
    _patch_client(monkeypatch, lambda url, params: _json_response(url, {"studies": []}))

    assert await search_clinicaltrials("zzz") == []


async def test_search_clinicaltrials_timeout_raises_unavailable(monkeypatch):
    def handler(url, params):
        raise httpx.TimeoutException("simulated timeout")

    _patch_client(monkeypatch, handler)

    with pytest.raises(EvidenceSourceUnavailableError):
        await search_clinicaltrials("hypertension")


# --------------------------------------------------------------------------
# _strip_html -- regression for text run together after tag removal
# --------------------------------------------------------------------------


def test_strip_html_keeps_section_labels_and_paragraphs_apart():
    # Real Europe PMC shape: no whitespace between the label tag and the text.
    raw = "<h4>Background</h4>Pneumothorax is a common emergency.<h4>Methods</h4>A rat model was established."

    assert evidence_agent._strip_html(raw) == (
        "Background Pneumothorax is a common emergency. Methods A rat model was established."
    )


def test_strip_html_joins_list_items_instead_of_gluing_them():
    # Real MedlinePlus shape: paragraphs and list items with no whitespace.
    raw = (
        "<p>It is called pneumothorax.</p><p>Causes include:</p>"
        "<ul><li>Lung diseases such as pneumonia</li><li>Being on a breathing machine</li></ul>"
    )

    assert evidence_agent._strip_html(raw) == (
        "It is called pneumothorax. Causes include: Lung diseases such as pneumonia; Being on a breathing machine."
    )


def test_strip_html_does_not_split_words_at_inline_tags_or_leave_stray_spaces():
    assert evidence_agent._strip_html("Candesartan for <i>migraine</i>s, and <b>CO</b><sub>2</sub>.") == (
        "Candesartan for migraines, and CO2."
    )
    assert evidence_agent._strip_html("a &amp; b &lt;c&gt;") == "a & b <c>"
    assert evidence_agent._strip_html(None) == "" and evidence_agent._strip_html("") == ""


# --------------------------------------------------------------------------
# _dedupe_by_title
# --------------------------------------------------------------------------


def _item(title, source="PubMed"):
    return {"title": title, "source": source, "url": f"https://x/{source}/{title[:5]}", "summary": "s",
            "evidence_level": "low", "publication_date": None}


def test_dedupe_drops_same_paper_across_sources_keeping_the_first():
    items = [
        _item("Candesartan for Migraine Prophylaxis: A Randomized Trial", "PubMed"),
        _item("Candesartan for migraine prophylaxis: a randomized trial.", "Europe PMC"),  # case/punctuation only
        _item("Candesartan for migraine prophylaxis; a randomised trial", "Europe PMC"),   # near-identical spelling
        _item("An unrelated paper on vestibular rehabilitation", "Europe PMC"),
    ]

    kept = _dedupe_by_title(items)

    assert [i["source"] for i in kept] == ["PubMed", "Europe PMC"]
    assert kept[0]["title"].startswith("Candesartan")
    assert kept[1]["title"].startswith("An unrelated")


def test_dedupe_keeps_distinct_titles_and_preserves_order():
    items = [_item("Headache in hypertension"), _item("Dizziness in the elderly"), _item("Migraine and stroke risk")]

    assert _dedupe_by_title(items) == items


# --------------------------------------------------------------------------
# gather_live_evidence: four sources, concurrency, failure semantics, cache
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    evidence_agent._live_evidence_cache.clear()
    yield
    evidence_agent._live_evidence_cache.clear()


def _src_item(source, title, level="moderate"):
    return {"title": title, "source": source, "url": f"https://example.org/{source}/{title}", "summary": "summary",
            "evidence_level": level, "publication_date": None}


def _patch_sources(monkeypatch, pubmed=None, medlineplus=None, europepmc=None, clinicaltrials=None):
    """Each arg is a list (returned) or an Exception (raised)."""

    def make(value):
        if isinstance(value, Exception):
            return AsyncMock(side_effect=value)
        return AsyncMock(return_value=value if value is not None else [])

    mocks = {
        "search_pubmed": make(pubmed), "search_medlineplus_guidelines": make(medlineplus),
        "search_europepmc": make(europepmc), "search_clinicaltrials": make(clinicaltrials),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(evidence_agent, name, mock)
    return mocks


async def test_gather_live_evidence_merges_all_four_sources(monkeypatch):
    _patch_sources(
        monkeypatch,
        pubmed=[_src_item("PubMed", "PubMed paper")],
        medlineplus=[_src_item("MedlinePlus", "Overview")],
        europepmc=[_src_item("Europe PMC", "EPMC paper")],
        clinicaltrials=[_src_item("ClinicalTrials.gov", "A trial")],
    )

    result = await gather_live_evidence("headache dizziness hypertension")

    assert isinstance(result, LiveEvidence)
    assert [s.source for s in result.sources] == ["PubMed", "MedlinePlus", "Europe PMC", "ClinicalTrials.gov"]
    assert result.sources_queried == 4 and result.failed_sources == []
    assert result.all_sources_failed is False


async def test_gather_live_evidence_runs_the_four_sources_concurrently(monkeypatch):
    """Every source blocks until ALL four have started -- only possible if
    they run concurrently. Sequential execution would time out."""
    started = 0
    all_started = asyncio.Event()

    def make(source):
        async def search(query):
            nonlocal started
            started += 1
            if started == 4:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=2)
            return [_src_item(source, f"{source} title")]

        return search

    monkeypatch.setattr(evidence_agent, "search_pubmed", make("PubMed"))
    monkeypatch.setattr(evidence_agent, "search_medlineplus_guidelines", make("MedlinePlus"))
    monkeypatch.setattr(evidence_agent, "search_europepmc", make("Europe PMC"))
    monkeypatch.setattr(evidence_agent, "search_clinicaltrials", make("ClinicalTrials.gov"))

    result = await gather_live_evidence("concurrency probe")

    assert len(result.sources) == 4 and result.failed_sources == []


async def test_gather_live_evidence_dedupes_pubmed_vs_europepmc(monkeypatch):
    _patch_sources(
        monkeypatch,
        pubmed=[_src_item("PubMed", "Candesartan for Migraine Prophylaxis")],
        europepmc=[_src_item("Europe PMC", "Candesartan for migraine prophylaxis."), _src_item("Europe PMC", "Another paper")],
    )

    result = await gather_live_evidence("migraine")

    assert [(s.source, s.title) for s in result.sources] == [
        ("PubMed", "Candesartan for Migraine Prophylaxis"), ("Europe PMC", "Another paper"),
    ]


async def test_gather_live_evidence_one_source_down_others_still_contribute(monkeypatch):
    _patch_sources(
        monkeypatch,
        pubmed=[_src_item("PubMed", "PubMed paper")],
        medlineplus=EvidenceSourceUnavailableError("MedlinePlus unavailable"),
        europepmc=EvidenceSourceUnavailableError("Europe PMC 503"),
        clinicaltrials=[_src_item("ClinicalTrials.gov", "A trial")],
    )

    result = await gather_live_evidence("headache")

    assert [s.source for s in result.sources] == ["PubMed", "ClinicalTrials.gov"]
    assert result.failed_sources == ["MedlinePlus", "Europe PMC"]
    assert result.all_sources_failed is False  # NOT unavailable: two sources answered


async def test_gather_live_evidence_unexpected_error_in_one_source_is_isolated(monkeypatch):
    _patch_sources(
        monkeypatch,
        pubmed=RuntimeError("a bug inside the pubmed parser"),
        medlineplus=[_src_item("MedlinePlus", "Overview")],
    )

    result = await gather_live_evidence("headache")  # must not raise

    assert [s.source for s in result.sources] == ["MedlinePlus"]
    assert result.failed_sources == ["PubMed"]


async def test_gather_live_evidence_all_four_down_is_unavailable_and_not_cached(monkeypatch):
    down = EvidenceSourceUnavailableError("down")
    mocks = _patch_sources(monkeypatch, pubmed=down, medlineplus=down, europepmc=down, clinicaltrials=down)

    result = await gather_live_evidence("an unreachable query")

    assert result.sources == []
    assert result.sources_queried == 4
    assert result.failed_sources == ["PubMed", "MedlinePlus", "Europe PMC", "ClinicalTrials.gov"]
    assert result.all_sources_failed is True
    # a total outage must not be pinned to this query: a later call retries
    assert "an unreachable query" not in evidence_agent._live_evidence_cache
    await gather_live_evidence("an unreachable query")
    assert mocks["search_pubmed"].await_count == 2


async def test_gather_live_evidence_all_sources_answer_with_nothing_is_not_unavailable(monkeypatch):
    _patch_sources(monkeypatch)  # all four return []

    result = await gather_live_evidence("a very rare query")

    assert result.sources == []
    assert result.failed_sources == []
    assert result.all_sources_failed is False  # answered-but-empty != failed


async def test_gather_live_evidence_a_hung_source_is_cut_off_by_its_deadline(monkeypatch):
    async def hangs(query):
        await asyncio.sleep(60)

    _patch_sources(monkeypatch, medlineplus=[_src_item("MedlinePlus", "Overview")])
    monkeypatch.setattr(evidence_agent, "search_pubmed", hangs)
    monkeypatch.setattr(evidence_agent, "SOURCE_DEADLINE_SECONDS", 0.05)

    result = await gather_live_evidence("slow query")

    assert [s.source for s in result.sources] == ["MedlinePlus"]
    assert result.failed_sources == ["PubMed"]


async def test_gather_live_evidence_caches_by_query(monkeypatch):
    mocks = _patch_sources(monkeypatch, pubmed=[_src_item("PubMed", "PubMed paper")])

    query = "headache dizziness hypertension"
    first = await gather_live_evidence(query)
    second = await gather_live_evidence(query)

    assert first == second
    for mock in mocks.values():
        mock.assert_awaited_once()


# --------------------------------------------------------------------------
# PubMed abstract summaries: opt-in LLM, concurrent, time-boxed.
# Regression: one real LLM call per abstract, in sequence, under the rate
# limiter, exceeded the 30s source deadline and PubMed's results were thrown
# away even though its HTTP calls had all succeeded (found live).
# --------------------------------------------------------------------------

LONG_ABSTRACT = "This abstract is long enough to need summarising. " * 12  # ~600 chars, over the 400-char threshold


async def test_llm_summaries_are_off_by_default_and_make_no_llm_call(monkeypatch):
    monkeypatch.delenv("EVIDENCE_LLM_SUMMARIES", raising=False)
    never = AsyncMock(side_effect=AssertionError("no LLM call is expected by default"))
    monkeypatch.setattr(evidence_agent, "_summarize_abstract", never)

    out = await evidence_agent._summarize_all([LONG_ABSTRACT, "A short abstract.", ""])

    never.assert_not_called()
    assert out[0].endswith("...") and len(out[0]) < len(LONG_ABSTRACT)  # truncated
    assert out[1] == "A short abstract."                                  # short ones kept whole
    assert out[2] == "(no abstract available)"


async def test_enabled_llm_summaries_run_concurrently_not_one_after_another(monkeypatch):
    monkeypatch.setenv("EVIDENCE_LLM_SUMMARIES", "true")
    started = 0
    all_started = asyncio.Event()

    async def fake_summarize(abstract):
        nonlocal started
        started += 1
        if started == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)  # only returns if all 3 run at once
        return f"LLM summary of {abstract[:1]}"

    monkeypatch.setattr(evidence_agent, "_summarize_abstract", fake_summarize)

    out = await evidence_agent._summarize_all(["a" * 500, "b" * 500, "c" * 500])

    assert out == ["LLM summary of a", "LLM summary of b", "LLM summary of c"]


async def test_slow_llm_summaries_are_cut_off_by_the_budget_and_fall_back_to_truncation(monkeypatch):
    monkeypatch.setenv("EVIDENCE_LLM_SUMMARIES", "true")
    monkeypatch.setattr(evidence_agent, "SUMMARY_BUDGET_SECONDS", 0.05)

    async def hangs(abstract):
        await asyncio.sleep(60)

    monkeypatch.setattr(evidence_agent, "_summarize_abstract", hangs)

    out = await asyncio.wait_for(evidence_agent._summarize_all([LONG_ABSTRACT, "short"]), timeout=5)

    assert out[0].endswith("...")      # fell back to truncation
    assert out[1] == "short"


async def test_one_failing_llm_summary_only_affects_that_abstract(monkeypatch):
    monkeypatch.setenv("EVIDENCE_LLM_SUMMARIES", "true")

    async def flaky(abstract):
        if abstract.startswith("BAD"):
            raise RuntimeError("503 UNAVAILABLE")
        return "LLM summary"

    monkeypatch.setattr(evidence_agent, "_summarize_abstract", flaky)

    out = await evidence_agent._summarize_all(["BAD" + "x" * 500, "good" + "y" * 500])

    assert out[0].endswith("...") and out[1] == "LLM summary"


async def test_search_pubmed_keeps_its_results_even_when_llm_summaries_hang(monkeypatch):
    """The live bug: PubMed's HTTP calls succeed, but slow LLM summaries used
    to take the whole source (and its results) down with them."""
    monkeypatch.setenv("EVIDENCE_LLM_SUMMARIES", "true")
    monkeypatch.setattr(evidence_agent, "SUMMARY_BUDGET_SECONDS", 0.05)

    async def hangs(abstract):
        await asyncio.sleep(60)

    monkeypatch.setattr(evidence_agent, "_summarize_abstract", hangs)
    long_efetch = (
        '<?xml version="1.0"?><PubmedArticleSet>'
        f'<PubmedArticle><MedlineCitation><PMID Version="1">111</PMID><Article><Abstract><AbstractText>{LONG_ABSTRACT}</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>'
        f'<PubmedArticle><MedlineCitation><PMID Version="1">222</PMID><Article><Abstract><AbstractText>{LONG_ABSTRACT}</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>'
        "</PubmedArticleSet>"
    )

    def handler(url, params):
        if url == evidence_agent.PUBMED_ESEARCH_URL:
            return _json_response(url, PUBMED_ESEARCH_JSON)
        if url == evidence_agent.PUBMED_ESUMMARY_URL:
            return _json_response(url, PUBMED_ESUMMARY_JSON)
        if url == evidence_agent.PUBMED_EFETCH_URL:
            return _text_response(url, long_efetch)
        raise AssertionError(f"unexpected URL {url}")

    _patch_client(monkeypatch, handler)

    results = await asyncio.wait_for(search_pubmed("headache", max_results=2), timeout=5)

    assert [r["url"] for r in results] == [
        "https://pubmed.ncbi.nlm.nih.gov/111/", "https://pubmed.ncbi.nlm.nih.gov/222/",
    ]
    assert all(r["summary"].endswith("...") for r in results)  # truncated, not lost


# --------------------------------------------------------------------------
# ClinicalTrials.gov query length + not retrying client errors.
# Regression (found live): the optimizer's refined 15-20 word queries made
# ClinicalTrials.gov answer 400 "Too complicated query" -- measured limit is
# 13 words -- and the 400 was then pointlessly retried.
# --------------------------------------------------------------------------


def test_clinicaltrials_query_is_capped_at_twelve_terms_keeping_the_first_ones():
    long_query = (
        "chest pain, shortness of breath, right lower lung opacity, subpulmonic fluid collection, "
        "differential pneumothorax vs large bulla vs pleural effusion"
    )

    capped = evidence_agent._clinicaltrials_query(long_query)

    assert len(capped.split()) == 12
    assert capped == "chest pain shortness of breath right lower lung opacity subpulmonic fluid collection"
    assert "," not in capped and ";" not in capped


def test_clinicaltrials_query_leaves_short_queries_alone_apart_from_punctuation():
    assert evidence_agent._clinicaltrials_query("hypertension headache") == "hypertension headache"
    assert evidence_agent._clinicaltrials_query("headache, dizziness; history of hypertension") == (
        "headache dizziness history of hypertension"
    )


async def test_search_clinicaltrials_sends_the_capped_query(monkeypatch):
    seen = {}

    def handler(url, params):
        seen.update(params)
        return _json_response(url, {"studies": []})

    _patch_client(monkeypatch, handler)

    await search_clinicaltrials("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen")

    assert len(seen["query.term"].split()) == 12


async def test_a_client_error_is_not_retried(monkeypatch):
    calls = 0

    def handler(url, params):
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="Too complicated query", request=httpx.Request("GET", url))

    _patch_client(monkeypatch, handler)

    with pytest.raises(EvidenceSourceUnavailableError, match="after 1 attempt"):
        await search_clinicaltrials("anything")
    assert calls == 1


@pytest.mark.parametrize("status", [429, 503])
async def test_rate_limit_and_server_errors_are_still_retried(monkeypatch, status):
    calls = 0

    def handler(url, params):
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="try later", request=httpx.Request("GET", url))

    _patch_client(monkeypatch, handler)

    with pytest.raises(EvidenceSourceUnavailableError):
        await search_clinicaltrials("anything")
    assert calls == evidence_agent.MAX_ATTEMPTS
