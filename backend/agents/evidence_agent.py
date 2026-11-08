"""Evidence agent: fetches CURRENT (live) evidence, distinct from the
offline knowledge base in rag_agent.py, and normalizes it into
LiveEvidence.

Sources:
  - PubMed, via NCBI E-utilities (esearch -> esummary -> efetch),
    no API key required at this call volume.
  - MedlinePlus health topics search (wsearch.nlm.nih.gov), chosen as
    the "one guideline source" over WHO/NICE because it is a keyless,
    reliable public API in the same NLM/NIH family as PubMed, with a
    predictable XML response -- WHO and NICE do not offer an equivalent
    simple, keyless search endpoint. It surfaces consumer/clinician
    health-topic overviews rather than primary guideline documents, but
    is the closest practical "current guideline-adjacent" source
    reachable without scraping or an API key.
  - Europe PMC REST API (keyless): indexes MEDLINE/PubMed plus PMC
    full-text, preprints and agricola. Its publication-type metadata
    (e.g. "Randomized Controlled Trial", "Practice Guideline") feeds the
    evidence level directly instead of relying on keywords alone.
  - ClinicalTrials.gov API v2 (keyless): registered trials -- study
    status and phase are surfaced in the summary since they are the
    clinically relevant facts for a trial record.
  No Google Scholar (no official API), no WHO/NICE feed (no public
  keyless search endpoint), no paid search wrapper.

Evidence level: `evidence_level` is assigned by a crude keyword-match
heuristic over each result's title+abstract/summary text (see
_assess_evidence_level below), upgraded by Europe PMC's publication-type
tags and ClinicalTrials.gov's phase/status. This is a PROTOTYPE
SIMPLIFICATION, not a validated evidence-grading system (e.g. GRADE) --
real grading requires structured study-design metadata and expert
appraisal.

Schema note: LiveEvidenceSource in backend/models/schemas.py gained an
additive, backward-compatible `publication_date: str | None = None`
field so a publication date has somewhere to go, without overloading the
`summary` field for it. LiveEvidence likewise gained `sources_queried`
and `failed_sources` (both defaulted) so "every source failed" can be
told apart from "sources answered but found nothing".

Resilience: every network call is wrapped in a timeout + one retry. A
source function RAISES EvidenceSourceUnavailableError when its primary
request fails after retrying; gather_live_evidence runs all four sources
concurrently, each under its own overall deadline, catches every
per-source failure, and records the failed source names on the returned
LiveEvidence. "Live evidence unavailable" is therefore signalled upstream
only when ALL sources failed (LiveEvidence.all_sources_failed) -- one
source being down just means the other three still contribute. It always
returns a valid LiveEvidence, worst case sources=[] with every source
listed as failed, so the pipeline can proceed on offline evidence alone.
Results are cached in memory by query (never when every source failed, so
a transient outage doesn't poison later calls).
"""

import asyncio
import difflib
import html
import logging
import re
import xml.etree.ElementTree as ET

import httpx
from anthropic import AsyncAnthropic

from backend.models.schemas import LiveEvidence, LiveEvidenceSource
from backend.services.llm_client import get_llm_client, get_model_name

logger = logging.getLogger(__name__)

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
MEDLINEPLUS_SEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CLINICALTRIALS_SEARCH_URL = "https://clinicaltrials.gov/api/v2/studies"

HTTP_TIMEOUT_SECONDS = 10.0  # per HTTP request
MAX_ATTEMPTS = 2  # 1 initial attempt + 1 retry
# Hard cap on ONE source's whole search (PubMed makes three sequential
# requests, each with its own timeout + retry) so a single slow source
# can never stall the pipeline; exceeding it counts as that source failing.
SOURCE_DEADLINE_SECONDS = 30.0
# Two titles at or above this difflib ratio (after normalization) are
# treated as the same paper (Europe PMC re-indexes most of PubMed).
TITLE_DUPLICATE_RATIO = 0.9

# Abstracts at or under this length are used as-is for `summary`;
# longer ones are auto-summarized via Claude (falling back to a
# truncated snippet if that call fails for any reason).
ABSTRACT_SUMMARIZE_THRESHOLD_CHARS = 400
CLAUDE_MODEL = get_model_name()

# --------------------------------------------------------------------------
# Evidence-level heuristic -- PROTOTYPE SIMPLIFICATION.
#
# This is crude keyword matching over title+abstract text, not a
# validated evidence hierarchy. It exists so the UI has *something* to
# sort/filter on until real structured evidence grading is built.
# --------------------------------------------------------------------------
_HIGH_EVIDENCE_KEYWORDS = (
    "guideline",
    "meta-analysis",
    "meta analysis",
    "systematic review",
    "randomized controlled trial",
    "randomised controlled trial",
    " rct ",
    " rct.",
)
_MODERATE_EVIDENCE_KEYWORDS = (
    "cohort",
    "case-control",
    "case control",
    "observational study",
    "review",
)


def _assess_evidence_level(text: str) -> str:
    lowered = f" {text.lower()} "
    if any(kw in lowered for kw in _HIGH_EVIDENCE_KEYWORDS):
        return "high"
    if any(kw in lowered for kw in _MODERATE_EVIDENCE_KEYWORDS):
        return "moderate"
    return "low"


class EvidenceSourceUnavailableError(RuntimeError):
    """Raised internally when a source's HTTP calls fail after retrying."""


_live_evidence_cache: dict[str, LiveEvidence] = {}

_anthropic_client: AsyncAnthropic | None = None


def _get_anthropic_client() -> AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        # See backend/services/llm_client.py -- provider-selectable via
        # LLM_PROVIDER (anthropic/bedrock/vertex/gemini/mistral) in .env. Kept as this
        # module's own _get_anthropic_client() so existing tests that
        # monkeypatch evidence_agent._get_anthropic_client keep working.
        _anthropic_client = get_llm_client()
    return _anthropic_client


_TAG_RE = re.compile(r"<[^>]+>")
# Block-level tags are a visual break: removing them with NO replacement glues
# neighbouring text together ("<h4>Background</h4>Pneumothorax..." ->
# "BackgroundPneumothorax...", "...cancer</li><li>Being on..." ->
# "cancerBeing on..."). Inline tags (<i>, <b>, <sub>) are removed with no
# space so "<i>migraine</i>s" stays one word.
_BLOCK_TAG_RE = re.compile(r"</?(?:h[1-6]|p|ul|ol|br|div|tr|td|th|table|section)\b[^>]*>", re.IGNORECASE)


def _strip_html(text: str | None) -> str:
    """HTML -> plain text for live-evidence titles/summaries: block tags
    become spaces, list items are joined with "; " (the last ending in "."),
    entities are decoded, and the space a removed tag leaves before
    punctuation is closed up. Wording is never changed."""
    if not text:
        return ""
    text = re.sub(r"</li>\s*(?=</[uo]l>)", ". ", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>", "; ", text, flags=re.IGNORECASE)
    text = _BLOCK_TAG_RE.sub(" ", text)
    text = html.unescape(_TAG_RE.sub("", text))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:?!)])", r"\1", text)
    return re.sub(r"\.\s*\.", ".", text)


def _truncate(text: str, limit: int = 300) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


async def _get_with_retries(
    client: httpx.AsyncClient, url: str, params: dict, source_name: str
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = await client.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            last_exc = exc
            logger.warning(
                "%s request failed (attempt %d/%d): %s", source_name, attempt, MAX_ATTEMPTS, exc
            )
    raise EvidenceSourceUnavailableError(
        f"{source_name} unavailable after {MAX_ATTEMPTS} attempts: {last_exc}"
    )


# --------------------------------------------------------------------------
# PubMed
# --------------------------------------------------------------------------


async def _esearch(client: httpx.AsyncClient, query: str, max_results: int) -> list[str]:
    response = await _get_with_retries(
        client,
        PUBMED_ESEARCH_URL,
        {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        },
        "PubMed esearch",
    )
    return response.json().get("esearchresult", {}).get("idlist", [])


async def _esummary(client: httpx.AsyncClient, pmids: list[str]) -> dict[str, dict]:
    response = await _get_with_retries(
        client,
        PUBMED_ESUMMARY_URL,
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
        "PubMed esummary",
    )
    result = response.json().get("result", {})
    return {pmid: result[pmid] for pmid in pmids if pmid in result}


def _parse_abstracts_xml(xml_text: str) -> dict[str, str]:
    abstracts: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return abstracts

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pieces = [el.text.strip() for el in article.findall(".//AbstractText") if el.text]
        if pieces:
            abstracts[pmid_el.text] = " ".join(pieces)
    return abstracts


async def _efetch_abstracts(client: httpx.AsyncClient, pmids: list[str]) -> dict[str, str]:
    response = await _get_with_retries(
        client,
        PUBMED_EFETCH_URL,
        {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"},
        "PubMed efetch",
    )
    return _parse_abstracts_xml(response.text)


async def _summarize_abstract(abstract: str) -> str:
    if not abstract:
        return "(no abstract available)"
    if len(abstract) <= ABSTRACT_SUMMARIZE_THRESHOLD_CHARS:
        return abstract

    try:
        client = _get_anthropic_client()
        response = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=150,
            system=(
                "Summarize the following PubMed abstract in 1-2 sentences "
                "for a clinician quickly scanning search results. Return "
                "only the summary text, no preamble."
            ),
            messages=[{"role": "user", "content": abstract}],
        )
        summary = response.content[0].text.strip()
        return summary or _truncate(abstract)
    except Exception as exc:  # any failure here must not crash the pipeline
        logger.warning("Claude abstract summarization failed, falling back to truncation: %s", exc)
        return _truncate(abstract)


async def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    async with httpx.AsyncClient() as client:
        try:
            pmids = await _esearch(client, query, max_results)
        except EvidenceSourceUnavailableError as exc:
            # The primary request failed: raise (don't return []), so the
            # caller can tell "PubMed is down" from "PubMed found nothing".
            logger.warning("PubMed live evidence unavailable for query %r: %s", query, exc)
            raise

        if not pmids:
            return []

        try:
            summaries = await _esummary(client, pmids)
        except EvidenceSourceUnavailableError as exc:
            logger.warning("PubMed esummary unavailable for query %r: %s", query, exc)
            summaries = {}

        try:
            abstracts = await _efetch_abstracts(client, pmids)
        except EvidenceSourceUnavailableError as exc:
            logger.warning("PubMed efetch unavailable for query %r: %s", query, exc)
            abstracts = {}

    results = []
    for pmid in pmids:
        summary_meta = summaries.get(pmid, {})
        title = (summary_meta.get("title") or "").strip() or "(title unavailable)"
        pubdate = summary_meta.get("pubdate") or None
        abstract = abstracts.get(pmid, "")

        evidence_summary = await _summarize_abstract(abstract)
        results.append(
            {
                "title": title,
                "source": "PubMed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "summary": evidence_summary,
                "evidence_level": _assess_evidence_level(f"{title} {abstract}"),
                "publication_date": pubdate,
            }
        )
    return results


# --------------------------------------------------------------------------
# MedlinePlus (guideline-adjacent source)
# --------------------------------------------------------------------------


def _parse_medlineplus_xml(xml_text: str, max_results: int) -> list[dict]:
    results = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    for document in root.findall(".//document")[:max_results]:
        url = document.get("url", "")
        title_el = document.find("content[@name='title']")
        summary_el = document.find("content[@name='FullSummary']")

        title = _strip_html(title_el.text if title_el is not None else "")
        summary = _strip_html(summary_el.text if summary_el is not None else "")

        results.append(
            {
                "title": title or "(title unavailable)",
                "source": "MedlinePlus",
                "url": url,
                "summary": _truncate(summary, limit=400) if summary else "(no summary available)",
                "evidence_level": _assess_evidence_level(f"{title} {summary}"),
                "publication_date": None,
            }
        )
    return results


async def search_medlineplus_guidelines(query: str, max_results: int = 3) -> list[dict]:
    async with httpx.AsyncClient() as client:
        try:
            response = await _get_with_retries(
                client,
                MEDLINEPLUS_SEARCH_URL,
                {"db": "healthTopics", "term": query, "retmax": max_results},
                "MedlinePlus",
            )
        except EvidenceSourceUnavailableError as exc:
            logger.warning("MedlinePlus live evidence unavailable for query %r: %s", query, exc)
            raise

    return _parse_medlineplus_xml(response.text, max_results)


# --------------------------------------------------------------------------
# Europe PMC
# --------------------------------------------------------------------------

_LEVEL_RANK = {"low": 0, "moderate": 1, "high": 2}
_EPMC_HIGH_PUBTYPES = (
    "practice guideline", "guideline", "meta-analysis", "systematic review", "randomized controlled trial",
)
_EPMC_MODERATE_PUBTYPES = (
    "review", "clinical trial", "observational study", "comparative study", "multicenter study", "cohort",
)


def _max_level(*levels: str) -> str:
    return max(levels, key=lambda level: _LEVEL_RANK.get(level, 0))


def _europepmc_evidence_level(pub_types: list[str], source_db: str, title: str, abstract: str) -> str:
    """Publication-type tags (when present) upgrade the keyword heuristic;
    preprints (source "PPR") are not peer reviewed and are capped at low."""
    if source_db == "PPR":
        return "low"
    lowered = " | ".join(pub_types).lower()
    from_types = "low"
    if any(t in lowered for t in _EPMC_HIGH_PUBTYPES):
        from_types = "high"
    elif any(t in lowered for t in _EPMC_MODERATE_PUBTYPES):
        from_types = "moderate"
    return _max_level(from_types, _assess_evidence_level(f"{title} {abstract}"))


def _parse_europepmc_json(payload: dict, max_results: int) -> list[dict]:
    results = []
    for item in (payload.get("resultList", {}) or {}).get("result", [])[:max_results]:
        record_id = item.get("id")
        if not record_id:
            continue
        source_db = item.get("source") or "MED"
        title = _strip_html(item.get("title")).rstrip(".").strip() or "(title unavailable)"
        abstract = _strip_html(item.get("abstractText"))
        pub_types = (item.get("pubTypeList", {}) or {}).get("pubType", []) or []
        results.append(
            {
                "title": title,
                "source": "Europe PMC",
                "url": f"https://europepmc.org/article/{source_db}/{record_id}",
                "summary": _truncate(abstract, limit=400) if abstract else "(no abstract available)",
                "evidence_level": _europepmc_evidence_level(pub_types, source_db, title, abstract),
                "publication_date": item.get("firstPublicationDate") or item.get("pubYear") or None,
            }
        )
    return results


async def search_europepmc(query: str, max_results: int = 5) -> list[dict]:
    async with httpx.AsyncClient() as client:
        try:
            response = await _get_with_retries(
                client,
                EUROPEPMC_SEARCH_URL,
                {"query": query, "format": "json", "pageSize": max_results, "resultType": "core"},
                "Europe PMC",
            )
        except EvidenceSourceUnavailableError as exc:
            logger.warning("Europe PMC live evidence unavailable for query %r: %s", query, exc)
            raise
    try:
        payload = response.json()
    except ValueError as exc:
        raise EvidenceSourceUnavailableError(f"Europe PMC returned a non-JSON response: {exc}") from exc
    return _parse_europepmc_json(payload, max_results)


# --------------------------------------------------------------------------
# ClinicalTrials.gov
# --------------------------------------------------------------------------

_LATE_PHASES = ("PHASE3", "PHASE4")


def _clinicaltrials_evidence_level(status: str, phases: list[str]) -> str:
    """Prototype heuristic: "moderate" by default (a registered trial is
    not itself a result), "high" for a COMPLETED Phase 3/4 trial."""
    if status == "COMPLETED" and any(p in _LATE_PHASES for p in phases):
        return "high"
    return "moderate"


def _parse_clinicaltrials_json(payload: dict, max_results: int) -> list[dict]:
    results = []
    for study in (payload.get("studies") or [])[:max_results]:
        protocol = study.get("protocolSection", {}) or {}
        ident = protocol.get("identificationModule", {}) or {}
        nct_id = ident.get("nctId")
        if not nct_id:
            continue
        status_module = protocol.get("statusModule", {}) or {}
        status = status_module.get("overallStatus") or "UNKNOWN"
        phases = (protocol.get("designModule", {}) or {}).get("phases") or []
        brief = _strip_html((protocol.get("descriptionModule", {}) or {}).get("briefSummary"))
        facts = f"Status: {status}. Phase: {', '.join(phases) if phases else 'not applicable'}."
        results.append(
            {
                "title": _strip_html(ident.get("briefTitle") or ident.get("officialTitle")) or "(title unavailable)",
                "source": "ClinicalTrials.gov",
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
                "summary": f"{facts} {_truncate(brief, limit=300)}" if brief else facts,
                "evidence_level": _clinicaltrials_evidence_level(status, phases),
                "publication_date": (status_module.get("startDateStruct", {}) or {}).get("date") or None,
            }
        )
    return results


async def search_clinicaltrials(query: str, max_results: int = 3) -> list[dict]:
    async with httpx.AsyncClient() as client:
        try:
            response = await _get_with_retries(
                client,
                CLINICALTRIALS_SEARCH_URL,
                {"query.term": query, "pageSize": max_results, "format": "json"},
                "ClinicalTrials.gov",
            )
        except EvidenceSourceUnavailableError as exc:
            logger.warning("ClinicalTrials.gov live evidence unavailable for query %r: %s", query, exc)
            raise
    try:
        payload = response.json()
    except ValueError as exc:
        raise EvidenceSourceUnavailableError(f"ClinicalTrials.gov returned a non-JSON response: {exc}") from exc
    return _parse_clinicaltrials_json(payload, max_results)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

_TITLE_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", _TITLE_NORMALIZE_RE.sub(" ", title.lower())).strip()


def _dedupe_by_title(items: list[dict]) -> list[dict]:
    """Drop later items whose normalized title matches (or nearly matches,
    difflib ratio >= TITLE_DUPLICATE_RATIO) an earlier one. Order is
    priority order, so the PubMed copy (listed first) wins over Europe
    PMC's re-indexed record of the same paper."""
    kept: list[dict] = []
    kept_titles: list[str] = []
    for item in items:
        normalized = _normalize_title(item["title"])
        is_duplicate = bool(normalized) and any(
            normalized == seen
            or difflib.SequenceMatcher(None, normalized, seen).ratio() >= TITLE_DUPLICATE_RATIO
            for seen in kept_titles
        )
        if is_duplicate:
            continue
        kept.append(item)
        kept_titles.append(normalized)
    return kept


async def _run_source(name: str, search_fn, query: str) -> tuple[str, list[dict], Exception | None]:
    """Run ONE source under its own deadline; never raises. A source that
    errors, times out, or hits an unexpected bug is reported as failed
    (with its exception) so the other sources are unaffected."""
    try:
        return name, await asyncio.wait_for(search_fn(query), timeout=SOURCE_DEADLINE_SECONDS), None
    except Exception as exc:  # incl. EvidenceSourceUnavailableError and TimeoutError
        logger.warning("Live evidence source %s failed for query %r: %s", name, query, exc)
        return name, [], exc


async def gather_live_evidence(query: str) -> LiveEvidence:
    if query in _live_evidence_cache:
        return _live_evidence_cache[query]

    # Resolved at call time (not import time) so each source can be
    # replaced independently, e.g. in tests. Listed in dedupe priority order.
    source_searches = [
        ("PubMed", search_pubmed),
        ("MedlinePlus", search_medlineplus_guidelines),
        ("Europe PMC", search_europepmc),
        ("ClinicalTrials.gov", search_clinicaltrials),
    ]
    outcomes = await asyncio.gather(*(_run_source(name, fn, query) for name, fn in source_searches))

    failed_sources = [name for name, _, error in outcomes if error is not None]
    merged = _dedupe_by_title([item for _, items, _ in outcomes for item in items])
    result = LiveEvidence(
        sources=[LiveEvidenceSource(**item) for item in merged],
        sources_queried=len(source_searches),
        failed_sources=failed_sources,
    )
    logger.info(
        "Live evidence for %r: %d item(s) from %d/%d source(s); failed: %s",
        query, len(result.sources), len(source_searches) - len(failed_sources), len(source_searches),
        failed_sources or "none",
    )

    # Don't cache a total outage -- it would pin "unavailable" to this
    # query for the life of the process even after the sources recover.
    if not result.all_sources_failed:
        _live_evidence_cache[query] = result
    return result
