"""Developer tool: (re)builds data/knowledge_base/**/*.json from real,
citable public sources. NOT run at ingest time or by the app -- ingest.py
reads the checked-in JSON files; this script exists so every passage's
provenance can be re-verified or refreshed:

    python -m backend.rag.build_knowledge_base            # add only the missing doc_ids
    python -m backend.rag.build_knowledge_base --refresh DOC_ID [DOC_ID ...]  # re-fetch just these
    python -m backend.rag.build_knowledge_base --rebuild  # re-fetch every passage

Sources (all free/keyless, same families the live Evidence Agent uses):

* MedlinePlus health topics + Medical Encyclopedia -- U.S. National
  Library of Medicine, a U.S. Government work (public domain). Text is
  copied VERBATIM from the page; only whole sections / whole sentences
  are kept, truncated at a sentence boundary to fit the ~200-400 token
  chunk band. Nothing is paraphrased or rewritten.
* PubMed abstracts (NCBI E-utilities) -- short VERBATIM abstract
  excerpts, each cited by PMID/journal/year. Abstracts remain the
  property of their publishers; they are excerpted here for prototype
  retrieval with attribution, per the build spec's "verifiable short
  PubMed abstract excerpts".

Every record keeps: doc_id, source (the human-readable citation shown in
reports), url, title, retrieved (ISO date), and the verbatim text.
Nothing here is synthetic.
"""

import html
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import httpx

KB_ROOT = Path("data/knowledge_base")
MAX_WORDS = 340  # < ingest.CHUNK_SIZE_TOKENS (350) so one passage stays one chunk; was 300
HTTP_TIMEOUT_SECONDS = 30
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# (doc_id, folder, MedlinePlus Medical Encyclopedia article id, section priority)
#
# Section order matters because a passage is cut at ~MAX_WORDS: whatever comes
# first survives. "Symptoms" is deliberately FIRST -- a clinical query describes
# what the patient presents with, and with "Causes" first the Symptoms section
# was cut off or dropped in 10 of 15 articles (e.g. the subdural hematoma
# passage ended up describing acute head-injury mechanics and never mentioned
# headache, confusion or balance/walking problems, so it was not retrieved for
# a patient with exactly those). Text is still verbatim; only the order changed.
ENCY_DOCS = [
    ("headache_tension_type", "symptom_disease", "000797", ["Symptoms", "Causes", "Exams and Tests"]),
    ("migraine_overview", "symptom_disease", "000709", ["Symptoms", "Causes", "Exams and Tests"]),
    ("bppv", "symptom_disease", "001420", ["Symptoms", "Causes", "Exams and Tests"]),
    ("menieres_disease", "symptom_disease", "000702", ["Symptoms", "Causes", "Exams and Tests"]),
    ("hypertension_overview", "symptom_disease", "000468", ["Symptoms", "Causes", "Exams and Tests"]),
    ("orthostatic_hypotension_low_blood_pressure", "symptom_disease", "007278", ["Symptoms", "Causes", "Exams and Tests"]),
    ("anemia_overview", "symptom_disease", "000560", ["Symptoms", "Causes", "Exams and Tests"]),
    ("tia_overview", "symptom_disease", "000730", ["Symptoms", "Causes", "Exams and Tests"]),
    ("chronic_subdural_hematoma", "symptom_disease", "000713", ["Symptoms", "Causes", "Exams and Tests"]),
    ("copd_chest_xray_context", "symptom_disease", "000091", ["Symptoms", "Causes", "Exams and Tests"]),
    ("pleural_effusion_chest_xray", "symptom_disease", "000086", ["Symptoms", "Causes", "Exams and Tests"]),
    ("brain_mri_overview", "symptom_disease", "003791", ["Why the Test is Performed", "What Abnormal Results Mean"]),
    ("head_ct_overview", "symptom_disease", "003786", ["Why the Test is Performed", "What Abnormal Results Mean"]),
    ("chest_xray_overview", "symptom_disease", "003804", ["Why the Test is Performed", "What Abnormal Results Mean"]),
    # Added 2026-11-12 after an LLM-judged run showed 0/5 relevant passages for a
    # memory-loss + gait case: the KB had nothing on dementia / hydrocephalus.
    ("alzheimer_disease", "symptom_disease", "000760", ["Symptoms", "Causes", "Exams and Tests"]),
    ("dementia_overview", "symptom_disease", "000739", ["Symptoms", "Causes", "Exams and Tests"]),
    ("subarachnoid_hemorrhage", "symptom_disease", "000701", ["Symptoms", "Causes", "Exams and Tests"]),
    ("pulmonary_tuberculosis", "symptom_disease", "000077", ["Symptoms", "Causes", "Exams and Tests"]),
    # Added 2026-11-13 (gap analysis, see data/knowledge_base/README.md): conditions the imaging models can
    # name (pulmonary edema, atelectasis) or that the answer key / live differentials reach for.
    ("pulmonary_edema", "symptom_disease", "000140", ["Symptoms", "Causes", "Exams and Tests"]),
    ("atelectasis", "symptom_disease", "000065", ["Symptoms", "Causes", "Exams and Tests"]),
    ("cluster_headache", "symptom_disease", "000786", ["Symptoms", "Causes", "Exams and Tests"]),
    ("brain_abscess", "symptom_disease", "000783", ["Symptoms", "Causes", "Exams and Tests"]),
]

# (doc_id, folder, MedlinePlus health-topic search term, expected title)
TOPIC_DOCS = [
    ("headache_overview", "symptom_disease", "headaches", "Headache"),
    ("dizziness_vertigo_overview", "symptom_disease", "dizziness and vertigo", "Dizziness and Vertigo"),
    ("ischemic_stroke_overview", "symptom_disease", "ischemic stroke", "Ischemic Stroke"),
    ("sinusitis_headache", "symptom_disease", "sinusitis", "Sinusitis"),
    ("pneumonia_chest_xray", "symptom_disease", "pneumonia", "Pneumonia"),
    ("pneumothorax_collapsed_lung", "symptom_disease", "collapsed lung", "Collapsed Lung"),
    ("hydrocephalus_overview", "symptom_disease", "hydrocephalus", "Hydrocephalus"),
    ("lung_cancer_overview", "symptom_disease", "lung cancer", "Lung Cancer"),
    ("hemorrhagic_stroke_overview", "symptom_disease", "hemorrhagic stroke", "Hemorrhagic Stroke"),
    ("traumatic_brain_injury_overview", "symptom_disease", "traumatic brain injury", "Traumatic Brain Injury"),
    ("chest_injuries_rib_fracture", "symptom_disease", "chest injuries and disorders", "Chest Injuries and Disorders"),
    # Added 2026-11-13 (gap analysis)
    ("heart_failure", "symptom_disease", "heart failure", "Heart Failure"),
    ("brain_tumors", "symptom_disease", "brain tumors", "Brain Tumors"),
    ("meningitis", "symptom_disease", "meningitis", "Meningitis"),
    ("chest_pain", "symptom_disease", "chest pain", "Chest Pain"),
    ("fainting", "symptom_disease", "fainting", "Fainting"),
    ("fatigue", "symptom_disease", "fatigue", "Fatigue"),
    ("multiple_sclerosis", "symptom_disease", "multiple sclerosis", "Multiple Sclerosis"),
    ("giant_cell_arteritis", "symptom_disease", "giant cell arteritis", "Giant Cell Arteritis"),
    ("concussion", "symptom_disease", "concussion", "Concussion"),
    ("brain_aneurysm", "symptom_disease", "brain aneurysm", "Brain Aneurysm"),
    ("arteriovenous_malformations", "symptom_disease", "arteriovenous malformations", "Arteriovenous Malformations"),
    ("aortic_aneurysm", "symptom_disease", "aortic aneurysm", "Aortic Aneurysm"),
    ("long_covid", "symptom_disease", "long covid", "Long COVID"),
    ("refractive_errors", "symptom_disease", "refractive errors", "Refractive Errors"),
]

# (doc_id, folder, PMID, expected title fragment) -- PMIDs were looked up
# via PubMed esearch and are re-verified against esummary at build time.
PUBMED_DOCS = [
    ("secondary_headache_red_flags_snnoop10", "guidelines", "30587518", "SNNOOP10"),
    ("bppv_aao_hnsf_guideline_2017", "guidelines", "28248609", "Benign Paroxysmal Positional Vertigo"),
    # The ACC/AHA 2017 and ISH 2020 primary guideline records carry no
    # abstract on PubMed, so this is the ISH 2020 guideline's key-messages
    # summary paper (it has one) -- cited as exactly that, not as the
    # guideline itself.
    ("hypertension_ish_2020_guideline_key_messages", "guidelines", "32972800", "key messages"),
    ("orthostatic_hypotension_in_hypertension_aha_2024", "guidelines", "38205630", "Orthostatic Hypotension"),
    ("hypertensive_emergency_review", "guidelines", "28372707", "Hypertensive Emergency"),
    ("vestibular_migraine_diagnostic_criteria", "guidelines", "23142830", "Vestibular migraine"),
    ("vestibular_neuritis_review", "symptom_disease", "30947184", "Vestibular Neuritis"),
    ("white_matter_hyperintensities_small_vessel", "symptom_disease", "28685621", "White matter hyperintensities"),
    ("pulmonary_nodule_fleischner_2017", "guidelines", "28240562", "Pulmonary Nodules"),
    ("cardiomegaly_chest_radiograph", "symptom_disease", "29073657", "cardiomegaly"),
    ("idiopathic_nph_guidelines_2021", "guidelines", "33455998", "Normal Pressure Hydrocephalus"),
    ("idiopathic_intracranial_hypertension_consensus_2018", "guidelines", "29903905", "Idiopathic intracranial hypertension"),
    ("vascular_contributions_to_cognitive_impairment_aha_2011", "guidelines", "21778438", "Vascular contributions to cognitive impairment"),
    # Added 2026-11-13 (gap analysis)
    ("vertebrobasilar_insufficiency_review", "symptom_disease", "28680502", "Vertebrobasilar Insufficiency"),
    ("pres_review_2025", "symptom_disease", "40818477", "Posterior reversible encephalopathy syndrome"),
    ("intracranial_calcification_vs_hemorrhage_ct", "symptom_disease", "37994049", "calcification"),
    ("long_covid_headache", "symptom_disease", "35915417", "Long COVID headache"),
    ("musculoskeletal_chest_pain_prevalence", "symptom_disease", "33838099", "musculoskeletal chest pain"),
    ("massive_cerebral_infarction_review", "symptom_disease", "15860137", "Massive cerebral infarction"),
    ("chest_pain_guideline_2021", "guidelines", "34756653", "Chest Pain"),
]


def _clean(fragment: str) -> str:
    """HTML -> plain text. Wording is never changed; only markup is
    removed, and two purely mechanical artifacts are repaired: (1) list
    items (which have no sentence punctuation of their own) are joined
    with "; " and the final item ends with "." so a list reads as one
    sentence, and (2) the space that stripping an inline tag (e.g. a link)
    leaves before punctuation / inside parentheses is removed."""
    fragment = re.sub(r"<(script|style).*?</\1>", "", fragment, flags=re.S)
    fragment = re.sub(r"</li>\s*(?=</[uo]l>)", ". ", fragment)
    fragment = re.sub(r"</li>", "; ", fragment)
    fragment = re.sub(r"</(p|h2|h3|div)>", " ", fragment)
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:?!)])", r"\1", text)  # "syndrome ." -> "syndrome."
    text = re.sub(r"\(\s+", "(", text)             # "( tinnitus" -> "(tinnitus"
    text = re.sub(r"\.\s*\.", ".", text)           # ". ." left by a list end before a period
    text = re.sub(r"([.!?]);", r"\1", text)        # ".;" when a bullet already ended in a period
    return text.strip()


def _truncate_at_sentence(text: str, max_words: int = MAX_WORDS) -> str:
    """Keep whole sentences only, up to ~max_words. Never cuts mid-sentence
    and never alters a kept sentence."""
    words = text.split()
    if len(words) <= max_words:
        return text
    window = " ".join(words[:max_words])
    cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if cut < len(window) // 2:  # no usable boundary in the second half
        cut = window.rfind(".")
    return window[: cut + 1].strip() if cut > 0 else window


def _get(client: httpx.Client, url: str, **params) -> httpx.Response:
    last_exc = None
    for _ in range(3):
        try:
            response = client.get(url, params=params or None)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            last_exc = exc
            time.sleep(1.0)
    raise RuntimeError(f"GET {url} failed after retries: {last_exc}")


def fetch_ency(client: httpx.Client, article_id: str, priority: list[str]) -> dict:
    url = f"https://medlineplus.gov/ency/article/{article_id}.htm"
    page = _get(client, url).text
    title = _clean(re.search(r"<title>(.*?)</title>", page, re.S).group(1)).replace(
        ": MedlinePlus Medical Encyclopedia", ""
    )
    summary_match = re.search(r'<div id="ency_summary">(.*?)</div>', page, re.S)
    parts = [_clean(summary_match.group(1))] if summary_match else []
    sections: dict[str, str] = {}
    for section in page.split("<section>")[1:]:
        heading = re.search(r"<h2>(.*?)</h2>", section, re.S)
        body = re.search(r'class="section-body"[^>]*>(.*?)</div>\s*</div>\s*</section>|class="section-body"[^>]*>(.*)', section, re.S)
        if heading and body:
            sections[_clean(heading.group(1))] = _clean(body.group(1) or body.group(2))
    for name in priority:
        if name in sections and sections[name]:
            parts.append(sections[name])
    text = _truncate_at_sentence(" ".join(p for p in parts if p))
    return {
        "title": title,
        "url": url,
        "source": f"MedlinePlus Medical Encyclopedia: {title} ({url})",
        "text": text,
    }


def fetch_topic(client: httpx.Client, term: str, expected_title: str) -> dict:
    response = _get(client, "https://wsearch.nlm.nih.gov/ws/query", db="healthTopics", term=term, retmax=5)
    for doc in ET.fromstring(response.text).iter("document"):
        fields = {c.get("name"): c.text or "" for c in doc.findall("content")}
        title = _clean(fields.get("title", ""))
        if title.lower() == expected_title.lower():
            url = doc.get("url")
            return {
                "title": title,
                "url": url,
                "source": f"MedlinePlus: {title} ({url})",
                "text": _truncate_at_sentence(_clean(fields["FullSummary"])),
            }
    raise RuntimeError(f"MedlinePlus topic '{expected_title}' not found for term '{term}'")


def fetch_pubmed(client: httpx.Client, pmid: str, expected_fragment: str) -> dict:
    summary = _get(client, EUTILS + "esummary.fcgi", db="pubmed", id=pmid, retmode="json").json()["result"][pmid]
    title = summary["title"].rstrip(".")
    if expected_fragment.lower() not in title.lower():
        raise RuntimeError(f"PMID {pmid} title mismatch: expected '{expected_fragment}' in '{title}'")
    time.sleep(0.4)
    root = ET.fromstring(_get(client, EUTILS + "efetch.fcgi", db="pubmed", id=pmid, rettype="abstract", retmode="xml").text)
    abstract = " ".join(
        "".join(node.itertext()).strip() for node in root.iter("AbstractText")
    )
    abstract = re.sub(r"\s+", " ", abstract).strip()
    if not abstract:
        raise RuntimeError(f"PMID {pmid} has no abstract text")
    author = summary["authors"][0]["name"] if summary.get("authors") else "Anon"
    year = summary["pubdate"][:4]
    journal = summary["source"]
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return {
        "title": title,
        "url": url,
        "source": f'PubMed PMID {pmid}: {author} et al. "{title}." {journal}. {year} (abstract excerpt; {url})',
        "text": _truncate_at_sentence(abstract),
    }


def _existing() -> dict[str, list[dict]]:
    """Records already checked in, by folder -- kept untouched in add mode so a
    passage's verified text and 'retrieved' date are not silently refreshed."""
    out: dict[str, list[dict]] = {"guidelines": [], "symptom_disease": []}
    for folder in out:
        path = KB_ROOT / folder / f"{folder}.json"
        if path.exists():
            out[folder] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _put(records: list[dict], record: dict) -> None:
    """Replace the record with the same doc_id in place (keeps file order and
    diffs small); append if it is new."""
    for i, old in enumerate(records):
        if old["doc_id"] == record["doc_id"]:
            records[i] = record
            return
    records.append(record)


def build(rebuild: bool = False, refresh: frozenset[str] = frozenset()) -> dict[str, list[dict]]:
    """Default (add mode): fetch only doc_ids not already on disk and append them.
    refresh: doc_ids to re-fetch and replace in place (everything else untouched).
    rebuild=True re-fetches EVERYTHING (refreshing every passage and date)."""
    retrieved = date.today().isoformat()
    folders = {"guidelines": [], "symptom_disease": []} if rebuild else _existing()
    have = {r["doc_id"] for records in folders.values() for r in records} - set(refresh)
    unknown = set(refresh) - {d[0] for d in ENCY_DOCS + TOPIC_DOCS + PUBMED_DOCS}
    if unknown:
        raise SystemExit(f"--refresh: unknown doc_id(s): {sorted(unknown)}")
    with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client:
        for doc_id, folder, article_id, priority in ENCY_DOCS:
            if doc_id in have:
                continue
            _put(folders[folder], {"doc_id": doc_id, "retrieved": retrieved, **fetch_ency(client, article_id, priority)})
            time.sleep(0.3)
        for doc_id, folder, term, title in TOPIC_DOCS:
            if doc_id in have:
                continue
            _put(folders[folder], {"doc_id": doc_id, "retrieved": retrieved, **fetch_topic(client, term, title)})
            time.sleep(0.3)
        for doc_id, folder, pmid, fragment in PUBMED_DOCS:
            if doc_id in have:
                continue
            _put(folders[folder], {"doc_id": doc_id, "retrieved": retrieved, **fetch_pubmed(client, pmid, fragment)})
            time.sleep(0.4)
    return folders


def main() -> None:
    import sys

    args = sys.argv[1:]
    refresh = frozenset(args[args.index("--refresh") + 1 :]) if "--refresh" in args else frozenset()
    folders = build(rebuild="--rebuild" in args, refresh=refresh)
    for folder, records in folders.items():
        path = KB_ROOT / folder / f"{folder}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {len(records)} real passages -> {path}")


if __name__ == "__main__":
    main()
