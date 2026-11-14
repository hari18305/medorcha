"""Second opinion on the zero-shot image read: the multimodal LLM looks at the raw image.

BiomedCLIP's read can be confidently wrong (a generated brain MRI scored "cerebral atrophy" 0.65, a normal
head CT scored "sinus abnormality" 0.89), and the indecisive-read rule only catches the unsure ones. The
fusion step's own look at the image is the better reader on the 88-image benchmark (68/88 against 50/88,
DEMO_READINESS.md section 5), so it is asked the SAME closed-set question BiomedCLIP is asked; when the two
readings disagree the zero-shot label is marked `contested` and, like an inconclusive read, kept out of
retrieval and the differential (the fusion step still sees the raw image itself).

Replayed offline on the 88 saved reads (no new LLM calls): of the 29 zero-shot reads with a top score >= 0.4
that were wrong, 25 disagreed with the LLM; of the 50 that were right, 5 did. Caveats: the images are public
teaching examples the LLM may have seen, the list is closed, one model and one pass, so the true rate is
probably lower. The rule is OFF unless IMAGING_CROSS_CHECK=on, because the reported evaluation was run without
it. It FAILS OPEN: a provider without vision, a call error or an unparseable answer leaves the zero-shot read
exactly as it was.
"""

import asyncio
import json
import logging
import os
import re

from backend.agents.imaging_labels import FINDING_LABELS_BY_MODALITY, NORMAL_FINDING_LABELS
from backend.services.llm_client import call_multimodal, supports_vision, throttle_llm_call_sync

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a careful radiology image reader helping to evaluate an AI system. Look only at the image. "
    "Reply with one JSON object and nothing else."
)


def cross_check_enabled() -> bool:
    return os.getenv("IMAGING_CROSS_CHECK", "off").strip().lower() in {"on", "1", "true", "yes"}


def build_prompt(modality: str) -> str:
    labels = FINDING_LABELS_BY_MODALITY[modality]
    return (
        f"This image is a {modality}. From this closed list choose the SINGLE label that best describes the main finding "
        f"(use a 'normal' label only if you see nothing abnormal):\n- " + "\n- ".join(labels) +
        '\nReply as {"label": "<exactly one label from the list>", "confidence": <0 to 1>}.'
    )


def parse(text: str, modality: str) -> str | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        label = json.loads(m.group(0)).get("label")
    except (json.JSONDecodeError, AttributeError):
        return None
    return label if label in FINDING_LABELS_BY_MODALITY[modality] else None


def _same(a: str, b: str) -> bool:
    # "normal ..." and "no significant abnormality" are the same statement
    return a == b or (a in NORMAL_FINDING_LABELS and b in NORMAL_FINDING_LABELS)


def agrees(llm_label: str, zero_shot_findings: list[str]) -> bool:
    """The LLM's label matches any label the zero-shot read reported (its top label and any close second)."""
    return any(_same(llm_label, f) for f in zero_shot_findings)


async def read_label(image_path: str, modality: str) -> str | None:
    """The LLM's closed-set label for the raw image, or None when it cannot be had (never raises)."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if modality not in FINDING_LABELS_BY_MODALITY or not supports_vision(provider):
        return None
    try:
        await asyncio.to_thread(throttle_llm_call_sync)
        raw = await call_multimodal(build_prompt(modality), image_path, provider=provider, role="deep", system=SYSTEM)
    except Exception as exc:  # noqa: BLE001 - fail open: the zero-shot read stands unchanged
        logger.warning("image cross-check unavailable (%s: %s); keeping the zero-shot read", type(exc).__name__, exc)
        return None
    label = parse(raw, modality)
    if label is None:
        logger.warning("image cross-check returned no usable label; keeping the zero-shot read")
    return label
