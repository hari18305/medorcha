"""Medical Imaging Agent: extracts ImagingFindings from an uploaded image.

IMPORTANT — scope and limitations:
This agent is a ZERO-SHOT CLASSIFIER built on BiomedCLIP, a pretrained
biomedical vision-language embedding model. It is bounded to a small,
hand-picked, prototype-scoped label set (see imaging_labels.py) — it is
NOT a comprehensive radiology ontology and NOT a clinically validated
diagnostic imaging model. Its output is presented to clinicians as
decision support only, never as a diagnosis. No LLM API is called in
this file; BiomedCLIP is a fixed, non-fine-tuned embedding model used
purely for image/text similarity scoring.

Named "Medical Imaging Agent" (not "MRI Agent") because MRI is only the
modality demoed first — the same zero-shot approach extends to CT,
X-ray, etc. by adding labels to imaging_labels.py, no code changes here.
"""

import asyncio
import os
from pathlib import Path

import open_clip
import torch
from PIL import Image, UnidentifiedImageError

from backend.agents.imaging_labels import (
    FINDING_LABELS_BY_MODALITY,
    MODALITY_LABELS,
    SUPPORTED_FINDING_MODALITIES,
)
from backend.models.schemas import ImagingFindings

MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# File extensions this agent will attempt to open. PNG/JPEG are decoded
# directly by Pillow. DICOM (.dcm/.dicom) is only "if-convertible": we
# attempt to open it the same way, and if the runtime's Pillow build
# can't decode it, that surfaces as a clear UnsupportedImageFormatError
# rather than a crash (this prototype does not bundle a DICOM codec).
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}

# Zero-shot softmax score FLOOR for including a finding label in the
# result; the single top-scoring label is always included regardless (see
# analyze_image), so `findings` is never empty.
FINDING_SCORE_THRESHOLD = 0.15

# A label BEYOND the top one must ALSO score at least this fraction of the
# top label's score. The scores are a softmax over ~a dozen labels, so they
# sum to 1: a decisive top label leaves little for the rest, and a secondary
# label that merely clears the floor is usually noise. Found live: a real
# pneumothorax film scored pneumothorax 0.68 and a spurious "rib fracture"
# 0.15 (22% as strong, 0.1514 against a 0.15 floor); fusion then ranked "rib
# fracture" at 0.85 in the differential though the film shows none.
# Measured on real images, genuine two-way ambiguity looks like 0.83-0.86 of
# the top, noise like 0.0-0.53, so 0.6 keeps the former and drops the latter.
# PROTOTYPE DEFAULT from a handful of images, not a validated value.
# Override with FINDING_RELATIVE_THRESHOLD.
def _relative_threshold_from_env() -> float:
    try:
        return float(os.getenv("FINDING_RELATIVE_THRESHOLD", "0.6"))
    except ValueError:
        return 0.6


FINDING_RELATIVE_THRESHOLD = _relative_threshold_from_env()

# CLIP-style logit scale applied to cosine similarities before softmax,
# matching BiomedCLIP's training configuration.
LOGIT_SCALE = 100.0

# Returned as the sole `findings` entry (with confidence 0.0) when the
# detected/hinted modality has no findings model configured -- see
# imaging_labels.FINDING_LABELS_BY_MODALITY. Deliberately does not
# resemble a real finding, and callers should prefer checking
# ImagingFindings.unsupported_modality over string-matching this.
UNSUPPORTED_MODALITY_MESSAGE = (
    "no findings model available for this modality (supported: "
    + ", ".join(SUPPORTED_FINDING_MODALITIES)
    + ")"
)


class ImagingAgentError(RuntimeError):
    """Base class for imaging agent errors."""


class ImageNotFoundError(ImagingAgentError):
    """Raised when the given image_path does not point to an existing file."""


class UnsupportedImageFormatError(ImagingAgentError):
    """Raised when the file extension or content is not a supported image."""


_model = None
_preprocess = None
_tokenizer = None


def _get_model_bundle():
    """Lazily load and cache the BiomedCLIP model, preprocessor, and tokenizer.

    Loaded once on first use (not at import time) so importing this
    module — e.g. for /health or unrelated tests — never triggers a
    network fetch or the multi-second model load. Cached in module
    globals thereafter so repeated calls to analyze_image reuse it.
    """
    global _model, _preprocess, _tokenizer
    if _model is None:
        model, preprocess = open_clip.create_model_from_pretrained(MODEL_ID)
        tokenizer = open_clip.get_tokenizer(MODEL_ID)
        model.eval()
        _model, _preprocess, _tokenizer = model, preprocess, tokenizer
    return _model, _preprocess, _tokenizer


def get_biomedclip_bundle():
    """Public accessor for the (model, preprocess, tokenizer) tuple this
    module already loads and caches -- Phase 13's visual case memory
    (backend/rag/visual_memory.py) reuses THIS loaded model to compute
    image embeddings rather than loading a second copy of the same
    ~400MB model in a different file."""
    return _get_model_bundle()


def _embed_image_sync(path: Path) -> list[float]:
    """Synchronous, CPU/GPU-bound BiomedCLIP image embedding. Run via an
    executor by embed_image() -- same pattern as
    _run_zero_shot_classification()."""
    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise UnsupportedImageFormatError(f"Could not decode image content at {path}: {exc}") from exc

    model, preprocess, _ = _get_model_bundle()
    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.squeeze(0).tolist()


async def embed_image(image_path: str) -> list[float]:
    """Computes a BiomedCLIP image embedding for `image_path`, using the
    SAME cached model/preprocessing as analyze_image() -- for Phase 13's
    visual case memory (backend/rag/visual_memory.py). This is a
    DIFFERENT embedding space from BioBERT's 768-dim text embeddings
    (backend/rag/embeddings.py) -- see get_image_embedding_dim() for the
    actual dimension of this space, never assumed.
    """
    path = _validate_image_path(image_path)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _embed_image_sync, path)


_image_embedding_dim: int | None = None


def get_image_embedding_dim() -> int:
    """The actual output dimension of BiomedCLIP's image encoder,
    determined by running one real embedding call against a throwaway
    blank image -- NEVER assumed to match BioBERT's 768-dim text space
    (a completely different embedding space). Phase 13's visual case
    memory Qdrant collection must be created with this size. Cached
    after the first call since it can't change at runtime for a given
    loaded model.
    """
    global _image_embedding_dim
    if _image_embedding_dim is None:
        model, preprocess, _ = _get_model_bundle()
        blank = Image.new("RGB", (224, 224))
        image_input = preprocess(blank).unsqueeze(0)
        with torch.no_grad():
            features = model.encode_image(image_input)
        _image_embedding_dim = int(features.shape[-1])
    return _image_embedding_dim


def _validate_image_path(image_path: str) -> Path:
    if not image_path:
        raise ValueError(
            "analyze_image requires a non-empty image_path; callers must "
            "confirm an image is present before calling this function."
        )

    path = Path(image_path)
    if not path.is_file():
        raise ImageNotFoundError(f"Image file not found: {image_path}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UnsupportedImageFormatError(
            f"Unsupported image file extension '{path.suffix}' for {image_path}. "
            f"Supported extensions: {sorted(ALLOWED_EXTENSIONS)}"
        )

    return path


def _zero_shot_scores(
    model, tokenizer, image_features: torch.Tensor, labels: list[str]
) -> list[tuple[str, float]]:
    """Cosine-similarity zero-shot scores between the image and each label, softmaxed."""
    text_tokens = tokenizer(labels)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = (LOGIT_SCALE * image_features @ text_features.T).softmax(dim=-1)
    scores = logits.squeeze(0).tolist()

    return sorted(zip(labels, scores), key=lambda pair: pair[1], reverse=True)


def rank_image(path: Path, modality_hint: str | None = None) -> dict:
    """Synchronous BiomedCLIP inference returning the FULL zero-shot rankings.

    {"modality": str,                       # the hint if given, else the top modality label
     "modality_scores": [(label, p), ...] | None,   # None when a hint skipped detection
     "finding_scores":  [(label, p), ...] | None}   # None when the modality has no findings model

    The production path (_run_zero_shot_classification) and the imaging benchmark
    (backend/tests/imaging_benchmark.py) both call this, so the benchmark measures
    exactly what the pipeline runs rather than a reimplementation of it.
    """
    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise UnsupportedImageFormatError(
            f"Could not decode image content at {path}: {exc}"
        ) from exc

    model, preprocess, tokenizer = _get_model_bundle()
    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        modality_scores = None
        if modality_hint is None:
            modality_scores = _zero_shot_scores(model, tokenizer, image_features, MODALITY_LABELS)
            modality = modality_scores[0][0]
        else:
            modality = modality_hint

        finding_labels = FINDING_LABELS_BY_MODALITY.get(modality)
        finding_scores = None
        if finding_labels is not None:
            finding_scores = _zero_shot_scores(model, tokenizer, image_features, finding_labels)

    return {"modality": modality, "modality_scores": modality_scores, "finding_scores": finding_scores}


def select_findings(finding_scores: list[tuple[str, float]]) -> list[str]:
    """Which ranked findings the pipeline reports: the top label always, plus any other
    that clears both FINDING_SCORE_THRESHOLD and FINDING_RELATIVE_THRESHOLD x the top."""
    top_label, top_score = finding_scores[0]
    findings = [
        label
        for label, score in finding_scores
        if score > FINDING_SCORE_THRESHOLD and score >= FINDING_RELATIVE_THRESHOLD * top_score
    ]
    return findings or [top_label]


def _run_zero_shot_classification(
    path: Path, modality_hint: str | None
) -> tuple[str, list[str], float, bool]:
    """Synchronous, CPU/GPU-bound BiomedCLIP inference. Run via an executor."""
    ranked = rank_image(path, modality_hint)
    modality, finding_scores = ranked["modality"], ranked["finding_scores"]

    if finding_scores is None:
        # No findings model for this modality -- do NOT classify
        # against some other modality's labels (that's exactly what
        # silently produced brain-related "findings" for a chest
        # X-ray before this check existed). Be honest instead.
        return modality, [UNSUPPORTED_MODALITY_MESSAGE], 0.0, True

    return modality, select_findings(finding_scores), float(finding_scores[0][1]), False


async def analyze_image(image_path: str, modality_hint: str | None = None) -> ImagingFindings:
    """Classify an image into ImagingFindings via BiomedCLIP zero-shot scoring.

    Only call this when image_path is actually present — CaseInput.image_path
    is optional, and callers upstream (e.g. the orchestrator) are responsible
    for skipping this agent entirely when no image was provided, rather than
    passing an empty/None path in here.

    If the detected/hinted modality has no findings model configured (see
    imaging_labels.FINDING_LABELS_BY_MODALITY), the returned
    ImagingFindings has unsupported_modality=True, confidence=0.0, and a
    findings list with one honest placeholder entry -- callers must check
    unsupported_modality rather than trusting `findings` at face value.
    """
    path = _validate_image_path(image_path)

    loop = asyncio.get_running_loop()
    modality, findings, confidence, unsupported = await loop.run_in_executor(
        None, _run_zero_shot_classification, path, modality_hint
    )

    return ImagingFindings(
        modality=modality,
        findings=findings,
        confidence=confidence,
        unsupported_modality=unsupported,
    )
