"""Candidate label sets for BiomedCLIP zero-shot classification.

All lists are plain Python lists of short clinical phrases used as the
text side of zero-shot classification (image embedding vs. each label's
text embedding). They are intentionally small and scoped to this
prototype's demo cases -- not a comprehensive radiology ontology.

To extend a modality's coverage: add another short, clinically phrased
string to that modality's list in FINDING_LABELS_BY_MODALITY. Prefer
phrases in the same style/length as the existing entries, since
BiomedCLIP's text encoder was trained on short biomedical captions --
long or multi-clause phrases embed less reliably.
"""

# Candidate imaging modalities. Extend this list to add support for a
# new modality (e.g. "abdominal ultrasound") -- the imaging agent
# classifies against whichever labels are here, no code changes needed
# elsewhere. Note: adding a modality here does NOT by itself give it a
# findings model -- see FINDING_LABELS_BY_MODALITY below.
MODALITY_LABELS = [
    "MRI brain scan",
    "CT scan",
    "chest X-ray",
    "unspecified medical image",
]

# Per-modality candidate findings. A modality only gets real
# finding-level classification if it has an entry here -- imaging_agent.py
# checks this explicitly and returns an honest "no findings model
# available for this modality" result (flagged via
# ImagingFindings.unsupported_modality) rather than forcing the image
# through an unrelated modality's label set. That mismatch is exactly
# what silently produced brain-related "findings" (and downstream
# brain-related RAG evidence) for a chest X-ray before this modality
# had its own labels -- confirmed against a real uploaded chest X-ray.
#
# "unspecified medical image" is deliberately absent: this prototype
# has no findings model for it. Extend by adding a new key + a
# genuinely-relevant label list for that modality, never by reusing
# another modality's labels.
FINDING_LABELS_BY_MODALITY: dict[str, list[str]] = {
    "MRI brain scan": [
        "normal brain MRI, no acute findings",
        "white matter hyperintensity",
        "mass lesion",
        "midline shift",
        "ventricular enlargement",
        "acute infarct",
        "chronic infarct",
        "hemorrhage",
        "sinus abnormality",
        "cerebral atrophy",
        "edema",
        "vascular malformation",
        "no significant abnormality",
    ],
    "CT scan": [
        "normal head CT, no acute findings",
        "hypodense lesion",
        "mass lesion",
        "midline shift",
        "ventricular enlargement",
        "acute infarct",
        "hemorrhage",
        "sinus abnormality",
        "calcification",
        "no significant abnormality",
    ],
    "chest X-ray": [
        "normal chest X-ray, no acute findings",
        "pulmonary nodule or mass",
        "consolidation",
        "pleural effusion",
        "pneumothorax",
        "cardiomegaly",
        "pulmonary edema",
        "hyperinflation consistent with emphysema",
        "atelectasis",
        "hilar or mediastinal widening",
        "rib fracture",
        "no significant abnormality",
    ],
}

# Modalities with a real findings model configured above. Used by
# imaging_agent.py to decide whether to run finding-level classification
# at all, and referenced in the "unsupported modality" message so it's
# accurate if this list grows.
SUPPORTED_FINDING_MODALITIES = list(FINDING_LABELS_BY_MODALITY.keys())

# Labels that all say "nothing abnormal"; two reads that land on different ones still agree.
NORMAL_FINDING_LABELS = {
    "normal chest X-ray, no acute findings",
    "normal head CT, no acute findings",
    "normal brain MRI, no acute findings",
    "no significant abnormality",
}


# --------------------------------------------------------------------------
# Finding PROMPTS (what is actually embedded as text) vs. finding LABELS (what is returned)
# --------------------------------------------------------------------------
# BiomedCLIP scores an image against short captions. A bare label such as "hemorrhage"
# is a weak caption; "head CT scan showing intracranial hemorrhage, hyperdense blood"
# reads like the figure captions the model was trained on. On a held-out half of the
# real-image benchmark this wording raised top-1 from 7/25 to 12/25 (MRR 0.49 -> 0.61;
# paired p = 0.18, i.e. NOT statistically significant at that size) - see
# data/benchmarks/imaging_real/PROMPT_EXPERIMENT.md. The prompt changes only what is
# embedded; the returned finding is still the label above, so nothing downstream changes.
# Modality prompts were also tried and NOT changed (gain of 1 image, below the bar of 2).
#
# Descriptions were written by a non-radiologist from general knowledge; do not edit them
# to chase the benchmark (the held-out half has been used once). Set
# IMAGING_FINDING_PROMPTS=label to fall back to the bare labels.

import os  # noqa: E402

MODALITY_PHRASE = {"chest X-ray": "chest X-ray", "CT scan": "head CT scan", "MRI brain scan": "brain MRI scan"}

FINDING_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "chest X-ray": {
        "normal chest X-ray, no acute findings": "no acute cardiopulmonary abnormality",
        "pulmonary nodule or mass": "a pulmonary nodule or lung mass",
        "consolidation": "airspace consolidation, pneumonia",
        "pleural effusion": "a pleural effusion",
        "pneumothorax": "a pneumothorax",
        "cardiomegaly": "an enlarged heart, cardiomegaly",
        "pulmonary edema": "pulmonary edema",
        "hyperinflation consistent with emphysema": "hyperinflated lungs, emphysema",
        "atelectasis": "atelectasis, collapsed lung",
        "hilar or mediastinal widening": "hilar or mediastinal widening",
        "rib fracture": "a rib fracture",
        "no significant abnormality": "no significant abnormality",
    },
    "CT scan": {
        "normal head CT, no acute findings": "no acute intracranial abnormality",
        "hypodense lesion": "a hypodense lesion",
        "mass lesion": "an intracranial mass lesion",
        "midline shift": "midline shift",
        "ventricular enlargement": "enlarged ventricles, hydrocephalus",
        "acute infarct": "an acute ischemic infarct, a hypodense region",
        "hemorrhage": "intracranial hemorrhage, hyperdense blood",
        "sinus abnormality": "sinus opacification, sinusitis",
        "calcification": "intracranial calcification",
        "no significant abnormality": "no significant abnormality",
    },
    "MRI brain scan": {
        "normal brain MRI, no acute findings": "no acute intracranial abnormality",
        "white matter hyperintensity": "white matter hyperintensities",
        "mass lesion": "a brain tumor, a mass lesion",
        "midline shift": "midline shift",
        "ventricular enlargement": "enlarged ventricles, hydrocephalus",
        "acute infarct": "an acute infarct with restricted diffusion",
        "chronic infarct": "a chronic infarct",
        "hemorrhage": "intracranial hemorrhage",
        "sinus abnormality": "sinus abnormality",
        "cerebral atrophy": "cerebral atrophy, volume loss",
        "edema": "cerebral edema",
        "vascular malformation": "a vascular malformation",
        "no significant abnormality": "no significant abnormality",
    },
}


def finding_prompt(modality: str, label: str) -> str:
    """The caption embedded for `label` under `modality`. Falls back to the bare label when
    IMAGING_FINDING_PROMPTS=label or when a (newly added) label has no description yet."""
    if os.getenv("IMAGING_FINDING_PROMPTS", "showing").strip().lower() == "label":
        return label
    desc = FINDING_DESCRIPTIONS.get(modality, {}).get(label)
    phrase = MODALITY_PHRASE.get(modality)
    if desc is None or phrase is None:
        return label
    return f"{phrase} showing {desc}"
