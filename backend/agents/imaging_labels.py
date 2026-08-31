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
# "CT scan" and "unspecified medical image" are deliberately absent:
# this prototype has no findings model for either yet.
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
