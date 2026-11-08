# Synthetic demo images

The image files in this directory (`synthetic_demo_brain_mri_*.png/.jpg`,
`synthetic_demo_ct_scan_*.png/.jpg`) are **programmatically generated
placeholder graphics** — simple shapes and gradients — used only to
exercise the imaging pipeline end to end, one demo image per supported
modality (MRI brain scan, CT scan, chest X-ray — see
`backend/agents/imaging_labels.py`'s `FINDING_LABELS_BY_MODALITY`).

They are **not real patient data**, not real scans, and not derived from
any real imaging study. Do not treat their filenames or content as
clinically meaningful.
