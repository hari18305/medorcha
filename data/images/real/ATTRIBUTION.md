# Real images: sources, licences and attribution

The six images in this folder are **real medical images** released by their
authors under open licences and retrieved from Wikimedia Commons on
2026-11-10. They are used only as test/demo inputs for the imaging pipeline.
They are **not** synthetic (unlike the `synthetic_demo_*` files one level up)
and the patients are not identified, but they are real people's scans — treat
them with the care that implies, and do not use them to imply anything about
any individual.

None of this is a clinical dataset and none of it validates the system.

| File here | Commons page | Author / credit | Licence | Changes made |
|---|---|---|---|---|
| `real_ct_scan_06.png` | [Computed tomography of human brain (6).png](https://commons.wikimedia.org/wiki/File:Computed_tomography_of_human_brain_(6).png) | Department of Radiology, Uppsala University Hospital; uploaded by Mikael Häggström | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | none |
| `real_ct_scan_10.png` | [Computed tomography of human brain (10).png](https://commons.wikimedia.org/wiki/File:Computed_tomography_of_human_brain_(10).png) | same | CC0 1.0 | none |
| `real_ct_scan_14.png` | [Computed tomography of human brain (14).png](https://commons.wikimedia.org/wiki/File:Computed_tomography_of_human_brain_(14).png) | same | CC0 1.0 | none |
| `real_brain_mri_t1.png` | [T1-weighted-MRI.png](https://commons.wikimedia.org/wiki/File:T1-weighted-MRI.png) | Mim.cis; source: Johns Hopkins University | Public domain (US medical imaging; [CC Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/)) | none |
| `real_chest_xray_normal_pa.jpg` | [Normal posteroanterior (PA) chest radiograph (X-ray).jpg](https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg) | Mikael Häggström (own work) | CC0 1.0 | none |
| `real_chest_xray_pneumothorax.png` | [Chest X-ray of pneumothorax.png](https://commons.wikimedia.org/wiki/File:Chest_X-ray_of_pneumothorax.png) | Alex Bonilla, Alexander J. Blair, et al. (10 authors, incl. Jatin M. Vyas), *Journal of Medical Case Reports* (2019), [doi:10.1186/s13256-019-2215-4](https://jmedicalcasereports.biomedcentral.com/articles/10.1186/s13256-019-2215-4); uploaded to Commons by Ameisenigel | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | **cropped** to panel A (top-left) of the multi-panel figure; no other change |

Licences were read from each file's own Commons page (not a mirror).
CC0 and the public-domain mark require no attribution; the credits above are
given anyway. CC BY 4.0 (the pneumothorax image) **requires** this
attribution, the licence link, and an indication of changes — kept here.

## Caveats about using these for evaluation

- The pneumothorax figure comes from an open-access PubMed Central article,
  and BiomedCLIP was trained on PubMed Central figure–caption pairs, so a
  correct "pneumothorax" label on it may reflect training-data overlap rather
  than generalisation.
- A handful of hand-picked, clean, well-known images is a plausibility check
  of the pipeline's routing, not an accuracy measurement.
- `python scripts/check_real_images.py` re-runs the real BiomedCLIP on each
  image and reports how it is classified.
