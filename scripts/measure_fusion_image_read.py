"""How well does the multimodal LLM read the RAW image, compared with BiomedCLIP, on the same benchmark images?

    python -m scripts.measure_fusion_image_read [--limit N] [--out eval_run/fusion_image_read.json]

The fusion step attaches the raw image beside the text, so its own visual reading is a second, unmeasured imaging signal. This asks the
LLM the SAME closed-set question BiomedCLIP is asked (the true modality is given; pick the best finding label for that modality from
the same label list) on every image of the real-image benchmark, one call per image, then compares top-1 with BiomedCLIP's on the
same images (paired, exact McNemar) and reports sensitivity and specificity for normal versus abnormal.

Limits, stated wherever a number from here is quoted: the labels are the uploaders', not a radiologist's; the images are public
teaching examples that a large model may have seen in training; the LLM is asked to choose from a closed list, which flatters it
compared with an open question; and it is one model, one prompt, one pass (it is not deterministic). It measures a capability, not
the pipeline.
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from backend.agents.imaging_labels import FINDING_LABELS_BY_MODALITY  # noqa: E402
from backend.services.llm_client import call_multimodal, throttle_llm_call_sync  # noqa: E402
from backend.tests.imaging_benchmark import NORMAL_LABELS, load_manifest, wilson  # noqa: E402
from backend.tests.imaging_prompt_experiment import mcnemar_exact  # noqa: E402

# The prompt, system message and parser are shared with the production cross-check, so this measures exactly it.
from backend.agents.image_crosscheck import SYSTEM, build_prompt, parse  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="eval_run/fusion_image_read.json")
    args = ap.parse_args()
    manifest = load_manifest()
    if args.limit:
        manifest = manifest[:: max(1, len(manifest) // args.limit)][: args.limit]
    bc = {r["file"]: r for r in json.loads(Path("data/benchmarks/imaging_real/results.json").read_text(encoding="utf-8"))["rows"]}

    rows = []
    for i, e in enumerate(manifest, 1):
        accepted = list(e["labels"])
        try:
            await asyncio.to_thread(throttle_llm_call_sync)
            raw = await call_multimodal(build_prompt(e["modality"]), e["file"], role="deep", system=SYSTEM)
            label = parse(raw, e["modality"])
            err = None if label else f"unparseable: {str(raw)[:80]!r}"
        except Exception as exc:  # noqa: BLE001
            label, err = None, f"{type(exc).__name__}"
        ok = label in accepted if label else False
        rows.append({"file": e["file"], "modality": e["modality"], "accepted": accepted, "label": label, "llm_ok": ok,
                     "bc_ok": bc[e["file"]]["top1_ok"], "set": e.get("set", "original"), "error": err,
                     "true_abnormal": accepted[0] not in NORMAL_LABELS, "llm_abnormal": (label not in NORMAL_LABELS) if label else None})
        print(f"[{i:>2}/{len(manifest)}] {'OK ' if ok else 'no '} {e['modality'][:6]:<6} true={accepted[0][:26]:<26} llm={str(label)[:26]:<26} {err or ''}", flush=True)

    scored = [r for r in rows if r["label"]]
    n = len(scored)
    k = sum(r["llm_ok"] for r in scored)
    kb = sum(r["bc_ok"] for r in scored)
    gained = sum(r["llm_ok"] and not r["bc_ok"] for r in scored)
    lost = sum(r["bc_ok"] and not r["llm_ok"] for r in scored)
    ab = [r for r in scored if r["true_abnormal"]]
    nor = [r for r in scored if not r["true_abnormal"]]
    result = {
        "n_images": len(rows), "n_scored": n, "unscored": len(rows) - n,
        "llm_top1": {"k": k, "n": n, "ci": wilson(k, n)}, "biomedclip_top1_same_images": {"k": kb, "n": n, "ci": wilson(kb, n)},
        "paired": {"llm_only_right": gained, "biomedclip_only_right": lost, "mcnemar_p": mcnemar_exact(gained, lost)},
        "sensitivity_abnormal": {"k": sum(r["llm_abnormal"] for r in ab), "n": len(ab)},
        "specificity_normal": {"k": sum(not r["llm_abnormal"] for r in nor), "n": len(nor)},
        "by_modality": {m: {"k": sum(r["llm_ok"] for r in scored if r["modality"] == m), "n": sum(1 for r in scored if r["modality"] == m),
                            "biomedclip_k": sum(r["bc_ok"] for r in scored if r["modality"] == m)} for m in FINDING_LABELS_BY_MODALITY},
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k_: v for k_, v in result.items() if k_ != "rows"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
