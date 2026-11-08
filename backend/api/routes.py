"""API routes.

POST /diagnose calls orchestrator.run_case() SYNCHRONOUSLY and returns
the FinalReport directly in the response body (not a
{"status": "processing"} placeholder) -- this is the simplest version
that proves the pipeline works end to end over HTTP. Async
polling (kick off run_case() in a background task/worker, let clients
poll GET /diagnose/{case_id} until it's ready) is a natural fast-follow
once this works, not implemented here.
"""

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from backend.agents.clinical_agent import extract_clinical_findings
from backend.agents.imaging_agent import ALLOWED_EXTENSIONS, analyze_image
from backend.api.schemas import ReviewRequest, ReviewResponse
from backend.models.schemas import CaseInput, ClinicalFindings, FinalReport, Patient
from backend.rag import case_memory, visual_memory
from backend.services import orchestrator, storage
from backend.services.logging import log_event
from backend.services.orchestrator import run_case

router = APIRouter()

UPLOAD_DIR = Path("data/images/uploads")


def _parse_json_field(raw: str | None, field_name: str, default):
    if raw is None or raw.strip() == "":
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"'{field_name}' must be valid JSON: {exc}"
        ) from exc


def _save_uploaded_image(case_id: str, image: UploadFile) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "upload").suffix or ".png"
    dest = UPLOAD_DIR / f"{case_id}{suffix}"
    dest.write_bytes(image.file.read())
    return str(dest)


@router.post("/diagnose", response_model=FinalReport)
def diagnose(
    case_id: str | None = Form(None),
    age: int = Form(...),
    sex: str = Form(...),
    clinical_text: str | None = Form(None),
    medical_report: str | None = Form(None),
    labs: str | None = Form(None),
    history: str | None = Form(None),
    image: UploadFile | None = File(None),
) -> FinalReport:
    """multipart/form-data so an optional image file can travel alongside
    the JSON-ish case fields in one request. `labs` and `history` are
    JSON-encoded strings (a dict and a list of strings respectively) --
    multipart form fields are flat, so nested/array values are passed
    as JSON text and parsed here, with a 422 on malformed JSON.

    Deliberately a plain `def` (not `async def`): run_case() is a
    synchronous, potentially slow pipeline, and FastAPI runs sync route
    handlers in a threadpool -- keeping this sync avoids blocking the
    event loop for the whole request, which `await`-ing a slow sync
    call from an async def would otherwise do.
    """
    resolved_case_id = case_id or str(uuid.uuid4())

    labs_dict = _parse_json_field(labs, "labs", {})
    if not isinstance(labs_dict, dict):
        raise HTTPException(status_code=422, detail="'labs' must be a JSON object")

    history_list = _parse_json_field(history, "history", [])
    if not isinstance(history_list, list) or not all(isinstance(h, str) for h in history_list):
        raise HTTPException(status_code=422, detail="'history' must be a JSON array of strings")

    image_path: str | None = None
    if image is not None and image.filename:
        suffix = Path(image.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            # A validation error (422), not a 500 -- this is a malformed
            # request, not a server failure. imaging_agent.analyze_image()
            # has its own extension/content check too, but that one only
            # ever produces a *graceful degradation* (no imaging findings,
            # request still succeeds) since it runs deep inside run_case()
            # where failures are intentionally absorbed. Rejecting an
            # obviously-wrong file type here, before we even attempt to
            # process the case, gives the caller a clear, immediate signal
            # instead of a silently-degraded report.
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported image file extension '{suffix}'. Supported: {sorted(ALLOWED_EXTENSIONS)}",
            )
        try:
            image_path = _save_uploaded_image(resolved_case_id, image)
        except Exception as exc:
            log_event("api_error", endpoint="/diagnose", case_id=resolved_case_id, error=str(exc))
            raise HTTPException(status_code=500, detail="Failed to save uploaded image") from exc

    try:
        case = CaseInput(
            case_id=resolved_case_id,
            patient=Patient(age=age, sex=sex),
            clinical_text=clinical_text,
            medical_report=medical_report,
            image_path=image_path,
            labs=labs_dict,
            history=history_list,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    storage.save_case(case, status="pending")

    try:
        report = run_case(case)
    except Exception as exc:
        # Never leak internals to the client -- log server-side, return a
        # generic 500. run_case() already degrades gracefully internally
        # (see orchestrator.py), so reaching here means something truly
        # unexpected happened.
        storage.save_case(case, status="failed")
        log_event("api_error", endpoint="/diagnose", case_id=resolved_case_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal server error while processing the case") from exc

    storage.save_case(case, status="completed")
    # Phase 14: persist the confidence breakdown alongside the report so
    # weight_tuner.py can later recalibrate w1/w2/w3 from real
    # accept/override outcomes joined against these exact numbers.
    breakdown = orchestrator.get_confidence_breakdown(resolved_case_id)
    storage.save_report(resolved_case_id, report, confidence_breakdown=breakdown)
    return report


@router.get("/diagnose/{case_id}", response_model=FinalReport)
def get_diagnosis(case_id: str) -> FinalReport:
    report = storage.get_report(case_id)
    if report is None:
        # In this synchronous-only first version /diagnose only returns
        # once run_case() has fully finished, so a missing report here
        # always means "no such case_id was ever submitted" (or a typo)
        # -- not "still processing". storage.get_case(case_id).status
        # ("pending"/"completed"/"failed") now exists and could
        # distinguish those states for a future async version; not
        # wired in here since this endpoint doesn't need it yet.
        raise HTTPException(status_code=404, detail=f"No report found for case_id '{case_id}'")
    return report


def _write_case_memory_for_override(case_id: str, review_id: int | None, diagnosis_text: str, reviewer: str) -> None:
    """Phase 13: on a clinician override, capture the correction as
    retrievable evidence for future similar cases -- ALWAYS a text
    case-memory entry, and ADDITIONALLY a visual entry when the case had
    an image in a supported modality at diagnosis time.

    Clinical/imaging findings aren't persisted from the original
    /diagnose call, so they're re-derived here from the stored
    CaseInput -- this only runs on the (comparatively rare) override
    path, not the hot request path. Every step is wrapped so a failure
    here degrades to "no case-memory entry written this time", never a
    500 on the review submission itself, which has already been
    persisted by the time this runs.
    """
    case = storage.get_case(case_id)
    if case is None:
        return  # defensive only -- submit_review() already 404s otherwise

    try:
        clinical = asyncio.run(extract_clinical_findings(case.clinical_text, case.medical_report))
    except Exception as exc:
        log_event("api_error", endpoint="/review", case_id=case_id, error=f"case-memory clinical re-extraction failed: {exc}")
        clinical = ClinicalFindings(demographics={}, symptoms=[], history=[], labs={}, imaging_text_findings=[])

    imaging = None
    if case.image_path:
        try:
            imaging = asyncio.run(analyze_image(case.image_path))
        except Exception as exc:
            log_event("api_error", endpoint="/review", case_id=case_id, error=f"case-memory imaging re-analysis failed: {exc}")

    try:
        case_memory.add_clinician_verified_case(
            case_id=case_id, clinical=clinical, imaging=imaging,
            diagnosis_text=diagnosis_text, reviewer=reviewer, review_id=review_id,
        )
    except Exception as exc:
        log_event("api_error", endpoint="/review", case_id=case_id, error=f"text case-memory write failed: {exc}")

    if case.image_path and imaging is not None and not imaging.unsupported_modality:
        try:
            asyncio.run(
                visual_memory.add_visual_case_memory(
                    case_id=case_id, image_path=case.image_path, modality=imaging.modality,
                    diagnosis_text=diagnosis_text, reviewer=reviewer, review_id=review_id,
                )
            )
        except Exception as exc:
            log_event("api_error", endpoint="/review", case_id=case_id, error=f"visual case-memory write failed: {exc}")


@router.post("/review/{case_id}", response_model=ReviewResponse)
def submit_review(case_id: str, review: ReviewRequest) -> ReviewResponse:
    if storage.get_report(case_id) is None:
        raise HTTPException(status_code=404, detail=f"No report found for case_id '{case_id}'")

    stored = storage.save_review(case_id, review.model_dump())

    if review.decision == "override" and review.annotation and review.annotation.strip():
        _write_case_memory_for_override(case_id, stored["id"], review.annotation.strip(), review.reviewer)

    storage.close_case(case_id)

    return ReviewResponse(
        case_id=stored["case_id"],
        decision=stored["decision"],
        reviewer=stored["reviewer"],
        annotation=stored["annotation"],
        rating=stored.get("rating"),
        timestamp=stored["timestamp"],
        recorded=True,
    )


@router.get("/review/{case_id}", response_model=ReviewResponse)
def get_review(case_id: str) -> ReviewResponse:
    """Returns the most recent review for case_id (a case can be
    reviewed more than once; storage.get_reviews returns the full
    history, most recent first), or 404 if it has never been reviewed."""
    reviews = storage.get_reviews(case_id)
    if not reviews:
        raise HTTPException(status_code=404, detail=f"No review found for case_id '{case_id}'")
    review = reviews[0]

    return ReviewResponse(
        case_id=review["case_id"],
        decision=review["decision"],
        reviewer=review["reviewer"],
        annotation=review["annotation"],
        rating=review.get("rating"),
        timestamp=review["timestamp"],
        recorded=True,
    )
