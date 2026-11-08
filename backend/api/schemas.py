"""API-layer request/response models.

Core domain contracts (CaseInput, FinalReport, etc.) live in
backend/models/schemas.py -- import from there rather than redefining
shapes here. This module holds only API-specific wrappers that no
agent needs to know about, like the human review decision below.
"""

from typing import Literal

from pydantic import BaseModel, model_validator


class ReviewRequest(BaseModel):
    decision: Literal["accept", "override", "annotate"]
    # For "annotate", this is the free-text note. For "override", this
    # carries the reviewer's replacement diagnosis (there's no separate
    # structured field for it -- the frontend puts whatever diagnosis
    # the reviewer picked/typed in here, e.g. "Override diagnosis:
    # Migraine with aura"), which is why both require it non-empty.
    # Phase 13: on "override", this exact string is what gets embedded
    # as the case-memory diagnosis text (both text and, for image
    # cases, visual).
    annotation: str | None = None
    reviewer: str
    # Optional 1-5 clinician confidence-in-the-review rating (Phase 11).
    # Persisted but NOT consumed by Phase 14's recalibration (see
    # weight_tuner.py's collect_review_outcomes()) -- only the
    # accept/override decision itself feeds that.
    rating: int | None = None

    @model_validator(mode="after")
    def _annotation_required_for_annotate_or_override(self) -> "ReviewRequest":
        if self.decision in ("annotate", "override") and not (self.annotation and self.annotation.strip()):
            raise ValueError(
                "'annotation' is required when decision is 'annotate' (the note) "
                "or 'override' (the replacement diagnosis)"
            )
        return self

    @model_validator(mode="after")
    def _rating_bounds(self) -> "ReviewRequest":
        if self.rating is not None and not (1 <= self.rating <= 5):
            raise ValueError("'rating' must be between 1 and 5")
        return self


class ReviewResponse(BaseModel):
    case_id: str
    decision: str
    reviewer: str
    annotation: str | None
    rating: int | None = None
    timestamp: str
    recorded: bool
