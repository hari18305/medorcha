"""Case, report, review, and agent-log persistence.

Backend: stdlib `sqlite3`, not SQLAlchemy. Documented choice: this
module has always used the standard library sqlite3 module (Phase 11's
reviews table introduced it), the schema here is small and fixed (four
tables, no relationships beyond simple FKs, "no migration framework
needed yet" per the spec), and pulling in SQLAlchemy would mean an ORM
layer, a new dependency, and a session/engine lifecycle to manage for
no real benefit at this size. If the schema grows substantially or a
migration tool becomes necessary, SQLAlchemy + Alembic is the natural
next step -- nothing here should make that harder to introduce later,
since every table access is already funneled through this module's
functions.

Tables (see the CREATE TABLE statements below for the authoritative
definitions):
  - cases: one row per submitted case (case_id PK).
  - reports: one row per case's current FinalReport (case_id PK, so
    saving a new report for the same case_id replaces it -- matches
    the Phase 9 in-memory dict's overwrite semantics).
  - reviews: one row per review event (a case can be reviewed more than
    once -- e.g. re-reviewed after a correction -- so this is a proper
    one-to-many table with its own `id`, not one row per case).
  - agent_logs: one row per orchestrator agent-call event (per Phase 8;
    also written to stdout via services/logging.py -- this table is
    the persisted, queryable copy of the same events).

`REFERENCES cases(case_id)` in reports/reviews/agent_logs documents the
intended relationship but is NOT enforced (SQLite has foreign keys off
by default, and this module deliberately leaves them off) -- agents can
be logged, and reports/reviews queried, independently of exactly when
save_case() ran, which matters for unit-testing individual pieces (e.g.
backend/tests/test_orchestrator.py) without needing to seed a `cases`
row first.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

from dotenv import load_dotenv

from backend.models.schemas import CaseInput, FinalReport

load_dotenv()


def _resolve_sqlite_path() -> str:
    database_url = os.getenv("DATABASE_URL", "sqlite:///./medorchestrate.db")
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        return database_url[len(prefix) :]
    # Not a sqlite:// URL (e.g. a real Postgres URL from a future
    # storage upgrade) -- this module only speaks SQLite, so fall back
    # to a local file rather than trying to connect to it.
    return "medorchestrate.db"


# Module-level so tests can monkeypatch it to an isolated temp file and
# call _init_tables() again, without touching the real dev database.
DB_PATH = _resolve_sqlite_path()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl_type: str) -> None:
    """Lightweight migration helper: this module has no migration
    framework (see module docstring), but Phase 13/14 add new columns
    (reviews.rating, reports.confidence_breakdown_json) to tables that
    may already exist with data in a developer's local .db file from an
    earlier version of this app -- CREATE TABLE IF NOT EXISTS alone
    can't add a column to an already-created table, so this checks via
    PRAGMA table_info and ALTER TABLE ADD COLUMN only if it's actually
    missing.
    """
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def _init_tables() -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                input_json TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                case_id TEXT PRIMARY KEY REFERENCES cases (case_id),
                report_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                iteration_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        # Phase 14 needs each report's confidence *breakdown* (s_rag,
        # s_web, agreement), not just the final float, joined against
        # many past cases' review outcomes -- persisted here since
        # weight_tuner.py runs long after the original request/response
        # cycle that computed it.
        _add_column_if_missing(conn, "reports", "confidence_breakdown_json", "TEXT")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL REFERENCES cases (case_id),
                decision TEXT NOT NULL,
                annotation TEXT,
                reviewer TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        # Phase 11's optional 1-5 rating -- persisted but (per spec) not
        # consumed by Phase 14's recalibration; bounds-checking happens
        # at the API layer (backend/api/schemas.py), not here.
        _add_column_if_missing(conn, "reviews", "rating", "INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_case_id ON reviews (case_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL REFERENCES cases (case_id),
                agent_name TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                status TEXT NOT NULL,
                latency REAL,
                confidence REAL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_logs_case_id ON agent_logs (case_id)")

        # Phase 13: one row per Qdrant collection an override actually
        # wrote to -- a text-only override writes one "text" row; an
        # override on an image case writes both a "text" and a "visual"
        # row (two separate collections, two separate embeddings).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS case_memory_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL REFERENCES cases (case_id),
                review_id INTEGER REFERENCES reviews (id),
                embedded_at TEXT NOT NULL,
                qdrant_point_id TEXT NOT NULL,
                memory_type TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_case_memory_log_case_id ON case_memory_log (case_id)")

        # Phase 14: an audit trail of every weight recalibration --
        # confidence.get_current_weights() reads the most recent row
        # here (falling back to DEFAULT_WEIGHTS if this table is empty).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS confidence_weight_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                updated_at TEXT NOT NULL,
                w1 REAL NOT NULL,
                w2 REAL NOT NULL,
                w3 REAL NOT NULL,
                num_reviews_used INTEGER NOT NULL,
                notes TEXT
            )
            """
        )


_init_tables()


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

# Conventional status values (not DB-enforced via CHECK, to keep the
# schema easy to extend without a migration): "pending" while a case is
# being processed, "completed" once a report is saved, "failed" if
# run_case() raised. GET /diagnose/{case_id} does not yet branch on
# this (it still just checks whether a report exists), but the column
# is here for that distinction once an async version needs it.


def save_case(case: CaseInput, status: str = "pending") -> None:
    """Insert a case, or update its input_json/status if case_id already
    exists -- created_at is set once, on first insert, and preserved on
    later calls (e.g. status transitioning pending -> completed/failed
    for the same case_id).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO cases (case_id, created_at, input_json, status)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
                input_json = excluded.input_json,
                status = excluded.status
            """,
            (case.case_id, created_at, case.model_dump_json(), status),
        )


def get_case(case_id: str) -> CaseInput | None:
    with _get_connection() as conn:
        row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    return CaseInput.model_validate_json(row["input_json"])


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------


def save_report(case_id: str, report: FinalReport, confidence_breakdown: dict | None = None) -> None:
    """Upserts the report for case_id -- a case has at most one current
    report; a new save_report() call replaces the previous one, same
    as the old in-memory dict's `_reports[case_id] = report`.

    confidence_breakdown (Phase 14): Phase 6's compute_confidence_breakdown()
    dict ({"s_rag", "s_web", "agreement", "confidence"}), persisted
    alongside the report since weight_tuner.py needs it across many past
    cases, long after the original request/response cycle. Optional --
    callers that don't have it yet (or don't care) can omit it, in which
    case that case simply won't be eligible for Phase 14's recalibration.
    """
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO reports (case_id, report_json, confidence, iteration_count, confidence_breakdown_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
                report_json = excluded.report_json,
                confidence = excluded.confidence,
                iteration_count = excluded.iteration_count,
                confidence_breakdown_json = excluded.confidence_breakdown_json
            """,
            (
                case_id,
                report.model_dump_json(),
                report.confidence,
                report.iteration_count,
                json.dumps(confidence_breakdown) if confidence_breakdown is not None else None,
            ),
        )


def get_report(case_id: str) -> FinalReport | None:
    with _get_connection() as conn:
        row = conn.execute("SELECT * FROM reports WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    return FinalReport.model_validate_json(row["report_json"])


def get_report_breakdown(case_id: str) -> dict | None:
    """The confidence breakdown persisted alongside case_id's report
    (see save_report()'s confidence_breakdown param), or None if it was
    never saved (older report, or saved without a breakdown)."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT confidence_breakdown_json FROM reports WHERE case_id = ?", (case_id,)
        ).fetchone()
    if row is None or row["confidence_breakdown_json"] is None:
        return None
    return json.loads(row["confidence_breakdown_json"])


# --------------------------------------------------------------------------
# reviews
# --------------------------------------------------------------------------


def save_review(case_id: str, review: dict) -> dict:
    """Insert a review row for case_id (a case can be reviewed more than
    once, so this appends rather than overwrites). Returns the stored
    row, including the generated id and server-assigned timestamp, so
    callers can echo back exactly what was persisted.

    review["rating"] (optional, 1-5): persisted but NOT consumed by
    Phase 14's recalibration (see weight_tuner.py) -- bounds-checking
    happens at the API layer (backend/api/schemas.py), not here.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO reviews (case_id, decision, annotation, reviewer, timestamp, rating) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (case_id, review["decision"], review.get("annotation"), review["reviewer"], timestamp, review.get("rating")),
        )
        review_id = cursor.lastrowid

    return {
        "id": review_id,
        "case_id": case_id,
        "decision": review["decision"],
        "annotation": review.get("annotation"),
        "reviewer": review["reviewer"],
        "timestamp": timestamp,
        "rating": review.get("rating"),
    }


def get_reviews(case_id: str) -> list[dict]:
    """Every review recorded for case_id, most recent first."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE case_id = ? ORDER BY timestamp DESC, id DESC",
            (case_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------
# agent_logs
# --------------------------------------------------------------------------


def log_agent_event(
    case_id: str,
    agent_name: str,
    iteration: int,
    status: str,
    latency: float | None = None,
    confidence: float | None = None,
) -> None:
    """Persists one orchestrator agent-call event (see
    backend/services/orchestrator.py's _log_agent_call, which calls
    this alongside the stdout structured log from services/logging.py).
    """
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO agent_logs (case_id, agent_name, iteration, status, latency, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (case_id, agent_name, iteration, status, latency, confidence),
        )


def get_agent_logs(case_id: str) -> list[dict]:
    """Every agent_logs row for case_id, in the order they were recorded."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM agent_logs WHERE case_id = ? ORDER BY id ASC",
            (case_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def close_case(case_id: str) -> None:
    """Marks case_id's status "closed" -- called once a review has been
    successfully recorded for it (see backend/api/routes.py's
    POST /review/{case_id})."""
    with _get_connection() as conn:
        conn.execute("UPDATE cases SET status = ? WHERE case_id = ?", ("closed", case_id))


# --------------------------------------------------------------------------
# case_memory_log (Phase 13)
# --------------------------------------------------------------------------


def log_case_memory_embedding(
    case_id: str, review_id: int | None, qdrant_point_id: str, memory_type: str
) -> None:
    """Records one case-memory write -- one row per Qdrant collection an
    override actually wrote to. memory_type is "text" or "visual"; an
    override with no image writes only a "text" row (see
    backend/rag/case_memory.py / backend/rag/visual_memory.py).
    """
    if memory_type not in ("text", "visual"):
        raise ValueError(f"memory_type must be 'text' or 'visual', got {memory_type!r}")
    embedded_at = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO case_memory_log (case_id, review_id, embedded_at, qdrant_point_id, memory_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (case_id, review_id, embedded_at, qdrant_point_id, memory_type),
        )


def get_case_memory_log(case_id: str) -> list[dict]:
    """Every case_memory_log row for case_id, in the order they were
    recorded -- e.g. an override on an image case produces two rows,
    one memory_type="text" and one memory_type="visual"."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM case_memory_log WHERE case_id = ? ORDER BY id ASC",
            (case_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------
# confidence_weight_history (Phase 14)
# --------------------------------------------------------------------------


def save_weight_update(w1: float, w2: float, w3: float, num_reviews_used: int, notes: str | None = None) -> dict:
    """Persists one weight recalibration event. confidence.py's
    get_current_weights() reads the most recent row here."""
    updated_at = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO confidence_weight_history (updated_at, w1, w2, w3, num_reviews_used, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (updated_at, w1, w2, w3, num_reviews_used, notes),
        )
        row_id = cursor.lastrowid
    return {
        "id": row_id,
        "updated_at": updated_at,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "num_reviews_used": num_reviews_used,
        "notes": notes,
    }


def get_current_weight_row() -> dict | None:
    """The most recent confidence_weight_history row, or None if no
    recalibration has ever happened (confidence.py's get_current_weights()
    falls back to DEFAULT_WEIGHTS in that case)."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM confidence_weight_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row is not None else None


def get_weight_history() -> list[dict]:
    """Every confidence_weight_history row, oldest first -- the full
    recalibration audit trail."""
    with _get_connection() as conn:
        rows = conn.execute("SELECT * FROM confidence_weight_history ORDER BY id ASC").fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------
# Review-outcome join (Phase 14's collect_review_outcomes() reads this)
# --------------------------------------------------------------------------


def get_reports_with_reviews() -> list[dict]:
    """Every case with BOTH a stored confidence breakdown AND at least
    one review decision, joined into
    {case_id, s_rag, s_web, agreement, predicted_confidence, decision}
    rows (one row per review -- a case reviewed twice yields two rows).
    Used by weight_tuner.collect_review_outcomes(); kept here (rather
    than reimplementing the join in weight_tuner.py) since this module
    owns the schema both tables live in.
    """
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT reports.case_id AS case_id,
                   reports.confidence_breakdown_json AS confidence_breakdown_json,
                   reviews.decision AS decision
            FROM reports
            JOIN reviews ON reviews.case_id = reports.case_id
            WHERE reports.confidence_breakdown_json IS NOT NULL
            ORDER BY reviews.id ASC
            """
        ).fetchall()

    results = []
    for row in rows:
        breakdown = json.loads(row["confidence_breakdown_json"])
        results.append(
            {
                "case_id": row["case_id"],
                "s_rag": breakdown.get("s_rag"),
                "s_web": breakdown.get("s_web"),
                "agreement": breakdown.get("agreement"),
                "predicted_confidence": breakdown.get("confidence"),
                "decision": row["decision"],
            }
        )
    return results
