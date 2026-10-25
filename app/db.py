"""SQLite persistence layer.

SQLite keeps the deployment footprint small (the project had no database
before this stage).  All clinically sensitive text is stored encrypted in
``ciphertext``/``payload`` columns; indexed columns hold only non-sensitive
routing metadata.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schemas import (
    CaseRecord,
    ClinicalSummary,
    Discrepancy,
    DraftNote,
    EscalationRecord,
    Fact,
    FollowUpAnswer,
    FollowUpRecord,
    PatientMessage,
    ProcessingRun,
    ReviewEvent,
    SourceRecord,
    StaffingState,
    Warning,
)

ISO = "%Y-%m-%dT%H:%M:%S.%f%z"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(model) -> str:
    if hasattr(model, "model_dump"):
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=False)
    return json.dumps(model, ensure_ascii=False, default=str)


_SENSITIVE_METADATA_KEYS = {
    "text",
    "transcript",
    "translated_text",
    "content",
    "block_content",
    "quote",
    "original_text",
    "normalized_text",
    "body",
    "message",
    "answer",
}


def _sanitize_metadata(value):
    """Remove source text that upstream components may nest in metadata."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in _SENSITIVE_METADATA_KEYS:
                if key in {"original_text", "normalized_text"} and item == "":
                    cleaned[key] = ""
                continue
            cleaned[key] = _sanitize_metadata(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata(item) for item in value]
    return value


def _restore_translation(record, texts: dict) -> None:
    """Reattach the encrypted translation without putting text in metadata."""
    translated = texts.get("translated_text")
    if translated:
        record.raw_metadata = {**record.raw_metadata, "translated_text": translated}


class Database:
    def __init__(self, path: Path, cipher=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cipher = cipher

    def _encode_payload(self, value) -> str:
        data = (
            value.model_dump(mode="json")
            if hasattr(value, "model_dump")
            else value
        )
        if self.cipher is not None:
            return self.cipher.encrypt_json(data)
        return json.dumps(data, ensure_ascii=False, default=str)

    def _decode_payload(self, payload: str):
        if self.cipher is not None:
            return self.cipher.decrypt_json(payload)
        return json.loads(payload)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA secure_delete=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    workflow_status TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    assigned_reviewer TEXT,
                    consent_granted_at TEXT,
                    consent_withdrawn_at TEXT,
                    deleted_at TEXT,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
                CREATE INDEX IF NOT EXISTS idx_cases_patient ON cases(patient_id);

                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    language TEXT,
                    content_hash TEXT,
                    usable INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT,
                    ciphertext TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sources_case ON sources(case_id);

                CREATE TABLE IF NOT EXISTS facts (
                    fact_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    source_id TEXT,
                    fact_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    fact_status TEXT NOT NULL,
                    assertion TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    ciphertext TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_facts_case ON facts(case_id);

                CREATE TABLE IF NOT EXISTS warnings (
                    warning_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    code TEXT NOT NULL,
                    outcome TEXT,
                    rule_id TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_warnings_case ON warnings(case_id);

                CREATE TABLE IF NOT EXISTS discrepancies (
                    discrepancy_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_disc_case ON discrepancies(case_id);

                CREATE TABLE IF NOT EXISTS summaries (
                    case_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS notes (
                    case_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS processing_runs (
                    run_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    error_code TEXT,
                    error_detail TEXT,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_case ON processing_runs(case_id);

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    case_id TEXT,
                    actor TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_log(case_id);

                CREATE TABLE IF NOT EXISTS reviews (
                    event_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    role TEXT NOT NULL,
                    original_status TEXT,
                    final_status TEXT,
                    reason TEXT,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reviews_case ON reviews(case_id);

                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    mime TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_docs_case ON documents(case_id);

                CREATE TABLE IF NOT EXISTS deletion_jobs (
                    job_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_deletion_case ON deletion_jobs(case_id);

                CREATE TABLE IF NOT EXISTS communications (
                    message_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    ciphertext TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_communications_case ON communications(case_id);

                CREATE TABLE IF NOT EXISTS followups (
                    followup_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    ciphertext TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    UNIQUE(case_id, question_id)
                );
                CREATE INDEX IF NOT EXISTS idx_followups_case ON followups(case_id);

                CREATE TABLE IF NOT EXISTS followup_answers (
                    answer_id TEXT PRIMARY KEY,
                    followup_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    source_id TEXT,
                    answered_by TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    ciphertext TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_answers_case ON followup_answers(case_id);

                CREATE TABLE IF NOT EXISTS escalations (
                    escalation_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    trigger_code TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    delivery TEXT NOT NULL,
                    last_error TEXT,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_escalations_case ON escalations(case_id);
                CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status);

                CREATE TABLE IF NOT EXISTS operations_state (
                    state_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            migrated = self._encrypt_legacy_payloads(conn)
            migrated += self._sanitize_legacy_source_metadata(conn)
        if migrated:
            # UPDATE leaves old plaintext pages in the freelist. Rewrite the
            # database and truncate the WAL so the migrated copies are gone.
            with self.connect() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            with self.connect() as conn:
                conn.execute("VACUUM")
            with self.connect() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _encrypt_legacy_payloads(self, conn) -> int:
        """Upgrade plaintext JSON payloads written by older builds.

        Ciphertext begins with a Fernet prefix rather than ``{``/``[``, so the
        detection is unambiguous. The update is idempotent.
        """
        if self.cipher is None:
            return 0
        migrated = 0
        for table in ("warnings", "discrepancies", "summaries", "notes", "escalations"):
            rows = conn.execute(f"SELECT rowid, payload FROM {table}").fetchall()
            for row in rows:
                raw = row["payload"]
                if not isinstance(raw, str) or not raw.lstrip().startswith(("{", "[")):
                    continue
                try:
                    data = json.loads(raw)
                except ValueError:
                    continue
                conn.execute(
                    f"UPDATE {table} SET payload=? WHERE rowid=?",
                    (self.cipher.encrypt_json(data), row["rowid"]),
                )
                migrated += 1
        return migrated

    def _sanitize_legacy_source_metadata(self, conn) -> int:
        migrated = 0
        rows = conn.execute("SELECT rowid, metadata FROM sources").fetchall()
        for row in rows:
            try:
                data = json.loads(row["metadata"])
            except (TypeError, ValueError):
                continue
            sanitized = _sanitize_metadata(data)
            if sanitized == data:
                continue
            conn.execute(
                "UPDATE sources SET metadata=? WHERE rowid=?",
                (json.dumps(sanitized, ensure_ascii=False), row["rowid"]),
            )
            migrated += 1
        return migrated

    # ----------------------------------------------------------------- cases
    def save_case(self, record: CaseRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO cases (case_id, patient_id, status, workflow_status,
                    received_at, assigned_reviewer, consent_granted_at,
                    consent_withdrawn_at, deleted_at, payload)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(case_id) DO UPDATE SET
                    status=excluded.status,
                    workflow_status=excluded.workflow_status,
                    assigned_reviewer=excluded.assigned_reviewer,
                    consent_granted_at=excluded.consent_granted_at,
                    consent_withdrawn_at=excluded.consent_withdrawn_at,
                    deleted_at=excluded.deleted_at,
                    payload=excluded.payload
                """,
                (
                    record.case_id,
                    record.patient_id,
                    record.status.value,
                    record.workflow_status.value,
                    record.received_at.isoformat(),
                    record.assigned_reviewer,
                    record.consent_granted_at.isoformat()
                    if record.consent_granted_at
                    else None,
                    record.consent_withdrawn_at.isoformat()
                    if record.consent_withdrawn_at
                    else None,
                    record.deleted_at.isoformat() if record.deleted_at else None,
                    _dump(record),
                ),
            )

    def get_case(self, case_id: str, include_deleted: bool = False) -> Optional[CaseRecord]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload, deleted_at FROM cases WHERE case_id=?", (case_id,)
            ).fetchone()
        if not row:
            return None
        if row["deleted_at"] and not include_deleted:
            return None
        return CaseRecord.model_validate_json(row["payload"])

    def update_case(self, case_id: str, **fields) -> CaseRecord:
        record = self.get_case(case_id, include_deleted=True)
        if record is None:
            raise KeyError(case_id)
        for key, value in fields.items():
            setattr(record, key, value)
        self.save_case(record)
        return record

    def list_cases(self, include_deleted: bool = False) -> List[CaseRecord]:
        sql = "SELECT payload, deleted_at FROM cases"
        if not include_deleted:
            sql += " WHERE deleted_at IS NULL"
        sql += " ORDER BY received_at ASC"
        with self.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [CaseRecord.model_validate_json(r["payload"]) for r in rows]

    # --------------------------------------------------------------- sources
    def upsert_source(self, source: SourceRecord, cipher) -> bool:
        """Returns True when inserted, False when an identical duplicate was skipped."""
        ciphertext = cipher.encrypt_json(
            {
                "original_text": source.original_text,
                "normalized_text": source.normalized_text,
                # Translation is source text too, so keep it encrypted rather
                # than in the unencrypted metadata column.
                "translated_text": source.raw_metadata.get("translated_text"),
            }
        )
        metadata = _dump(
            _sanitize_metadata(
                source.model_copy(
                    update={"original_text": "", "normalized_text": ""}
                ).model_dump(mode="json")
            )
        )
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT content_hash FROM sources WHERE source_id=?",
                (source.source_id,),
            ).fetchone()
            if existing and existing["content_hash"] == source.content_hash:
                return False
            conn.execute(
                """
                INSERT INTO sources (source_id, case_id, modality, segment_id,
                    language, content_hash, usable, created_at, ciphertext, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source_id) DO UPDATE SET
                    ciphertext=excluded.ciphertext,
                    metadata=excluded.metadata,
                    content_hash=excluded.content_hash,
                    language=excluded.language,
                    usable=excluded.usable
                """,
                (
                    source.source_id,
                    source.case_id,
                    source.modality.value,
                    source.segment_id,
                    source.language,
                    source.content_hash,
                    1 if source.usable else 0,
                    _now_iso(),
                    ciphertext,
                    metadata,
                ),
            )
        return True

    def get_source(self, source_id: str, cipher) -> Optional[SourceRecord]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT metadata, ciphertext, deleted_at FROM sources WHERE source_id=?",
                (source_id,),
            ).fetchone()
        if not row or row["deleted_at"]:
            return None
        record = SourceRecord.model_validate_json(row["metadata"])
        texts = cipher.decrypt_json(row["ciphertext"])
        record.original_text = texts["original_text"]
        record.normalized_text = texts["normalized_text"]
        _restore_translation(record, texts)
        return record

    def get_sources(self, case_id: str, cipher) -> List[SourceRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT metadata, ciphertext FROM sources WHERE case_id=? AND deleted_at IS NULL ORDER BY created_at ASC",
                (case_id,),
            ).fetchall()
        result = []
        for row in rows:
            record = SourceRecord.model_validate_json(row["metadata"])
            texts = cipher.decrypt_json(row["ciphertext"])
            record.original_text = texts["original_text"]
            record.normalized_text = texts["normalized_text"]
            _restore_translation(record, texts)
            result.append(record)
        return result

    # ----------------------------------------------------------------- facts
    def upsert_fact(self, fact: Fact, cipher) -> bool:
        ciphertext = cipher.encrypt_json(fact.model_dump(mode="json"))
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT ciphertext FROM facts WHERE fact_id=?", (fact.fact_id,)
            ).fetchone()
            if existing:
                old = Fact.model_validate(cipher.decrypt_json(existing["ciphertext"]))
                if old.verification_status.value in {
                    "verified",
                    "corrected",
                    "rejected",
                    "disputed",
                }:
                    return False
                conn.execute(
                    "UPDATE facts SET ciphertext=?, updated_at=?, fact_status=?, assertion=? WHERE fact_id=?",
                    (
                        ciphertext,
                        fact.updated_at.isoformat(),
                        fact.status.value,
                        fact.assertion.value,
                        fact.fact_id,
                    ),
                )
                return True
            conn.execute(
                """
                INSERT INTO facts (fact_id, case_id, source_id, fact_type, label,
                    fact_status, assertion, verification_status, created_at,
                    updated_at, ciphertext)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    fact.fact_id,
                    fact.case_id,
                    fact.source_refs[0].source_id if fact.source_refs else None,
                    fact.fact_type.value,
                    fact.label,
                    fact.status.value,
                    fact.assertion.value,
                    fact.verification_status.value,
                    fact.created_at.isoformat(),
                    fact.updated_at.isoformat(),
                    ciphertext,
                ),
            )
        return True

    def get_facts(self, case_id: str, cipher) -> List[Fact]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT ciphertext FROM facts WHERE case_id=? ORDER BY created_at ASC",
                (case_id,),
            ).fetchall()
        return [
            Fact.model_validate(cipher.decrypt_json(r["ciphertext"])) for r in rows
        ]

    def get_fact(self, case_id: str, fact_id: str, cipher) -> Optional[Fact]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT ciphertext FROM facts WHERE case_id=? AND fact_id=?",
                (case_id, fact_id),
            ).fetchone()
        if not row:
            return None
        return Fact.model_validate(cipher.decrypt_json(row["ciphertext"]))

    def delete_source(self, source_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE sources SET deleted_at=?, ciphertext='' WHERE source_id=?",
                (_now_iso(), source_id),
            )

    # -------------------------------------------------------------- warnings
    def save_warnings(self, case_id: str, warnings: Iterable[Warning]) -> None:
        """Merge warnings without clearing prior flags.

        Reprocessing can add context but must never make an earlier warning
        silently disappear. Acknowledged warnings stay acknowledged; new
        payloads only refresh warnings with the same deterministic id.
        """
        with self.connect() as conn:
            for warning in warnings:
                existing = conn.execute(
                    "SELECT payload FROM warnings WHERE warning_id=?",
                    (warning.warning_id,),
                ).fetchone()
                if existing:
                    previous = Warning.model_validate(
                        self._decode_payload(existing["payload"])
                    )
                    warning.status = previous.status
                conn.execute(
                    """
                    INSERT INTO warnings (warning_id, case_id, kind, code,
                        outcome, rule_id, status, created_at, payload)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(warning_id) DO UPDATE SET
                        kind=excluded.kind,
                        code=excluded.code,
                        outcome=excluded.outcome,
                        rule_id=excluded.rule_id,
                        status=excluded.status,
                        payload=excluded.payload
                    """,
                    (
                        warning.warning_id,
                        case_id,
                        warning.kind.value,
                        warning.code,
                        warning.outcome.value if warning.outcome else None,
                        warning.rule_id,
                        warning.status.value,
                        warning.created_at.isoformat(),
                        self._encode_payload(warning),
                    ),
                )

    def get_warnings(self, case_id: str) -> List[Warning]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM warnings WHERE case_id=? ORDER BY created_at ASC",
                (case_id,),
            ).fetchall()
        return [
            Warning.model_validate(self._decode_payload(r["payload"]))
            for r in rows
        ]

    def acknowledge_warning(self, warning_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM warnings WHERE warning_id=?", (warning_id,)
            ).fetchone()
            if not row:
                return False
            warning = Warning.model_validate(self._decode_payload(row["payload"]))
            warning.status = warning.status.__class__.acknowledged
            conn.execute(
                "UPDATE warnings SET status='acknowledged', payload=? WHERE warning_id=?",
                (self._encode_payload(warning), warning_id),
            )
        return True

    # ---------------------------------------------------------- discrepancies
    def save_discrepancies(self, case_id: str, items: Iterable[Discrepancy]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM discrepancies WHERE case_id=?", (case_id,))
            for item in items:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO discrepancies (discrepancy_id, case_id,
                        category, severity, created_at, payload)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        item.discrepancy_id,
                        case_id,
                        item.category.value,
                        item.severity.value,
                        item.created_at.isoformat(),
                        self._encode_payload(item),
                    ),
                )

    def get_discrepancies(self, case_id: str) -> List[Discrepancy]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM discrepancies WHERE case_id=? ORDER BY created_at ASC",
                (case_id,),
            ).fetchall()
        return [
            Discrepancy.model_validate(self._decode_payload(r["payload"]))
            for r in rows
        ]

    # --------------------------------------------------------------- summary
    def save_summary(self, summary: ClinicalSummary) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO summaries (case_id, updated_at, payload) VALUES (?,?,?)",
                (
                    summary.case_id,
                    summary.generated_at.isoformat(),
                    self._encode_payload(summary),
                ),
            )

    def get_summary(self, case_id: str) -> Optional[ClinicalSummary]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM summaries WHERE case_id=?", (case_id,)
            ).fetchone()
        if not row:
            return None
        return ClinicalSummary.model_validate(self._decode_payload(row["payload"]))

    def save_note(self, case_id: str, note: DraftNote) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO notes (case_id, updated_at, payload) VALUES (?,?,?)",
                (case_id, note.updated_at.isoformat(), self._encode_payload(note)),
            )

    def get_note(self, case_id: str) -> Optional[DraftNote]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM notes WHERE case_id=?", (case_id,)
            ).fetchone()
        if not row:
            return None
        return DraftNote.model_validate(self._decode_payload(row["payload"]))

    # ------------------------------------------------------- processing runs
    def save_run(self, run: ProcessingRun) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO processing_runs (run_id, case_id, stage, status, attempt,
                    error_code, error_detail, started_at, finished_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status, attempt=excluded.attempt,
                    error_code=excluded.error_code, error_detail=excluded.error_detail,
                    started_at=excluded.started_at, finished_at=excluded.finished_at
                """,
                (
                    run.run_id,
                    run.case_id,
                    run.stage,
                    run.status,
                    run.attempt,
                    run.error_code,
                    run.error_detail,
                    run.started_at.isoformat() if run.started_at else None,
                    run.finished_at.isoformat() if run.finished_at else None,
                ),
            )

    def get_runs(self, case_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM processing_runs WHERE case_id=? ORDER BY started_at ASC",
                (case_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_run(self, case_id: str) -> Optional[Dict[str, Any]]:
        runs = self.get_runs(case_id)
        return runs[-1] if runs else None

    # ----------------------------------------------------------------- audit
    def audit(
        self,
        actor: str,
        role: str,
        action: str,
        case_id: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        import uuid

        with self.connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (event_id, case_id, actor, role, action, detail, created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    uuid.uuid4().hex,
                    case_id,
                    actor,
                    role,
                    action,
                    json.dumps(detail or {}, ensure_ascii=False, default=str),
                    _now_iso(),
                ),
            )

    def get_audit(self, case_id: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            if case_id:
                rows = conn.execute(
                    "SELECT * FROM audit_log WHERE case_id=? ORDER BY id DESC LIMIT ?",
                    (case_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- review
    def add_review(self, event: ReviewEvent) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO reviews (event_id, case_id, action, actor, role,
                    original_status, final_status, reason, detail, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event.event_id,
                    event.case_id,
                    event.action.value,
                    event.actor,
                    event.role,
                    event.original_status.value if event.original_status else None,
                    event.final_status.value if event.final_status else None,
                    event.reason,
                    json.dumps(event.detail, ensure_ascii=False, default=str),
                    event.created_at.isoformat(),
                ),
            )

    def get_reviews(self, case_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reviews WHERE case_id=? ORDER BY created_at ASC",
                (case_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- documents
    def add_document(
        self,
        document_id: str,
        case_id: str,
        stored_name: str,
        original_name: str,
        mime: str,
        size: int,
        sha256: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (document_id, case_id, stored_name, original_name,
                    mime, size, sha256, created_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    document_id,
                    case_id,
                    stored_name,
                    original_name,
                    mime,
                    size,
                    sha256,
                    _now_iso(),
                ),
            )

    def get_documents(self, case_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE case_id=? AND deleted_at IS NULL ORDER BY created_at ASC",
                (case_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id=? AND deleted_at IS NULL",
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def find_document_by_hash(self, case_id: str, digest: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE case_id=? AND sha256=? AND deleted_at IS NULL",
                (case_id, digest),
            ).fetchone()
        return dict(row) if row else None

    def mark_document_deleted(self, document_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE documents SET deleted_at=? WHERE document_id=?",
                (_now_iso(), document_id),
            )

    # ---------------------------------------------------------- deletion jobs
    def save_deletion_job(self, job: Dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO deletion_jobs (job_id, case_id, status, steps,
                    created_at, updated_at, last_error)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    job["job_id"],
                    job["case_id"],
                    job["status"],
                    json.dumps(job["steps"], ensure_ascii=False),
                    job["created_at"],
                    job["updated_at"],
                    job.get("last_error"),
                ),
            )

    def get_deletion_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM deletion_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if not row:
            return None
        job = dict(row)
        job["steps"] = json.loads(job["steps"])
        return job

    def list_deletion_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM deletion_jobs WHERE status=? ORDER BY created_at ASC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM deletion_jobs ORDER BY created_at ASC"
                ).fetchall()
        result = []
        for row in rows:
            job = dict(row)
            job["steps"] = json.loads(job["steps"])
            result.append(job)
        return result

    # ------------------------------------------------------- review updates
    def update_fact_review(self, fact: Fact, cipher) -> None:
        ciphertext = cipher.encrypt_json(fact.model_dump(mode="json"))
        with self.connect() as conn:
            conn.execute(
                "UPDATE facts SET ciphertext=?, updated_at=?, verification_status=?, fact_status=? WHERE fact_id=?",
                (
                    ciphertext,
                    fact.updated_at.isoformat(),
                    fact.verification_status.value,
                    fact.status.value,
                    fact.fact_id,
                ),
            )

    # -------------------------------------------------------- communications
    def save_communication(self, message: PatientMessage, cipher) -> None:
        metadata = _dump(message.model_copy(update={"body": ""}))
        ciphertext = cipher.encrypt_json({"body": message.body})
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO communications (message_id, case_id, intent, language,
                    status, created_at, ciphertext, metadata)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(message_id) DO UPDATE SET
                    status=excluded.status,
                    ciphertext=excluded.ciphertext,
                    metadata=excluded.metadata
                """,
                (
                    message.message_id,
                    message.case_id,
                    message.intent.value,
                    message.language,
                    message.status.value,
                    message.created_at.isoformat(),
                    ciphertext,
                    metadata,
                ),
            )

    def get_communication(self, message_id: str, cipher) -> Optional[PatientMessage]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT metadata, ciphertext FROM communications WHERE message_id=?",
                (message_id,),
            ).fetchone()
        if not row:
            return None
        message = PatientMessage.model_validate_json(row["metadata"])
        message.body = cipher.decrypt_json(row["ciphertext"])["body"]
        return message

    def get_communications(self, case_id: str, cipher) -> List[PatientMessage]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT metadata, ciphertext FROM communications WHERE case_id=? ORDER BY created_at ASC",
                (case_id,),
            ).fetchall()
        messages = []
        for row in rows:
            message = PatientMessage.model_validate_json(row["metadata"])
            message.body = cipher.decrypt_json(row["ciphertext"])["body"]
            messages.append(message)
        return messages

    def count_communications(self, statuses: Iterable[str]) -> int:
        values = list(statuses)
        if not values:
            return 0
        placeholders = ",".join("?" for _ in values)
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS total FROM communications WHERE status IN ({placeholders})",
                values,
            ).fetchone()
        return int(row["total"])

    # ------------------------------------------------------------ follow-ups
    def save_followup(self, followup: FollowUpRecord, cipher) -> None:
        metadata = _dump(followup.model_copy(update={"text": ""}))
        ciphertext = cipher.encrypt_json({"text": followup.text})
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO followups (followup_id, case_id, question_id, language,
                    status, created_at, ciphertext, metadata)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(case_id, question_id) DO UPDATE SET
                    status=excluded.status,
                    ciphertext=excluded.ciphertext,
                    metadata=excluded.metadata
                """,
                (
                    followup.followup_id,
                    followup.case_id,
                    followup.question_id,
                    followup.language,
                    followup.status.value,
                    followup.created_at.isoformat(),
                    ciphertext,
                    metadata,
                ),
            )

    def get_followup(self, case_id: str, followup_id: str, cipher) -> Optional[FollowUpRecord]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT metadata, ciphertext FROM followups WHERE case_id=? AND followup_id=?",
                (case_id, followup_id),
            ).fetchone()
        if not row:
            return None
        followup = FollowUpRecord.model_validate_json(row["metadata"])
        followup.text = cipher.decrypt_json(row["ciphertext"])["text"]
        return followup

    def get_followup_by_question(
        self, case_id: str, question_id: str, cipher
    ) -> Optional[FollowUpRecord]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT metadata, ciphertext FROM followups WHERE case_id=? AND question_id=?",
                (case_id, question_id),
            ).fetchone()
        if not row:
            return None
        followup = FollowUpRecord.model_validate_json(row["metadata"])
        followup.text = cipher.decrypt_json(row["ciphertext"])["text"]
        return followup

    def get_followups(self, case_id: str, cipher) -> List[FollowUpRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT metadata, ciphertext FROM followups WHERE case_id=? ORDER BY created_at ASC",
                (case_id,),
            ).fetchall()
        records = []
        for row in rows:
            followup = FollowUpRecord.model_validate_json(row["metadata"])
            followup.text = cipher.decrypt_json(row["ciphertext"])["text"]
            records.append(followup)
        return records

    def count_followups(self, statuses: Iterable[str]) -> int:
        values = list(statuses)
        if not values:
            return 0
        placeholders = ",".join("?" for _ in values)
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS total FROM followups WHERE status IN ({placeholders})",
                values,
            ).fetchone()
        return int(row["total"])

    def save_followup_answer(self, answer: FollowUpAnswer, cipher) -> None:
        ciphertext = cipher.encrypt_json(
            {
                "text": answer.text,
                "language": answer.language,
                "source_id": answer.source_id,
                "answered_by": answer.answered_by,
                "role": answer.role,
                "translated_text": answer.translated_text,
                "document_id": answer.document_id,
                "metadata": answer.metadata,
            }
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO followup_answers (answer_id, followup_id, case_id,
                    source_id, answered_by, role, created_at, ciphertext)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    answer.answer_id,
                    answer.followup_id,
                    answer.case_id,
                    answer.source_id,
                    answer.answered_by,
                    answer.role,
                    answer.created_at.isoformat(),
                    ciphertext,
                ),
            )

    def get_followup_answers(self, case_id: str, cipher) -> List[FollowUpAnswer]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM followup_answers WHERE case_id=? ORDER BY created_at ASC",
                (case_id,),
            ).fetchall()
        answers = []
        for row in rows:
            data = cipher.decrypt_json(row["ciphertext"])
            answers.append(
                FollowUpAnswer(
                    answer_id=row["answer_id"],
                    followup_id=row["followup_id"],
                    case_id=row["case_id"],
                    text=data["text"],
                    language=data.get("language") or "unknown",
                    source_id=data.get("source_id") or row["source_id"],
                    answered_by=data.get("answered_by") or row["answered_by"],
                    role=data.get("role") or row["role"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    translated_text=data.get("translated_text"),
                    document_id=data.get("document_id"),
                    metadata=data.get("metadata") or {},
                )
            )
        return answers

    # ------------------------------------------------------------ escalations
    def save_escalation(self, escalation: EscalationRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO escalations (escalation_id, case_id, trigger_code,
                    severity, status, dedupe_key, created_at, updated_at,
                    attempts, delivery, last_error, payload)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(escalation_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    attempts=excluded.attempts,
                    last_error=excluded.last_error,
                    payload=excluded.payload
                """,
                (
                    escalation.escalation_id,
                    escalation.case_id,
                    escalation.trigger_code,
                    escalation.severity.value,
                    escalation.status.value,
                    escalation.dedupe_key,
                    escalation.created_at.isoformat(),
                    escalation.updated_at.isoformat(),
                    escalation.attempts,
                    escalation.delivery,
                    escalation.last_error,
                    self._encode_payload(escalation),
                ),
            )

    def get_escalation(self, escalation_id: str) -> Optional[EscalationRecord]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM escalations WHERE escalation_id=?",
                (escalation_id,),
            ).fetchone()
        if not row:
            return None
        return EscalationRecord.model_validate(
            self._decode_payload(row["payload"])
        )

    def find_escalation(self, dedupe_key: str) -> Optional[EscalationRecord]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM escalations WHERE dedupe_key=?",
                (dedupe_key,),
            ).fetchone()
        if not row:
            return None
        return EscalationRecord.model_validate(
            self._decode_payload(row["payload"])
        )

    def get_escalations(
        self, case_id: Optional[str] = None, status: Optional[str] = None
    ) -> List[EscalationRecord]:
        clauses, values = [], []
        if case_id:
            clauses.append("case_id=?")
            values.append(case_id)
        if status:
            clauses.append("status=?")
            values.append(status)
        query = "SELECT payload FROM escalations"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC"
        with self.connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [
            EscalationRecord.model_validate(self._decode_payload(row["payload"]))
            for row in rows
        ]

    def count_escalations(self, statuses: Iterable[str]) -> int:
        values = list(statuses)
        if not values:
            return 0
        placeholders = ",".join("?" for _ in values)
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS total FROM escalations WHERE status IN ({placeholders})",
                values,
            ).fetchone()
        return int(row["total"])

    # ------------------------------------------------------ operations state
    def save_operations_state(self, state: StaffingState) -> None:
        updated = state.updated_at or datetime.now(timezone.utc)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO operations_state (state_key, payload, updated_at)
                VALUES ('staffing', ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (_dump(state), updated.isoformat()),
            )

    def get_operations_state(self) -> StaffingState:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM operations_state WHERE state_key='staffing'"
            ).fetchone()
        return StaffingState.model_validate_json(row["payload"]) if row else StaffingState()

    # ---------------------------------------------------- deletion / purge
    def purge_case_sources(self, case_id: str) -> int:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM sources WHERE case_id=?", (case_id,))
            return cursor.rowcount

    def purge_case_facts(self, case_id: str) -> int:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM facts WHERE case_id=?", (case_id,))
            return cursor.rowcount

    def purge_case_derived(self, case_id: str) -> int:
        removed = 0
        with self.connect() as conn:
            for table, key in (
                ("summaries", "case_id"),
                ("notes", "case_id"),
                ("warnings", "case_id"),
                ("discrepancies", "case_id"),
                ("communications", "case_id"),
                ("followups", "case_id"),
                ("followup_answers", "case_id"),
                ("escalations", "case_id"),
            ):
                cursor = conn.execute(f"DELETE FROM {table} WHERE {key}=?", (case_id,))
                removed += cursor.rowcount
        return removed

    def finalize_case_deletion(self, case_id: str) -> None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM cases WHERE case_id=?", (case_id,)
            ).fetchone()
            if not row:
                return
            record = CaseRecord.model_validate_json(row["payload"])
            record.patient_id = "deleted"
            record.status = record.status.__class__.deleted
            record.workflow_status = record.workflow_status.__class__.unassessed
            record.deleted_at = datetime.now(timezone.utc)
            record.assigned_reviewer = None
            record.meta = {
                "deletion": {
                    "performed_at": record.deleted_at.isoformat(),
                    "consent_withdrawn_at": (
                        record.consent_withdrawn_at.isoformat()
                        if record.consent_withdrawn_at
                        else None
                    ),
                }
            }
            conn.execute(
                """
                UPDATE cases SET patient_id=?, status=?, workflow_status=?,
                    assigned_reviewer=NULL, deleted_at=?, payload=?
                WHERE case_id=?
                """,
                (
                    record.patient_id,
                    record.status.value,
                    record.workflow_status.value,
                    record.deleted_at.isoformat(),
                    _dump(record),
                    case_id,
                ),
            )
