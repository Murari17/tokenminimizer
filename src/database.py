"""Thread-safe parameterized SQLite persistence layer with atomic reservation (`status='PENDING'`)
and `UNIQUE(event_id)` replay/concurrency protection.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.schemas import (
    CleanedPayload,
    DispatchStatus,
    DraftOutput,
    RuleEvaluationResult,
)


class DuplicateEventError(ValueError):
    """Raised when an event_id has already been reserved or processed (replay/race protection)."""


class Database:
    """Thread-safe SQLite persistence layer enforcing parameterized queries and atomic reservations."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("PRAGMA busy_timeout = 5000")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    sender TEXT NOT NULL,
                    clean_subject TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    assigned_queue TEXT NOT NULL,
                    sla_hours INTEGER NOT NULL,
                    escalate_to_human INTEGER NOT NULL,
                    tokens_used INTEGER NOT NULL,
                    draft_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def has_event_id(self, event_id: UUID | str) -> bool:
        """Check whether `event_id` has already been reserved or recorded in `audit_logs`."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT 1 FROM audit_logs WHERE event_id = ? LIMIT 1",
                (str(event_id),),
            )
            return cursor.fetchone() is not None

    def reserve_event(
        self,
        event_id: UUID | str,
        sender: str,
        clean_subject: str = "",
    ) -> int:
        """Atomically insert a `PENDING` row for `event_id` before any SLM call or transport send.

        Guarantees that under concurrent webhook retries, only a single thread can win the
        reservation; all other concurrent threads immediately fail with `DuplicateEventError`
        before dispatching duplicate messages.
        """
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO audit_logs (
                        event_id,
                        sender,
                        clean_subject,
                        status,
                        priority,
                        assigned_queue,
                        sla_hours,
                        escalate_to_human,
                        tokens_used,
                        draft_json,
                        created_at
                    ) VALUES (?, ?, ?, 'PENDING', 'LOW', 'pending', 0, 0, 0, NULL, ?)
                    """,
                    (str(event_id), sender, clean_subject, created_at),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                raise DuplicateEventError(
                    f"Replay detected: event_id {event_id} is already reserved or processed."
                ) from exc

            row_id = cursor.lastrowid
            if row_id is None:
                raise RuntimeError("Failed to retrieve SQLite lastrowid after event reservation.")
            return int(row_id)

    def finalize_dispatch(
        self,
        audit_log_id: int,
        *,
        data: CleanedPayload,
        rules: RuleEvaluationResult,
        status: DispatchStatus,
        draft: Optional[DraftOutput],
        escalate_to_human: bool,
        tokens_used: int,
    ) -> int:
        """Update a previously reserved `PENDING` audit row to its terminal status after dispatch."""
        draft_serialized: Optional[str] = (
            json.dumps(draft.model_dump(mode="json")) if draft is not None else None
        )
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                UPDATE audit_logs
                SET sender = ?,
                    clean_subject = ?,
                    status = ?,
                    priority = ?,
                    assigned_queue = ?,
                    sla_hours = ?,
                    escalate_to_human = ?,
                    tokens_used = ?,
                    draft_json = ?
                WHERE id = ?
                """,
                (
                    data.sender,
                    data.clean_subject,
                    status.value,
                    rules.priority.value,
                    rules.assigned_queue,
                    int(rules.sla_hours),
                    1 if escalate_to_human else 0,
                    int(tokens_used),
                    draft_serialized,
                    int(audit_log_id),
                ),
            )
            self._conn.commit()
            return int(audit_log_id)

    def record_dispatch(
        self,
        *,
        data: CleanedPayload,
        rules: RuleEvaluationResult,
        status: DispatchStatus,
        draft: Optional[DraftOutput],
        escalate_to_human: bool,
        tokens_used: int,
        reserved_log_id: Optional[int] = None,
    ) -> int:
        """Reserve (if not already reserved) and finalize the pipeline execution trace."""
        log_id = (
            reserved_log_id
            if reserved_log_id is not None
            else self.reserve_event(
                event_id=data.event_id,
                sender=data.sender,
                clean_subject=data.clean_subject,
            )
        )
        return self.finalize_dispatch(
            log_id,
            data=data,
            rules=rules,
            status=status,
            draft=draft,
            escalate_to_human=escalate_to_human,
            tokens_used=tokens_used,
        )

    def get_audit_log(self, audit_log_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single audit log record by primary key."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM audit_logs WHERE id = ?",
                (int(audit_log_id),),
            )
            row = cursor.fetchone()
            return dict(row) if row is not None else None

    def get_logs_by_event_id(self, event_id: UUID | str) -> List[Dict[str, Any]]:
        """Retrieve all audit log records for a given event_id."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM audit_logs WHERE event_id = ? ORDER BY id ASC",
                (str(event_id),),
            )
            return [dict(r) for r in cursor.fetchall()]

    def count_logs(self) -> int:
        """Return total number of rows in `audit_logs`."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM audit_logs")
            row = cursor.fetchone()
            return int(row["cnt"]) if row is not None else 0

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()
