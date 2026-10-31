"""Thread-safe parameterized SQLite persistence layer with atomic reservation (`status='PENDING'`),
lease-based stale `PENDING` reclamation, mid-pipeline failure finalization (`status='FAILED'`),
and pre-send reconciliation logging for at-most-once delivery.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.config import DEFAULT_CONFIG
from src.schemas import (
    CleanedPayload,
    DispatchStatus,
    DraftOutput,
    RuleEvaluationResult,
)


class DuplicateEventError(ValueError):
    """Raised when an event_id has already been reserved or processed (replay/race protection)."""


class Database:
    """Thread-safe SQLite persistence layer enforcing parameterized queries and lease-backed reservations."""

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
                    error_detail TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dispatch_reconciliation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_log_id INTEGER NOT NULL,
                    event_id TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    referenced_ticket_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def has_event_id(self, event_id: UUID | str) -> bool:
        """Check whether `event_id` exists in `audit_logs`."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT 1 FROM audit_logs WHERE event_id = ? LIMIT 1",
                (str(event_id),),
            )
            return cursor.fetchone() is not None

    @staticmethod
    def _is_lease_expired(updated_at_iso: str, lease_seconds: float, now_dt: datetime) -> bool:
        try:
            ts = datetime.fromisoformat(updated_at_iso)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return (now_dt - ts).total_seconds() >= lease_seconds
        except Exception:
            return True

    def reserve_event(
        self,
        event_id: UUID | str,
        sender: str,
        clean_subject: str = "",
        lease_seconds: float = DEFAULT_CONFIG.pending_lease_seconds,
        now: Optional[datetime] = None,
    ) -> int:
        """Atomically reserve `event_id` with `status='PENDING'` or reclaim if `FAILED` / lease-expired.

        - If no row exists for `event_id`: inserts `status='PENDING'` and returns `id`.
        - If an existing row has `status == 'FAILED'` (from an earlier unhandled exception before
          dispatch) OR has `status == 'PENDING'` whose lease (`lease_seconds`) has expired (from
          a hard process crash before `send()`): atomically reclaims the row back to `PENDING` with
          a fresh timestamp and returns its `id`.
        - If an existing row is in `DISPATCHING` (crash occurred while `transport.send()` was in
          flight) or a terminal status (`SENT`, `DROPPED_SPAM`, `ESCALATED`, `BLOCKED_VALIDATION`),
          or is an active non-expired `PENDING` reservation: raises `DuplicateEventError` to guarantee
          at-most-once delivery.
        """
        reference_now = now or datetime.now(timezone.utc)
        now_iso = reference_now.isoformat()
        event_id_str = str(event_id)

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
                        error_detail,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, 'PENDING', 'LOW', 'pending', 0, 0, 0, NULL, NULL, ?, ?)
                    """,
                    (event_id_str, sender, clean_subject, now_iso, now_iso),
                )
                self._conn.commit()
                row_id = cursor.lastrowid
                if row_id is None:
                    raise RuntimeError("Failed to retrieve SQLite lastrowid after event reservation.")
                return int(row_id)
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                # Check if existing row is eligible for reclaim (FAILED or expired PENDING lease)
                cursor.execute(
                    "SELECT id, status, updated_at FROM audit_logs WHERE event_id = ? LIMIT 1",
                    (event_id_str,),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    row_status = existing["status"]
                    reclaimable = row_status == "FAILED" or (
                        row_status == "PENDING"
                        and self._is_lease_expired(
                            existing["updated_at"],
                            lease_seconds=lease_seconds,
                            now_dt=reference_now,
                        )
                    )
                    if reclaimable:
                        cursor.execute(
                            """
                            UPDATE audit_logs
                            SET sender = ?,
                                clean_subject = ?,
                                status = 'PENDING',
                                priority = 'LOW',
                                assigned_queue = 'pending',
                                sla_hours = 0,
                                escalate_to_human = 0,
                                tokens_used = 0,
                                draft_json = NULL,
                                error_detail = NULL,
                                updated_at = ?
                            WHERE id = ?
                            """,
                            (sender, clean_subject, now_iso, int(existing["id"])),
                        )
                        self._conn.commit()
                        return int(existing["id"])

                raise DuplicateEventError(
                    f"Replay detected: event_id {event_id_str} is already reserved or processed."
                ) from exc

    def mark_failed(
        self,
        audit_log_id: int,
        error_detail: str,
        tokens_used: int = 0,
    ) -> None:
        """Transition a `PENDING` reservation to `FAILED` (escalated to `dlq_human_recovery`) on unhandled error."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                UPDATE audit_logs
                SET status = 'FAILED',
                    priority = 'HIGH',
                    assigned_queue = 'dlq_human_recovery',
                    sla_hours = 1,
                    escalate_to_human = 1,
                    tokens_used = ?,
                    error_detail = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'PENDING'
                """,
                (int(tokens_used), error_detail[:500], now_iso, int(audit_log_id)),
            )
            self._conn.commit()

    def begin_dispatch_reconciliation(
        self,
        audit_log_id: int,
        event_id: UUID | str,
        recipient: str,
        referenced_ticket_id: str,
    ) -> int:
        """Record an `IN_FLIGHT` reconciliation entry and transition row to `DISPATCHING` prior to `send()`."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "UPDATE audit_logs SET status = 'DISPATCHING', updated_at = ? WHERE id = ?",
                (now_iso, int(audit_log_id)),
            )
            cursor.execute(
                """
                INSERT INTO dispatch_reconciliation (
                    audit_log_id,
                    event_id,
                    recipient,
                    referenced_ticket_id,
                    state,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, 'IN_FLIGHT', ?, ?)
                """,
                (
                    int(audit_log_id),
                    str(event_id),
                    recipient,
                    referenced_ticket_id,
                    now_iso,
                    now_iso,
                ),
            )
            self._conn.commit()
            rec_id = cursor.lastrowid
            if rec_id is None:
                raise RuntimeError("Failed to create dispatch reconciliation record.")
            return int(rec_id)

    def complete_dispatch_reconciliation(
        self,
        reconciliation_id: int,
        state: str = "DELIVERED",
    ) -> None:
        """Mark a pre-send reconciliation entry as `DELIVERED` or `SEND_ERROR`."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "UPDATE dispatch_reconciliation SET state = ?, updated_at = ? WHERE id = ?",
                (state, now_iso, int(reconciliation_id)),
            )
            self._conn.commit()

    def sweep_stale_pending(
        self,
        lease_seconds: float = DEFAULT_CONFIG.pending_lease_seconds,
        now: Optional[datetime] = None,
    ) -> int:
        """Sweep stale `PENDING` or `DISPATCHING` rows older than `lease_seconds` to `ESCALATED`."""
        reference_now = now or datetime.now(timezone.utc)
        now_iso = reference_now.isoformat()
        swept = 0
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT id, status, updated_at FROM audit_logs WHERE status IN ('PENDING', 'DISPATCHING')"
            )
            rows = cursor.fetchall()
            for row in rows:
                if self._is_lease_expired(row["updated_at"], lease_seconds=lease_seconds, now_dt=reference_now):
                    reason = (
                        "Stale DISPATCHING lease swept (at-most-once reconciliation required)"
                        if row["status"] == "DISPATCHING"
                        else "Stale PENDING lease swept to human recovery"
                    )
                    cursor.execute(
                        """
                        UPDATE audit_logs
                        SET status = 'ESCALATED',
                            priority = 'HIGH',
                            assigned_queue = 'dlq_reconciliation',
                            sla_hours = 1,
                            escalate_to_human = 1,
                            error_detail = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (reason, now_iso, int(row["id"])),
                    )
                    swept += 1
            if swept > 0:
                self._conn.commit()
        return swept

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
        """Update a reserved `PENDING`/`DISPATCHING` audit row to its terminal status."""
        draft_serialized: Optional[str] = (
            json.dumps(draft.model_dump(mode="json")) if draft is not None else None
        )
        now_iso = datetime.now(timezone.utc).isoformat()
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
                    draft_json = ?,
                    error_detail = NULL,
                    updated_at = ?
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
                    now_iso,
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

    def get_reconciliation_logs(self, event_id: UUID | str) -> List[Dict[str, Any]]:
        """Retrieve pre-send reconciliation records for a given event_id."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM dispatch_reconciliation WHERE event_id = ? ORDER BY id ASC",
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
