"""Thread-safe parameterized SQLite persistence layer with fencing-token (`lease_token`)
compare-and-swap (CAS), lease heartbeats, stale `PENDING` reclamation, and pre-send
reconciliation logging for strict at-most-once delivery.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from src.config import DEFAULT_CONFIG
from src.schemas import (
    CleanedPayload,
    DispatchStatus,
    DraftOutput,
    RuleEvaluationResult,
)


class DuplicateEventError(ValueError):
    """Raised when an event_id has already been reserved or processed (replay/race protection)."""


class StaleLeaseError(DuplicateEventError):
    """Raised when a worker's lease_token has been superseded by a newer attempt or sweeper."""


class LeaseReservation(int):
    """An `int` row ID that also carries the unique fencing `lease_token` for compare-and-swap."""

    lease_token: str

    def __new__(cls, row_id: int, lease_token: str) -> "LeaseReservation":
        instance = super().__new__(cls, row_id)
        instance.lease_token = lease_token
        return instance

    @property
    def audit_log_id(self) -> int:
        return int(self)


def _extract_id_and_token(
    reservation: int | LeaseReservation,
    explicit_token: Optional[str] = None,
) -> tuple[int, Optional[str]]:
    token = explicit_token or getattr(reservation, "lease_token", None)
    return int(reservation), token


class Database:
    """Thread-safe SQLite persistence layer enforcing parameterized queries and fencing-token CAS."""

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
                    lease_token TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 1,
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
                    lease_token TEXT NOT NULL,
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
    ) -> LeaseReservation:
        """Atomically reserve `event_id` with a fresh fencing `lease_token`, or reclaim if expired/FAILED.

        Returns a `LeaseReservation(row_id, lease_token)` (which is an `int` subclass for full
        backward compatibility while carrying `.lease_token` for compare-and-swap verification).
        """
        reference_now = now or datetime.now(timezone.utc)
        now_iso = reference_now.isoformat()
        event_id_str = str(event_id)
        new_token = uuid4().hex

        with self._lock:
            cursor = self._conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO audit_logs (
                        event_id,
                        lease_token,
                        attempt,
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
                    ) VALUES (?, ?, 1, ?, ?, 'PENDING', 'LOW', 'pending', 0, 0, 0, NULL, NULL, ?, ?)
                    """,
                    (event_id_str, new_token, sender, clean_subject, now_iso, now_iso),
                )
                self._conn.commit()
                row_id = cursor.lastrowid
                if row_id is None:
                    raise RuntimeError("Failed to retrieve SQLite lastrowid after event reservation.")
                return LeaseReservation(int(row_id), new_token)
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                cursor.execute(
                    "SELECT id, status, updated_at, attempt FROM audit_logs WHERE event_id = ? LIMIT 1",
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
                        next_attempt = int(existing["attempt"]) + 1
                        cursor.execute(
                            """
                            UPDATE audit_logs
                            SET lease_token = ?,
                                attempt = ?,
                                sender = ?,
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
                            WHERE id = ? AND status = ? AND updated_at = ?
                            """,
                            (
                                new_token,
                                next_attempt,
                                sender,
                                clean_subject,
                                now_iso,
                                int(existing["id"]),
                                row_status,
                                existing["updated_at"],
                            ),
                        )
                        if cursor.rowcount == 1:
                            self._conn.commit()
                            return LeaseReservation(int(existing["id"]), new_token)
                        self._conn.rollback()

                raise DuplicateEventError(
                    f"Replay detected: event_id {event_id_str} is already reserved or processed."
                ) from exc

    def heartbeat_lease(
        self,
        audit_log_id: int | LeaseReservation,
        lease_token: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """Refresh `updated_at` between pipeline stages if and only if `lease_token` still matches."""
        row_id, token = _extract_id_and_token(audit_log_id, lease_token)
        if not token:
            return True
        now_iso = (now or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                UPDATE audit_logs
                SET updated_at = ?
                WHERE id = ? AND status = 'PENDING' AND lease_token = ?
                """,
                (now_iso, row_id, token),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def mark_failed(
        self,
        audit_log_id: int | LeaseReservation,
        error_detail: str,
        tokens_used: int = 0,
        lease_token: Optional[str] = None,
    ) -> bool:
        """Transition a `PENDING` reservation to `FAILED` only if `lease_token` still matches.

        Prevents a slow/stale worker whose lease expired from clobbering a newer worker's state.
        """
        row_id, token = _extract_id_and_token(audit_log_id, lease_token)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            if token is not None:
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
                    WHERE id = ? AND status = 'PENDING' AND lease_token = ?
                    """,
                    (int(tokens_used), error_detail[:500], now_iso, row_id, token),
                )
            else:
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
                    (int(tokens_used), error_detail[:500], now_iso, row_id),
                )
            self._conn.commit()
            return cursor.rowcount == 1

    def begin_dispatch_reconciliation(
        self,
        audit_log_id: int | LeaseReservation,
        event_id: UUID | str,
        recipient: str,
        referenced_ticket_id: str,
        lease_token: Optional[str] = None,
    ) -> int:
        """Compare-and-swap `PENDING` -> `DISPATCHING` using `lease_token` before `transport.send()`.

        If `rowcount == 0` (meaning the lease expired and was reclaimed by another worker or swept),
        raises `StaleLeaseError` so the stale worker aborts BEFORE calling `transport.send()`.
        """
        row_id, token = _extract_id_and_token(audit_log_id, lease_token)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            if token is not None:
                cursor.execute(
                    """
                    UPDATE audit_logs
                    SET status = 'DISPATCHING', updated_at = ?
                    WHERE id = ? AND status = 'PENDING' AND lease_token = ?
                    """,
                    (now_iso, row_id, token),
                )
            else:
                cursor.execute(
                    """
                    UPDATE audit_logs
                    SET status = 'DISPATCHING', updated_at = ?
                    WHERE id = ? AND status = 'PENDING'
                    """,
                    (now_iso, row_id),
                )
            if cursor.rowcount == 0:
                self._conn.rollback()
                raise StaleLeaseError(
                    f"Stale lease token for audit_log_id={row_id}; aborting send to prevent duplicate dispatch."
                )

            cursor.execute(
                """
                INSERT INTO dispatch_reconciliation (
                    audit_log_id,
                    lease_token,
                    event_id,
                    recipient,
                    referenced_ticket_id,
                    state,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, 'IN_FLIGHT', ?, ?)
                """,
                (
                    row_id,
                    token or "",
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
                "SELECT id, status, lease_token, updated_at FROM audit_logs WHERE status IN ('PENDING', 'DISPATCHING')"
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
                            lease_token = ?,
                            priority = 'HIGH',
                            assigned_queue = 'dlq_reconciliation',
                            sla_hours = 1,
                            escalate_to_human = 1,
                            error_detail = ?,
                            updated_at = ?
                        WHERE id = ? AND lease_token = ?
                        """,
                        (uuid4().hex, reason, now_iso, int(row["id"]), row["lease_token"]),
                    )
                    swept += cursor.rowcount
            if swept > 0:
                self._conn.commit()
        return swept

    def finalize_dispatch(
        self,
        audit_log_id: int | LeaseReservation,
        *,
        data: CleanedPayload,
        rules: RuleEvaluationResult,
        status: DispatchStatus,
        draft: Optional[DraftOutput],
        escalate_to_human: bool,
        tokens_used: int,
        lease_token: Optional[str] = None,
    ) -> int:
        """Compare-and-swap a reserved `PENDING`/`DISPATCHING` row to its terminal status using `lease_token`."""
        row_id, token = _extract_id_and_token(audit_log_id, lease_token)
        draft_serialized: Optional[str] = (
            json.dumps(draft.model_dump(mode="json")) if draft is not None else None
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.cursor()
            if token is not None:
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
                    WHERE id = ? AND lease_token = ? AND status IN ('PENDING', 'DISPATCHING')
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
                        row_id,
                        token,
                    ),
                )
            else:
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
                    WHERE id = ? AND status IN ('PENDING', 'DISPATCHING')
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
                        row_id,
                    ),
                )
            if cursor.rowcount == 0:
                self._conn.rollback()
                raise StaleLeaseError(
                    f"Stale lease token when finalizing audit_log_id={row_id}; state was superseded."
                )
            self._conn.commit()
            return row_id

    def record_dispatch(
        self,
        *,
        data: CleanedPayload,
        rules: RuleEvaluationResult,
        status: DispatchStatus,
        draft: Optional[DraftOutput],
        escalate_to_human: bool,
        tokens_used: int,
        reserved_log_id: Optional[int | LeaseReservation] = None,
        lease_token: Optional[str] = None,
    ) -> int:
        """Reserve (if not already reserved) and finalize the pipeline execution trace."""
        reservation = (
            reserved_log_id
            if reserved_log_id is not None
            else self.reserve_event(
                event_id=data.event_id,
                sender=data.sender,
                clean_subject=data.clean_subject,
            )
        )
        return self.finalize_dispatch(
            reservation,
            data=data,
            rules=rules,
            status=status,
            draft=draft,
            escalate_to_human=escalate_to_human,
            tokens_used=tokens_used,
            lease_token=lease_token,
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
