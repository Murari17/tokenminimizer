"""Parameterized SQLite persistence layer for audit and dispatch logs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.schemas import (
    CleanedPayload,
    DispatchStatus,
    DraftOutput,
    RuleEvaluationResult,
)


class Database:
    """SQLite persistence layer enforcing strictly parameterized queries (`?` placeholders)."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
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

    def record_dispatch(
        self,
        *,
        data: CleanedPayload,
        rules: RuleEvaluationResult,
        status: DispatchStatus,
        draft: Optional[DraftOutput],
        escalate_to_human: bool,
        tokens_used: int,
    ) -> int:
        """Persist pipeline execution trace using strictly parameterized SQL."""
        draft_serialized: Optional[str] = (
            json.dumps(draft.model_dump(mode="json")) if draft is not None else None
        )
        created_at = datetime.now(timezone.utc).isoformat()

        cursor = self._conn.cursor()
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(data.event_id),
                data.sender,
                data.clean_subject,
                status.value,
                rules.priority.value,
                rules.assigned_queue,
                int(rules.sla_hours),
                1 if escalate_to_human else 0,
                int(tokens_used),
                draft_serialized,
                created_at,
            ),
        )
        self._conn.commit()
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("Failed to retrieve SQLite lastrowid after audit insert.")
        return int(row_id)

    def get_audit_log(self, audit_log_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single audit log record by primary key."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM audit_logs WHERE id = ?",
            (int(audit_log_id),),
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def get_logs_by_event_id(self, event_id: UUID | str) -> List[Dict[str, Any]]:
        """Retrieve all audit log records for a given event_id."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM audit_logs WHERE event_id = ? ORDER BY id ASC",
            (str(event_id),),
        )
        return [dict(r) for r in cursor.fetchall()]

    def count_logs(self) -> int:
        """Return total number of rows in `audit_logs`."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) AS cnt FROM audit_logs")
        row = cursor.fetchone()
        return int(row["cnt"]) if row is not None else 0

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

