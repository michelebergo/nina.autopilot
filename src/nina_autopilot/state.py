"""SQLite session store — durable record of phase + events for the Conductor.

Phase 2 minimum: enough to inspect a session after the fact and resume
basic state if the orchestrator restarts mid-night. Replay-grade detail
comes in Phase 3+.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS session (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    sequence_file TEXT,
    phase TEXT NOT NULL DEFAULT 'BOOT',
    ended_at TEXT,
    end_reason TEXT
);

CREATE TABLE IF NOT EXISTS event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES session(id)
);

CREATE INDEX IF NOT EXISTS idx_event_session ON event(session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- sessions ----

    def start_session(self, sequence_file: Optional[str] = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO session (started_at, sequence_file, phase) VALUES (?, ?, 'BOOT')",
            (_now(), sequence_file),
        )
        self._conn.commit()
        return cur.lastrowid

    def set_phase(self, session_id: int, phase: str) -> None:
        self._conn.execute("UPDATE session SET phase = ? WHERE id = ?", (phase, session_id))
        self._conn.commit()

    def end_session(self, session_id: int, reason: str) -> None:
        self._conn.execute(
            "UPDATE session SET ended_at = ?, end_reason = ? WHERE id = ?",
            (_now(), reason, session_id),
        )
        self._conn.commit()

    def get_session(self, session_id: int) -> Optional[dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def current_session(self) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM session WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # ---- events ----

    def record_event(self, session_id: int, kind: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO event (session_id, timestamp, kind, payload_json) VALUES (?, ?, ?, ?)",
            (session_id, _now(), kind, json.dumps(payload)),
        )
        self._conn.commit()

    def list_events(self, session_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM event WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "timestamp": r["timestamp"],
                "kind": r["kind"],
                "payload": json.loads(r["payload_json"]),
            }
            for r in rows
        ]


def open_store(path: str | Path) -> SessionStore:
    """Open (or create) the session store at `path`."""
    conn = sqlite3.connect(str(path))
    return SessionStore(conn)
