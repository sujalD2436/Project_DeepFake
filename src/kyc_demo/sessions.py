from __future__ import annotations

import json
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DB_PATH, DEFAULT_TOKEN_TTL_SECONDS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_expired(expires_at_iso: str) -> bool:
    try:
        exp = datetime.fromisoformat(expires_at_iso)
        return datetime.now(timezone.utc) > exp
    except Exception:
        return True


class SessionManager:
    """Manages KYC verification sessions in SQLite with expiring security tokens."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kyc_sessions (
                    session_id TEXT PRIMARY KEY,
                    verification_token TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    clip_score REAL,
                    usable_frames INTEGER,
                    sampled_frames INTEGER,
                    risk_band TEXT,
                    explanation_note TEXT,
                    explanation_png_base64 TEXT,
                    reviewer_verdict TEXT,
                    reviewer_notes TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    verified_at TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_token ON kyc_sessions (verification_token);"
            )

    def create_session(
        self,
        session_id: str | None = None,
        ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
        initial_status: str = "PENDING",
    ) -> dict[str, Any]:
        """Generate a session ID and an unpredictable expiring verification token."""
        sid = session_id or f"kyc_{uuid.uuid4().hex[:12]}"
        token = secrets.token_urlsafe(32)
        created_at = datetime.now(timezone.utc)
        expires_at = datetime.fromtimestamp(
            created_at.timestamp() + ttl_seconds, tz=timezone.utc
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO kyc_sessions (
                    session_id, verification_token, status, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?);
                """,
                (sid, token, initial_status, created_at.isoformat(), expires_at.isoformat()),
            )

        return self.get_session(sid) or {}

    def attach_score(
        self,
        session_id: str,
        score_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Store video scoring result into the session record."""
        clip_score = score_result.get("clip_score")
        status = score_result.get("status", "unknown")
        usable_frames = score_result.get("usable_frames", 0)
        sampled_frames = score_result.get("sampled_frames", 0)
        explanation_note = score_result.get("explanation_note", "")
        explanation_base64 = score_result.get("explanation_png_base64", "")

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE kyc_sessions SET
                    status = ?,
                    clip_score = ?,
                    usable_frames = ?,
                    sampled_frames = ?,
                    risk_band = ?,
                    explanation_note = ?,
                    explanation_png_base64 = ?
                WHERE session_id = ?;
                """,
                (
                    status,
                    clip_score,
                    usable_frames,
                    sampled_frames,
                    status,
                    explanation_note,
                    explanation_base64,
                    session_id,
                ),
            )
        return self.get_session(session_id) or {}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM kyc_sessions WHERE session_id = ?;", (session_id,)
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            data["is_expired"] = _is_expired(data["expires_at"])
            return data

    def get_session_by_token(self, token: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM kyc_sessions WHERE verification_token = ?;", (token,)
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            data["is_expired"] = _is_expired(data["expires_at"])
            return data

    def verify_token(self, token: str) -> tuple[bool, str, dict[str, Any] | None]:
        """Validate token and record reviewer verification timestamp."""
        session = self.get_session_by_token(token)
        if not session:
            return False, "Verification token not found.", None
        if session["is_expired"]:
            return False, "Verification token has expired.", session

        # Mark verified timestamp
        now = _utc_now()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE kyc_sessions SET verified_at = ? WHERE verification_token = ?;",
                (now, token),
            )
        session["verified_at"] = now
        return True, "Session verified successfully.", session

    def record_reviewer_verdict(
        self, session_id: str, verdict: str, notes: str = ""
    ) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE kyc_sessions SET
                    reviewer_verdict = ?,
                    reviewer_notes = ?
                WHERE session_id = ?;
                """,
                (verdict, notes, session_id),
            )
        return self.get_session(session_id)
