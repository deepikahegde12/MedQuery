"""SQLite persistence for query results (history + GET /api/query/{id})."""

import sqlite3
import threading
from pathlib import Path

from answer_processing.models import QueryResult


class QueryStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS queries (
                       id TEXT PRIMARY KEY,
                       question TEXT NOT NULL,
                       status TEXT NOT NULL,
                       created_at TEXT NOT NULL,
                       payload TEXT NOT NULL
                   )"""
            )
            self._conn.commit()

    def save(self, result: QueryResult) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO queries (id, question, status, created_at, payload) VALUES (?, ?, ?, ?, ?)",
                (result.id, result.question, result.status.value, result.created_at.isoformat(),
                 result.model_dump_json()),
            )
            self._conn.commit()

    def get(self, query_id: str) -> QueryResult | None:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM queries WHERE id = ?", (query_id,)).fetchone()
        return QueryResult.model_validate_json(row[0]) if row else None

    def history(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, question, status, created_at FROM queries ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"id": r[0], "question": r[1], "status": r[2], "created_at": r[3]} for r in rows]

    def clear(self) -> int:
        with self._lock:
            count = self._conn.execute("DELETE FROM queries").rowcount
            self._conn.commit()
        return count

    def close(self) -> None:
        with self._lock:
            self._conn.close()
