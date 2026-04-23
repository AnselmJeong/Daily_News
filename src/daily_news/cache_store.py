"""SQLite-backed cache store for Daily News.

Embeddings and living clusters intentionally stay in their existing stores:
``embeddings.npz`` and ``living_clusters/*.json``. This module owns the
append/update-heavy cache tables that were previously JSONL-only.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable


DB_NAME = "cache.db"


def db_path(cache_dir: Path) -> Path:
    return cache_dir / DB_NAME


def connect(cache_dir: Path) -> sqlite3.Connection:
    cache_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path(cache_dir))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cache_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS abstracts (
            fname TEXT PRIMARY KEY,
            abstract TEXT NOT NULL DEFAULT '',
            abstract_method TEXT NOT NULL DEFAULT '',
            tldr TEXT NOT NULL DEFAULT '',
            paper_summary TEXT NOT NULL DEFAULT '',
            summary_model TEXT NOT NULL DEFAULT '',
            summary_prompt_version TEXT NOT NULL DEFAULT '',
            extracted_at TEXT,
            updated_at TEXT,
            raw_json TEXT
        );

        CREATE TABLE IF NOT EXISTS orphan_pool (
            category TEXT NOT NULL,
            fname TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (category, fname)
        );
        CREATE INDEX IF NOT EXISTS idx_orphan_pool_category_last_seen
            ON orphan_pool(category, last_seen);

        CREATE TABLE IF NOT EXISTS run_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            date TEXT,
            rec_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_run_log_date ON run_log(date);

        CREATE TABLE IF NOT EXISTS cluster_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            at TEXT,
            kind TEXT,
            category TEXT,
            fname TEXT,
            uid TEXT,
            rec_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cluster_decisions_kind_at
            ON cluster_decisions(kind, at);
        CREATE INDEX IF NOT EXISTS idx_cluster_decisions_category
            ON cluster_decisions(category);

        CREATE TABLE IF NOT EXISTS daily_summaries (
            date TEXT PRIMARY KEY,
            headline TEXT NOT NULL DEFAULT '',
            lede TEXT NOT NULL DEFAULT '',
            n_articles INTEGER,
            n_categories INTEGER,
            n_themes INTEGER,
            n_recent_year INTEGER,
            saved_at TEXT,
            rec_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS theme_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            category TEXT,
            cluster_id INTEGER,
            living_uid TEXT,
            status TEXT,
            rising_score REAL,
            theme_name TEXT,
            rec_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_theme_snapshots_date
            ON theme_snapshots(date);
        """
    )
    conn.commit()


def load_abstracts(cache_dir: Path, _legacy_path: Path) -> dict[str, dict]:
    with connect(cache_dir) as conn:
        rows = conn.execute("SELECT * FROM abstracts").fetchall()
    out: dict[str, dict] = {}
    for row in rows:
        fname = str(row["fname"])
        out[fname] = {
            "fname": fname,
            "abstract": row["abstract"] or "",
            "abstract_method": row["abstract_method"] or "",
            "tldr": row["tldr"] or "",
            "paper_summary": row["paper_summary"] or "",
            "summary_model": row["summary_model"] or "",
            "summary_prompt_version": row["summary_prompt_version"] or "",
            "extracted_at": row["extracted_at"],
            "updated_at": row["updated_at"],
        }
    return out


def upsert_abstract(conn: sqlite3.Connection, obj: dict) -> None:
    fname = str(obj.get("fname", "")).strip()
    if not fname:
        return
    raw = json.dumps(obj, ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO abstracts (
            fname, abstract, abstract_method, tldr, paper_summary,
            summary_model, summary_prompt_version, extracted_at, updated_at,
            raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fname) DO UPDATE SET
            abstract = CASE
                WHEN excluded.abstract != '' THEN excluded.abstract
                ELSE abstracts.abstract
            END,
            abstract_method = CASE
                WHEN excluded.abstract_method != '' THEN excluded.abstract_method
                ELSE abstracts.abstract_method
            END,
            tldr = CASE
                WHEN excluded.tldr != '' THEN excluded.tldr
                ELSE abstracts.tldr
            END,
            paper_summary = CASE
                WHEN excluded.paper_summary != '' THEN excluded.paper_summary
                ELSE abstracts.paper_summary
            END,
            summary_model = CASE
                WHEN excluded.summary_model != '' THEN excluded.summary_model
                ELSE abstracts.summary_model
            END,
            summary_prompt_version = CASE
                WHEN excluded.summary_prompt_version != ''
                    THEN excluded.summary_prompt_version
                ELSE abstracts.summary_prompt_version
            END,
            extracted_at = COALESCE(excluded.extracted_at, abstracts.extracted_at),
            updated_at = COALESCE(excluded.updated_at, abstracts.updated_at),
            raw_json = excluded.raw_json
        """,
        (
            fname,
            str(obj.get("abstract", "") or ""),
            str(obj.get("abstract_method", "") or ""),
            str(obj.get("tldr", "") or ""),
            str(obj.get("paper_summary", "") or ""),
            str(obj.get("summary_model", "") or ""),
            str(obj.get("summary_prompt_version", "") or ""),
            obj.get("extracted_at"),
            obj.get("updated_at"),
            raw,
        ),
    )


def append_abstract(cache_dir: Path, obj: dict) -> None:
    with connect(cache_dir) as conn:
        upsert_abstract(conn, obj)
        conn.commit()


def load_orphans(cache_dir: Path, _root: Path, category: str) -> list[dict]:
    with connect(cache_dir) as conn:
        rows = conn.execute(
            """
            SELECT fname, category, first_seen, last_seen, attempts
            FROM orphan_pool
            WHERE category = ?
            ORDER BY last_seen DESC, first_seen DESC, fname
            """,
            (category,),
        ).fetchall()
    return [dict(row) for row in rows]


def load_all_orphans(cache_dir: Path, _root: Path) -> dict[str, list[dict]]:
    with connect(cache_dir) as conn:
        rows = conn.execute(
            """
            SELECT fname, category, first_seen, last_seen, attempts
            FROM orphan_pool
            ORDER BY category, last_seen DESC, first_seen DESC, fname
            """
        ).fetchall()
    out: dict[str, list[dict]] = {}
    for row in rows:
        rec = dict(row)
        out.setdefault(str(rec["category"]), []).append(rec)
    return out


def upsert_orphan_records(conn: sqlite3.Connection, records: Iterable[dict]) -> None:
    for obj in records:
        fname = str(obj.get("fname", "") or "").strip()
        category = str(obj.get("category", "") or "").strip()
        if not fname or not category:
            continue
        conn.execute(
            """
            INSERT INTO orphan_pool(category, fname, first_seen, last_seen, attempts)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(category, fname) DO UPDATE SET
                first_seen = excluded.first_seen,
                last_seen = excluded.last_seen,
                attempts = excluded.attempts
            """,
            (
                category,
                fname,
                str(obj.get("first_seen", "") or "")[:10],
                str(obj.get("last_seen", "") or "")[:10],
                int(obj.get("attempts", 1) or 1),
            ),
        )


def replace_orphans(cache_dir: Path, category: str, records: Iterable[dict]) -> None:
    with connect(cache_dir) as conn:
        conn.execute("DELETE FROM orphan_pool WHERE category = ?", (category,))
        upsert_orphan_records(conn, records)
        conn.commit()


def append_run_log(cache_dir: Path, rec: dict) -> None:
    raw = json.dumps(rec, ensure_ascii=False)
    with connect(cache_dir) as conn:
        conn.execute(
            "INSERT INTO run_log(ts, date, rec_json) VALUES (?, ?, ?)",
            (rec.get("ts"), rec.get("date"), raw),
        )
        conn.commit()


def append_decision(cache_dir: Path, rec: dict) -> None:
    raw = json.dumps(rec, ensure_ascii=False)
    uid = rec.get("uid") or rec.get("best_uid") or rec.get("a")
    with connect(cache_dir) as conn:
        conn.execute(
            """
            INSERT INTO cluster_decisions(at, kind, category, fname, uid, rec_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                rec.get("at"),
                rec.get("kind"),
                rec.get("category"),
                rec.get("fname"),
                uid,
                raw,
            ),
        )
        conn.commit()


def save_daily_summary(cache_dir: Path, rec: dict) -> None:
    raw = json.dumps(rec, ensure_ascii=False)
    with connect(cache_dir) as conn:
        conn.execute(
            """
            INSERT INTO daily_summaries(
                date, headline, lede, n_articles, n_categories, n_themes,
                n_recent_year, saved_at, rec_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                headline = excluded.headline,
                lede = excluded.lede,
                n_articles = excluded.n_articles,
                n_categories = excluded.n_categories,
                n_themes = excluded.n_themes,
                n_recent_year = excluded.n_recent_year,
                saved_at = excluded.saved_at,
                rec_json = excluded.rec_json
            """,
            (
                rec.get("date"),
                rec.get("headline", ""),
                rec.get("lede", ""),
                rec.get("n_articles"),
                rec.get("n_categories"),
                rec.get("n_themes"),
                rec.get("n_recent_year"),
                rec.get("saved_at"),
                raw,
            ),
        )
        conn.commit()


def load_daily_summaries(cache_dir: Path, _legacy_path: Path) -> dict[str, dict]:
    with connect(cache_dir) as conn:
        rows = conn.execute("SELECT rec_json FROM daily_summaries").fetchall()
    out: dict[str, dict] = {}
    for row in rows:
        try:
            obj = json.loads(row["rec_json"])
        except Exception:
            continue
        if obj.get("date"):
            out[str(obj["date"])] = obj
    return out


def append_theme_snapshot(cache_dir: Path, rec: dict) -> None:
    raw = json.dumps(rec, ensure_ascii=False)
    with connect(cache_dir) as conn:
        conn.execute(
            """
            INSERT INTO theme_snapshots(
                date, category, cluster_id, living_uid, status,
                rising_score, theme_name, rec_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.get("date"),
                rec.get("category"),
                rec.get("cluster_id"),
                rec.get("living_uid"),
                rec.get("status"),
                rec.get("rising_score"),
                rec.get("theme_name"),
                raw,
            ),
        )
        conn.commit()


def load_theme_snapshots(cache_dir: Path, _legacy_path: Path) -> list[dict]:
    with connect(cache_dir) as conn:
        rows = conn.execute(
            "SELECT rec_json FROM theme_snapshots ORDER BY date, id"
        ).fetchall()
    out: list[dict] = []
    for row in rows:
        try:
            obj = json.loads(row["rec_json"])
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def load_theme_snapshots_for_date(cache_dir: Path, _legacy_path: Path, day: str) -> list[dict]:
    with connect(cache_dir) as conn:
        rows = conn.execute(
            "SELECT rec_json FROM theme_snapshots WHERE date = ? ORDER BY id",
            (day,),
        ).fetchall()
    out: list[dict] = []
    for row in rows:
        try:
            obj = json.loads(row["rec_json"])
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out
