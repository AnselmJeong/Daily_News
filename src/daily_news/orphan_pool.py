"""
Orphan pool — per-category store of unclustered papers that are re-evaluated
at the start of each weekly run, so a paper that looked lonely one week may
still join a cluster weeks later.

Storage layout:

    <articles>/_news/.cache/cache.db  # primary SQLite store
    <category slug>.jsonl             # legacy import source only

Each record:

    {"fname": "2026 - Doe - Title.pdf",
     "category": "neurobiology of psychiatric disorders",
     "first_seen": "2026-04-05",
     "last_seen":  "2026-04-19",
     "attempts":   3}

The embedding itself is not duplicated here — it lives in
`_news/.cache/embeddings.npz` and is looked up by fname. If an orphan has no
embedding entry it is treated as unusable and dropped silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


ORPHAN_TTL_WEEKS = 8
ORPHAN_POOL_MAX = 500


@dataclass
class OrphanRecord:
    fname: str
    category: str
    first_seen: str          # ISO date when the paper first entered the pool
    last_seen: str           # ISO date of the most recent re-evaluation
    attempts: int = 1        # number of weekly runs this orphan survived

    def to_json(self) -> dict:
        return {
            "fname": self.fname,
            "category": self.category,
            "first_seen": self.first_seen[:10],
            "last_seen": self.last_seen[:10],
            "attempts": int(self.attempts),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "OrphanRecord":
        return cls(
            fname=obj.get("fname", ""),
            category=obj.get("category", ""),
            first_seen=str(obj.get("first_seen", ""))[:10],
            last_seen=str(obj.get("last_seen", ""))[:10],
            attempts=int(obj.get("attempts", 1)),
        )


def _slug(cat: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", cat).strip("-").lower()
    return s or "uncategorized"


def pool_path(root: Path, category: str) -> Path:
    return root / f"{_slug(category)}.jsonl"


def load_pool(root: Path, category: str) -> list[OrphanRecord]:
    from . import cache_store
    rows = cache_store.load_orphans(root.parent, root, category)
    return [OrphanRecord.from_json(row) for row in rows]


def load_all_pools(root: Path) -> dict[str, list[OrphanRecord]]:
    """Return {category -> [OrphanRecord, ...]}. Category key is the human form
    stored on the record (not the slug)."""
    from . import cache_store
    rows_by_cat = cache_store.load_all_orphans(root.parent, root)
    return {
        cat: [OrphanRecord.from_json(row) for row in rows]
        for cat, rows in rows_by_cat.items()
    }


def save_pool(root: Path, category: str, records: list[OrphanRecord]) -> None:
    from . import cache_store
    cache_store.replace_orphans(root.parent, category, [r.to_json() for r in records])


def prune(
    records: list[OrphanRecord],
    today: date,
    ttl_weeks: int = ORPHAN_TTL_WEEKS,
    max_size: int = ORPHAN_POOL_MAX,
) -> list[OrphanRecord]:
    """Drop orphans older than TTL, then cap the list size (keep most-recent
    last_seen). Returns a new list; input is not mutated."""
    cutoff_ord = today.toordinal() - ttl_weeks * 7
    kept: list[OrphanRecord] = []
    for r in records:
        try:
            d = date.fromisoformat(r.last_seen[:10])
        except Exception:
            continue
        if d.toordinal() < cutoff_ord:
            continue
        kept.append(r)
    if len(kept) > max_size:
        kept.sort(key=lambda r: r.last_seen, reverse=True)
        kept = kept[:max_size]
    return kept


def touch(
    records: list[OrphanRecord],
    fnames: Iterable[str],
    today_iso: str,
    category: str,
) -> list[OrphanRecord]:
    """Upsert `fnames` into `records` as of today. Existing entries get their
    ``last_seen`` updated and ``attempts`` incremented when today is a new week.
    Returns a new list; input is not mutated."""
    today10 = today_iso[:10]
    by_fname = {r.fname: OrphanRecord(**r.__dict__) for r in records}
    for fname in fnames:
        if not fname:
            continue
        if fname in by_fname:
            r = by_fname[fname]
            if r.last_seen != today10:
                r.attempts += 1
            r.last_seen = today10
        else:
            by_fname[fname] = OrphanRecord(
                fname=fname,
                category=category,
                first_seen=today10,
                last_seen=today10,
                attempts=1,
            )
    return list(by_fname.values())


def drop(records: list[OrphanRecord], fnames: Iterable[str]) -> list[OrphanRecord]:
    """Remove records whose fname is in `fnames`."""
    drop_set = set(f for f in fnames if f)
    return [r for r in records if r.fname not in drop_set]


def group_by_month(records: list[OrphanRecord]) -> dict[str, list[OrphanRecord]]:
    """Bucket records into `YYYY-MM` keys by ``first_seen``. Used by the
    monthly orphan index page (Phase 4)."""
    out: dict[str, list[OrphanRecord]] = {}
    for r in records:
        key = r.first_seen[:7] or "unknown"
        out.setdefault(key, []).append(r)
    return out
