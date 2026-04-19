"""
Living-cluster persistence layer.

A "living cluster" is a persistent, per-category cluster that survives across
daily runs. Today's papers are first matched against existing living clusters
(Step B: join); anything left over is clustered by HDBSCAN (Step C: born).
Each cluster carries its own lineage: members, centroid, name history, events.

Storage layout (under `<articles>/_news/.cache/living_clusters/`):

    <category slug>/<uid>.json      # one cluster per file
    registry.json                   # quick index: category -> [uid, ...]

MVP scope: join / born + atomic save. Drift-rename, merge, split, revive,
dormant are stubs for future extension.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

TAU_JOIN = 0.60          # cosine sim >= τ_join → absorb into existing cluster
TAU_CANDIDATE = 0.45     # below this: definitely orphan (reserved; MVP treats
                         # anything < τ_join as orphan)
TAU_REVIVE = 0.65        # dormant cluster revival threshold
TAU_MERGE = 0.85         # two active clusters collapse into one
BORN_MIN_SIZE = 2        # HDBSCAN cluster must have ≥ N today members to "be born"
DELTA_DRIFT = 0.15       # cos distance; exceed → re-name candidate
NAME_GROWTH_RATIO = 1.5  # members grew ×N since last naming → re-name candidate
DORMANCY_DAYS = 30       # no new members for N days → dormant
SPLIT_MIN_SIZE = 3       # a sub-cluster must hold ≥ N members to qualify for split
SPLIT_SILHOUETTE = 0.5   # silhouette score threshold for 2-way split decision


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LivingCluster:
    uid: str
    category: str
    created_at: str
    updated_at: str
    status: str = "active"             # active | dormant | merged_into:<uid> | split
    theme_name: str = ""
    theme_summary: str = ""
    keywords: list[str] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)
    centroid_at_last_name: list[float] = field(default_factory=list)
    members: list[dict] = field(default_factory=list)      # {"fname":..., "added":"YYYY-MM-DD"}
    name_history: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    def to_json(self) -> dict:
        return {
            "uid": self.uid,
            "category": self.category,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "theme_name": self.theme_name,
            "theme_summary": self.theme_summary,
            "keywords": self.keywords,
            "centroid": [round(float(x), 6) for x in self.centroid],
            "centroid_at_last_name": [round(float(x), 6) for x in self.centroid_at_last_name],
            "members": self.members,
            "name_history": self.name_history,
            "events": self.events,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "LivingCluster":
        return cls(
            uid=obj["uid"],
            category=obj["category"],
            created_at=obj.get("created_at", ""),
            updated_at=obj.get("updated_at", ""),
            status=obj.get("status", "active"),
            theme_name=obj.get("theme_name", ""),
            theme_summary=obj.get("theme_summary", ""),
            keywords=list(obj.get("keywords", [])),
            centroid=list(obj.get("centroid", [])),
            centroid_at_last_name=list(obj.get("centroid_at_last_name", [])),
            members=list(obj.get("members", [])),
            name_history=list(obj.get("name_history", [])),
            events=list(obj.get("events", [])),
        )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def slugify_category(cat: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", cat).strip("-").lower()
    return s or "uncategorized"


def category_dir(root: Path, category: str) -> Path:
    return root / slugify_category(category)


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_cluster(root: Path, lc: LivingCluster) -> None:
    path = category_dir(root, lc.category) / f"{lc.uid}.json"
    _atomic_write_json(path, lc.to_json())


def load_all_clusters(root: Path) -> dict[str, list[LivingCluster]]:
    """Return {category -> [LivingCluster, ...]} for every json under root."""
    result: dict[str, list[LivingCluster]] = {}
    if not root.exists():
        return result
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        for p in sorted(cat_dir.glob("*.json")):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                lc = LivingCluster.from_json(obj)
                result.setdefault(lc.category, []).append(lc)
            except Exception:
                continue
    return result


def save_registry_index(root: Path, by_cat: dict[str, list[LivingCluster]]) -> None:
    """Human-readable index, not strictly needed for correctness."""
    idx = {
        cat: [
            {
                "uid": lc.uid,
                "status": lc.status,
                "size": lc.size,
                "theme_name": lc.theme_name,
                "updated_at": lc.updated_at,
            }
            for lc in sorted(lcs, key=lambda l: -l.size)
        ]
        for cat, lcs in by_cat.items()
    }
    _atomic_write_json(root / "registry.json", idx)


# ---------------------------------------------------------------------------
# UID allocation
# ---------------------------------------------------------------------------

def next_uid(category: str, existing: list[LivingCluster]) -> str:
    slug = slugify_category(category)
    # take the largest numeric suffix and bump
    max_n = 0
    pat = re.compile(rf"^{re.escape(slug)}-(\d+)$")
    for lc in existing:
        m = pat.match(lc.uid)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return f"{slug}-{max_n + 1:04d}"


# ---------------------------------------------------------------------------
# Similarity + centroid math
# ---------------------------------------------------------------------------

def cosine_sim(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def best_match(
    embedding,
    candidates: list[LivingCluster],
    include_dormant: bool = False,
) -> tuple[Optional[LivingCluster], float]:
    best: Optional[LivingCluster] = None
    best_sim = -1.0
    for lc in candidates:
        if not lc.centroid:
            continue
        if lc.status == "active":
            pass
        elif include_dormant and lc.status == "dormant":
            pass
        else:
            continue
        s = cosine_sim(embedding, lc.centroid)
        if s > best_sim:
            best_sim = s
            best = lc
    return best, best_sim


def incremental_centroid(old_centroid: list[float], old_n: int, new_emb) -> list[float]:
    import numpy as np
    if not old_centroid or old_n <= 0:
        return [float(x) for x in np.asarray(new_emb, dtype=np.float32).tolist()]
    c = np.asarray(old_centroid, dtype=np.float32)
    e = np.asarray(new_emb, dtype=np.float32)
    merged = (c * old_n + e) / (old_n + 1)
    return [float(x) for x in merged.tolist()]


# ---------------------------------------------------------------------------
# Bootstrap — seed registry from themes_history.jsonl
# ---------------------------------------------------------------------------

def bootstrap_from_themes_history(
    themes_history_path: Path,
    root: Path,
    window_days: int = 14,
    reference_date: Optional[date] = None,
    before_date: Optional[date] = None,
) -> dict[str, list[LivingCluster]]:
    """One-shot migration: take the most recent daily snapshot per
    (category, cluster_id) within window_days and create a LivingCluster for it.
    Safe to run only when `root` is empty — if any clusters exist, we skip.

    `before_date` excludes snapshots from the day currently being generated.
    Otherwise a first run can seed an LC from today's own snapshot and then
    immediately report the same paper as an extension of that LC.
    """
    existing = load_all_clusters(root)
    if any(existing.values()):
        return existing
    if not themes_history_path.exists():
        return {}

    from collections import defaultdict
    # latest snapshot wins per (category, cluster_id, date-based lineage)
    # We can't reconstruct true lineage from daily snapshots, so we just take
    # the latest snapshot and treat each as a seed.
    by_key: dict[tuple, dict] = {}
    for line in themes_history_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        key = (rec.get("category", ""), rec.get("cluster_id"), rec.get("date", ""))
        # use the date as tiebreaker; later date wins
        prev = by_key.get((rec.get("category", ""), rec.get("cluster_id")))
        if prev is None or rec.get("date", "") >= prev.get("date", ""):
            by_key[(rec.get("category", ""), rec.get("cluster_id"))] = rec

    # Filter to recent window.
    ref = reference_date or date.today()
    cutoff = ref.toordinal() - window_days
    by_cat: dict[str, list[LivingCluster]] = defaultdict(list)
    for (cat, _cid), rec in by_key.items():
        try:
            d = date.fromisoformat(rec.get("date", ""))
            if before_date is not None and d >= before_date:
                continue
            if d.toordinal() < cutoff:
                continue
        except Exception:
            continue
        if not rec.get("centroid") or not rec.get("members_today"):
            continue
        existing_lcs = by_cat[cat]
        uid = next_uid(cat, existing_lcs)
        now = datetime.now().isoformat(timespec="seconds")
        added = rec["date"]
        members = [{"fname": f, "added": added} for f in rec["members_today"]]
        lc = LivingCluster(
            uid=uid,
            category=cat,
            created_at=rec["date"] + "T00:00:00",
            updated_at=now,
            status="active",
            theme_name=rec.get("theme_name", ""),
            theme_summary="",
            keywords=[],
            centroid=list(rec["centroid"]),
            centroid_at_last_name=list(rec["centroid"]),
            members=members,
            name_history=[{"at": rec["date"], "name": rec.get("theme_name", "")}],
            events=[{
                "at": rec["date"],
                "type": "bootstrapped",
                "from": "themes_history.jsonl",
                "seed_files": rec["members_today"],
            }],
        )
        by_cat[cat].append(lc)
        save_cluster(root, lc)

    if by_cat:
        save_registry_index(root, dict(by_cat))
    return dict(by_cat)


# ---------------------------------------------------------------------------
# Event/absorb helpers for the daily pipeline
# ---------------------------------------------------------------------------

def absorb(lc: LivingCluster, today_fnames: list[str], today_embeddings, today_iso: str) -> dict:
    """Mutate `lc` to include today's papers. Returns the emitted event dict."""
    import numpy as np
    n_before = lc.size
    n = n_before
    c = np.asarray(lc.centroid, dtype=np.float32) if lc.centroid else None
    for fname, emb in zip(today_fnames, today_embeddings):
        e = np.asarray(emb, dtype=np.float32)
        if c is None or n == 0:
            c = e.copy()
        else:
            c = (c * n + e) / (n + 1)
        n += 1
        lc.members.append({"fname": fname, "added": today_iso})

    shift = 0.0
    if lc.centroid and c is not None:
        shift = 1.0 - cosine_sim(lc.centroid, c.tolist())
    if c is not None:
        lc.centroid = [float(x) for x in c.tolist()]
    lc.updated_at = datetime.now().isoformat(timespec="seconds")
    ev = {
        "at": today_iso,
        "type": "extended",
        "added_files": list(today_fnames),
        "n_before": n_before,
        "n_after": lc.size,
        "centroid_shift_cos": round(shift, 4),
    }
    lc.events.append(ev)
    return ev


def _add_to_centroid(c, e, n):
    import numpy as np
    if c is None:
        return np.asarray(e, dtype=np.float32).copy()
    return (c * n + np.asarray(e, dtype=np.float32)) / (n + 1)


def size_at_last_name(lc: LivingCluster) -> int:
    """How many members the cluster had at the time of the latest rename."""
    # Walk events backward to find the most recent renamed/born event
    target_at = None
    for ev in reversed(lc.events):
        if ev.get("type") in ("born", "renamed"):
            target_at = ev.get("at")
            break
    if target_at is None:
        return lc.size
    # Count members with added <= target_at (ISO date compare works lexicographically)
    n = 0
    for m in lc.members:
        if str(m.get("added", ""))[:10] <= str(target_at)[:10]:
            n += 1
    return max(1, n)


def should_rename(lc: LivingCluster) -> tuple[bool, str, float]:
    """Return (yes_rename, reason, drift_cos). reason ∈ {drift, growth, both, none}."""
    drift = 0.0
    if lc.centroid and lc.centroid_at_last_name:
        drift = 1.0 - cosine_sim(lc.centroid, lc.centroid_at_last_name)
    grew = False
    base = size_at_last_name(lc)
    if base > 0 and lc.size / base >= NAME_GROWTH_RATIO:
        grew = True
    drift_hit = drift >= DELTA_DRIFT
    if drift_hit and grew:
        return True, "drift+growth", drift
    if drift_hit:
        return True, "drift", drift
    if grew:
        return True, "growth", drift
    return False, "none", drift


def apply_rename(
    lc: LivingCluster,
    new_name: str,
    new_summary: str,
    new_keywords: list[str],
    today_iso: str,
    reason: str,
    drift_cos: float,
) -> dict:
    """Mutate `lc` with new naming + record name_history + event. Returns event."""
    old_name = lc.theme_name
    lc.theme_name = new_name
    lc.theme_summary = new_summary
    lc.keywords = list(new_keywords or [])
    lc.centroid_at_last_name = list(lc.centroid)
    lc.name_history.append({
        "at": today_iso,
        "name": new_name,
        "reason": reason,
        "drift_cos": round(drift_cos, 4),
    })
    ev = {
        "at": today_iso,
        "type": "renamed",
        "from": old_name,
        "to": new_name,
        "reason": reason,
        "drift_cos": round(drift_cos, 4),
    }
    lc.events.append(ev)
    lc.updated_at = datetime.now().isoformat(timespec="seconds")
    return ev


def mark_dormant(lc: LivingCluster, today_iso: str) -> dict:
    lc.status = "dormant"
    lc.updated_at = datetime.now().isoformat(timespec="seconds")
    ev = {"at": today_iso, "type": "dormant"}
    lc.events.append(ev)
    return ev


def revive(lc: LivingCluster, today_iso: str) -> dict:
    lc.status = "active"
    lc.updated_at = datetime.now().isoformat(timespec="seconds")
    ev = {"at": today_iso, "type": "revived"}
    lc.events.append(ev)
    return ev


def merge_into(loser: LivingCluster, winner: LivingCluster, today_iso: str) -> tuple[dict, dict]:
    """Collapse `loser` into `winner`: move members, recompute centroid,
    keep loser as a stub with status='merged_into:<winner.uid>'."""
    import numpy as np
    # Recompute centroid as the weighted average
    wa = winner.size or 1
    la = loser.size or 1
    if winner.centroid and loser.centroid:
        wc = np.asarray(winner.centroid, dtype=np.float32)
        lc_ = np.asarray(loser.centroid, dtype=np.float32)
        merged_c = (wc * wa + lc_ * la) / (wa + la)
        winner.centroid = [float(x) for x in merged_c.tolist()]
    # Append members (dedup by fname)
    seen = {m.get("fname") for m in winner.members}
    for m in loser.members:
        if m.get("fname") not in seen:
            winner.members.append(m)
            seen.add(m.get("fname"))
    now = datetime.now().isoformat(timespec="seconds")
    winner.updated_at = now
    ev_w = {
        "at": today_iso, "type": "merged_with",
        "other": loser.uid, "absorbed_n": la,
    }
    winner.events.append(ev_w)
    loser.status = f"merged_into:{winner.uid}"
    loser.updated_at = now
    ev_l = {
        "at": today_iso, "type": "merged_into",
        "into": winner.uid, "at_size": la,
    }
    loser.events.append(ev_l)
    return ev_w, ev_l


def last_activity_date(lc: LivingCluster) -> Optional[date]:
    best: Optional[date] = None
    for ev in lc.events:
        if ev.get("type") in ("born", "extended", "revived", "bootstrapped"):
            try:
                d = date.fromisoformat(str(ev.get("at", ""))[:10])
            except Exception:
                continue
            if best is None or d > best:
                best = d
    return best


def growth_rate_7d(lc: LivingCluster, ref_date: Optional[date] = None) -> float:
    """Fraction of members added in the 7 days ending at `ref_date` (inclusive).

    Returns 0.0 when the cluster has no members. Clamped to [0, 1]. This is
    the "rising theme" signal from organic_cluster.md §3F — high values mean
    a lot of the cluster's mass was picked up very recently.
    """
    ref = ref_date or date.today()
    total = len(lc.members)
    if total <= 0:
        return 0.0
    cutoff = ref.toordinal() - 6  # 7-day window inclusive of ref
    recent = 0
    for m in lc.members:
        try:
            d = date.fromisoformat(str(m.get("added", ""))[:10])
        except Exception:
            continue
        if d.toordinal() >= cutoff and d <= ref:
            recent += 1
    return max(0.0, min(1.0, recent / total))


def split_cluster(
    original: LivingCluster,
    existing: list[LivingCluster],
    assignments: list[int],                # per-member 0/1 labels aligned with original.members
    member_embeddings: list,               # aligned embeddings (np-like vectors)
    today_iso: str,
    silhouette: float,
) -> Optional[LivingCluster]:
    """Cleave `original` into two clusters along `assignments` (0=keep, 1=spawn).

    Recomputes both centroids from the assigned embeddings, mutates `original`
    in place to retain only label-0 members, and returns a newly allocated
    LivingCluster holding label-1 members. Both sides emit a `split` event.

    Returns None when the split would leave either side below SPLIT_MIN_SIZE.
    """
    import numpy as np
    if len(assignments) != len(original.members):
        return None
    keep_idx = [i for i, a in enumerate(assignments) if a == 0]
    spawn_idx = [i for i, a in enumerate(assignments) if a == 1]
    if len(keep_idx) < SPLIT_MIN_SIZE or len(spawn_idx) < SPLIT_MIN_SIZE:
        return None

    keep_members = [original.members[i] for i in keep_idx]
    spawn_members = [original.members[i] for i in spawn_idx]
    keep_embs = np.stack([np.asarray(member_embeddings[i], dtype=np.float32)
                          for i in keep_idx])
    spawn_embs = np.stack([np.asarray(member_embeddings[i], dtype=np.float32)
                           for i in spawn_idx])
    keep_centroid = [float(x) for x in keep_embs.mean(axis=0).tolist()]
    spawn_centroid = [float(x) for x in spawn_embs.mean(axis=0).tolist()]

    now = datetime.now().isoformat(timespec="seconds")
    spawn_uid = next_uid(original.category, existing)
    child = LivingCluster(
        uid=spawn_uid,
        category=original.category,
        created_at=now,
        updated_at=now,
        status="active",
        theme_name=original.theme_name,     # inherit; drift-rename will refresh
        theme_summary=original.theme_summary,
        keywords=list(original.keywords),
        centroid=spawn_centroid,
        centroid_at_last_name=spawn_centroid,
        members=spawn_members,
        name_history=([{"at": today_iso, "name": original.theme_name,
                        "reason": "split_from", "parent": original.uid}]
                      if original.theme_name else []),
        events=[{
            "at": today_iso,
            "type": "split_from",
            "parent": original.uid,
            "size": len(spawn_members),
            "silhouette": round(float(silhouette), 4),
        }],
    )

    original.members = keep_members
    original.centroid = keep_centroid
    # Treat the retained side's naming baseline as "now" — the cluster's
    # character has shifted enough that drift should be measured afresh.
    original.centroid_at_last_name = list(keep_centroid)
    original.updated_at = now
    original.events.append({
        "at": today_iso,
        "type": "split",
        "into": [original.uid, spawn_uid],
        "kept": len(keep_members),
        "spawned": len(spawn_members),
        "silhouette": round(float(silhouette), 4),
    })
    return child


def create_born(
    category: str,
    existing: list[LivingCluster],
    today_fnames: list[str],
    centroid: list[float],
    today_iso: str,
    theme_name: str = "",
    theme_summary: str = "",
    keywords: Optional[list[str]] = None,
) -> LivingCluster:
    now = datetime.now().isoformat(timespec="seconds")
    uid = next_uid(category, existing)
    lc = LivingCluster(
        uid=uid,
        category=category,
        created_at=now,
        updated_at=now,
        status="active",
        theme_name=theme_name,
        theme_summary=theme_summary,
        keywords=list(keywords or []),
        centroid=list(centroid),
        centroid_at_last_name=list(centroid),
        members=[{"fname": f, "added": today_iso} for f in today_fnames],
        name_history=[{"at": today_iso, "name": theme_name}] if theme_name else [],
        events=[{
            "at": today_iso,
            "type": "born",
            "seed_files": list(today_fnames),
            "from": None,
        }],
    )
    return lc
