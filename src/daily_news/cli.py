#!/usr/bin/env python3
"""
weekly-news  —  Weekly news page generator for the Articles folder.

Pipeline (idempotent, cache-backed):
  1. Harvest   : PDFs added in the target weekly window from index.json.
  2. Extract   : title + abstract (PyMuPDF, heuristic).
  3. Embed     : sentence-transformers (all-MiniLM-L6-v2).
  4. Cluster   : HDBSCAN per category, combining today + last 14 days.
  5. Name      : Ollama (minimax-m2.7:cloud) labels each cluster.
  6. Rising    : Poisson-residual score vs baseline.
  7. Render    : weekly issue HTML + refresh _news/index.html rollup.

Typical invocations:
  weekly-news                              # today, evening run
  weekly-news --date 2026-04-18            # rebuild a specific day
  weekly-news --from 2026-04-01 --to 2026-04-19  # ingest a date range
  weekly-news --since-hours 3              # ingest only pdfs added in last 3h
  weekly-news --no-llm                     # skip theme naming (cluster labels only)
  weekly-news --articles-root /path/to/Articles   # point at a specific root

Articles-root resolution order:
  1. --articles-root CLI flag
  2. $ARTICLES_ROOT env var
  3. current working directory (default)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import html as html_module
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date, time as dtime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths — populated by configure_paths() before run()
# ---------------------------------------------------------------------------

ARTICLES_ROOT: Path = Path()
NEWS_DIR: Path = Path()
CACHE_DIR: Path = Path()
INDEX_JSON: Path = Path()
ABSTRACTS_PATH: Path = Path()
EMBEDDINGS_PATH: Path = Path()
THEMES_HISTORY_PATH: Path = Path()
RUN_LOG_PATH: Path = Path()
DAILY_SUMMARIES_PATH: Path = Path()
LIVING_CLUSTERS_DIR: Path = Path()
DECISIONS_LOG_PATH: Path = Path()
ORPHAN_POOL_DIR: Path = Path()


def configure_paths(articles_root: Path) -> None:
    global ARTICLES_ROOT, NEWS_DIR, CACHE_DIR, INDEX_JSON
    global ABSTRACTS_PATH, EMBEDDINGS_PATH, THEMES_HISTORY_PATH, RUN_LOG_PATH
    global DAILY_SUMMARIES_PATH, LIVING_CLUSTERS_DIR, DECISIONS_LOG_PATH
    global ORPHAN_POOL_DIR
    ARTICLES_ROOT = articles_root.resolve()
    NEWS_DIR = ARTICLES_ROOT / "_news"
    CACHE_DIR = NEWS_DIR / ".cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_JSON = ARTICLES_ROOT / "index.json"
    ABSTRACTS_PATH = CACHE_DIR / "abstracts.jsonl"
    EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npz"
    THEMES_HISTORY_PATH = CACHE_DIR / "themes_history.jsonl"
    RUN_LOG_PATH = CACHE_DIR / "run_log.jsonl"
    DAILY_SUMMARIES_PATH = CACHE_DIR / "daily_summaries.jsonl"
    LIVING_CLUSTERS_DIR = CACHE_DIR / "living_clusters"
    DECISIONS_LOG_PATH = CACHE_DIR / "cluster_decisions.jsonl"
    ORPHAN_POOL_DIR = CACHE_DIR / "orphan_pool"


def resolve_articles_root(cli_arg: Optional[str]) -> Path:
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()
    env = os.environ.get("ARTICLES_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


# tunables
BASELINE_DAYS = 14              # window for clustering context + baseline
MIN_CLUSTER_SIZE = 3
RISING_ACTIVE = 1.0
RISING_HOT = 2.0
MAX_CLUSTERS_PER_CATEGORY = 6
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "minimax-m2.7:cloud")
OLLAMA_TIMEOUT = 90             # seconds per call

log = logging.getLogger("daily_news")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    fname: str
    category: str
    added: datetime
    rel_path: str        # path relative to ARTICLES_ROOT
    pub_year: str = ""
    source: str = "indexed"  # or "root"
    confidence: str = ""
    title: str = ""
    authors: str = ""
    abstract: str = ""
    abstract_method: str = ""  # pymupdf / heuristic / none
    tldr: str = ""             # per-paper key-finding one-liner (LLM)
    embedding: Optional["np.ndarray"] = field(default=None, repr=False)  # type: ignore

    @property
    def text_for_embed(self) -> str:
        if self.abstract:
            return f"{self.title}\n\n{self.abstract}"
        return self.title


@dataclass
class Cluster:
    cluster_id: int
    category: str
    members_today: list[Entry] = field(default_factory=list)
    members_recent: list[Entry] = field(default_factory=list)
    centroid: Optional["np.ndarray"] = field(default=None, repr=False)  # type: ignore
    theme_name: str = ""
    theme_name_fallback: bool = False
    theme_summary: str = ""
    keywords: list[str] = field(default_factory=list)
    keywords_are_fallback: bool = False
    coherent: bool = True
    rising_score: float = 0.0
    status: str = "mentioned"  # rising / active / mentioned / new
    # Living-cluster linkage (MVP: join / born)
    living_uid: Optional[str] = None
    lineage: str = "fresh"      # "extended" (joined existing LC) / "born" / "fresh"
    prior_member_count: int = 0 # members in the LC before today
    added_today_count: Optional[int] = None # actual new LC members added today
    renamed: bool = False       # true if theme_name changed today
    name_before: str = ""       # prior theme_name, if renamed
    revived: bool = False       # true if a dormant LC was revived today
    rename_reason: str = ""     # drift | growth | drift+growth
    growth_rate_7d: float = 0.0 # fraction of LC members added in last 7 days

    @property
    def n_today(self) -> int:
        return len(self.members_today)

    @property
    def n_recent(self) -> int:
        return len(self.members_recent)

    @property
    def n_added_today(self) -> int:
        return self.n_today if self.added_today_count is None else self.added_today_count


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_PUB_YEAR_RE = re.compile(r"^\s*(\d{4})\s*-\s*")
_AUTHOR_TITLE_RE = re.compile(r"^\s*\d{4}\s*-\s*(.+?)\s*-\s*(.+?)$")


def parse_pub_year(fname: str) -> str:
    m = _PUB_YEAR_RE.match(fname)
    return m.group(1) if m else ""


def parse_authors_title(fname: str) -> tuple[str, str]:
    base = fname[:-4] if fname.lower().endswith(".pdf") else fname
    m = _AUTHOR_TITLE_RE.match(base)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", base


def rel_href(rel: str) -> str:
    parts = rel.split("/")
    return "../" + "/".join(urllib.parse.quote(p) for p in parts)


def ensure_logger(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Step 1 — harvest
# ---------------------------------------------------------------------------

def day_bounds(d: date) -> tuple[datetime, datetime]:
    start = datetime.combine(d, dtime.min)
    end = datetime.combine(d + timedelta(days=1), dtime.min)
    return start, end


def harvest(
    target_date: date,
    since_hours: Optional[float],
    from_date: Optional[date] = None,
) -> tuple[list[Entry], list[Entry]]:
    """Return (today_entries, recent_entries_for_context)."""
    if not INDEX_JSON.exists():
        log.error("index.json not found at %s", INDEX_JSON)
        sys.exit(2)
    idx = json.loads(INDEX_JSON.read_text())
    files = idx["files"]

    if since_hours is not None:
        today_start = datetime.now() - timedelta(hours=since_hours)
        today_end = datetime.now()
    elif from_date is not None:
        today_start = datetime.combine(from_date, dtime.min)
        today_end = datetime.combine(target_date + timedelta(days=1), dtime.min)
    else:
        today_start, today_end = day_bounds(target_date)

    baseline_start = today_start - timedelta(days=BASELINE_DAYS)

    today_entries: list[Entry] = []
    recent_entries: list[Entry] = []

    for fname, meta in files.items():
        added_raw = meta.get("added")
        if not added_raw:
            continue
        try:
            added = datetime.fromisoformat(added_raw)
        except ValueError:
            continue
        if added < baseline_start:
            continue
        cat = meta.get("category", "_uncategorized")
        if str(cat).strip().lower() in {"_uncategorized", "uncategorized"}:
            continue
        rel = f"{cat}/{fname}"
        authors, title = parse_authors_title(fname)
        entry = Entry(
            fname=fname,
            category=cat,
            added=added,
            rel_path=rel,
            pub_year=parse_pub_year(fname),
            source="indexed",
            confidence=meta.get("confidence", ""),
            title=title or fname[:-4],
            authors=authors,
        )
        if today_start <= added < today_end:
            today_entries.append(entry)
        else:
            recent_entries.append(entry)

    log.info("harvest: today=%d  recent_context=%d  window=%s..%s",
             len(today_entries), len(recent_entries),
             today_start.isoformat(timespec="minutes"),
             today_end.isoformat(timespec="minutes"))
    return today_entries, recent_entries


# ---------------------------------------------------------------------------
# Step 2 — abstract extraction (PyMuPDF + heuristic)
# ---------------------------------------------------------------------------

_ABSTRACT_RE = re.compile(
    r"(?ims)\babstract\b[\s:—\-\.]*\n?(?P<body>.{120,3500}?)"
    r"(?=\n\s*(?:keywords?|key\s*words?|introduction|background|1[\.\s]+introduction|"
    r"\b1[\.\s]+background|significance|main\s+text|\bmethods?\b|\bresults?\b)\b)",
)
_CLEAN_WS_RE = re.compile(r"\s+")
_BAD_CHARS_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]")


def load_abstracts_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if ABSTRACTS_PATH.exists():
        for line in ABSTRACTS_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cache[obj["fname"]] = obj
            except Exception:
                continue
    return cache


def append_abstract_cache(obj: dict) -> None:
    with ABSTRACTS_PATH.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def persist_tldrs(entries: list["Entry"]) -> None:
    """Append updated tldr records for entries where tldr was newly set.

    The cache is a jsonl where later lines override earlier ones by fname,
    so a plain append is sufficient to update.
    """
    cache = load_abstracts_cache()
    written = 0
    for e in entries:
        if not e.tldr:
            continue
        prev = cache.get(e.fname, {})
        if prev.get("tldr") == e.tldr:
            continue
        append_abstract_cache({
            "fname": e.fname,
            "abstract": e.abstract,
            "abstract_method": e.abstract_method,
            "tldr": e.tldr,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        written += 1
    if written:
        log.info("tldrs: persisted %d updates to abstracts.jsonl", written)


def _clean_extracted(text: str) -> str:
    text = _BAD_CHARS_RE.sub(" ", text)
    text = _CLEAN_WS_RE.sub(" ", text).strip()
    return text


def extract_abstract(pdf_path: Path) -> tuple[str, str]:
    """Return (abstract_text, method)."""
    try:
        import pymupdf  # type: ignore
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError:
            return "", "nomodule"

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        log.debug("pymupdf open failed for %s: %s", pdf_path.name, e)
        return "", "open_error"

    try:
        n_pages = min(3, doc.page_count)
        text_parts = []
        for i in range(n_pages):
            try:
                text_parts.append(doc.load_page(i).get_text("text"))
            except Exception:
                continue
        full = "\n".join(text_parts)
    finally:
        doc.close()

    if not full or len(full) < 100:
        return "", "empty"

    # strategy 1: explicit "Abstract" header
    m = _ABSTRACT_RE.search(full)
    if m:
        abs_text = _clean_extracted(m.group("body"))
        if 120 <= len(abs_text) <= 4000:
            return abs_text, "pymupdf_header"

    # strategy 2: heuristic — take first "paragraph" block 120–2500 chars
    paragraphs = re.split(r"\n\s*\n", full)
    for para in paragraphs:
        p = _clean_extracted(para)
        if 150 <= len(p) <= 2500 and _looks_like_abstract(p):
            return p, "pymupdf_heuristic"

    # fallback: first big block of text
    trimmed = _clean_extracted(full[:4000])
    if len(trimmed) >= 300:
        return trimmed[:1500], "pymupdf_fallback"

    return "", "nomatch"


def _looks_like_abstract(p: str) -> bool:
    # reject headers, author lists, affiliation blocks
    lower = p.lower()
    if lower.startswith(("keywords", "key words", "copyright", "doi", "©")):
        return False
    if "@" in p and len(p) < 400:
        return False   # likely an author/email block
    word_count = len(p.split())
    if word_count < 30 or word_count > 500:
        return False
    # abstracts usually have at least one of these verbs somewhere
    triggers = ("study", "we ", "this ", "our ", "results", "method", "here we",
                "background", "objective", "patients", "findings", "conclusion")
    return any(t in lower for t in triggers)


def extract_abstracts(entries: list[Entry]) -> None:
    """Fill entry.abstract/abstract_method in-place using cache when possible."""
    cache = load_abstracts_cache()
    extracted_now = 0
    for e in entries:
        hit = cache.get(e.fname)
        if hit and hit.get("abstract_method") not in (None, "", "open_error", "empty"):
            e.abstract = hit.get("abstract", "")
            e.abstract_method = hit.get("abstract_method", "")
            e.tldr = hit.get("tldr", "") or ""
            continue
        pdf_path = ARTICLES_ROOT / e.rel_path
        if not pdf_path.exists():
            # try re-resolving via root if moved
            alt = ARTICLES_ROOT / e.fname
            pdf_path = alt if alt.exists() else pdf_path
        if not pdf_path.exists():
            e.abstract = ""
            e.abstract_method = "missing"
            continue
        abstract, method = extract_abstract(pdf_path)
        e.abstract = abstract
        e.abstract_method = method
        extracted_now += 1
        append_abstract_cache({
            "fname": e.fname,
            "abstract": abstract,
            "abstract_method": method,
            "extracted_at": datetime.now().isoformat(timespec="seconds"),
        })
    log.info("abstracts: %d cached, %d extracted now, %d with abstract",
             len(entries) - extracted_now, extracted_now,
             sum(1 for e in entries if e.abstract))


# ---------------------------------------------------------------------------
# Step 3 — embeddings (sentence-transformers)
# ---------------------------------------------------------------------------

def load_embeddings_cache() -> dict[str, "np.ndarray"]:
    import numpy as np
    if not EMBEDDINGS_PATH.exists():
        return {}
    try:
        arc = np.load(EMBEDDINGS_PATH, allow_pickle=False)
        keys = arc["keys"]
        vecs = arc["vecs"]
        return {str(k): vecs[i] for i, k in enumerate(keys)}
    except Exception as e:
        log.warning("embedding cache load failed: %s", e)
        return {}


def save_embeddings_cache(cache: dict[str, "np.ndarray"]) -> None:
    import numpy as np
    if not cache:
        return
    keys = np.array(list(cache.keys()))
    vecs = np.stack(list(cache.values())).astype("float32")
    np.savez_compressed(EMBEDDINGS_PATH, keys=keys, vecs=vecs)


def embed_entries(entries: list[Entry]) -> None:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        log.error("sentence-transformers missing. pip install sentence-transformers")
        sys.exit(3)
    cache = load_embeddings_cache()
    to_embed_idx: list[int] = []
    for i, e in enumerate(entries):
        if e.fname in cache:
            e.embedding = cache[e.fname]
        else:
            to_embed_idx.append(i)
    if to_embed_idx:
        log.info("embedding: loading model %s for %d new texts",
                 EMBEDDING_MODEL, len(to_embed_idx))
        model = SentenceTransformer(EMBEDDING_MODEL)
        texts = [entries[i].text_for_embed for i in to_embed_idx]
        new_vecs = model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)
        for j, idx in enumerate(to_embed_idx):
            entries[idx].embedding = new_vecs[j]
            cache[entries[idx].fname] = new_vecs[j]
        save_embeddings_cache(cache)
    else:
        log.info("embedding: all %d hits from cache", len(entries))


# ---------------------------------------------------------------------------
# Step 4 — clustering per category
# ---------------------------------------------------------------------------

def cluster_category(today: list[Entry], recent: list[Entry]) -> list[Cluster]:
    """Cluster today+recent entries. Every today-paper ends up in some cluster."""
    import numpy as np
    if not today:
        return []
    all_entries = today + recent

    if len(all_entries) < MIN_CLUSTER_SIZE:
        # Too few for HDBSCAN — each today paper becomes its own singleton cluster
        result = []
        for i, e in enumerate(today):
            c = Cluster(cluster_id=i, category=e.category,
                        members_today=[e], members_recent=[])
            if e.embedding is not None:
                c.centroid = e.embedding.copy()
            result.append(c)
        return result

    matrix = np.stack([e.embedding for e in all_entries])
    labels = _hdbscan_or_fallback(matrix)

    clusters_by_id: dict[int, Cluster] = {}
    today_set = {id(e) for e in today}
    noise_today: list[Entry] = []

    for e, lbl in zip(all_entries, labels):
        if lbl == -1:
            if id(e) in today_set:
                noise_today.append(e)
            continue
        c = clusters_by_id.setdefault(
            lbl, Cluster(cluster_id=int(lbl), category=e.category))
        if id(e) in today_set:
            c.members_today.append(e)
        else:
            c.members_recent.append(e)

    out = [c for c in clusters_by_id.values() if c.n_today >= 1]

    # Rescue noise today-papers as singleton clusters
    next_id = max((c.cluster_id for c in out), default=-1) + 1
    for e in noise_today:
        c = Cluster(cluster_id=next_id, category=e.category,
                    members_today=[e], members_recent=[])
        if e.embedding is not None:
            c.centroid = e.embedding.copy()
        out.append(c)
        next_id += 1

    for c in out:
        if c.centroid is None:
            vecs = [e.embedding for e in c.members_today + c.members_recent
                    if e.embedding is not None]
            if vecs:
                c.centroid = np.stack(vecs).mean(axis=0)
    return out


def _hdbscan_or_fallback(matrix) -> list[int]:
    try:
        import hdbscan  # type: ignore
        min_cluster = max(MIN_CLUSTER_SIZE, len(matrix) // 25)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster,
            min_samples=2,
            metric="euclidean",   # vectors already normalized → ≈ cosine
            cluster_selection_method="eom",
        )
        return list(clusterer.fit_predict(matrix))
    except ImportError:
        log.info("hdbscan not installed — falling back to agglomerative")
        return _agglomerative_cluster(matrix)


def _agglomerative_cluster(matrix) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering  # type: ignore
    n = len(matrix)
    if n < MIN_CLUSTER_SIZE:
        return [-1] * n
    n_clusters = max(2, min(MAX_CLUSTERS_PER_CATEGORY, n // 4))
    alg = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average")
    return list(alg.fit_predict(matrix))


# ---------------------------------------------------------------------------
# Step 4b — Living-cluster hybrid: join existing, then HDBSCAN the orphans
# ---------------------------------------------------------------------------

def hybrid_cluster_category(
    today: list[Entry],
    recent: list[Entry],
    living: list,   # list[LivingCluster], untyped to avoid import cycle in header
) -> list[Cluster]:
    """Step B (join) then Step C (born) for one category.

    - Each today entry is matched to the nearest active living cluster.
      If cos-sim ≥ τ_join it is absorbed → we emit an 'extended' Cluster that
      wraps the existing living centroid and carries only today's additions.
    - Remaining today entries go through the stock HDBSCAN with recent context.
      Clusters with ≥ BORN_MIN_SIZE today members become 'born' living clusters.
      Smaller ones are 'fresh' (displayed as normal, not persisted yet).
    """
    import numpy as np
    from . import living_cluster as lcmod

    absorbed_by_uid: dict[str, list[Entry]] = {}
    orphans: list[Entry] = []
    living_by_uid = {lc.uid: lc for lc in living}

    for e in today:
        if e.embedding is None:
            orphans.append(e)
            continue
        # First try active clusters with τ_join
        best, sim = lcmod.best_match(e.embedding, living, include_dormant=False)
        decision = {
            "kind": "join_check",
            "fname": e.fname,
            "category": e.category,
            "best_uid": best.uid if best else None,
            "best_sim": round(sim, 4) if best else None,
            "outcome": None,
        }
        if best is not None and sim >= lcmod.TAU_JOIN:
            absorbed_by_uid.setdefault(best.uid, []).append(e)
            decision["outcome"] = f"joined:{best.uid}"
        else:
            # Try dormant clusters with τ_revive
            best_d, sim_d = lcmod.best_match(e.embedding, living, include_dormant=True)
            if best_d is not None and best_d.status == "dormant" and sim_d >= lcmod.TAU_REVIVE:
                absorbed_by_uid.setdefault(best_d.uid, []).append(e)
                decision["outcome"] = f"revived:{best_d.uid}"
                decision["best_uid"] = best_d.uid
                decision["best_sim"] = round(sim_d, 4)
            else:
                orphans.append(e)
                decision["outcome"] = "orphan"
        if DECISIONS_LOG_PATH and str(DECISIONS_LOG_PATH) not in ("", "."):
            try:
                _append_jsonl(DECISIONS_LOG_PATH, {**decision,
                                                   "at": datetime.now().isoformat(timespec="seconds")})
            except Exception:
                pass

    result: list[Cluster] = []

    # Extended clusters — one Cluster per living cluster that picked up papers today
    next_id = 0
    for uid, entries in absorbed_by_uid.items():
        lc = living_by_uid[uid]
        c = Cluster(
            cluster_id=next_id,
            category=entries[0].category,
            members_today=entries,
            members_recent=[],
        )
        c.centroid = np.asarray(lc.centroid, dtype=np.float32) if lc.centroid else None
        c.theme_name = lc.theme_name
        c.theme_name_fallback = False
        c.theme_summary = lc.theme_summary
        c.keywords = list(lc.keywords)
        c.keywords_are_fallback = False
        c.living_uid = lc.uid
        c.lineage = "extended"
        c.prior_member_count = lc.size
        result.append(c)
        next_id += 1

    # HDBSCAN the orphans. Recent papers already claimed by a living cluster
    # are excluded so old LC members do not steer new orphan births.
    living_member_fnames = {
        m.get("fname")
        for lc in living
        for m in getattr(lc, "members", [])
        if m.get("fname")
    }
    recent_orphans = [e for e in recent if e.fname not in living_member_fnames]
    orphan_clusters = cluster_category(orphans, recent_orphans)
    for c in orphan_clusters:
        c.cluster_id = next_id
        # "born" if ≥ BORN_MIN_SIZE today papers clustered together AND the
        # cluster's centroid doesn't match any existing living cluster
        if c.n_today >= lcmod.BORN_MIN_SIZE and c.centroid is not None:
            best_d, sim_d = lcmod.best_match(c.centroid, living, include_dormant=True)
            if best_d is not None and best_d.status == "dormant" and sim_d >= lcmod.TAU_REVIVE:
                c.living_uid = best_d.uid
                c.lineage = "extended"
                c.prior_member_count = best_d.size
            else:
                best, sim = lcmod.best_match(c.centroid, living)
                if best is not None and sim >= lcmod.TAU_JOIN:
                    # HDBSCAN happened to recreate an existing living cluster —
                    # treat as extended.
                    c.living_uid = best.uid
                    c.lineage = "extended"
                    c.prior_member_count = best.size
                else:
                    c.lineage = "born"
        result.append(c)
        next_id += 1

    return result


def backfill_orphan_pool(
    category: str,
    living: list,        # list[LivingCluster]
    pool: list,          # list[OrphanRecord]
    emb_cache: dict,     # {fname: np.ndarray}
    today_iso: str,
) -> tuple[list, list[str]]:
    """Re-evaluate each orphan in `pool` against the given living clusters.

    For every orphan whose embedding cosine-sim against the best-matching
    active LC is ≥ τ_join, mutate that LC in-place via
    :func:`living_cluster.absorb_backfill` (original `first_seen` date is
    preserved on the new member). Returns ``(touched_lcs, absorbed_fnames)``.

    The caller is responsible for (a) saving the touched LCs and (b) removing
    `absorbed_fnames` from the pool before persisting it.
    """
    from . import living_cluster as lcmod

    if not pool or not living:
        return [], []

    by_lc: dict[str, list[tuple[str, "np.ndarray", str]]] = {}
    absorbed_fnames: list[str] = []

    for rec in pool:
        emb = emb_cache.get(rec.fname)
        if emb is None:
            continue
        best, sim = lcmod.best_match(emb, living, include_dormant=False)
        if best is None or sim < lcmod.TAU_JOIN:
            continue
        by_lc.setdefault(best.uid, []).append((rec.fname, emb, rec.first_seen))
        absorbed_fnames.append(rec.fname)
        if DECISIONS_LOG_PATH and str(DECISIONS_LOG_PATH) not in ("", "."):
            try:
                _append_jsonl(DECISIONS_LOG_PATH, {
                    "kind": "backfill",
                    "fname": rec.fname,
                    "category": category,
                    "best_uid": best.uid,
                    "best_sim": round(sim, 4),
                    "original_date": rec.first_seen,
                    "at": datetime.now().isoformat(timespec="seconds"),
                })
            except Exception:
                pass

    touched: list = []
    living_by_uid = {lc.uid: lc for lc in living}
    for uid, triples in by_lc.items():
        lc = living_by_uid.get(uid)
        if lc is None:
            continue
        lcmod.absorb_backfill(lc, triples, today_iso=today_iso)
        touched.append(lc)

    return touched, absorbed_fnames


def apply_living_updates(
    clusters: list[Cluster],
    living_by_cat: dict[str, list],   # dict[str, list[LivingCluster]]
    today_iso: str,
) -> list:
    """Mutate living clusters in memory: absorb today's papers into extended
    LCs, create new LCs for born clusters. Returns the list of touched LCs
    (not yet saved)."""
    from . import living_cluster as lcmod

    touched: list = []  # LivingCluster

    for c in clusters:
        if c.n_today == 0:
            continue
        cat_list = living_by_cat.setdefault(c.category, [])

        if c.lineage == "extended" and c.living_uid:
            lc = next((l for l in cat_list if l.uid == c.living_uid), None)
            if lc is None:
                continue
            seen = {m.get("fname") for m in lc.members}
            pairs = [(e.fname, e.embedding) for e in c.members_today
                     if e.embedding is not None and e.fname not in seen]
            c.added_today_count = len(pairs)
            if not pairs:
                c.lineage = "fresh"
                continue
            # If previously dormant, revive only when there are genuinely new
            # members to add. A duplicate rebuild should not change LC state.
            if lc.status == "dormant":
                lcmod.revive(lc, today_iso)
                c.revived = True
            fnames = [p[0] for p in pairs]
            embs = [p[1] for p in pairs]
            lcmod.absorb(lc, fnames, embs, today_iso)
            touched.append(lc)

        elif c.lineage == "born":
            fnames = [e.fname for e in c.members_today]
            if c.centroid is None or not fnames:
                continue
            # Skip if a prior run already created an LC with these exact seed files
            already = None
            for lc_existing in cat_list:
                seeds = {m.get("fname") for m in lc_existing.members}
                if set(fnames).issubset(seeds):
                    already = lc_existing
                    break
            if already is not None:
                c.living_uid = already.uid
                c.added_today_count = 0
                c.lineage = "fresh"
                continue
            lc = lcmod.create_born(
                category=c.category,
                existing=cat_list,
                today_fnames=fnames,
                centroid=[float(x) for x in c.centroid.tolist()],
                today_iso=today_iso,
                theme_name=c.theme_name,
                theme_summary=c.theme_summary,
                keywords=c.keywords,
            )
            cat_list.append(lc)
            c.living_uid = lc.uid
            c.added_today_count = len(fnames)
            touched.append(lc)

    return touched


def save_living_clusters(
    touched: list,   # list[LivingCluster]
    living_by_cat: dict,
) -> None:
    from . import living_cluster as lcmod
    for lc in touched:
        lcmod.save_cluster(LIVING_CLUSTERS_DIR, lc)
    if touched:
        lcmod.save_registry_index(LIVING_CLUSTERS_DIR, living_by_cat)


def _living_cluster_prompt_lines(lc, abstracts_cache: dict, max_members: int = 8) -> list[str]:
    """Build bullet lines (title + abstract snippet) for an LC's most recent members."""
    # Most recent first
    ordered = sorted(lc.members, key=lambda m: str(m.get("added", "")), reverse=True)
    lines = []
    for m in ordered[:max_members]:
        fname = m.get("fname", "")
        _, title = parse_authors_title(fname)
        rec = abstracts_cache.get(fname, {})
        abstract = str(rec.get("abstract", "") or "")
        snippet = (abstract[:260] + "…") if len(abstract) > 260 else abstract
        lines.append(f"- {title}\n  {snippet}" if snippet else f"- {title}")
    return lines


def rename_drifted_clusters(
    clusters: list[Cluster],
    living_by_cat: dict,
    use_llm: bool,
    today_iso: str,
    decisions_log_path: Optional[Path] = None,
) -> list[dict]:
    """For each 'extended' cluster whose LC crossed drift or growth thresholds,
    re-query the LLM with the full membership and record a 'renamed' event.
    Returns the list of rename events (for rendering)."""
    from . import living_cluster as lcmod
    abstracts = load_abstracts_cache()
    rename_events: list[dict] = []

    # We also consider any LC that got absorbed today (lineage=extended) and
    # any LC just created (lineage=born) — born clusters were already named
    # during the main naming pass and their centroid_at_last_name equals the
    # current centroid, so should_rename() will return False.
    touched_uids: set[str] = set()
    for c in clusters:
        if c.lineage != "extended" or not c.living_uid or c.n_added_today <= 0:
            continue
        lc = next((l for l in living_by_cat.get(c.category, []) if l.uid == c.living_uid), None)
        if lc is None:
            continue
        if lc.uid in touched_uids:
            continue
        touched_uids.add(lc.uid)
        should, reason, drift = lcmod.should_rename(lc)
        decision = {
            "at": today_iso,
            "kind": "rename_check",
            "uid": lc.uid,
            "category": lc.category,
            "should_rename": should,
            "reason": reason,
            "drift_cos": round(drift, 4),
            "size": lc.size,
            "size_at_last_name": lcmod.size_at_last_name(lc),
        }
        if decisions_log_path is not None:
            _append_jsonl(decisions_log_path, decision)
        if not should:
            continue

        # Build prompt from LC's own membership (not just today's)
        bullets = _living_cluster_prompt_lines(lc, abstracts, max_members=8)
        if not bullets:
            continue
        new_name = ""
        new_summary = ""
        new_kw: list[str] = []
        if use_llm:
            prompt = OLLAMA_CLUSTER_PROMPT.format(papers="\n".join(bullets))
            try:
                resp = _call_ollama(prompt)
                obj = _parse_json_loose(resp)
                new_name = str(obj.get("theme", "")).strip()[:120]
                new_summary = str(obj.get("summary", "")).strip()[:600]
                kw = obj.get("keywords", [])
                if isinstance(kw, list):
                    new_kw = [str(k).strip() for k in kw if str(k).strip()][:5]
            except Exception as ex:
                log.warning("rename llm failed for %s: %s", lc.uid, ex)
        if not new_name:
            # keep the old name rather than risk a bad rename
            continue
        if new_name == lc.theme_name:
            # LLM confirmed the same name — still update centroid_at_last_name
            # so we don't keep re-asking on every run
            lc.centroid_at_last_name = list(lc.centroid)
            continue
        ev = lcmod.apply_rename(
            lc, new_name, new_summary, new_kw,
            today_iso=today_iso, reason=reason, drift_cos=drift,
        )
        rename_events.append({
            "uid": lc.uid, "category": lc.category,
            "from": ev.get("from", ""), "to": ev.get("to", ""),
            "reason": reason, "drift_cos": drift,
        })
        # Propagate the new name into the Cluster object for today's render
        c.name_before = str(ev.get("from", ""))
        c.theme_name = lc.theme_name
        c.theme_name_fallback = False
        c.theme_summary = lc.theme_summary
        c.keywords = list(lc.keywords)
        c.keywords_are_fallback = False
        c.renamed = True
        c.rename_reason = reason
        log.info("renamed %s: %r → %r (reason=%s drift=%.3f)",
                 lc.uid, ev.get("from", ""), ev.get("to", ""), reason, drift)

    return rename_events


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Step 5 — theme naming via Ollama
# ---------------------------------------------------------------------------

OLLAMA_CLUSTER_PROMPT = """\
아래 [0], [1], ... 번호의 논문들은 의미적으로 유사한 클러스터를 이룹니다.
신경과학/정신의학 전문가 관점에서 다음을 작성하세요.

- theme: 이 클러스터를 관통하는 공통 주제명 (3-8 단어, 한국어).
- overview: 이 주제 **자체**가 무엇인지를 설명하는 개념적 개요 2문장
  (한국어). 개별 논문의 결과를 서술하지 말고, 분야·문제·접근의 맥락을
  서술하세요. "이 연구는", "본 논문은", "논문들은" 같은 표현은 금지.
- keywords: 영어 키워드 3-5개.
- coherent: 묶임이 실제로 일관된 주제인가?
- tldrs: 각 논문이 주장 또는 발견한 **핵심 한 가지**를 한 문장으로
  (한국어). 방법 나열 금지, 저널 머리말·홈페이지 정보 포함 금지.

논문:
{papers}

마크다운 없이 JSON으로만 응답:
{{
  "theme": "...",
  "overview": "...",
  "keywords": ["kw1","kw2","kw3"],
  "coherent": true,
  "tldrs": [
    {{"idx": 0, "tldr": "..."}},
    {{"idx": 1, "tldr": "..."}}
  ]
}}
"""

OLLAMA_SINGLETON_PROMPT = """\
아래 논문에 대해 신경과학/정신의학 전문가 관점에서 다음을 작성하세요.

- theme: 논문이 다루는 핵심 주제 (3-6 단어, 한국어).
- tldr: 논문이 주장하거나 발견한 핵심 한 가지를 한 문장으로 (한국어).
  방법 나열·저널 머리말·홈페이지 URL 포함 금지.

제목: {title}
초록: {abstract}

마크다운 없이 JSON으로만 응답:
{{
  "theme": "...",
  "tldr": "..."
}}
"""


def name_cluster_llm(cluster: Cluster, use_llm: bool) -> None:
    # Extended clusters inherit name/summary from their living cluster.
    # We only (re)name born/fresh clusters. Drift-based rename runs later.
    if cluster.lineage == "extended":
        # Still ensure per-paper tldrs are present for today's members.
        _fill_missing_tldrs(cluster.members_today, use_llm)
        return
    if not use_llm:
        _fallback_theme_name(cluster)
        _fill_missing_tldrs(cluster.members_today, use_llm=False)
        return
    if cluster.n_today == 1 and not cluster.members_recent:
        _name_singleton_ollama(cluster)
    else:
        _name_multi_ollama(cluster)


def _name_singleton_ollama(cluster: Cluster) -> None:
    e = cluster.members_today[0]
    snippet = (e.abstract[:500] + "…") if len(e.abstract) > 500 else e.abstract
    prompt = OLLAMA_SINGLETON_PROMPT.format(
        title=e.title,
        abstract=snippet or "(초록 없음)",
    )
    try:
        resp = _call_ollama(prompt)
        obj = _parse_json_loose(resp)
        cluster.theme_name = str(obj.get("theme", "")).strip()[:120]
        cluster.theme_name_fallback = False
        tldr = str(obj.get("tldr", "")).strip()[:400]
        if tldr:
            e.tldr = tldr
        # For a 1-paper cluster, the "cluster summary" is not a concept
        # overview (there's no common theme to abstract over), so we leave
        # cluster.theme_summary empty — the renderer skips showing it.
        cluster.theme_summary = ""
        cluster.coherent = True
        if not cluster.theme_name:
            _fallback_theme_name(cluster)
    except Exception as ex:
        log.warning("ollama singleton failed for %s: %s", e.fname, ex)
        _fallback_theme_name(cluster)


def _name_multi_ollama(cluster: Cluster) -> None:
    members = (cluster.members_today + cluster.members_recent)[:8]
    bullets = []
    for i, e in enumerate(members):
        snippet = (e.abstract[:260] + "…") if len(e.abstract) > 260 else e.abstract
        if snippet:
            bullets.append(f"[{i}] {e.title}\n  {snippet}")
        else:
            bullets.append(f"[{i}] {e.title}")
    prompt = OLLAMA_CLUSTER_PROMPT.format(papers="\n".join(bullets))
    try:
        resp = _call_ollama(prompt)
        obj = _parse_json_loose(resp)
        cluster.theme_name = str(obj.get("theme", "")).strip()[:120]
        cluster.theme_name_fallback = False
        # Accept both "overview" (new) and "summary" (legacy) keys.
        overview = (obj.get("overview") or obj.get("summary") or "")
        overview = str(overview).strip()[:600]
        # Guard: reject overviews that read like a paper TLDR.
        if overview and re.match(r"^\s*(이 연구는|본 연구는|본 논문은|이 논문은|논문들은)", overview):
            overview = re.sub(
                r"^\s*(이 연구는|본 연구는|본 논문은|이 논문은|논문들은)\s*",
                "", overview,
            )
        cluster.theme_summary = overview
        kw = obj.get("keywords", [])
        if isinstance(kw, list):
            cluster.keywords = [str(k).strip() for k in kw if str(k).strip()][:5]
            cluster.keywords_are_fallback = False
        cluster.coherent = bool(obj.get("coherent", True))
        # Per-paper tldrs
        tldrs = obj.get("tldrs") or []
        if isinstance(tldrs, list):
            for item in tldrs:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("idx", -1))
                except (TypeError, ValueError):
                    continue
                tldr = str(item.get("tldr", "")).strip()[:400]
                if 0 <= idx < len(members) and tldr:
                    members[idx].tldr = tldr
        if not cluster.theme_name:
            _fallback_theme_name(cluster)
        # Any members still missing a tldr → fill individually
        _fill_missing_tldrs(cluster.members_today, use_llm=True)
    except Exception as ex:
        log.warning("ollama cluster failed for %s/%d: %s",
                    cluster.category, cluster.cluster_id, ex)
        _fallback_theme_name(cluster)
        _fill_missing_tldrs(cluster.members_today, use_llm=True)


def _fill_missing_tldrs(entries: list["Entry"], use_llm: bool) -> None:
    """For entries without a tldr, either call Ollama singleton-style or
    fall back to a cleaned abstract snippet."""
    for e in entries:
        if e.tldr:
            continue
        if not use_llm:
            e.tldr = _clean_abstract_snippet(e.abstract, max_chars=220)
            continue
        snippet = (e.abstract[:500] + "…") if len(e.abstract) > 500 else e.abstract
        prompt = OLLAMA_SINGLETON_PROMPT.format(
            title=e.title,
            abstract=snippet or "(초록 없음)",
        )
        try:
            resp = _call_ollama(prompt)
            obj = _parse_json_loose(resp)
            tldr = str(obj.get("tldr", "")).strip()[:400]
            if tldr:
                e.tldr = tldr
            else:
                e.tldr = _clean_abstract_snippet(e.abstract, max_chars=220)
        except Exception as ex:
            log.warning("tldr fill failed for %s: %s", e.fname, ex)
            e.tldr = _clean_abstract_snippet(e.abstract, max_chars=220)


def _fallback_theme_name(cluster: Cluster) -> None:
    # Offline/debug fallback only. Do not present title words as real keywords.
    members = cluster.members_today + cluster.members_recent
    if len(members) == 1:
        cluster.theme_name = members[0].title
    else:
        cluster.theme_name = cluster.category
    cluster.theme_name_fallback = True
    cluster.keywords = []
    cluster.keywords_are_fallback = True
    cluster.theme_summary = ""
    cluster.coherent = True


_STOP = {
    "with", "from", "that", "this", "using", "based", "study", "analysis", "review",
    "effects", "effect", "role", "after", "during", "through", "systematic",
    "associated", "between", "novel", "human", "humans", "patients", "evidence",
    "across", "within", "without", "among", "against", "about", "their", "into",
    "brain", "neural", "neurons", "neuron", "data", "model", "models",
}


OLLAMA_THROUGHLINE_PROMPT = """\
오늘 수집된 논문들의 주요 테마를 검토하고, 여러 카테고리를 관통하는 핵심 연구 흐름을 찾아 편집자 시각에서 헤드라인과 리드를 한국어로 작성하세요.

오늘의 주요 테마:
{themes}

전체 논문 수: {total}편

마크다운 없이 JSON으로만 응답:
{{
  "headline": "임팩트 있는 헤드라인 (10-20자, 한국어, 핵심 연구 흐름을 압축)",
  "lede": "2-3문장 리드 (한국어, 구체적 저자명·발견을 언급하며 오늘의 연구 흐름 전달)"
}}
"""


def generate_throughline(all_clusters: list["Cluster"], total: int, use_llm: bool) -> tuple[str, str]:
    fallback_h = "오늘의 논문"
    fallback_l = f"총 {total}편의 논문이 오늘 도착했습니다."
    if not use_llm:
        return fallback_h, fallback_l
    theme_lines = []
    for c in sorted(all_clusters, key=lambda x: -(x.n_today + x.n_recent))[:8]:
        if c.theme_name:
            snippet = c.theme_summary[:100] if c.theme_summary else ""
            theme_lines.append(f"- {c.theme_name}: {snippet} ({c.n_today}편)")
    if not theme_lines:
        return fallback_h, fallback_l
    prompt = OLLAMA_THROUGHLINE_PROMPT.format(
        themes="\n".join(theme_lines),
        total=total,
    )
    try:
        resp = _call_ollama(prompt)
        obj = _parse_json_loose(resp)
        headline = str(obj.get("headline", "")).strip() or fallback_h
        lede = str(obj.get("lede", "")).strip() or fallback_l
        return headline, lede
    except Exception as ex:
        log.warning("throughline generation failed: %s", ex)
        return fallback_h, fallback_l


def _call_ollama(prompt: str) -> str:
    import json as _json
    body = _json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        payload = resp.read().decode()
    obj = _json.loads(payload)
    return obj.get("response", "")


def _parse_json_loose(s: str) -> dict:
    s = s.strip()
    # strip markdown fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).rstrip("`").strip()
    # find first { ... last }
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]
    return json.loads(s)


# ---------------------------------------------------------------------------
# Step 6 — rising detection
# ---------------------------------------------------------------------------

def score_rising(clusters: list[Cluster]) -> None:
    import math
    for c in clusters:
        baseline_per_day = c.n_recent / max(1, BASELINE_DAYS)
        residual = (c.n_today - baseline_per_day) / math.sqrt(baseline_per_day + 1.0)
        c.rising_score = round(residual, 2)
        if c.n_recent == 0 and c.n_today >= 2:
            c.status = "new"
        elif residual >= RISING_HOT:
            c.status = "rising"
        elif residual >= RISING_ACTIVE:
            c.status = "active"
        else:
            c.status = "mentioned"


def save_theme_snapshot(target_date: date, clusters: list[Cluster]) -> None:
    with THEMES_HISTORY_PATH.open("a") as f:
        for c in clusters:
            if c.centroid is None:
                continue
            rec = {
                "date": target_date.isoformat(),
                "category": c.category,
                "cluster_id": c.cluster_id,
                "living_uid": c.living_uid,
                "lineage": c.lineage,
                "theme_name": c.theme_name,
                "theme_name_fallback": c.theme_name_fallback,
                "n_today": c.n_today,
                "n_added_today": c.n_added_today,
                "n_recent": c.n_recent,
                "prior_member_count": c.prior_member_count,
                "rising_score": c.rising_score,
                "status": c.status,
                "renamed": c.renamed,
                "keywords_are_fallback": c.keywords_are_fallback,
                "name_before": c.name_before,
                "rename_reason": c.rename_reason,
                "revived": c.revived,
                "growth_rate_7d": round(float(c.growth_rate_7d), 4),
                "centroid": [round(float(x), 5) for x in c.centroid.tolist()],
                "members_today": [e.fname for e in c.members_today],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Step 7 — render daily HTML + rollup index
# ---------------------------------------------------------------------------

CSS = """
:root{--paper:#f4efe6;--paper-2:#ebe5d8;--ink:#1a1613;--ink-2:#3a332d;
--mute:#7a7065;--rule:#d9d1c1;--card:#fbf7ee;
--accent:#9c2a1a;--accent-2:#2d4a2b;--accent-3:#b5821f;--accent-4:#3c5f7a;
--cols:2}
*{box-sizing:border-box}html,body{margin:0;padding:0}
body{background:var(--paper);color:var(--ink);
font-family:"IBM Plex Sans KR","Newsreader",-apple-system,BlinkMacSystemFont,sans-serif;
font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased;
background-image:radial-gradient(rgba(0,0,0,0.018) 1px,transparent 1px);
background-size:3px 3px}
a{color:inherit}
.wrap{max-width:1280px;margin:0 auto;padding:36px 40px 80px}
.mast{border-top:3px solid var(--ink);border-bottom:1px solid var(--ink);
padding:14px 0 10px;display:flex;align-items:baseline;justify-content:space-between;
gap:20px;flex-wrap:wrap}
.mast-left{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.mast-title{font-family:"Newsreader",serif;font-weight:700;font-size:42px;
line-height:1;letter-spacing:-0.02em;font-style:italic}
.mast-sub{font-family:"JetBrains Mono",monospace;font-size:11px;
letter-spacing:0.12em;text-transform:uppercase;color:var(--mute)}
.mast-right{font-family:"JetBrains Mono",monospace;font-size:11px;
letter-spacing:0.08em;color:var(--mute);text-transform:uppercase;text-align:right}
.mast-right b{color:var(--ink);font-weight:500}
.strap{display:flex;justify-content:space-between;align-items:center;
padding:8px 0 24px;border-bottom:1px solid var(--rule);margin-bottom:28px;
font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.15em;text-transform:uppercase;color:var(--mute)}
.strap .dot{display:inline-block;width:6px;height:6px;border-radius:50%;
background:var(--accent);margin-right:6px;vertical-align:middle;animation:pulse 2.4s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.hero{display:grid;grid-template-columns:1.4fr 1fr;gap:0;
border-top:1px solid var(--ink);border-bottom:1px solid var(--ink);margin-bottom:40px}
.hero-main{padding:28px 36px 32px 0;border-right:1px solid var(--rule)}
.hero-side{padding:28px 0 32px 36px}
.kicker{font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.18em;text-transform:uppercase;color:var(--accent);
font-weight:500;margin-bottom:14px;display:flex;align-items:center;gap:10px}
.kicker::after{content:"";flex:1;height:1px;background:var(--accent);opacity:.3}
.hero-h{font-family:"Newsreader",serif;font-weight:500;font-size:52px;
line-height:1.02;letter-spacing:-0.02em;margin:0 0 16px;color:var(--ink)}
.hero-h em{font-style:italic;color:var(--accent)}
.hero-lede{font-family:"Newsreader",serif;font-size:18px;line-height:1.5;
color:var(--ink-2);margin:0 0 22px;max-width:46ch;text-wrap:pretty}
.hero-stats{display:flex;gap:28px;padding-top:18px;border-top:1px solid var(--rule)}
.stat{display:flex;flex-direction:column;gap:2px}
.stat b{font-family:"Newsreader",serif;font-size:32px;font-weight:600;
line-height:1;font-feature-settings:"tnum"}
.stat span{font-family:"JetBrains Mono",monospace;font-size:10px;
letter-spacing:0.12em;text-transform:uppercase;color:var(--mute)}
.side-h{font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.18em;text-transform:uppercase;color:var(--mute);
margin:0 0 14px;display:flex;align-items:center;gap:10px}
.side-h::after{content:"";flex:1;height:1px;background:var(--rule)}
.pulse-list{display:flex;flex-direction:column;gap:10px}
.pulse-item{display:grid;grid-template-columns:32px 1fr auto;align-items:center;
gap:12px;padding:10px 0;border-bottom:1px dotted var(--rule);
cursor:pointer;transition:background .15s}
.pulse-item:last-child{border-bottom:none}
.pulse-item:hover{background:rgba(156,42,26,0.04)}
.pulse-rank{font-family:"Newsreader",serif;font-size:22px;font-style:italic;
color:var(--mute);font-feature-settings:"tnum"}
.pulse-name{font-weight:500;font-size:14.5px;line-height:1.25}
.pulse-name small{display:block;color:var(--mute);font-size:12px;font-weight:400;margin-top:2px}
.pulse-bar{width:72px;height:6px;background:var(--paper-2);border-radius:3px;
position:relative;overflow:hidden}
.pulse-bar::after{content:"";position:absolute;inset:0;width:var(--w,50%);
background:var(--accent);border-radius:3px}
.pulse-count{font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--mute);margin-top:4px}
.themes{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
background:var(--rule);border:1px solid var(--rule);margin-bottom:48px}
.t-card{background:var(--paper);padding:20px 22px 22px;
display:flex;flex-direction:column;gap:10px;min-height:220px;position:relative}
.t-label{font-family:"JetBrains Mono",monospace;font-size:10px;
letter-spacing:0.16em;text-transform:uppercase;color:var(--mute)}
.t-title{font-family:"Newsreader",serif;font-weight:500;font-size:22px;
line-height:1.1;letter-spacing:-0.01em;margin:2px 0 4px}
.t-desc{font-size:13.5px;line-height:1.5;color:var(--ink-2);flex:1;text-wrap:pretty}
.t-bar{height:4px;background:var(--paper-2);border-radius:2px;
position:relative;overflow:hidden;margin-top:6px}
.t-bar::after{content:"";position:absolute;inset:0;width:var(--w,50%);background:var(--accent)}
.t-count{font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--mute);
display:flex;justify-content:space-between}
.t-count b{color:var(--ink);font-weight:500}
.sec-head{display:flex;align-items:baseline;justify-content:space-between;
margin:48px 0 18px;padding-bottom:10px;border-bottom:2px solid var(--ink)}
.sec-h{font-family:"Newsreader",serif;font-weight:600;font-size:30px;
letter-spacing:-0.015em;margin:0}
.sec-meta{font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.15em;text-transform:uppercase;color:var(--mute)}
.cat-section{margin-top:56px;scroll-margin-top:20px}
.cat-header{display:grid;grid-template-columns:auto 1fr auto;align-items:end;
gap:20px;padding-bottom:12px;border-bottom:1px solid var(--ink);margin-bottom:24px}
.cat-num{font-family:"Newsreader",serif;font-style:italic;font-size:56px;
font-weight:400;line-height:0.9;color:var(--accent);font-feature-settings:"tnum"}
.cat-title{font-family:"Newsreader",serif;font-weight:500;font-size:28px;
line-height:1.1;letter-spacing:-0.015em;margin:0}
.cat-title small{display:block;font-family:"JetBrains Mono",monospace;
font-size:10.5px;font-weight:400;letter-spacing:0.15em;text-transform:uppercase;
color:var(--mute);margin-top:6px}
.cat-pill{font-family:"JetBrains Mono",monospace;font-size:11px;
padding:4px 10px;background:var(--ink);color:var(--paper);
border-radius:999px;white-space:nowrap}
.col-toggle{position:fixed;right:18px;bottom:18px;display:flex;align-items:center;
gap:4px;background:var(--paper);border:1px solid var(--ink);padding:4px;z-index:50;
font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:0.1em;
text-transform:uppercase;box-shadow:0 8px 22px -14px rgba(26,22,19,0.35)}
.col-toggle .lbl{padding:0 8px 0 6px;color:var(--mute)}
.col-toggle button{appearance:none;font:inherit;background:transparent;color:var(--ink);
border:none;padding:6px 10px;cursor:pointer;display:flex;align-items:center;gap:4px;
transition:background .15s,color .15s}
.col-toggle button:hover{background:var(--paper-2)}
.col-toggle button[aria-pressed="true"]{background:var(--ink);color:var(--paper)}
.col-toggle .glyph{display:inline-flex;gap:2px}
.col-toggle .glyph i{display:block;width:3px;height:11px;background:currentColor}
.cluster-group{margin-bottom:28px}
.cluster-head{display:block;background:var(--paper-2);border-left:3px solid var(--accent);
padding:12px 16px;margin-bottom:1px;position:relative;text-decoration:none;color:inherit;
transition:background .15s,border-color .15s}
.cluster-head:hover{background:#e4ddcf;border-left-color:var(--ink)}
.cluster-badge{position:absolute;top:10px;right:12px;
font-family:"JetBrains Mono",monospace;font-size:9.5px;letter-spacing:0.12em;
text-transform:uppercase;padding:3px 8px;border-radius:3px;font-weight:600}
.cluster-badge{margin-left:6px}
.cluster-badge.new{background:var(--accent);color:var(--paper)}
.cluster-badge.ext{background:var(--ink);color:var(--paper)}
.cluster-badge.ren{background:var(--accent-3);color:var(--paper)}
.cluster-badge.rev{background:#4a7a3c;color:var(--paper)}
.cluster-badge.rising{background:#c7521a;color:var(--paper)}
.cluster-badge{position:static;display:inline-block;vertical-align:middle}
.cluster-head .cluster-badge:first-of-type{position:absolute;top:10px;right:12px}
.cluster-sparkline{display:flex;align-items:flex-end;gap:2px;height:22px;
margin-top:8px;max-width:180px}
.cluster-sparkline span{display:block;flex:1;background:var(--ink-2);
opacity:0.25;min-height:2px;border-radius:1px}
.cluster-sparkline span.has{opacity:0.85;background:var(--accent)}
.evo-item.evo-ren{border-left:3px solid var(--accent-3)}
.evo-item.evo-rev{border-left:3px solid #4a7a3c}
.evo-item.evo-rising{border-left:3px solid #c7521a}
.evo-item.evo-split{border-left:3px solid #7a3c7a}
.evo-list{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:36px}
.evo-item{display:grid;grid-template-columns:auto minmax(220px,1fr) minmax(0,auto);
align-items:center;gap:10px;padding:10px 14px;
background:var(--card);border:1px solid var(--rule);border-radius:4px;
font-size:13.5px}
.evo-item.evo-born{border-left:3px solid var(--accent)}
.evo-item.evo-ext{border-left:3px solid var(--ink)}
.evo-icon{font-size:16px}
.evo-kind{font-family:"JetBrains Mono",monospace;font-size:9.5px;
letter-spacing:0.12em;text-transform:uppercase;color:var(--mute);
padding:2px 6px;border:1px solid var(--rule);border-radius:3px}
.evo-name{font-family:"Newsreader",serif;font-weight:500;min-width:0;
white-space:normal;overflow:visible;text-overflow:clip}
.evo-meta{font-family:"JetBrains Mono",monospace;font-size:10.5px;color:var(--mute);
text-align:right;white-space:normal}
@media(max-width:700px){
.evo-list{grid-template-columns:1fr}
.evo-item{grid-template-columns:auto 1fr;align-items:flex-start}
.evo-meta{grid-column:2;width:100%;text-align:left}
}
.cluster-label{font-family:"JetBrains Mono",monospace;font-size:9.5px;
letter-spacing:0.14em;text-transform:uppercase;color:var(--mute)}
.cluster-name{display:block;font-family:"Newsreader",serif;font-weight:500;
font-size:17px;margin:4px 0 0}
.cluster-summary{font-size:13.5px;color:var(--ink-2);margin:6px 0 0;line-height:1.5}
.clusters-grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));
gap:24px;align-items:start;margin-top:8px}
.clusters-grid>.cluster-group{margin-bottom:0}
.clusters-grid .papers{grid-template-columns:1fr}
.papers{display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));
gap:1px;background:var(--rule);border:1px solid var(--rule)}
.paper{background:var(--card);padding:22px 24px;
display:flex;flex-direction:column;gap:8px;
text-decoration:none;color:inherit;transition:background .18s;position:relative}
.paper:hover{background:var(--paper-2)}
.paper-year{position:absolute;top:18px;right:22px;
font-family:"Newsreader",serif;font-style:italic;
font-size:14px;color:var(--mute);font-feature-settings:"tnum"}
.paper-tags{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:2px}
.tag{font-family:"JetBrains Mono",monospace;font-size:9.5px;
letter-spacing:0.1em;text-transform:uppercase;
padding:2px 7px;border:1px solid var(--rule);border-radius:3px;color:var(--mute)}
.tag.hot{background:var(--accent);color:var(--paper);border-color:var(--accent)}
.paper-title{font-family:"Newsreader",serif;font-weight:500;font-size:18px;
line-height:1.2;letter-spacing:-0.005em;margin:0;padding-right:44px;text-wrap:pretty}
.paper-authors{font-family:"Newsreader",serif;font-style:italic;
font-size:13.5px;color:var(--mute)}
.paper-summary{font-size:13.5px;line-height:1.55;color:var(--ink-2);margin-top:4px;text-wrap:pretty}
.paper-read{margin-top:auto;padding-top:10px;
font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.12em;text-transform:uppercase;color:var(--mute);
display:flex;align-items:center;gap:6px}
.paper:hover .paper-read{color:var(--accent)}
.paper-read .arr{transition:transform .2s}
.paper:hover .paper-read .arr{transform:translateX(4px)}
.papers.single{grid-template-columns:1fr}
.cluster-count{display:inline-flex;align-items:baseline;gap:6px;margin-top:6px;
font-family:"JetBrains Mono",monospace;font-size:11px;color:var(--mute);
letter-spacing:0.05em}
.cluster-count b{font-family:"Newsreader",serif;font-size:18px;font-weight:500;
color:var(--ink);font-feature-settings:"tnum"}
.cluster-count .added{color:var(--accent);font-weight:600}
.cluster-preview{display:grid;grid-template-columns:1fr;gap:1px;
background:var(--rule);border:1px solid var(--rule);margin-top:10px}
.cluster-preview .rep{background:var(--card);padding:10px 14px;
text-decoration:none;color:inherit;display:flex;flex-direction:column;gap:2px;
transition:background .15s}
.cluster-preview .rep:hover{background:var(--paper-2)}
.rep-title{font-family:"Newsreader",serif;font-weight:500;font-size:14.5px;
line-height:1.3;color:var(--ink);text-wrap:pretty}
.rep-authors{font-family:"JetBrains Mono",monospace;font-size:10px;
color:var(--mute);letter-spacing:0.04em}
.orphan-badge{display:inline-flex;align-items:center;gap:6px;
font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.1em;text-transform:uppercase;
background:var(--card);border:1px solid var(--rule);
padding:4px 10px;border-radius:999px;color:var(--ink-2);
text-decoration:none;white-space:nowrap;margin-left:10px}
.orphan-badge:hover{border-color:var(--ink);background:var(--ink);color:var(--paper)}
footer{margin-top:80px;padding-top:20px;border-top:3px double var(--ink);
font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.12em;text-transform:uppercase;color:var(--mute);
display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px}
@media(min-width:1400px){:root{--cols:3}}
@media(max-width:1100px){:root{--cols:2}}
@media(max-width:760px){:root{--cols:1}}
html[data-cols="1"]{--cols:1!important}
html[data-cols="2"]{--cols:2!important}
html[data-cols="3"]{--cols:3!important}
@media print{.col-toggle{display:none}}
@media(max-width:980px){
.hero{grid-template-columns:1fr}
.hero-main{border-right:none;border-bottom:1px solid var(--rule);padding:24px 0}
.hero-side{padding:24px 0}
.themes{grid-template-columns:repeat(2,1fr)}
.hero-h{font-size:40px}}
@media(max-width:600px){
.wrap{padding:20px 18px 60px}
.mast-title{font-size:30px}
.hero-h{font-size:32px}
.themes{grid-template-columns:1fr}
.cat-num{font-size:42px}
.cat-title{font-size:22px}
.paper-title{font-size:17px}
.col-toggle{right:10px;bottom:10px}}
"""


def anchor_of(cat: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", cat).strip("-").lower()


def esc(s) -> str:
    return html_module.escape(str(s), quote=True)


_JOURNAL_BOILERPLATE_RES = [
    # Bounded so we don't eat the actual abstract that follows.
    re.compile(r"Contents\s+lists\s+available\s+at\s+\S+", re.IGNORECASE),
    re.compile(r"journal\s+homepage\s*:\s*\S+", re.IGNORECASE),
    re.compile(r"\bwww\.[^\s)]+", re.IGNORECASE),
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"\bScienceDirect\b", re.IGNORECASE),
    re.compile(r"\bElsevier(?:\s+(?:Ltd|Inc|B\.V\.|Science))?\b", re.IGNORECASE),
    re.compile(r"\bSpringer(?:\s+Nature)?\b", re.IGNORECASE),
    re.compile(r"\bWiley(?:\s+Periodicals)?\b", re.IGNORECASE),
    re.compile(r"\bNature\s+Publishing\s+Group\b", re.IGNORECASE),
    # Journal names that appear in headers (best-effort, add as seen)
    re.compile(r"Neuroscience\s+and\s+Biobehavioral\s+Reviews", re.IGNORECASE),
    # PDF-extractor artifacts and article-history stubs
    re.compile(r"\[TD\$[A-Z]+\]"),
    re.compile(r"©\s*\d{4}(?:\s*[-–]\s*\d{4})?"),
    re.compile(r"All rights reserved\.?", re.IGNORECASE),
    re.compile(r"\bdoi\s*:\s*\S+", re.IGNORECASE),
    re.compile(r"\bArticle\s+history\b[:.]?", re.IGNORECASE),
    re.compile(r"\bReview\s+article\b", re.IGNORECASE),
    re.compile(r"\bResearch\s+article\b", re.IGNORECASE),
    re.compile(r"Received\s+\d+\s+\w+\s+\d{4}", re.IGNORECASE),
    re.compile(r"Accepted\s+\d+\s+\w+\s+\d{4}", re.IGNORECASE),
    re.compile(r"Available\s+online\s+\d+\s+\w+\s+\d{4}", re.IGNORECASE),
]


def _clean_abstract_snippet(abstract: str, max_chars: int = 220) -> str:
    """Strip common journal/header boilerplate before truncating."""
    if not abstract:
        return ""
    s = abstract
    for rx in _JOURNAL_BOILERPLATE_RES:
        s = rx.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" .-—•·")
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    last_dot = max(cut.rfind(". "), cut.rfind("。"))
    if last_dot > 100:
        return cut[:last_dot + 1]
    return cut.rstrip() + "…"


# Legacy name kept for any other callers.
def _abstract_snippet(abstract: str, max_chars: int = 220) -> str:
    return _clean_abstract_snippet(abstract, max_chars)


def _paper_tags_html(e: "Entry", keywords: list[str], keywords_are_fallback: bool = False) -> str:
    tags = []
    if e.pub_year.isdigit() and int(e.pub_year) >= 2025:
        tags.append(f'<span class="tag hot">New {esc(e.pub_year)}</span>')
    if not keywords_are_fallback:
        for kw in keywords[:2]:
            tags.append(f'<span class="tag">{esc(kw)}</span>')
    return f'<div class="paper-tags">{"".join(tags)}</div>' if tags else ""


def _top_representative_papers(
    c: "Cluster",
    lc,
    emb_cache: dict,
    k: int = 3,
) -> list[tuple[str, str, str]]:
    """Return up to k (title, authors, href) tuples for the cluster's most
    centroid-representative papers. If centroid/emb lookups fail, fall back to
    newest members_today by filename order.

    `lc` may be the living-cluster (preferred: uses its full member list) or
    None (fresh/unmatched cluster → pick from members_today).
    """
    from . import living_cluster as lcmod
    picks: list[tuple[float, str, str, str]] = []  # (sim, fname, title, authors)
    centroid = None
    if c.centroid is not None:
        centroid = c.centroid
    elif lc is not None and getattr(lc, "centroid", None):
        try:
            import numpy as np
            centroid = np.asarray(lc.centroid, dtype="float32")
        except Exception:
            centroid = None

    if lc is not None and getattr(lc, "members", None):
        for m in lc.members:
            fn = m.get("fname", "")
            if not fn:
                continue
            v = emb_cache.get(fn) if emb_cache else None
            sim = lcmod.cosine_sim(centroid, v) if (centroid is not None and v is not None) else 0.0
            authors, title = parse_authors_title(fn)
            picks.append((sim, fn, title or fn, authors))
    else:
        for e in c.members_today:
            sim = (lcmod.cosine_sim(centroid, e.embedding)
                   if (centroid is not None and e.embedding is not None) else 0.0)
            picks.append((sim, e.fname, e.title or e.fname, e.authors))

    if any(p[0] > 0 for p in picks):
        picks.sort(key=lambda p: -p[0])
    ranked = picks[:k]
    out: list[tuple[str, str, str]] = []
    for _, fn, title, authors in ranked:
        href = f"../{urllib.parse.quote(c.category)}/{urllib.parse.quote(fn)}"
        out.append((title, authors, href))
    return out


def _orphan_pool_href(cat: str) -> str:
    from . import orphan_pool as orphmod
    return f"orphans-{orphmod._slug(cat)}.html"


def render_daily_html(
    target_date: date,
    today: list[Entry],
    clusters_by_cat: dict[str, list[Cluster]],
    use_llm: bool = True,
    living_by_cat: Optional[dict] = None,
    emb_cache: Optional[dict] = None,
    orphan_counts_by_cat: Optional[dict[str, int]] = None,
) -> Path:
    living_by_cat = living_by_cat or {}
    emb_cache = emb_cache or {}
    orphan_counts_by_cat = orphan_counts_by_cat or {}
    living_by_uid: dict[str, object] = {}
    for lcs in living_by_cat.values():
        for lc in lcs:
            living_by_uid[lc.uid] = lc
    total = len(today)
    all_clusters: list[Cluster] = [
        c for cs in clusters_by_cat.values() for c in cs
        if c.coherent and c.n_today > 0
    ]
    cats_sorted = sorted(
        clusters_by_cat.keys(),
        key=lambda c: (-sum(cl.n_today for cl in clusters_by_cat[c]), c.lower()),
    )
    n_recent_year = sum(
        1 for e in today
        if e.pub_year.isdigit() and int(e.pub_year) >= target_date.year - 1
    )

    # Through-line headline + lede
    headline, lede = generate_throughline(all_clusters, total, use_llm)

    def _cluster_key(c: Cluster) -> str:
        return c.living_uid or f"{c.category}:{c.cluster_id}"

    def _canonical_name(c: Cluster) -> str:
        if c.living_uid:
            lc = living_by_uid.get(c.living_uid)
            if lc and lc.theme_name:
                return lc.theme_name
        if c.theme_name:
            return c.theme_name
        if c.category:
            return c.category
        return c.living_uid or "Untitled Cluster"

    def _canonical_summary(c: Cluster) -> str:
        if c.living_uid:
            lc = living_by_uid.get(c.living_uid)
            if lc and lc.theme_summary:
                return lc.theme_summary
        return c.theme_summary

    def _best_cluster(group: list[Cluster]) -> Cluster:
        return max(
            group,
            key=lambda c: (
                bool(c.theme_name),
                bool(c.theme_summary),
                c.n_today + c.n_recent,
                c.n_today,
            ),
        )

    def _group_clusters(clusters: list[Cluster]) -> list[list[Cluster]]:
        grouped: dict[str, list[Cluster]] = {}
        for c in clusters:
            grouped.setdefault(_cluster_key(c), []).append(c)
        return list(grouped.values())

    def _unique_entries(entries: list[Entry]) -> list[Entry]:
        seen: set[str] = set()
        out: list[Entry] = []
        for e in entries:
            key = e.rel_path or f"{e.category}/{e.fname}"
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out

    def _merged_cluster(group: list[Cluster]) -> Cluster:
        best = _best_cluster(group)
        added_sum = sum(c.n_added_today for c in group)
        total_after = max(
            [c.prior_member_count + c.n_added_today for c in group]
            + [c.n_today + c.n_recent for c in group]
            + [added_sum]
        )
        merged = Cluster(
            cluster_id=best.cluster_id,
            category=best.category,
            members_today=_unique_entries([e for c in group for e in c.members_today]),
            members_recent=_unique_entries([e for c in group for e in c.members_recent]),
            centroid=best.centroid,
            theme_name=_canonical_name(best),
            theme_name_fallback=all(c.theme_name_fallback for c in group),
            theme_summary=_canonical_summary(best),
            keywords=best.keywords,
            keywords_are_fallback=best.keywords_are_fallback,
            coherent=best.coherent,
            rising_score=max(c.rising_score for c in group),
            status=best.status,
            living_uid=best.living_uid,
            lineage=best.lineage,
            prior_member_count=max(0, total_after - added_sum),
            added_today_count=added_sum,
            renamed=any(c.renamed for c in group),
            name_before=next((c.name_before for c in group if c.name_before), ""),
            revived=any(c.revived for c in group),
            rename_reason=next((c.rename_reason for c in group if c.rename_reason), ""),
            growth_rate_7d=max(c.growth_rate_7d for c in group),
        )
        if any(c.lineage == "born" for c in group):
            merged.lineage = "born"
        elif any(c.revived for c in group):
            merged.lineage = "extended"
        elif any(c.lineage == "extended" for c in group):
            merged.lineage = "extended"
        return merged

    unique_all_clusters = [_merged_cluster(group) for group in _group_clusters(all_clusters)]

    # Formatting
    try:
        date_str = target_date.strftime("%-d %B %Y")
    except ValueError:
        date_str = target_date.isoformat()
    iss_str = str(target_date.month).zfill(2)
    gen_time = datetime.now().strftime("%H:%M KST")

    # Top themes sidebar — top clusters by total papers
    pulse_clusters = sorted(unique_all_clusters, key=lambda c: -(c.n_today + c.n_recent))[:4]
    pulse_max = max((c.n_today + c.n_recent for c in pulse_clusters), default=1)
    pulse_items_html = ""
    for i, c in enumerate(pulse_clusters, 1):
        cnt = c.n_today + c.n_recent
        pct = int(cnt / pulse_max * 100)
        pulse_items_html += (
            f'<div class="pulse-item" onclick="location.href=\'#cat-{esc(anchor_of(c.category))}\'">'
            f'<div class="pulse-rank">{i:02d}</div>'
            f'<div class="pulse-name">{esc(_canonical_name(c))}'
            f'<small>{esc(c.category)}</small></div>'
            f'<div><div class="pulse-bar" style="--w:{pct}%"></div>'
            f'<div class="pulse-count">{cnt}편</div></div>'
            f'</div>'
        )

    # Living-cluster evolution highlights — one row per living cluster.
    born_clusters = [c for c in unique_all_clusters if c.lineage == "born" and c.n_added_today > 0]
    extended_clusters = [c for c in unique_all_clusters if c.lineage == "extended" and c.n_added_today > 0]
    extended_uids = {c.living_uid for c in extended_clusters if c.living_uid}
    renamed_clusters = [
        c for c in unique_all_clusters
        if c.renamed and (not c.living_uid or c.living_uid not in extended_uids)
    ]
    revived_clusters = [
        c for c in unique_all_clusters
        if c.revived and (not c.living_uid or c.living_uid not in extended_uids)
    ]

    def _uid_link(c: Cluster) -> str:
        if c.living_uid:
            return f'<a class="evo-name" href="cluster-{esc(c.living_uid)}.html">{esc(_canonical_name(c))}</a>'
        return f'<a class="evo-name" href="#cat-{esc(anchor_of(c.category))}">{esc(_canonical_name(c))}</a>'

    evo_items_html = ""
    for c in born_clusters:
        evo_items_html += (
            f'<div class="evo-item evo-born">'
            f'<span class="evo-kind">Born</span>'
            f'{_uid_link(c)}'
            f'<span class="evo-meta">새 주제 · {c.n_today}편</span>'
            f'</div>'
        )
    for c in sorted(extended_clusters, key=lambda c: -c.n_added_today):
        cluster_total = c.prior_member_count + c.n_added_today
        meta_bits = [f'+{c.n_added_today} → 총 {cluster_total}편']
        if c.renamed:
            meta_bits.append(f'renamed{(" · " + esc(c.rename_reason)) if c.rename_reason else ""}')
        if c.revived:
            meta_bits.append('revived')
        if c.growth_rate_7d >= 0.3:
            meta_bits.append(f'🔥 7일 {int(round(c.growth_rate_7d * 100))}%')
        evo_items_html += (
            f'<div class="evo-item evo-ext">'
            f'<span class="evo-kind">Extended</span>'
            f'{_uid_link(c)}'
            f'<span class="evo-meta">{" · ".join(meta_bits)}</span>'
            f'</div>'
        )
    for c in renamed_clusters:
        evo_items_html += (
            f'<div class="evo-item evo-ren">'
            f'<span class="evo-kind">Renamed</span>'
            f'{_uid_link(c)}'
            f'<span class="evo-meta">renamed'
            f'{(" · " + esc(c.rename_reason)) if c.rename_reason else ""}</span>'
            f'</div>'
        )
    for c in revived_clusters:
        evo_items_html += (
            f'<div class="evo-item evo-rev">'
            f'<span class="evo-kind">Revived</span>'
            f'{_uid_link(c)}'
            f'<span class="evo-meta">휴면에서 돌아옴 · +{c.n_added_today}편</span>'
            f'</div>'
        )

    # Rising — top clusters where ≥30% of mass arrived in the last 7 days,
    # excluding ones already surfaced above as Born/Extended.
    surfaced_uids = {
        c.living_uid
        for c in born_clusters + extended_clusters + renamed_clusters + revived_clusters
        if c.living_uid
    }
    rising_clusters = sorted(
        [c for c in unique_all_clusters
         if c.growth_rate_7d >= 0.3 and c.living_uid
         and c.living_uid not in surfaced_uids],
        key=lambda c: -c.growth_rate_7d,
    )[:5]
    for c in rising_clusters:
        evo_items_html += (
            f'<div class="evo-item evo-rising">'
            f'<span class="evo-kind">Rising</span>'
            f'{_uid_link(c)}'
            f'<span class="evo-meta">🔥 최근 7일 '
            f'{int(round(c.growth_rate_7d * 100))}%</span>'
            f'</div>'
        )

    # Split events dated today — structural changes in LC registry.
    split_events_today = []
    target_iso = target_date.isoformat()
    for lcs in (living_by_cat or {}).values():
        for lc in lcs:
            for ev in lc.events:
                if ev.get("type") == "split" and str(ev.get("at", ""))[:10] == target_iso:
                    split_events_today.append((lc, ev))
    for lc, ev in split_events_today:
        into = ev.get("into") or []
        child_uid = next((u for u in into if u != lc.uid), "")
        link_parent = (f'<a class="evo-name" href="cluster-{esc(lc.uid)}.html">'
                       f'{esc(lc.theme_name or lc.uid)}</a>')
        link_child = (f'<a class="evo-name" href="cluster-{esc(child_uid)}.html">'
                      f'{esc(child_uid)}</a>') if child_uid else ""
        evo_items_html += (
            f'<div class="evo-item evo-split">'
            f'<span class="evo-kind">Split</span>'
            f'{link_parent}'
            f'<span class="evo-meta">✂️ {ev.get("kept", "?")} + {ev.get("spawned", "?")}편'
            + (f' → {link_child}' if link_child else '')
            + f' · silhouette={ev.get("silhouette", 0):.2f}</span>'
            f'</div>'
        )

    evo_meta_bits = []
    if born_clusters: evo_meta_bits.append(f"Born {len(born_clusters)}")
    if extended_clusters: evo_meta_bits.append(f"Extended {len(extended_clusters)}")
    renamed_count = sum(1 for c in unique_all_clusters if c.renamed)
    revived_count = sum(1 for c in unique_all_clusters if c.revived)
    if renamed_count: evo_meta_bits.append(f"Renamed {renamed_count}")
    if revived_count: evo_meta_bits.append(f"Revived {revived_count}")
    if rising_clusters: evo_meta_bits.append(f"Rising {len(rising_clusters)}")
    if split_events_today: evo_meta_bits.append(f"Split {len(split_events_today)}")
    evo_block_html = (
        '<div class="sec-head">'
        '<h2 class="sec-h">Cluster Evolution</h2>'
        f'<div class="sec-meta">{" · ".join(evo_meta_bits)}</div>'
        '</div>'
        f'<div class="evo-list">{evo_items_html}</div>'
    ) if evo_items_html else ""

    # Sparkline helper — weekly member counts over the last 10 weeks
    def _sparkline_for(c: Cluster, ref_date: date, n_weeks: int = 10) -> str:
        if not c.living_uid:
            return ""
        lc = living_by_uid.get(c.living_uid)
        if lc is None:
            return ""
        buckets = [0] * n_weeks
        for m in getattr(lc, "members", []):
            try:
                d = date.fromisoformat(str(m.get("added", ""))[:10])
            except Exception:
                continue
            delta_days = (ref_date - d).days
            wk = delta_days // 7
            if 0 <= wk < n_weeks:
                buckets[n_weeks - 1 - wk] += 1
        peak = max(buckets) or 1
        bars = []
        for n in buckets:
            pct = max(8, int(n / peak * 100)) if n > 0 else 6
            cls = "has" if n > 0 else ""
            bars.append(f'<span class="{cls}" style="height:{pct}%" title="{n}편"></span>')
        return f'<div class="cluster-sparkline">{"".join(bars)}</div>'

    # Paper card helper
    def paper_card(e: Entry, summary: str, keywords: list[str],
                   keywords_are_fallback: bool = False) -> str:
        tags_html = _paper_tags_html(e, keywords, keywords_are_fallback)
        return (
            f'<a class="paper" href="{esc(rel_href(e.rel_path))}">'
            f'<span class="paper-year">{esc(e.pub_year)}</span>'
            f'{tags_html}'
            f'<h4 class="paper-title">{esc(e.title)}</h4>'
            f'<div class="paper-authors">{esc(e.authors or "—")}</div>'
            + (f'<p class="paper-summary">{esc(summary)}</p>' if summary else '') +
            f'<div class="paper-read">Read PDF <span class="arr">→</span></div>'
            f'</a>'
        )

    # Category sections
    sections_html = ""
    for cat_i, cat in enumerate(cats_sorted, 1):
        cl_list_raw = sorted(clusters_by_cat[cat], key=lambda c: (-c.rising_score, -c.n_today))
        n = sum(cl.n_today for cl in cl_list_raw)
        if n == 0:
            continue
        cl_list = sorted(
            [_merged_cluster(group) for group in _group_clusters(cl_list_raw)],
            key=lambda c: (-c.rising_score, -c.n_today),
        )
        anchor = anchor_of(cat)

        # subtitle from top cluster keywords
        all_kw: list[str] = []
        for c in cl_list:
            if not c.keywords_are_fallback:
                all_kw.extend(c.keywords[:2])
        subtitle = " · ".join(dict.fromkeys(all_kw[:4]))

        cluster_cards: list[str] = []
        for c in cl_list:
            if c.n_today == 0:
                continue
            # "Orphan" = a today-paper that did not join/spawn a living cluster
            # (extended/born). Matches the criterion in run()'s orphan-pool
            # touch step, so the `🗂 N unclustered` badge count stays consistent
            # with what actually goes to orphans-<slug>.html.
            is_bare_singleton = (c.n_today == 1
                                 and c.lineage not in ("extended", "born"))
            if is_bare_singleton:
                # Unclustered singletons are rolled up into the category's
                # orphan-index page via the badge below — no inline render.
                continue

            lc = living_by_uid.get(c.living_uid) if c.living_uid else None
            reps = _top_representative_papers(c, lc, emb_cache, k=3)
            cluster_preview = "".join(
                f'<a class="rep" href="{esc(href)}">'
                f'<span class="rep-title">{esc(title)}</span>'
                + (f'<span class="rep-authors">{esc(authors)}</span>' if authors else '') +
                f'</a>'
                for (title, authors, href) in reps
            )
            badge_bits = []
            if c.lineage == "born":
                badge_bits.append('<span class="cluster-badge new">NEW</span>')
            elif c.lineage == "extended" and c.n_added_today > 0:
                badge_bits.append(f'<span class="cluster-badge ext">+{c.n_added_today} this week</span>')
            if c.renamed:
                badge_bits.append('<span class="cluster-badge ren">RENAMED</span>')
            if c.revived:
                badge_bits.append('<span class="cluster-badge rev">REVIVED</span>')
            if (c.growth_rate_7d >= 0.3 and c.lineage != "born"):
                pct = int(round(c.growth_rate_7d * 100))
                badge_bits.append(
                    f'<span class="cluster-badge rising">🔥 {pct}% · 7d</span>'
                )
            badge = "".join(badge_bits)
            spark_html = _sparkline_for(c, target_date) if c.living_uid else ""
            # Suppress the cluster-level concept overview when the cluster
            # has only a single paper in total — there's nothing to abstract
            # across, so the "summary" would just duplicate that paper.
            total_size = c.n_today + c.n_recent
            display_name = _canonical_name(c)
            display_summary = _canonical_summary(c)
            total_size = max(total_size, c.prior_member_count + c.n_added_today)
            show_summary = bool(display_summary) and total_size >= 2
            cluster_uid_link = (f'cluster-{esc(c.living_uid)}.html'
                                if c.living_uid else f'#cat-{esc(anchor)}')
            added_bit = (f' <span class="added">+{c.n_added_today} this week</span>'
                         if c.n_added_today > 0 else "")
            count_html = (
                f'<span class="cluster-count">'
                f'📄 <b>{total_size}</b> papers{added_bit}'
                f'</span>'
            )
            head_tag = "a" if c.living_uid else "div"
            head_href = f' href="{cluster_uid_link}"' if c.living_uid else ""
            cluster_cards.append(
                '<div class="cluster-group">'
                f'<{head_tag} class="cluster-head"{head_href}>'
                + badge +
                f'<span class="cluster-label">cluster</span>'
                f'<span class="cluster-name">{esc(display_name)}</span>'
                + (f'<p class="cluster-summary">{esc(display_summary)}</p>' if show_summary else '') +
                count_html +
                spark_html +
                f'</{head_tag}>'
                + (f'<div class="cluster-preview">{cluster_preview}</div>' if cluster_preview else '') +
                '</div>'
            )

        orphan_n = int(orphan_counts_by_cat.get(cat, 0))
        orphan_badge_html = (
            f'<a class="orphan-badge" href="{esc(_orphan_pool_href(cat))}" '
            f'title="Unclustered papers awaiting future grouping">'
            f'🗂 {orphan_n} unclustered</a>'
        ) if orphan_n > 0 else ""

        cat_body = ""
        if cluster_cards:
            cat_body += '<div class="clusters-grid">' + "".join(cluster_cards) + '</div>'

        sections_html += (
            f'<section class="cat-section" id="cat-{esc(anchor)}">'
            f'<div class="cat-header">'
            f'<div class="cat-num">{cat_i:02d}</div>'
            f'<div><h3 class="cat-title">{esc(cat)}'
            + (f'<small>{esc(subtitle)}</small>' if subtitle else '') +
            f'</h3></div>'
            f'<div><span class="cat-pill">{n} paper{"s" if n != 1 else ""}</span>{orphan_badge_html}</div>'
            f'</div>'
            f'{cat_body}'
            f'</section>'
        )

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weekly Brief · {esc(target_date.isoformat())}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400&family=IBM+Plex+Sans+KR:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="col-toggle" role="group" aria-label="Column layout">
  <span class="lbl">Layout</span>
  <button data-cols="1" aria-pressed="false" title="Single column">
    <span class="glyph"><i></i></span>1
  </button>
  <button data-cols="2" aria-pressed="false" title="Two columns">
    <span class="glyph"><i></i><i></i></span>2
  </button>
  <button data-cols="3" aria-pressed="false" title="Three columns">
    <span class="glyph"><i></i><i></i><i></i></span>3
  </button>
  <button data-cols="auto" aria-pressed="true" title="Responsive auto">Auto</button>
</div>
<script>
  (function() {{
    var key = 'weekly-brief-cols';
    var root = document.documentElement;
    var buttons = document.querySelectorAll('.col-toggle button');
    function apply(choice) {{
      if (choice === 'auto' || !choice) root.removeAttribute('data-cols');
      else root.setAttribute('data-cols', choice);
      buttons.forEach(function(button) {{
        button.setAttribute('aria-pressed', button.dataset.cols === (choice || 'auto') ? 'true' : 'false');
      }});
    }}
    try {{ apply(localStorage.getItem(key) || 'auto'); }} catch (err) {{ apply('auto'); }}
    buttons.forEach(function(button) {{
      button.addEventListener('click', function() {{
        var choice = button.dataset.cols;
        apply(choice);
        try {{ localStorage.setItem(key, choice); }} catch (err) {{}}
      }});
    }});
  }})();
</script>
<div class="wrap">

  <div class="mast">
    <div class="mast-left">
      <div class="mast-title">Weekly Brief</div>
      <div class="mast-sub">Neuroscience · 주간 논문 요약</div>
    </div>
    <div class="mast-right">
      Vol.&nbsp;{target_date.year} · Iss.&nbsp;{iss_str} · <b>{esc(date_str)}</b><br>
      Generated {esc(gen_time)}
    </div>
  </div>

  <div class="strap">
    <span><span class="dot"></span>This Week's Reading List · {total} articles</span>
    <span>Window: this week &nbsp;·&nbsp; Baseline: last {BASELINE_DAYS} days</span>
  </div>

  <section class="hero">
    <div class="hero-main">
      <div class="kicker">This Week's Through-line</div>
      <h1 class="hero-h">{esc(headline)}</h1>
      <p class="hero-lede">{esc(lede)}</p>
      <div class="hero-stats">
        <div class="stat"><b>{total}</b><span>new articles this week</span></div>
        <div class="stat"><b>{len(cats_sorted)}</b><span>categories</span></div>
        <div class="stat"><b>{len(all_clusters)}</b><span>themes</span></div>
        <div class="stat"><b>{n_recent_year}</b><span>papers {target_date.year - 1}–{target_date.year}</span></div>
      </div>
    </div>
    <div class="hero-side">
      <div class="side-h">Top Themes · 주제 밀도</div>
      <div class="pulse-list">{pulse_items_html}</div>
    </div>
  </section>

  {evo_block_html}

  <div class="sec-head">
    <h2 class="sec-h">This Week's Reading</h2>
    <div class="sec-meta">{len(cats_sorted)} categories · {total} articles</div>
  </div>

  {sections_html}

  <footer>
    <span>Weekly Brief · Neuroscience Edition</span>
    <span>Window: this week · Baseline: last {BASELINE_DAYS} days</span>
    <span>{total} articles · {len(cats_sorted)} categories · {len(all_clusters)} themes</span>
    <a href="index.html" style="color:inherit">↑ index</a>
  </footer>

</div>
</body>
</html>
"""
    out = NEWS_DIR / f"daily-{target_date.isoformat()}.html"
    out.write_text(html_doc, encoding="utf-8")
    log.info("rendered %s (%d bytes)", out.name, out.stat().st_size)
    save_daily_summary(
        target_date, headline, lede,
        n_articles=total,
        n_categories=len(cats_sorted),
        n_themes=len(all_clusters),
        n_recent_year=n_recent_year,
    )
    return out


INDEX_CSS = """
:root{--paper:#f4efe6;--paper-2:#ebe5d8;--ink:#1a1613;--ink-2:#3a332d;
--mute:#7a7065;--rule:#d9d1c1;--card:#fbf7ee;--accent:#9c2a1a;--accent-3:#b5821f}
*{box-sizing:border-box}html,body{margin:0;padding:0}
body{background:var(--paper);color:var(--ink);
font-family:"IBM Plex Sans KR","Newsreader",-apple-system,BlinkMacSystemFont,sans-serif;
font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased;
background-image:radial-gradient(rgba(0,0,0,0.018) 1px,transparent 1px);
background-size:3px 3px}
a{color:inherit;text-decoration:none}
.wrap{max-width:1180px;margin:0 auto;padding:36px 40px 80px}
.mast{border-top:3px solid var(--ink);border-bottom:1px solid var(--ink);
padding:14px 0 10px;display:flex;align-items:baseline;justify-content:space-between;
gap:20px;flex-wrap:wrap}
.mast-left{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.mast-title{font-family:"Newsreader",serif;font-weight:700;font-size:42px;
line-height:1;letter-spacing:-0.02em;font-style:italic}
.mast-sub{font-family:"JetBrains Mono",monospace;font-size:11px;
letter-spacing:0.12em;text-transform:uppercase;color:var(--mute)}
.mast-right{font-family:"JetBrains Mono",monospace;font-size:11px;
letter-spacing:0.08em;color:var(--mute);text-transform:uppercase;text-align:right}
.mast-right b{color:var(--ink);font-weight:500}
.strap{display:flex;justify-content:space-between;align-items:center;
padding:8px 0 24px;border-bottom:1px solid var(--rule);margin-bottom:40px;
font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.15em;text-transform:uppercase;color:var(--mute)}
.strap .dot{display:inline-block;width:6px;height:6px;border-radius:50%;
background:var(--accent);margin-right:6px;vertical-align:middle;animation:pulse 2.4s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.title-row{display:grid;grid-template-columns:1fr auto;align-items:end;gap:24px;
border-bottom:2px solid var(--ink);padding-bottom:14px;margin-bottom:36px}
.page-h{font-family:"Newsreader",serif;font-weight:500;font-size:68px;
line-height:0.95;letter-spacing:-0.025em;margin:0}
.page-h em{font-style:italic;color:var(--accent)}
.page-sub{font-family:"JetBrains Mono",monospace;font-size:11px;
letter-spacing:0.15em;text-transform:uppercase;color:var(--mute);text-align:right}
.page-sub b{color:var(--ink);font-weight:500;font-size:13px;display:block;margin-top:4px}
.latest{display:grid;grid-template-columns:180px 1fr auto;gap:32px;align-items:start;
border-top:1px solid var(--ink);border-bottom:1px solid var(--ink);
background:var(--card);padding:32px 36px;
transition:transform .2s,box-shadow .2s;cursor:pointer;margin-bottom:48px}
.latest:hover{transform:translateY(-2px);box-shadow:0 10px 24px -16px rgba(26,22,19,0.4)}
.latest-date{display:flex;flex-direction:column;gap:2px;
border-right:1px solid var(--rule);padding-right:24px}
.latest-date .y{font-family:"JetBrains Mono",monospace;font-size:11px;
letter-spacing:0.18em;color:var(--mute)}
.latest-date .d{font-family:"Newsreader",serif;font-weight:600;font-size:84px;
line-height:0.9;letter-spacing:-0.03em;font-feature-settings:"tnum"}
.latest-date .m{font-family:"Newsreader",serif;font-style:italic;
font-size:22px;color:var(--accent);margin-top:4px}
.latest-body .kicker{font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.18em;text-transform:uppercase;color:var(--accent);
font-weight:500;margin-bottom:10px;display:flex;align-items:center;gap:10px}
.latest-body .kicker::after{content:"";flex:1;height:1px;background:var(--accent);opacity:.3}
.latest-headline{font-family:"Newsreader",serif;font-weight:500;font-size:36px;
line-height:1.05;letter-spacing:-0.015em;margin:0 0 12px}
.latest-lede{font-family:"Newsreader",serif;font-size:16px;line-height:1.5;
color:var(--ink-2);margin:0 0 16px;max-width:58ch}
.latest-stats{display:flex;gap:24px;margin-top:14px;
padding-top:14px;border-top:1px solid var(--rule)}
.mini-stat{display:flex;flex-direction:column;gap:2px}
.mini-stat b{font-family:"Newsreader",serif;font-size:22px;font-weight:600;
line-height:1;font-feature-settings:"tnum"}
.mini-stat span{font-family:"JetBrains Mono",monospace;font-size:9.5px;
letter-spacing:0.12em;text-transform:uppercase;color:var(--mute)}
.latest-cta{align-self:center;display:flex;flex-direction:column;align-items:center;gap:8px;
padding:18px 22px;border:1px solid var(--ink);background:var(--paper);
transition:background .2s,color .2s;min-width:140px}
.latest:hover .latest-cta{background:var(--ink);color:var(--paper)}
.latest-cta .arr{font-family:"Newsreader",serif;font-style:italic;font-size:44px;line-height:1}
.latest-cta span{font-family:"JetBrains Mono",monospace;font-size:10px;
letter-spacing:0.18em;text-transform:uppercase}
.sec-head{display:flex;align-items:baseline;justify-content:space-between;
margin:40px 0 18px;padding-bottom:10px;border-bottom:2px solid var(--ink)}
.sec-h{font-family:"Newsreader",serif;font-weight:600;font-size:26px;
letter-spacing:-0.015em;margin:0}
.sec-meta{font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.15em;text-transform:uppercase;color:var(--mute)}
.archive{display:flex;flex-direction:column;border-top:1px solid var(--rule)}
.issue{display:grid;grid-template-columns:100px 120px 1fr auto;gap:28px;
align-items:baseline;padding:20px 16px;border-bottom:1px solid var(--rule);
transition:background .15s;cursor:pointer}
.issue:hover{background:var(--card)}
.issue.future{cursor:default;opacity:0.55}
.issue.future:hover{background:transparent}
.iss-no{font-family:"Newsreader",serif;font-style:italic;font-size:20px;
color:var(--mute);font-feature-settings:"tnum"}
.iss-date{font-family:"Newsreader",serif;font-weight:500;font-size:22px;
letter-spacing:-0.01em;font-feature-settings:"tnum"}
.iss-date small{display:block;font-family:"JetBrains Mono",monospace;
font-size:9.5px;letter-spacing:0.15em;text-transform:uppercase;
color:var(--mute);font-weight:400;margin-top:3px}
.iss-title{font-family:"Newsreader",serif;font-size:18px;
line-height:1.35;color:var(--ink-2)}
.iss-title b{color:var(--ink);font-weight:500}
.iss-meta{font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.08em;color:var(--mute);white-space:nowrap;text-align:right}
.iss-meta b{color:var(--ink);font-weight:500}
.iss-arr{font-family:"Newsreader",serif;font-style:italic;font-size:22px;
color:var(--mute);transition:transform .2s,color .2s;display:block;margin-top:4px}
.issue:hover .iss-arr{transform:translateX(4px);color:var(--accent)}
.skeleton-date{color:var(--mute);font-weight:400}
.skeleton-line{height:10px;background:var(--paper-2);border-radius:2px;width:65%}
.about{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
background:var(--rule);border:1px solid var(--rule);margin-top:56px}
.about-cell{background:var(--paper);padding:22px 24px}
.about-cell .h{font-family:"JetBrains Mono",monospace;font-size:10px;
letter-spacing:0.18em;text-transform:uppercase;color:var(--mute);margin-bottom:8px}
.about-cell .t{font-family:"Newsreader",serif;font-size:18px;
line-height:1.35;color:var(--ink-2);text-wrap:pretty}
.about-cell .t b{color:var(--ink);font-weight:500}
footer{margin-top:64px;padding-top:20px;border-top:3px double var(--ink);
font-family:"JetBrains Mono",monospace;font-size:10.5px;
letter-spacing:0.12em;text-transform:uppercase;color:var(--mute);
display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px}
@media(max-width:860px){
.wrap{padding:20px 18px 60px}
.page-h{font-size:44px}
.title-row{grid-template-columns:1fr;gap:12px}
.page-sub{text-align:left}
.latest{grid-template-columns:1fr;gap:20px;padding:24px}
.latest-date{border-right:none;border-bottom:1px solid var(--rule);padding:0 0 16px}
.latest-date .d{font-size:64px}
.latest-headline{font-size:26px}
.latest-cta{align-self:flex-start;flex-direction:row}
.latest-cta .arr{font-size:22px}
.issue{grid-template-columns:60px 1fr auto;gap:14px}
.iss-no{font-size:14px}
.iss-date{font-size:18px}
.iss-title{font-size:15px;grid-column:1 / -1}
.iss-meta{white-space:normal;overflow-wrap:anywhere;font-size:9.5px}
.about{grid-template-columns:1fr}}
"""

_MONTHS_EN = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
_DAYS_EN = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


CLUSTER_PAGE_CSS = """
:root{--paper:#f4efe6;--paper-2:#ebe5d8;--ink:#1a1613;--ink-2:#3a332d;
--mute:#7a7065;--rule:#d9d1c1;--card:#fbf7ee;--accent:#9c2a1a;--accent-3:#b5821f}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:"IBM Plex Sans KR","Newsreader",sans-serif;font-size:15px;line-height:1.55}
a{color:inherit}
.wrap{max-width:980px;margin:0 auto;padding:36px 40px 80px}
.mast{border-top:3px solid var(--ink);border-bottom:1px solid var(--ink);
padding:14px 0 10px;display:flex;align-items:baseline;justify-content:space-between}
.mast-title{font-family:"Newsreader",serif;font-weight:700;font-size:32px;font-style:italic}
.back{font-family:"JetBrains Mono",monospace;font-size:11px;
letter-spacing:0.1em;text-transform:uppercase;color:var(--mute)}
h1.ct{font-family:"Newsreader",serif;font-weight:500;font-size:44px;
margin:28px 0 8px;letter-spacing:-0.02em}
.ct-meta{font-family:"JetBrains Mono",monospace;font-size:11px;
color:var(--mute);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:24px}
.ct-summary{font-size:16px;color:var(--ink-2);margin-bottom:32px}
.ct-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
background:var(--rule);border:1px solid var(--rule);margin-bottom:36px}
.ct-stat{background:var(--card);padding:16px 20px}
.ct-stat b{display:block;font-family:"Newsreader",serif;font-weight:500;
font-size:26px}
.ct-stat span{font-family:"JetBrains Mono",monospace;font-size:10px;
letter-spacing:0.12em;text-transform:uppercase;color:var(--mute)}
.ct-section{margin:40px 0}
.ct-section h2{font-family:"JetBrains Mono",monospace;font-size:11px;
letter-spacing:0.14em;text-transform:uppercase;color:var(--mute);
border-bottom:1px solid var(--rule);padding-bottom:6px;margin-bottom:14px}
.name-hist{display:flex;flex-direction:column;gap:4px}
.name-hist-row{display:flex;gap:12px;align-items:baseline;padding:4px 0}
.name-hist-date{font-family:"JetBrains Mono",monospace;font-size:11px;
color:var(--mute);min-width:100px}
.name-hist-name{font-family:"Newsreader",serif;font-size:16px}
.name-hist-reason{font-family:"JetBrains Mono",monospace;font-size:10px;
color:var(--accent-3);margin-left:auto}
.event-row{display:flex;gap:12px;align-items:baseline;padding:8px 0;
border-bottom:1px dashed var(--rule)}
.event-date{font-family:"JetBrains Mono",monospace;font-size:11px;
color:var(--mute);min-width:100px}
.event-type{font-family:"JetBrains Mono",monospace;font-size:10px;
letter-spacing:0.12em;text-transform:uppercase;padding:2px 7px;
border:1px solid var(--rule);border-radius:3px;min-width:80px;text-align:center}
.event-type.born{background:var(--accent);color:var(--paper);border-color:var(--accent)}
.event-type.extended,.event-type.backfill{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.event-type.renamed{background:var(--accent-3);color:var(--paper);border-color:var(--accent-3)}
.event-type.revived{background:#4a7a3c;color:var(--paper);border-color:#4a7a3c}
.event-type.split{background:#7a3c7a;color:var(--paper);border-color:#7a3c7a}
.event-type.split_from{background:#7a3c7a;color:var(--paper);border-color:#7a3c7a}
.event-body{flex:1;font-size:13.5px;color:var(--ink-2)}
.members-weekly{display:flex;flex-direction:column;gap:14px}
.week-group{border:1px solid var(--rule);background:var(--card);
border-radius:3px;overflow:hidden}
.week-group[open]>.week-header{border-bottom:1px solid var(--rule)}
.week-header{list-style:none;cursor:pointer;padding:10px 14px;
display:flex;align-items:baseline;gap:10px;font-family:"JetBrains Mono",monospace;
font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:var(--ink-2)}
.week-header::-webkit-details-marker{display:none}
.week-header::marker{display:none}
.week-header::before{content:"▸";font-size:10px;color:var(--mute);
transition:transform .15s ease}
.week-group[open]>.week-header::before{transform:rotate(90deg);display:inline-block}
.week-emoji{font-size:13px}
.week-label{color:var(--ink)}
.week-count{margin-left:auto;color:var(--mute);font-size:11px}
.paper-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
gap:1px;background:var(--rule)}
.paper-card{background:var(--paper);padding:14px 16px;display:flex;
flex-direction:column;gap:6px;min-height:92px}
.paper-title{font-family:"Newsreader",serif;font-weight:500;font-size:16px;
line-height:1.35;margin:0}
.paper-title a{text-decoration:none;color:var(--ink)}
.paper-title a:hover{color:var(--accent);text-decoration:underline}
.paper-authors{font-family:"JetBrains Mono",monospace;font-size:10.5px;
color:var(--mute);letter-spacing:0.04em}
.paper-snippet{margin:4px 0 0;font-size:13px;color:var(--ink-2);
line-height:1.45}
.ct-empty{color:var(--mute);font-family:"JetBrains Mono",monospace;font-size:11px}
.spark-big{display:flex;align-items:flex-end;gap:3px;height:60px;
background:var(--card);padding:10px;border:1px solid var(--rule)}
.spark-big span{display:block;flex:1;background:var(--ink-2);
opacity:0.2;border-radius:2px;min-height:3px}
.spark-big span.has{opacity:0.85;background:var(--accent)}
.spark-big span.future{opacity:0.05}
.spark-label{font-family:"JetBrains Mono",monospace;font-size:9.5px;
letter-spacing:0.12em;text-transform:uppercase;color:var(--mute);
margin-top:6px;display:flex;justify-content:space-between}
.bc-list{display:flex;flex-wrap:wrap;gap:8px;align-items:center;
font-family:"JetBrains Mono",monospace;font-size:11px}
.bc-link{padding:4px 10px;border:1px solid var(--rule);
background:var(--card);color:var(--ink);text-decoration:none;border-radius:3px;
letter-spacing:0.05em}
.bc-link:hover{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.bc-missing{color:var(--mute);border-style:dashed;cursor:default}
.bc-missing:hover{background:var(--card);color:var(--mute);border-color:var(--rule)}
.bc-more{color:var(--mute);font-size:10px;letter-spacing:0.1em;text-transform:uppercase}
"""


def render_orphan_index_pages() -> list[Path]:
    """Write monthly orphan pages + a per-category index.

    Output:
      - NEWS_DIR/orphans-<slug>-YYYY-MM.html  (one per (category, month) pair)
      - NEWS_DIR/orphans-<slug>.html          (month picker for the category)
    """
    from . import orphan_pool as orphmod

    pools = orphmod.load_all_pools(ORPHAN_POOL_DIR)
    if not pools:
        return []
    abstracts_cache = load_abstracts_cache() or {}
    written: list[Path] = []
    today = date.today()

    for cat, records in pools.items():
        if not records:
            continue
        slug = orphmod._slug(cat)
        by_month = orphmod.group_by_month(records)
        # Render each monthly page.
        month_keys = sorted(by_month.keys(), reverse=True)
        for mk in month_keys:
            recs = by_month[mk]
            page_path = _render_orphan_month_page(cat, slug, mk, recs, abstracts_cache, today)
            written.append(page_path)
        # Index page
        idx_path = _render_orphan_category_index(cat, slug, by_month, month_keys)
        written.append(idx_path)

    if written:
        log.info("rendered %d orphan pages", len(written))
    return written


def _orphan_ttl_weeks() -> int:
    from . import orphan_pool as orphmod
    return orphmod.ORPHAN_TTL_WEEKS


def _render_orphan_month_page(
    cat: str, slug: str, month_key: str,
    records: list, abstracts_cache: dict, today: date,
) -> Path:
    # Group by monday-anchored week, newest week open by default.
    week_groups: dict[str, list] = {}
    for r in records:
        try:
            d = date.fromisoformat(r.first_seen[:10])
        except Exception:
            continue
        monday = d - timedelta(days=d.weekday())
        week_groups.setdefault(monday.isoformat(), []).append(r)
    sorted_weeks = sorted(week_groups.keys(), reverse=True)

    week_html_parts: list[str] = []
    for i, wk_key in enumerate(sorted_weeks):
        recs = sorted(week_groups[wk_key], key=lambda r: r.first_seen, reverse=True)
        open_attr = " open" if i < 2 else ""
        cards = ""
        for r in recs:
            authors, title = parse_authors_title(r.fname)
            href = f"../{urllib.parse.quote(r.category)}/{urllib.parse.quote(r.fname)}"
            snippet = ""
            info = abstracts_cache.get(r.fname) or {}
            snippet = _abstract_snippet(info.get("abstract", "") or "", max_chars=220)
            meta_bits = [
                f'first seen {esc(r.first_seen)}',
                f'{r.attempts} attempt{"s" if r.attempts != 1 else ""}',
            ]
            card_bits = [
                f'<h4 class="paper-title"><a href="{esc(href)}">{esc(title or r.fname)}</a></h4>'
            ]
            if authors:
                card_bits.append(f'<div class="paper-authors">{esc(authors)}</div>')
            if snippet:
                card_bits.append(f'<p class="paper-snippet">{esc(snippet)}</p>')
            card_bits.append(
                f'<div class="paper-meta">{" · ".join(meta_bits)}</div>'
            )
            cards += f'<article class="paper-card">{"".join(card_bits)}</article>'
        week_html_parts.append(
            f'<details class="week-group"{open_attr}>'
            f'<summary class="week-header">'
            f'<span class="week-emoji">📅</span>'
            f'<span class="week-label">Week of {esc(wk_key)}</span>'
            f'<span class="week-count">{len(recs)} orphan{"s" if len(recs) != 1 else ""}</span>'
            f'</summary>'
            f'<div class="paper-grid">{cards}</div>'
            f'</details>'
        )
    weeks_html = "".join(week_html_parts) or "<p class=\"ct-empty\">—</p>"

    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orphans · {esc(cat)} · {esc(month_key)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,700&family=IBM+Plex+Sans+KR:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CLUSTER_PAGE_CSS}
.paper-meta{{font-family:"JetBrains Mono",monospace;font-size:10px;
color:var(--mute);letter-spacing:0.04em;margin-top:4px}}</style>
</head><body><div class="wrap">
  <div class="mast">
    <div class="mast-title">Orphans</div>
    <a class="back" href="orphans-{esc(slug)}.html">← {esc(cat)} index</a>
  </div>
  <h1 class="ct">{esc(cat)} · {esc(month_key)}</h1>
  <div class="ct-meta">
    Unclustered papers · {len(records)} paper{"s" if len(records) != 1 else ""}
    · pool TTL {_orphan_ttl_weeks()} weeks
  </div>
  <p class="ct-summary">이 페이지의 논문들은 아직 어떤 cluster에도 속하지 못한 상태입니다.
  다음 주 실행에서 기존 cluster로 재흡수되거나 새 cluster로 승격될 수 있습니다.</p>
  <div class="ct-section"><h2>Members by week</h2>
    <div class="members-weekly">{weeks_html}</div>
  </div>
</div></body></html>
"""
    out = NEWS_DIR / f"orphans-{slug}-{month_key}.html"
    out.write_text(doc, encoding="utf-8")
    return out


def _render_orphan_category_index(
    cat: str, slug: str,
    by_month: dict[str, list], month_keys: list[str],
) -> Path:
    rows = ""
    for mk in month_keys:
        n = len(by_month[mk])
        rows += (
            f'<a class="month-row" href="orphans-{esc(slug)}-{esc(mk)}.html">'
            f'<span class="month-key">{esc(mk)}</span>'
            f'<span class="month-count">{n} orphan{"s" if n != 1 else ""}</span>'
            f'<span class="month-arr">→</span>'
            f'</a>'
        )
    if not rows:
        rows = '<p class="ct-empty">no orphans</p>'
    total = sum(len(v) for v in by_month.values())
    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orphans · {esc(cat)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,700&family=IBM+Plex+Sans+KR:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CLUSTER_PAGE_CSS}
.month-row{{display:grid;grid-template-columns:160px 1fr 24px;
align-items:center;gap:14px;padding:14px 18px;
background:var(--card);border:1px solid var(--rule);
text-decoration:none;color:inherit;margin-bottom:6px;
transition:background .15s}}
.month-row:hover{{background:var(--paper-2)}}
.month-key{{font-family:"JetBrains Mono",monospace;font-size:14px;
letter-spacing:0.1em;color:var(--ink)}}
.month-count{{font-family:"JetBrains Mono",monospace;font-size:11px;
color:var(--mute);letter-spacing:0.1em;text-transform:uppercase}}
.month-arr{{font-family:"Newsreader",serif;font-style:italic;
font-size:20px;color:var(--mute)}}</style>
</head><body><div class="wrap">
  <div class="mast">
    <div class="mast-title">Orphans</div>
    <a class="back" href="index.html">← index</a>
  </div>
  <h1 class="ct">{esc(cat)}</h1>
  <div class="ct-meta">Unclustered paper pool · {total} paper{"s" if total != 1 else ""}</div>
  <p class="ct-summary">이 카테고리에서 아직 cluster에 편입되지 않은 논문들의 월별 보관소입니다.
  매 주 실행마다 기존 cluster centroid에 재평가되며, 승격된 논문은 사라집니다.</p>
  <div class="ct-section"><h2>By month</h2>
    {rows}
  </div>
</div></body></html>
"""
    out = NEWS_DIR / f"orphans-{slug}.html"
    out.write_text(doc, encoding="utf-8")
    return out


def render_cluster_detail_pages(
    living_by_cat: dict, force: bool = False,
) -> list[Path]:
    """One HTML page per living cluster. Written to NEWS_DIR/cluster-<uid>.html.

    Skips LCs whose on-disk HTML is newer than ``lc.updated_at`` unless
    ``force`` is set.
    """
    abstracts_cache = load_abstracts_cache()
    written: list[Path] = []
    skipped = 0
    for cat, lcs in living_by_cat.items():
        for lc in lcs:
            out_path = NEWS_DIR / f"cluster-{lc.uid}.html"
            if not force and not _cluster_page_is_stale(lc, out_path):
                skipped += 1
                continue
            path = _render_single_cluster_page(lc, abstracts_cache=abstracts_cache)
            if path is not None:
                written.append(path)
    if written or skipped:
        log.info("cluster detail pages: %d rendered, %d skipped (up-to-date)",
                 len(written), skipped)
    return written


def _cluster_page_is_stale(lc, out_path: Path) -> bool:
    """Return True if the cluster page needs regeneration."""
    if not out_path.exists():
        return True
    try:
        mtime = datetime.fromtimestamp(out_path.stat().st_mtime)
        updated = datetime.fromisoformat(lc.updated_at)
        return updated > mtime
    except Exception:
        return True


def _render_single_cluster_page(lc, abstracts_cache: Optional[dict] = None) -> Optional[Path]:
    # weekly histogram — last 26 weeks
    n_weeks = 26
    today = date.today()
    buckets = [0] * n_weeks
    for m in lc.members:
        try:
            d = date.fromisoformat(str(m.get("added", ""))[:10])
        except Exception:
            continue
        delta = (today - d).days
        wk = delta // 7
        if 0 <= wk < n_weeks:
            buckets[n_weeks - 1 - wk] += 1
    peak = max(buckets) or 1
    spark_cells = ""
    for n in buckets:
        pct = max(6, int(n / peak * 100)) if n > 0 else 4
        cls = "has" if n > 0 else ""
        spark_cells += f'<span class="{cls}" style="height:{pct}%" title="{n}편"></span>'
    spark_label_left = (today - timedelta(days=n_weeks * 7)).isoformat()

    # Name history rows
    hist_rows = ""
    for h in lc.name_history:
        reason = h.get("reason", "")
        tag = f'<span class="name-hist-reason">{esc(reason)}</span>' if reason else ""
        hist_rows += (
            f'<div class="name-hist-row">'
            f'<span class="name-hist-date">{esc(h.get("at", ""))}</span>'
            f'<span class="name-hist-name">{esc(h.get("name", ""))}</span>'
            f'{tag}'
            f'</div>'
        )
    if not hist_rows and lc.theme_name:
        hist_rows = (
            f'<div class="name-hist-row">'
            f'<span class="name-hist-date">{esc(lc.created_at[:10])}</span>'
            f'<span class="name-hist-name">{esc(lc.theme_name)}</span>'
            f'</div>'
        )

    # Events
    ev_rows = ""
    for ev in lc.events:
        t = ev.get("type", "")
        body = _describe_event(ev)
        ev_rows += (
            f'<div class="event-row">'
            f'<span class="event-date">{esc(ev.get("at", ""))}</span>'
            f'<span class="event-type {esc(t)}">{esc(t)}</span>'
            f'<span class="event-body">{body}</span>'
            f'</div>'
        )

    # Members grouped by calendar week (Mon-anchored), newest week first. Each
    # week is collapsible; the two most recent weeks are expanded by default so
    # the page opens to the active work without burying older cohorts.
    week_groups: dict[str, list[dict]] = {}
    for m in lc.members:
        try:
            d = date.fromisoformat(str(m.get("added", ""))[:10])
        except Exception:
            continue
        monday = d - timedelta(days=d.weekday())
        week_groups.setdefault(monday.isoformat(), []).append(m)
    sorted_weeks = sorted(week_groups.keys(), reverse=True)
    week_html_parts: list[str] = []
    for i, wk_key in enumerate(sorted_weeks):
        members = sorted(
            week_groups[wk_key],
            key=lambda m: str(m.get("added", "")),
            reverse=True,
        )
        open_attr = " open" if i < 2 else ""
        cards = ""
        for m in members:
            fname = m.get("fname", "")
            authors, title = parse_authors_title(fname)
            href = f"../{urllib.parse.quote(lc.category)}/{urllib.parse.quote(fname)}"
            snippet = ""
            if abstracts_cache:
                info = abstracts_cache.get(fname) or {}
                snippet = _abstract_snippet(info.get("abstract", "") or "", max_chars=220)
            card_bits = [
                f'<h4 class="paper-title"><a href="{esc(href)}">{esc(title or fname)}</a></h4>'
            ]
            if authors:
                card_bits.append(f'<div class="paper-authors">{esc(authors)}</div>')
            if snippet:
                card_bits.append(f'<p class="paper-snippet">{esc(snippet)}</p>')
            cards += f'<article class="paper-card">{"".join(card_bits)}</article>'
        week_html_parts.append(
            f'<details class="week-group"{open_attr}>'
            f'<summary class="week-header">'
            f'<span class="week-emoji">📅</span>'
            f'<span class="week-label">Week of {esc(wk_key)}</span>'
            f'<span class="week-count">added: {len(members)}</span>'
            f'</summary>'
            f'<div class="paper-grid">{cards}</div>'
            f'</details>'
        )
    members_html = "".join(week_html_parts) or "<p class=\"ct-empty\">—</p>"

    # Breadcrumb — weekly snapshot pages this cluster appeared in.
    breadcrumb_dates: set[str] = set()
    for m in lc.members:
        d_str = str(m.get("added", ""))[:10]
        if d_str:
            breadcrumb_dates.add(d_str)
    breadcrumb_html = ""
    if breadcrumb_dates:
        sorted_dates = sorted(breadcrumb_dates, reverse=True)[:12]
        links = []
        for d_str in sorted_dates:
            daily_path = NEWS_DIR / f"daily-{d_str}.html"
            if daily_path.exists():
                links.append(
                    f'<a class="bc-link" href="daily-{esc(d_str)}.html">{esc(d_str)}</a>'
                )
            else:
                links.append(f'<span class="bc-link bc-missing">{esc(d_str)}</span>')
        more = (f' <span class="bc-more">+{len(breadcrumb_dates) - len(sorted_dates)} earlier</span>'
                if len(breadcrumb_dates) > len(sorted_dates) else "")
        breadcrumb_html = (
            '<div class="ct-section"><h2>Appeared in weekly issues</h2>'
            f'<div class="bc-list">{" · ".join(links)}{more}</div>'
            '</div>'
        )

    summary_html = f'<p class="ct-summary">{esc(lc.theme_summary)}</p>' if lc.theme_summary else ""
    keywords_html = (
        '<div class="ct-section"><h2>Keywords</h2><p>'
        + " · ".join(esc(k) for k in lc.keywords) + '</p></div>'
    ) if lc.keywords else ""

    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(lc.theme_name or lc.uid)} · Cluster</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,700&family=IBM+Plex+Sans+KR:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CLUSTER_PAGE_CSS}</style>
</head><body><div class="wrap">
  <div class="mast">
    <div class="mast-title">Cluster</div>
    <a class="back" href="index.html">← index</a>
  </div>
  <h1 class="ct">{esc(lc.theme_name or "(무제)")}</h1>
  <div class="ct-meta">
    {esc(lc.category)} · uid {esc(lc.uid)} · status {esc(lc.status)}
    · created {esc(lc.created_at[:10])} · updated {esc(lc.updated_at[:10])}
  </div>
  {summary_html}
  <div class="ct-stats">
    <div class="ct-stat"><b>{lc.size}</b><span>members</span></div>
    <div class="ct-stat"><b>{len(lc.events)}</b><span>events</span></div>
    <div class="ct-stat"><b>{len(lc.name_history)}</b><span>names</span></div>
    <div class="ct-stat"><b>{(today - date.fromisoformat(lc.created_at[:10])).days if lc.created_at else 0}</b><span>days old</span></div>
  </div>

  <div class="ct-section"><h2>Weekly cadence · last {n_weeks} weeks</h2>
    <div class="spark-big">{spark_cells}</div>
    <div class="spark-label"><span>{esc(spark_label_left)}</span><span>now</span></div>
  </div>

  {keywords_html}

  <div class="ct-section"><h2>Name history</h2><div class="name-hist">{hist_rows}</div></div>

  <div class="ct-section"><h2>Events</h2>{ev_rows or '<p>—</p>'}</div>

  {breadcrumb_html}

  <div class="ct-section"><h2>Members · {lc.size}</h2>
    <div class="members-weekly">{members_html}</div>
  </div>
</div></body></html>
"""
    out = NEWS_DIR / f"cluster-{lc.uid}.html"
    out.write_text(doc, encoding="utf-8")
    return out


def _describe_event(ev: dict) -> str:
    t = ev.get("type", "")
    if t == "born":
        seeds = ev.get("seed_files") or []
        return f"seeded with {len(seeds)} paper(s)"
    if t == "extended" or t == "backfill":
        return (f'+{len(ev.get("added_files") or [])} → '
                f'{ev.get("n_before", 0)} → {ev.get("n_after", 0)} '
                f'(drift Δ={ev.get("centroid_shift_cos", 0):.3f})')
    if t == "renamed":
        return (f'{esc(ev.get("from", ""))} → {esc(ev.get("to", ""))} '
                f'(reason={esc(ev.get("reason", ""))}, drift={ev.get("drift_cos", 0):.3f})')
    if t == "merged_with":
        return f'merged with {esc(ev.get("other", ""))}'
    if t == "merged_into":
        return f'merged into {esc(ev.get("into", ""))}'
    if t == "split":
        into = ev.get("into") or []
        return (f'split into {len(into)} (kept {ev.get("kept", "?")} + '
                f'spawned {ev.get("spawned", "?")}, '
                f'silhouette={ev.get("silhouette", 0):.2f})')
    if t == "split_from":
        return (f'spawned from {esc(ev.get("parent", ""))} '
                f'(size {ev.get("size", "?")}, '
                f'silhouette={ev.get("silhouette", 0):.2f})')
    if t == "revived":
        return f'revived after dormancy'
    if t == "dormant":
        return f'marked dormant'
    if t == "bootstrapped":
        return f'seeded from themes_history.jsonl'
    return esc(str(ev))


def render_rollup_index() -> Path:
    summaries = load_daily_summaries()

    # Collect all daily HTML files, sorted newest first
    dailies = []
    for p in sorted(NEWS_DIR.glob("daily-*.html"), reverse=True):
        m = re.match(r"daily-(\d{4}-\d{2}-\d{2})\.html", p.name)
        if m:
            dailies.append((m.group(1), p.name))

    total_issues = len(dailies)
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M KST")

    # Latest issue card
    latest_html = ""
    if dailies:
        latest_d, latest_fname = dailies[0]
        try:
            d_obj = date.fromisoformat(latest_d)
            day_num = d_obj.day
            month_name = _MONTHS_EN[d_obj.month - 1]
            year_str = f"{d_obj.year} · {month_name[:3]}"
            dow_str = _DAYS_EN[d_obj.weekday()]
        except Exception:
            day_num, month_name, year_str, dow_str = "—", "—", latest_d, ""
        s = summaries.get(latest_d, {})
        headline = s.get("headline", "오늘의 논문")
        lede = s.get("lede", "")
        n_art = s.get("n_articles", "—")
        n_cat = s.get("n_categories", "—")
        n_th = s.get("n_themes", "—")
        n_recent_year = s.get("n_recent_year", "—")
        latest_html = (
            f'<a class="latest" href="{esc(latest_fname)}" role="link">'
            f'<div class="latest-date">'
            f'<div class="y">{esc(year_str)}</div>'
            f'<div class="d">{day_num}</div>'
            f'<div class="m">{esc(dow_str)}</div>'
            f'</div>'
            f'<div class="latest-body">'
            f'<div class="kicker">Today\'s Through-line</div>'
            f'<h3 class="latest-headline">{esc(headline)}</h3>'
            + (f'<p class="latest-lede">{esc(lede)}</p>' if lede else '') +
            f'<div class="latest-stats">'
            f'<div class="mini-stat"><b>{n_art}</b><span>articles</span></div>'
            f'<div class="mini-stat"><b>{n_cat}</b><span>categories</span></div>'
            f'<div class="mini-stat"><b>{n_th}</b><span>themes</span></div>'
            + (f'<div class="mini-stat"><b>{n_recent_year}</b><span>papers recent</span></div>'
               if n_recent_year != "—" and n_recent_year is not None else '') +
            f'</div></div>'
            f'<div class="latest-cta"><span>Read issue</span><span class="arr">→</span></div>'
            f'</a>'
        )

    # Archive rows (all issues, numbered from oldest=001)
    archive_rows = ""
    for iss_num, (d_str, fname) in enumerate(reversed(dailies), 1):
        try:
            d_obj = date.fromisoformat(d_str)
            month_abbr = _MONTHS_EN[d_obj.month - 1][:3]
            dow_abbr = _DAYS_EN[d_obj.weekday()][:3]
            date_display = f"{d_obj.month:02d} · {d_obj.day:02d}"
            small_display = f"{dow_abbr} · {month_abbr} {d_obj.day}"
        except Exception:
            date_display = d_str
            small_display = ""
            d_obj = None

        s = summaries.get(d_str, {})
        headline = s.get("headline", "")
        top_themes = _peek_day_themes(d_str)
        iss_title_text = headline or top_themes or d_str
        n_art = s.get("n_articles", "")
        n_cat = s.get("n_categories", "")
        meta_html = (
            f'<b>{n_art}</b> articles · <b>{n_cat}</b> cats<br>'
            if n_art else ""
        )

        archive_rows += (
            f'<a class="issue" href="{esc(fname)}">'
            f'<div class="iss-no">Vol. {d_obj.year if d_obj else "—"} · № {iss_num:03d}</div>'
            f'<div class="iss-date">{esc(date_display)}'
            + (f'<small>{esc(small_display)}</small>' if small_display else '') +
            f'</div>'
            f'<div class="iss-title"><b>{esc(iss_title_text)}</b></div>'
            f'<div class="iss-meta">{meta_html}<span class="iss-arr">→</span></div>'
            f'</a>'
        )

    # Reverse the numbering so newest = highest number
    # (rebuild with correct descending order display)
    archive_rows = ""
    for idx, (d_str, fname) in enumerate(dailies):
        iss_num = total_issues - idx
        try:
            d_obj = date.fromisoformat(d_str)
            month_abbr = _MONTHS_EN[d_obj.month - 1][:3]
            dow_abbr = _DAYS_EN[d_obj.weekday()][:3]
            date_display = f"{d_obj.month:02d} · {d_obj.day:02d}"
            small_display = f"{dow_abbr} · {month_abbr} {d_obj.day}"
            year_n = d_obj.year
        except Exception:
            date_display, small_display, year_n = d_str, "", "—"

        s = summaries.get(d_str, {})
        headline = s.get("headline", "")
        top_themes = _peek_day_themes(d_str)
        iss_title_text = headline or top_themes or d_str
        n_art = s.get("n_articles", "")
        n_cat = s.get("n_categories", "")
        meta_html = (
            f'<b>{n_art}</b> articles · <b>{n_cat}</b> cats<br>'
            if n_art else ""
        )
        archive_rows += (
            f'<a class="issue" href="{esc(fname)}">'
            f'<div class="iss-no">Vol. {year_n} · № {iss_num:03d}</div>'
            f'<div class="iss-date">{esc(date_display)}'
            + (f'<small>{esc(small_display)}</small>' if small_display else '') +
            f'</div>'
            f'<div class="iss-title"><b>{esc(iss_title_text)}</b></div>'
            f'<div class="iss-meta">{meta_html}<span class="iss-arr">→</span></div>'
            f'</a>'
        )

    if not archive_rows:
        archive_rows = '<div style="padding:20px;color:var(--mute)">아직 발행된 호가 없습니다.</div>'

    future_rows = ""
    if dailies:
        try:
            latest_date_obj = date.fromisoformat(dailies[0][0])
        except Exception:
            latest_date_obj = date.today()
        for i in range(1, 3):
            fd = latest_date_obj + timedelta(days=i)
            future_num = total_issues + i
            month_abbr = _MONTHS_EN[fd.month - 1][:3]
            dow_abbr = _DAYS_EN[fd.weekday()][:3]
            width = 65 if i == 1 else 48
            future_rows += (
                f'<div class="issue future" aria-hidden="true">'
                f'<div class="iss-no">Vol. {fd.year} · № {future_num:03d}</div>'
                f'<div class="iss-date skeleton-date">{fd.month:02d} · {fd.day:02d}'
                f'<small>{dow_abbr} · {month_abbr} {fd.day}</small></div>'
                f'<div class="iss-title"><div class="skeleton-line" style="width:{width}%"></div></div>'
                f'<div class="iss-meta">forthcoming</div>'
                f'</div>'
            )

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weekly Brief · Archive</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400;1,6..72,500&family=IBM+Plex+Sans+KR:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{INDEX_CSS}</style>
</head>
<body>
<div class="wrap">

  <div class="mast">
    <div class="mast-left">
      <div class="mast-title">Weekly Brief</div>
      <div class="mast-sub">Neuroscience · 주간 논문 요약</div>
    </div>
    <div class="mast-right">
      Archive · All issues<br>
      <b>Updated weekly</b>
    </div>
  </div>

  <div class="strap">
    <span><span class="dot"></span>Live feed · Rollup of weekly digests</span>
    <span>{total_issues} issue{"s" if total_issues != 1 else ""} published</span>
  </div>

  <div class="title-row">
    <h1 class="page-h">The <em>Archive</em></h1>
    <div class="page-sub">
      Weekly digests of new arrivals in<br>
      computational, clinical &amp; molecular neuroscience.<br>
      <b>Click any issue to read →</b>
    </div>
  </div>

  <div class="sec-head">
    <h2 class="sec-h">Latest Issue</h2>
    <div class="sec-meta">Most recent digest</div>
  </div>

  {latest_html or '<p style="color:var(--mute)">아직 발행된 호가 없습니다.</p>'}

  <div class="sec-head">
    <h2 class="sec-h">All Issues</h2>
    <div class="sec-meta">Chronological · Newest first</div>
  </div>

  <div class="archive">{archive_rows}{future_rows}</div>

  <div class="about">
    <div class="about-cell">
      <div class="h">What is this</div>
      <div class="t">
        <b>Weekly Brief</b>는 주간 단위로 수집된 신경과학 논문들의 요약과 공통 주제를
        정리한 아카이브입니다. 각 호는 직전 {BASELINE_DAYS}일을 baseline으로 주제 흐름을 감지합니다.
      </div>
    </div>
    <div class="about-cell">
      <div class="h">Cadence</div>
      <div class="t">
        주간 업데이트되는 PDF 기반 자동 브리프입니다. 카테고리, 주제, 요약은 로컬 아카이브를 기준으로 구성됩니다.
      </div>
    </div>
    <div class="about-cell">
      <div class="h">How to read</div>
      <div class="t">
        각 호는 <b>Through-line</b>으로 시작해 <b>Top Themes</b>와 카테고리별 논문 카드로 이어집니다.
        제목을 클릭하면 PDF 원문으로 이동합니다.
      </div>
    </div>
  </div>

  <footer>
    <span>Weekly Brief · Neuroscience Edition</span>
    <span>weekly_news.py · Window: this week · Baseline: {BASELINE_DAYS} days</span>
    <span>Generated {esc(gen_time)}</span>
  </footer>

</div>
</body>
</html>
"""
    out = NEWS_DIR / "index.html"
    out.write_text(html_doc, encoding="utf-8")
    log.info("rollup index → %s", out)
    return out


def _peek_day_themes(d: str) -> str:
    """Read themes_history.jsonl and grab up to 2 headline themes for the date."""
    if not THEMES_HISTORY_PATH.exists():
        return ""
    hot: list[tuple[float, str]] = []
    with THEMES_HISTORY_PATH.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("date") != d:
                continue
            if obj.get("status") in ("rising", "new") and obj.get("theme_name"):
                hot.append((obj.get("rising_score", 0), obj["theme_name"]))
    hot.sort(reverse=True)
    names = [name for _, name in hot[:2]]
    return " · ".join(names)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    target_date: date,
    since_hours: Optional[float],
    use_llm: bool,
    from_date: Optional[date] = None,
) -> None:
    t0 = time.time()
    log.info("=" * 60)
    log.info("weekly_news run  date=%s  from=%s  to=%s  since_hours=%s  llm=%s  articles_root=%s",
             target_date, from_date, target_date if from_date else None,
             since_hours, use_llm, ARTICLES_ROOT)

    today, recent = harvest(target_date, since_hours, from_date=from_date)

    if not today:
        log.warning("No new articles for %s — still refreshing rollup.", target_date)
        render_rollup_index()
        _log_run({"date": target_date.isoformat(), "today_count": 0,
                  "note": "no new articles"})
        return

    extract_abstracts(today + recent)
    embed_entries(today + recent)

    # Living-cluster registry — bootstrap on first run, then load
    from . import living_cluster as lcmod
    living_by_cat = lcmod.load_all_clusters(LIVING_CLUSTERS_DIR)
    if not living_by_cat:
        living_by_cat = lcmod.bootstrap_from_themes_history(
            THEMES_HISTORY_PATH,
            LIVING_CLUSTERS_DIR,
            window_days=BASELINE_DAYS,
            reference_date=target_date,
            before_date=target_date,
        )
        if living_by_cat:
            log.info("bootstrapped %d living clusters across %d categories",
                     sum(len(v) for v in living_by_cat.values()),
                     len(living_by_cat))

    # Orphan pool: re-evaluate previously unclustered papers against active LCs
    # BEFORE matching today's entries, so that a paper which looked lonely in an
    # earlier week can now be back-filled into a living cluster. Pool is
    # organised per category; non-current categories are left untouched until
    # their next activity.
    from . import orphan_pool as orphmod
    emb_cache_all = load_embeddings_cache()
    today_iso = target_date.isoformat()
    backfill_touched_by_uid: dict[str, "lcmod.LivingCluster"] = {}
    pool_by_cat: dict[str, list] = {}  # category -> list[OrphanRecord] AFTER this run

    clusters_by_cat: dict[str, list[Cluster]] = {}
    for cat, members in _group_by_cat(today).items():
        recent_same_cat = [e for e in recent if e.category == cat]
        living_same_cat = living_by_cat.get(cat, [])

        # (1) Load pool for this category & try back-fill absorption.
        pool = orphmod.load_pool(ORPHAN_POOL_DIR, cat)
        if pool:
            touched, absorbed = backfill_orphan_pool(
                cat, living_same_cat, pool, emb_cache_all, today_iso,
            )
            if absorbed:
                log.info("backfill: absorbed %d pool orphans into %d LCs (cat=%s)",
                         len(absorbed), len(touched), cat)
                pool = orphmod.drop(pool, absorbed)
            for lc in touched:
                backfill_touched_by_uid[lc.uid] = lc

        # (2) Hybrid clustering on today's entries (pool entries are not added
        # to HDBSCAN here — a future phase may extend this to include pool
        # orphans for co-clustering).
        cs = hybrid_cluster_category(members, recent_same_cat, living_same_cat)
        clusters_by_cat[cat] = cs

        # (3) Identify today's new orphans — today papers that remained
        # singletons and did not join any living cluster. These join the pool
        # for re-evaluation next week.
        today_orphan_fnames: list[str] = []
        for c in cs:
            if c.lineage in ("extended", "born"):
                continue
            # fresh singleton/noise — treat as orphan
            if c.n_today == 1:
                today_orphan_fnames.extend(e.fname for e in c.members_today)
        if today_orphan_fnames:
            pool = orphmod.touch(pool, today_orphan_fnames, today_iso, cat)

        # (4) TTL-prune and stash for later persistence.
        pool = orphmod.prune(pool, target_date)
        pool_by_cat[cat] = pool

    total_clusters = sum(len(v) for v in clusters_by_cat.values())
    log.info("naming %d clusters via Ollama=%s", total_clusters, use_llm)
    named = 0
    for cs in clusters_by_cat.values():
        for c in cs:
            name_cluster_llm(c, use_llm=use_llm)
            named += 1
            if named % 5 == 0:
                log.info("  … named %d / %d", named, total_clusters)

    for cs in clusters_by_cat.values():
        score_rising(cs)

    all_clusters_flat = [c for cs in clusters_by_cat.values() for c in cs]

    # Living-cluster: absorb + born (mutate in memory, don't save yet)
    touched_lcs = apply_living_updates(
        all_clusters_flat, living_by_cat,
        today_iso=target_date.isoformat(),
    )

    # Drift-gated rename — runs AFTER absorb so centroid reflects today's additions
    rename_events = rename_drifted_clusters(
        all_clusters_flat, living_by_cat,
        use_llm=use_llm,
        today_iso=target_date.isoformat(),
        decisions_log_path=DECISIONS_LOG_PATH,
    )
    if rename_events:
        log.info("renamed %d living clusters", len(rename_events))

    # Any LC touched by absorb/born OR by rename OR by backfill must be saved
    touched_by_uid = {lc.uid: lc for lc in touched_lcs}
    for uid, lc in backfill_touched_by_uid.items():
        touched_by_uid[uid] = lc
    for ev in rename_events:
        lcs = living_by_cat.get(ev["category"], [])
        for lc in lcs:
            if lc.uid == ev["uid"]:
                touched_by_uid[lc.uid] = lc
                break
    save_living_clusters(list(touched_by_uid.values()), living_by_cat)

    # Persist the updated orphan pools (one JSONL per category that saw
    # activity this run). Categories not in `pool_by_cat` keep whatever was
    # already on disk.
    for cat, records in pool_by_cat.items():
        try:
            orphmod.save_pool(ORPHAN_POOL_DIR, cat, records)
        except Exception as e:
            log.warning("orphan_pool save failed (%s): %s", cat, e)

    # Stamp each daily cluster with its LC's current 7-day growth rate so
    # renderers can surface rising themes without recomputing from members.
    from . import living_cluster as lcmod
    living_index: dict[str, "lcmod.LivingCluster"] = {}
    for lcs in living_by_cat.values():
        for lc in lcs:
            living_index[lc.uid] = lc
    for c in all_clusters_flat:
        if c.living_uid and c.living_uid in living_index:
            c.growth_rate_7d = lcmod.growth_rate_7d(
                living_index[c.living_uid], ref_date=target_date,
            )

    # Persist any newly generated per-paper TLDRs to the abstracts cache
    # so future runs don't re-call the LLM.
    persist_tldrs(today)

    save_theme_snapshot(target_date, [c for cs in clusters_by_cat.values() for c in cs])
    # Surface orphan counts. For categories touched today, use the in-memory
    # post-run state; for the rest, read whatever pool is still on disk.
    orphan_counts_by_cat = {cat: len(records) for cat, records in pool_by_cat.items()}
    on_disk = orphmod.load_all_pools(ORPHAN_POOL_DIR)
    for cat, recs in on_disk.items():
        if cat not in orphan_counts_by_cat:
            orphan_counts_by_cat[cat] = len(recs)
    render_daily_html(target_date, today, clusters_by_cat, use_llm=use_llm,
                      living_by_cat=living_by_cat,
                      emb_cache=emb_cache_all,
                      orphan_counts_by_cat=orphan_counts_by_cat)
    render_cluster_detail_pages(living_by_cat)
    render_orphan_index_pages()
    render_rollup_index()

    elapsed = time.time() - t0
    _log_run({
        "date": target_date.isoformat(),
        "from_date": from_date.isoformat() if from_date else None,
        "to_date": target_date.isoformat() if from_date else None,
        "today_count": len(today),
        "recent_count": len(recent),
        "clusters": total_clusters,
        "llm": use_llm,
        "elapsed_sec": round(elapsed, 1),
    })
    log.info("done in %.1fs", elapsed)


def _group_by_cat(entries: list[Entry]) -> dict[str, list[Entry]]:
    d: dict[str, list[Entry]] = defaultdict(list)
    for e in entries:
        d[e.category].append(e)
    return d


def run_consolidate(dry_run: bool = False) -> None:
    """Weekly housekeeping pass: merge near-duplicate clusters, mark stale ones
    dormant. Never runs from the daily pipeline — invoked via subcommand."""
    from . import living_cluster as lcmod
    from itertools import combinations

    today_iso = date.today().isoformat()
    by_cat = lcmod.load_all_clusters(LIVING_CLUSTERS_DIR)
    if not by_cat:
        log.info("consolidate: no living clusters")
        return

    touched: set[str] = set()
    merges = 0
    dormant_marked = 0
    splits = 0

    emb_cache = load_embeddings_cache()

    for cat, lcs in by_cat.items():
        # Only merge within category. Compare active clusters pairwise.
        actives = [l for l in lcs if l.status == "active" and l.centroid]
        # Sort by size desc so we keep the bigger side as winner
        actives.sort(key=lambda l: -l.size)
        # Track which have been absorbed this pass
        absorbed: set[str] = set()
        for a, b in combinations(actives, 2):
            if a.uid in absorbed or b.uid in absorbed:
                continue
            sim = lcmod.cosine_sim(a.centroid, b.centroid)
            decision = {
                "at": today_iso,
                "kind": "merge_check",
                "category": cat,
                "a": a.uid, "b": b.uid, "sim": round(sim, 4),
                "threshold": lcmod.TAU_MERGE,
            }
            if sim >= lcmod.TAU_MERGE:
                decision["outcome"] = f"merge:{a.uid}<-{b.uid}"
                if not dry_run:
                    lcmod.merge_into(b, a, today_iso)
                    touched.add(a.uid)
                    touched.add(b.uid)
                    absorbed.add(b.uid)
                    merges += 1
                log.info("merge %s ← %s (sim=%.3f)", a.uid, b.uid, sim)
            else:
                decision["outcome"] = "keep"
            try:
                _append_jsonl(DECISIONS_LOG_PATH, decision)
            except Exception:
                pass

        # Split sweep — try to cleave clusters whose members bifurcate cleanly.
        # Skip anything already merged-away in this pass.
        try:
            from sklearn.cluster import KMeans  # type: ignore
            from sklearn.metrics import silhouette_score  # type: ignore
            import numpy as np
        except Exception:
            KMeans = None  # type: ignore
            silhouette_score = None  # type: ignore

        if KMeans is not None and silhouette_score is not None:
            # Iterate over a snapshot since we may append to `lcs` below.
            for lc in list(lcs):
                if lc.status != "active":
                    continue
                if lc.uid in absorbed:
                    continue
                if lc.size < 2 * lcmod.SPLIT_MIN_SIZE:
                    continue
                embs = []
                missing = 0
                for m in lc.members:
                    fn = m.get("fname")
                    v = emb_cache.get(fn) if fn else None
                    if v is None:
                        missing += 1
                        break
                    embs.append(np.asarray(v, dtype=np.float32))
                if missing or len(embs) < 2 * lcmod.SPLIT_MIN_SIZE:
                    continue
                X = np.stack(embs)
                try:
                    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
                    labels = km.labels_.tolist()
                    if len(set(labels)) < 2:
                        continue
                    sil = float(silhouette_score(X, labels, metric="cosine"))
                except Exception as e:
                    log.debug("split kmeans failed for %s: %s", lc.uid, e)
                    continue

                n0 = sum(1 for l in labels if l == 0)
                n1 = len(labels) - n0
                decision = {
                    "at": today_iso,
                    "kind": "split_check",
                    "category": cat,
                    "uid": lc.uid,
                    "silhouette": round(sil, 4),
                    "sizes": [n0, n1],
                    "threshold": lcmod.SPLIT_SILHOUETTE,
                    "min_size": lcmod.SPLIT_MIN_SIZE,
                }
                if (sil >= lcmod.SPLIT_SILHOUETTE
                        and min(n0, n1) >= lcmod.SPLIT_MIN_SIZE):
                    decision["outcome"] = "split"
                    if not dry_run:
                        child = lcmod.split_cluster(
                            original=lc,
                            existing=lcs,
                            assignments=labels,
                            member_embeddings=embs,
                            today_iso=today_iso,
                            silhouette=sil,
                        )
                        if child is not None:
                            lcs.append(child)
                            touched.add(lc.uid)
                            touched.add(child.uid)
                            splits += 1
                            log.info("split %s → %s + %s (sil=%.2f, %d/%d)",
                                     lc.uid, lc.uid, child.uid, sil, n0, n1)
                        else:
                            decision["outcome"] = "split_aborted"
                else:
                    decision["outcome"] = "keep"
                try:
                    _append_jsonl(DECISIONS_LOG_PATH, decision)
                except Exception:
                    pass

        # Dormancy sweep
        for lc in lcs:
            if lc.status != "active":
                continue
            last = lcmod.last_activity_date(lc)
            if last is None:
                continue
            days = (date.today() - last).days
            if days >= lcmod.DORMANCY_DAYS:
                log.info("dormant %s (last activity %d days ago)", lc.uid, days)
                if not dry_run:
                    lcmod.mark_dormant(lc, today_iso)
                    touched.add(lc.uid)
                    dormant_marked += 1

    if not dry_run and touched:
        # Save all touched clusters
        lcs_to_save = []
        for cat, lcs in by_cat.items():
            for lc in lcs:
                if lc.uid in touched:
                    lcs_to_save.append(lc)
        save_living_clusters(lcs_to_save, by_cat)
        log.info("consolidate: %d merged · %d split · %d dormant · %d files updated",
                 merges, splits, dormant_marked, len(lcs_to_save))
    else:
        log.info("consolidate: merges=%d splits=%d dormant=%d (dry_run=%s)",
                 merges, splits, dormant_marked, dry_run)


def run_weekly_range(start_date: date, end_date: date, use_llm: bool) -> None:
    """Produce one weekly snapshot per 7-day window starting at ``start_date``.

    For ``start=2026-04-01`` the pages are dated 04-07 (covering 04-01..04-07),
    04-14 (04-08..04-14), 04-21, ... up to the last complete or partial week
    that ends on or before ``end_date``.
    """
    if start_date > end_date:
        log.warning("run_weekly_range: start (%s) after end (%s) — nothing to do",
                    start_date, end_date)
        return
    windows: list[tuple[date, date]] = []
    week_start = start_date
    while week_start <= end_date:
        week_end = min(week_start + timedelta(days=6), end_date)
        windows.append((week_start, week_end))
        week_start = week_end + timedelta(days=1)

    log.info("weekly range: %d windows from %s to %s",
             len(windows), start_date, end_date)
    for i, (ws, we) in enumerate(windows, 1):
        log.info("─── weekly window %d/%d: %s..%s ───", i, len(windows), ws, we)
        run(target_date=we, since_hours=None, use_llm=use_llm, from_date=ws)


def run_rebuild_clusters(force: bool = False, dry_run: bool = False) -> None:
    """Load every living cluster and rewrite its detail page. Also refreshes
    the per-category orphan index pages so the weekly snapshot's badges point
    somewhere valid."""
    from . import living_cluster as lcmod

    by_cat = lcmod.load_all_clusters(LIVING_CLUSTERS_DIR)
    total = sum(len(v) for v in by_cat.values())
    if not total:
        log.info("rebuild-clusters: no living clusters on disk")
        return

    if dry_run:
        stale = 0
        for lcs in by_cat.values():
            for lc in lcs:
                out = NEWS_DIR / f"cluster-{lc.uid}.html"
                if force or _cluster_page_is_stale(lc, out):
                    stale += 1
        log.info("rebuild-clusters (dry-run): %d/%d stale (force=%s)",
                 stale, total, force)
        return

    written = render_cluster_detail_pages(by_cat, force=force)
    orphan_pages = render_orphan_index_pages()
    log.info("rebuild-clusters: %d cluster pages, %d orphan pages",
             len(written), len(orphan_pages))


def _log_run(rec: dict) -> None:
    rec["ts"] = datetime.now().isoformat(timespec="seconds")
    with RUN_LOG_PATH.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_daily_summary(target_date: date, headline: str, lede: str,
                        n_articles: int, n_categories: int, n_themes: int,
                        n_recent_year: Optional[int] = None) -> None:
    records: dict[str, dict] = {}
    if DAILY_SUMMARIES_PATH.exists():
        for line in DAILY_SUMMARIES_PATH.read_text().splitlines():
            try:
                obj = json.loads(line)
                records[obj["date"]] = obj
            except Exception:
                continue
    records[target_date.isoformat()] = {
        "date": target_date.isoformat(),
        "headline": headline,
        "lede": lede,
        "n_articles": n_articles,
        "n_categories": n_categories,
        "n_themes": n_themes,
        "n_recent_year": n_recent_year,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with DAILY_SUMMARIES_PATH.open("w") as f:
        for obj in records.values():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_daily_summaries() -> dict[str, dict]:
    if not DAILY_SUMMARIES_PATH.exists():
        return {}
    records: dict[str, dict] = {}
    for line in DAILY_SUMMARIES_PATH.read_text().splitlines():
        try:
            obj = json.loads(line)
            records[obj["date"]] = obj
        except Exception:
            continue
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="weekly-news",
        description=__doc__.split("\n\n")[0] if __doc__ else "",
    )
    ap.add_argument("--articles-root", default=None,
                    help="Path to the Articles folder (contains index.json). "
                         "Defaults to $ARTICLES_ROOT or the current directory.")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--from", dest="from_date",
                    help="YYYY-MM-DD. Start date for a range ingest, inclusive.")
    ap.add_argument("--to", dest="to_date",
                    help="YYYY-MM-DD. End date for --from range ingest, inclusive. "
                         "Defaults to today.")
    ap.add_argument("--since-hours", type=float, default=None,
                    help="Ingest PDFs added in the last N hours instead of a calendar day")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip Ollama; use conservative title/category fallback labels")
    ap.add_argument("--consolidate", action="store_true",
                    help="Housekeeping pass: merge near-duplicate clusters and "
                         "mark stale ones dormant. Skips the daily run.")
    ap.add_argument("--rebuild-clusters", action="store_true",
                    help="Regenerate every cluster-<uid>.html from the current "
                         "living cluster registry. Skips the daily run.")
    ap.add_argument("--force-rebuild", action="store_true",
                    help="With --rebuild-clusters: bypass the dirty-check and "
                         "rewrite every cluster page regardless of mtime.")
    ap.add_argument("--dry-run", action="store_true",
                    help="With --consolidate or --rebuild-clusters: log only.")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    ensure_logger(args.verbose)

    articles_root = resolve_articles_root(args.articles_root)
    configure_paths(articles_root)

    if args.consolidate:
        run_consolidate(dry_run=args.dry_run)
        return

    if args.rebuild_clusters:
        run_rebuild_clusters(force=args.force_rebuild, dry_run=args.dry_run)
        return

    if args.from_date and args.since_hours is not None:
        ap.error("--from/--to cannot be combined with --since-hours")
    if args.to_date and not args.from_date:
        ap.error("--to requires --from")
    if args.from_date and args.date:
        ap.error("--from/--to cannot be combined with --date; use --to instead")

    # --from without --to → weekly cadence: publish one page per 7-day
    # window starting at --from, labeled by the window's end date. E.g.
    # `--from 2026-04-01` → pages at 04-07 (covers 04-01..04-07),
    # 04-14 (covers 04-08..04-14), 04-21, ... up to today.
    if args.from_date and not args.to_date and not args.date:
        start = datetime.strptime(args.from_date, "%Y-%m-%d").date()
        today_d = date.today()
        if start > today_d:
            ap.error("--from is in the future")
        run_weekly_range(start_date=start, end_date=today_d,
                         use_llm=not args.no_llm)
        return

    if args.from_date:
        target = datetime.strptime(args.to_date, "%Y-%m-%d").date() if args.to_date else date.today()
    elif args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target = date.today()
    from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date() if args.from_date else None
    if from_date and from_date > target:
        ap.error("--from must be on or before --to")
    run(target_date=target, since_hours=args.since_hours,
        use_llm=not args.no_llm, from_date=from_date)


if __name__ == "__main__":
    main()
