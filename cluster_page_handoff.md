# Cluster Page Refactor — Handoff (Phase 3 이후)

원본 계획: [cluster_page_plan.md](cluster_page_plan.md) — 반드시 먼저 읽을 것.
Worktree: `.claude/worktrees/happy-margulis-b2ef32` (branch `claude/happy-margulis-b2ef32`).

---

## ✅ 완료 (Phase 1 + 2)

### Phase 1 — Orphan carry-forward (commit 없음, working tree 상태)
- **신규**: [src/daily_news/orphan_pool.py](src/daily_news/orphan_pool.py)
  - `OrphanRecord` 데이터클래스, `load_pool/save_pool/load_all_pools`,
    `touch/drop/prune`, `group_by_month` (Phase 4 재사용), `pool_path`.
  - 상수: `ORPHAN_TTL_WEEKS = 8`, `ORPHAN_POOL_MAX = 500`.
  - 저장소: `_news/.cache/orphan_pool/<slug>.jsonl`.
- [src/daily_news/living_cluster.py:32](src/daily_news/living_cluster.py:32) —
  `TAU_JOIN` 0.60 → **0.50** (plan §열린 질문-1). `TAU_REVIVE`는 0.65 유지.
- [src/daily_news/living_cluster.py:368](src/daily_news/living_cluster.py:368) —
  `absorb_backfill(lc, triples, today_iso)` 추가. 원래 `first_seen` 날짜를
  `members[].added`로 보존하고 `events[]`에 `type: "backfill"` 추가.
- [src/daily_news/cli.py:58](src/daily_news/cli.py:58) — `ORPHAN_POOL_DIR` 전역 경로 등록.
- [src/daily_news/cli.py:733](src/daily_news/cli.py:733) —
  `backfill_orphan_pool(category, living, pool, emb_cache, today_iso)` 헬퍼.
  pool 각 레코드를 LC centroid에 대해 τ_join 재평가 → 흡수. 감사 로그는
  `DECISIONS_LOG_PATH`(`kind: "backfill"`)에 기록.
- [src/daily_news/cli.py:2782](src/daily_news/cli.py:2782) — `run()` 카테고리 루프에
  (1) pool 로드 (2) backfill (3) hybrid_cluster_category (4) today-orphan pool
  등록 (5) TTL prune 순서로 통합. 저장 단계에 backfill-touched LC + pool 병합.

**범위 축소**: plan §Phase 1 step 2의 "Step C' 확장 HDBSCAN (pool을 today와 함께
HDBSCAN)"은 Entry stub 재구성 복잡도 때문에 **미구현**. Phase 3+ 도중 필요하면
다시 검토할 것. 현재는 backfill(pool→기존 LC 재매칭)만 작동.

### Phase 2 — 주별 그룹 cluster page
- [src/daily_news/cli.py:2299](src/daily_news/cli.py:2299)
  `render_cluster_detail_pages`가 `abstracts_cache`를 한 번 로드해서 주입.
- [src/daily_news/cli.py:2312](src/daily_news/cli.py:2312)
  `_render_single_cluster_page(lc, abstracts_cache=None)` — 멤버를
  **월요일-앵커 주**로 그룹, 최근 2주만 `<details open>`, 각 논문은
  카드(제목 + 저자 + 초록 220자 스니펫)로 `grid-template-columns:
  repeat(auto-fill, minmax(320px, 1fr))` 배치.
- [src/daily_news/cli.py:2282](src/daily_news/cli.py:2282) CSS 개편
  (`.members-weekly/.week-group/.paper-grid/.paper-card` 등). 기존 flat
  `.members/.member` 클래스는 제거됨.
- [src/daily_news/cli.py:2441](src/daily_news/cli.py:2441) `_describe_event`에
  `backfill` 분기 추가(extended와 동일 포맷). CSS에서도 `.event-type.backfill`은
  `.extended`와 동일 스타일 (plan §열린 질문-4: UI 구분 없음).

---

## ⏳ 남은 과제

### Phase 3 — Weekly snapshot page 슬림화
**파일**: [src/daily_news/cli.py](src/daily_news/cli.py), `render_daily_html` (line 1562 부근).

**현재 동작** (삭제/변경 대상):
- Cluster card 내부에 `[:6]` 멤버 리스트 인라인 렌더링 — cli.py 1853 근처.
- Singleton/orphan 논문 섹션 인라인 렌더링 — cli.py 1829~1843 근처.
- sparkline 주간 히스토그램 — `_sparkline_for()` (cli.py 1773~1795).

**목표 동작** (plan §Phase 3 + §열린 질문-3):
1. Cluster card는:
   - theme name / summary / keywords / sparkline 유지
   - 멤버 논문 리스트 제거 → **centroid 기준 cosine sim 상위 3편** 미리보기 카드
     ("대표 논문"). 제목 + 저자만. 클릭 시 해당 paper PDF 링크.
   - count badge: `📄 N papers (+K this week)` (K는 `c.n_added_today`).
     클릭 시 `cluster-<uid>.html`로 이동 (이미 존재하는 링크 [cli.py:1655](src/daily_news/cli.py:1655) 재사용).
2. 대표 논문 선정: 해당 cluster의 centroid와 각 멤버(LC 기준이면 LC members의
   fname → emb_cache 조회) cosine sim 상위 3. LC가 없는 fresh 클러스터는
   `members_today` 기준 top-3. centroid가 없으면 그냥 newest 3 fallback.
3. Singleton/orphan 인라인 섹션 제거. 카테고리 헤더 옆에
   `🗂 N unclustered papers` 형태 badge + `orphans-<slug>.html` (Phase 4 출력)
   링크. count는 orphan_pool + 오늘 unclustered singleton 합.
4. "this week's new N"은 표시하지 않음 (성장 여부는 sparkline + `+N` badge만).

**힌트**:
- Cluster card HTML 블록 찾기: `grep -n "cluster · " src/daily_news/cli.py`
  (한국어 badge 문자 `cluster · N편` 근처 [cli.py:1876-1887](src/daily_news/cli.py:1876)).
- sim 계산 헬퍼: `living_cluster.cosine_sim(a, b)` 재사용.
- top-3 선정 헬퍼를 `cli.py`에 `_top_representative_papers(c, emb_cache, k=3)` 로 추가 권장.
- 대표 논문 카드 CSS는 Phase 2에서 만든 `.paper-card/.paper-title/...`와 통일.
  (weekly page CSS는 `CLUSTER_PAGE_CSS`가 아니라 daily HTML 템플릿 내부 `<style>`
  블록 — 따로 추가 필요.)

### Phase 4 — Orphan index page (월별 분할)
**신규 출력**: `_news/orphans-<slug>-YYYY-MM.html`, 그리고 카테고리별 인덱스
페이지 `_news/orphans-<slug>.html` (월 선택 UI).

**파일**: [src/daily_news/cli.py](src/daily_news/cli.py), 신규 함수.

**구현 포인트** (plan §Phase 4 + §열린 질문-2):
1. `orphan_pool.load_all_pools(ORPHAN_POOL_DIR)` → 카테고리별 records.
2. `orphan_pool.group_by_month(records)` → `{YYYY-MM: [records]}`.
3. 월별 페이지:
   - Phase 2와 **동일한 주별 `<details>` + paper-card 레이아웃 재활용**.
   - 각 카드에 "pool 진입 주" (`first_seen`) + `attempts` 배지.
   - 초록은 `load_abstracts_cache()` 참조.
4. 카테고리 인덱스 페이지: 월 목록 + 각 월 orphan 개수.
5. `run()` 마지막에 `render_orphan_index_pages()` 호출.
6. Cluster로 승격된 orphan은 이미 `run()`에서 `orphmod.drop(...)`으로 pool에서
   빠졌으므로 자동으로 사라짐.

**파일명 규칙**: `orphan_pool._slug(cat)` 재사용.

### Phase 5 — Dirty-check 증분 regeneration
**파일**: [src/daily_news/cli.py:2299 `render_cluster_detail_pages`](src/daily_news/cli.py:2299).

현재는 매 실행마다 모든 LC의 cluster page를 재생성. LC 수가 늘면 느려짐.

**변경**:
```python
def _cluster_page_is_stale(lc, out_path) -> bool:
    if not out_path.exists():
        return True
    try:
        mtime = datetime.fromtimestamp(out_path.stat().st_mtime)
        updated = datetime.fromisoformat(lc.updated_at)
        return updated > mtime
    except Exception:
        return True
```
`_render_single_cluster_page` 호출 전 체크. `--force-rebuild` CLI 플래그로
override 가능하게 (Phase 6과 연계).

### Phase 6 — `rebuild-clusters` CLI + breadcrumb back-links
**파일**: [src/daily_news/cli.py:3016 `main()`](src/daily_news/cli.py:3016),
`_render_single_cluster_page`, `__main__.py`.

1. `main()` argparse에 `--rebuild-clusters` 플래그 추가. 분기:
   - `load_all_clusters` → 전체 LC에 대해 `_render_single_cluster_page` 호출
     (dirty-check 우회). `run()` 없이 종료. Phase 2의 새 레이아웃으로
     일괄 재생성 용도.
   - 선택적으로 `--dry-run`과 결합 → 몇 개가 stale인지만 로그.
2. Cluster page에 breadcrumb 섹션 추가 (plan §Phase 5-3):
   - `lc.members` 날짜들을 주별로 unique화 → `weekly-YYYY-MM-DD.html`로 링크.
   - 현재 weekly 파일명 규칙은 `daily-YYYY-MM-DD.html` ([cli.py:1493](src/daily_news/cli.py:1493) 근처)임.
     **plan에서는 `weekly-YYYY-MM-DD.html`로 변경하라고 했지만** 변경 시
     기존 rollup/index에서 `daily-*.html` glob 쓰는 곳이 여러 군데
     ([cli.py:2472](src/daily_news/cli.py:2472) 등) — 마이그레이션 비용 큼.
     **권장**: 일단 `daily-*.html` 유지하고 breadcrumb 링크만 맞추기. 이름
     규칙 변경은 별도 과제로 분리.

---

## 🧭 실행/검증 순서 (새 context에서)

```bash
cd /Volumes/Aquatope/_DEV_/Daily_News/.claude/worktrees/happy-margulis-b2ef32

# 1. 현재 상태 확인
git status
git diff --stat

# 2. 문법 체크 (빠름)
python -c "import ast; [ast.parse(open(f).read()) for f in ['src/daily_news/cli.py','src/daily_news/living_cluster.py','src/daily_news/orphan_pool.py']]; print('OK')"

# 3. 계획 + 완료 현황 읽기
cat cluster_page_plan.md
cat cluster_page_handoff.md   # 이 파일

# 4. 실제 파이프라인 실행(선택) — 기존 articles root 세팅이 있어야 함
# uv run daily-news --date 2026-04-19
```

## ⚠️ 주의사항

1. **Phase 1의 Step C' 미구현** — 미래에 pool 멤버끼리 새 cluster를 형성시키고
   싶다면, `cluster_category`에 Entry stub을 흘려보내는 방식 + `create_born`에
   per-member `added` 날짜 보존 파라미터 추가가 필요.
2. **TAU_JOIN 0.50 리스크** — centroid drift 모니터링 필요. LC의 `name_history`에
   이미 drift cos가 기록되므로 `run_consolidate` 쪽에 경고 로직 추가 고려.
3. **CSS 공유** — Phase 3의 대표 논문 카드는 weekly daily HTML template 내부
   `<style>` 블록에 추가해야 함. `CLUSTER_PAGE_CSS`는 cluster detail page
   전용이라 공유되지 않음. Grep: `grep -n "<style>" src/daily_news/cli.py`.
4. **`abstracts_cache`가 None일 수 있는 경로** — Phase 4 orphan 페이지에서도
   방어적으로 체크. Phase 2는 이미 체크 중.

---

## 📋 Todo (복원용)

새 session에서 TodoWrite로 다음 그대로 등록:

```
1. Phase 3: Slim weekly snapshot page (top-3 preview + badge, no inline paper lists)
2. Phase 4: Monthly orphan index pages (orphans-<cat>-YYYY-MM.html)
3. Phase 5: Dirty-check incremental regen for cluster pages
4. Phase 6: rebuild-clusters CLI flag + breadcrumb back-links to weekly pages
```
