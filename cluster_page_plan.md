# Cluster Page 분리 구현 계획

## 배경과 목표

파이프라인은 **주 1회 실행** (weekly snapshot) 기준이다. 현재 snapshot HTML은 모든 cluster의 멤버 논문(최대 6개) + singleton orphan까지 한 페이지에 인라인으로 렌더링하여 용량이 크고 열람이 어렵다.

**용어**: 이하 모든 "snapshot"은 weekly snapshot을 의미하며, 파일명 규약은 `weekly-YYYY-MM-DD.html` (해당 주의 실행 날짜 또는 주 종료일 기준)로 변경한다. 내부적으로는 `run_date` 라벨 하나로 충분하며 cadence는 렌더링/표기 문제.

**목표:**
1. Weekly snapshot page는 **cluster 요약 카드만** 표시 (논문 리스트 제거)
2. Cluster card 클릭 → 해당 cluster 전용 page로 이동 (이미 존재하는 `cluster-<uid>.html` 확장)
3. Cluster page는 **주(week)별로 묶어서** 멤버 논문을 표시 (responsive 1–3 cols). 섹션 헤더는 `📅 Week of 2026-04-19 (added: N)` 형식.
4. Orphan 논문은 카테고리별 "orphan index" page에 모아두고, 주별 count badge로 접근
5. **과거 orphan을 미래 snapshot의 clustering에 포함** → 나중에라도 cluster로 승격 가능

---

## 현재 구조 (참고)

| 요소 | 위치 |
|---|---|
| Weekly(=기존 daily) page 렌더링 | [cli.py:1557](src/daily_news/cli.py:1557) |
| Cluster detail 렌더링 | [cli.py:2175](src/daily_news/cli.py:2175), `_render_single_cluster_page` [cli.py:2188](src/daily_news/cli.py:2188) |
| Hybrid clustering | `hybrid_cluster_category` [cli.py:613](src/daily_news/cli.py:613) |
| Living cluster 저장소 | `_news/.cache/living_clusters/<category-slug>/<uid>.json` |
| Orphan 처리 | [cli.py:664](src/daily_news/cli.py:664), [cli.py:705](src/daily_news/cli.py:705) |
| τ_join / τ_revive | 0.60 / 0.65 |
| Article ID | 파일명 (`YYYY - Author - Title.pdf`), `index.json` 기준 |

---

## 실현 가능성 분석

### ✅ 강점 (이미 확보된 것)
- **Article identity가 파일명으로 안정적** → orphan을 보관했다가 나중에 재사용해도 중복/충돌 없음
- **Living cluster JSON이 누적식**이라 매 실행 시 cluster page를 regenerate하는 것이 이미 기본 동작
- **`members[].added` 필드**에 추가된 날짜(= weekly snapshot 실행일)가 기록되어 있어 주별 grouping 자료가 이미 존재
- **Rollup index** ([cli.py:2334](src/daily_news/cli.py:2334))와 onclick navigation 패턴이 이미 있음

### ⚠️ 주의할 점
- Cluster page 수가 많아지면 regenerate 비용 증가 → **변경된 cluster만** regenerate하도록 dirty-flag 도입 필요
- Orphan pool이 무한정 쌓이면 HDBSCAN이 느려짐 → **TTL(예: 8주)** 필요
- 이전 daily snapshot HTML은 불변으로 두는 것이 맞음 (archival property). 사용자 구상과 일치.

### ❌ 하지 않아도 되는 것
- 과거 daily snapshot HTML 재생성 (링크만 cluster page로 가므로 그대로 둬도 됨)
- Article에 별도 UUID 발급 (파일명으로 충분)

---

## 구현 단계

### Phase 1 — Orphan Carry-Forward
**파일**: `src/daily_news/cli.py`, `src/daily_news/living_cluster.py` (또는 신규 `orphan_pool.py`)

1. `_news/.cache/orphan_pool/<category-slug>.jsonl` 저장소 신설
   - 레코드: `{fname, embedding_ref, first_seen, last_seen, attempts}`
   - Embedding은 `embeddings.npz`에서 참조 (중복 저장 피함)
2. `hybrid_cluster_category`에 orphan pool 병합 단계 추가:
   - **Step B' (τ_join 재시도)**: 오늘 논문을 매칭하기 전에, pool의 기존 orphan들도 centroid에 대해 재평가. sim ≥ τ_join이면 해당 living cluster에 **"extended" (back-fill)** 로 추가, event에 `{type: "backfill", fname, original_date}` 기록
   - **Step C' (확장 HDBSCAN)**: 오늘의 잔여 orphan + pool 내 orphan을 함께 HDBSCAN. 새 cluster가 태어나면 멤버 `added` 필드에 **원래 등장 날짜**를 보존
3. Pool 정리: 8주 이상 `last_seen`이 오래된 orphan 제거 (TTL)
4. Parameter: `ORPHAN_TTL_WEEKS = 8`, `ORPHAN_POOL_MAX = 500/category`

### Phase 2 — Cluster Page를 Snapshot 그룹 뷰로 개편
**파일**: `cli.py` `_render_single_cluster_page` 재설계

기존: 전체 members 플랫 리스트 (newest first)
신규:
```
[Header: theme name, summary, keywords, sparkline]
[Timeline events: born / extended / renamed / revived]

📅 Week of 2026-04-19 (added: 3)
  ┌────────────┬────────────┬────────────┐
  │ paper card │ paper card │ paper card │  ← responsive 1/2/3 cols
  └────────────┴────────────┴────────────┘

📅 Week of 2026-04-12 (added: 2)
  ...
```
- `members`를 `added`(= weekly run 날짜) 기준으로 groupby. 이미 주 1회 실행이므로 날짜가 곧 주를 식별
- Paper card: 제목, 저자, abstract 첫 1~2문장, PDF 링크
- CSS: `grid-template-columns: repeat(auto-fill, minmax(320px, 1fr))`
- 날짜 섹션을 접을 수 있게 `<details>` 사용 (최신 1~2개만 기본 펼침)

### Phase 3 — Weekly Snapshot Page 슬림화
**파일**: `cli.py:1557` 근처 템플릿

1. Cluster card에서 "최대 6개 논문 리스트" 제거 → **카운트 badge**로 대체
   - 예: `📄 12 papers (+3 this week)` 형태, 클릭 시 `cluster-<uid>.html`로 이동
   - Sparkline, theme name, summary, keywords는 유지 (요약 정보)
2. Singleton/orphan 섹션도 인라인 렌더링 제거 → **orphan index page**로 링크
   - 카테고리 카드에 `🗂 15 unclustered papers` 형태 badge 추가

### Phase 4 — Orphan Index Page
**신규 출력**: `_news/orphans-<category-slug>.html`

- 해당 카테고리의 현재 orphan pool을 **주별 grouping** (Phase 2와 동일한 레이아웃)
- 매 weekly run 시 regenerate (pool 전체 반영)
- 카드에는 "pool에 들어온 주" + "attempts 횟수"(= 몇 주 동안 살아남았는지) 부차 정보
- 미래에 cluster로 승격되면 이곳에서 사라짐 (원래 주의 weekly page에는 여전히 orphan으로 남아있지만, 사용자 구상대로 과거는 건드리지 않음)

### Phase 5 — 증분 Regeneration & 링크 보수
1. `_render_single_cluster_page`를 호출하기 전 **dirty check**:
   - 해당 cluster의 `updated_at` > 마지막 HTML mtime → regenerate
   - 그 외는 skip
2. Daily page의 cluster card `onclick`은 이미 `cluster-<uid>.html`을 가리킴 ([cli.py:1655](src/daily_news/cli.py:1655)) — 경로 그대로 유지
3. Cluster page에서 **"이 cluster가 등장한 주들"** 섹션 추가: `members` 날짜들을 unique화해서 weekly page들로 back-link (breadcrumb)

### Phase 6 — 테스트 & 마이그레이션
1. `uv run daily-news rebuild-clusters --dry-run` 같은 플래그로 기존 living cluster에서 cluster page만 일괄 재생성
2. Snapshot fixture 기반 회귀 테스트 (최소 2주치 데이터로 orphan → cluster 승격 시나리오)

---

## 스키마 변경 요약

**Living cluster JSON** (`<uid>.json`):
- `events[]`에 새 타입 `backfill` 추가
- `members[].added`는 그대로 (원래 날짜 보존 — back-fill된 경우에도 원래 등장일 사용)

**신규**: `_news/.cache/orphan_pool/<category-slug>.jsonl`
```json
{"fname": "2026 - ... .pdf", "first_seen": "2026-04-05", "last_seen": "2026-04-19", "attempts": 3}
```
(주 1회 실행 기준 `attempts`는 이 orphan이 몇 주 연속으로 재평가되었는지를 의미)
Embedding은 `embeddings.npz` 재사용 (fname → index 매핑).

---

## 열린 질문 → 결정 사항 (2026-04-19)

1. **τ_join 완화**: orphan을 최대한 줄이기 위해 `TAU_JOIN`을 **0.60 → 0.50** 으로 낮춘다. back-fill 재평가에도 동일 값 적용 (별도 중간 단계 불필요). `TAU_REVIVE`도 상응해서 조정할지 여부는 실제 데이터로 판단 — 일단 0.65 유지하고 dormant 부활이 너무 드물면 0.55로 재검토.
   - 주의: 0.50은 "느슨한 연관" 수준이라 false-positive 흡수(주제가 살짝 다른 논문이 섞이는 현상)가 늘 수 있음. Cluster centroid가 점점 모호해질 위험 → 주기적으로 "centroid drift" 모니터링 필요 (이미 있는 `name_history.drift` 지표 활용).
2. **Orphan page 월별 분할**: orphan이 계속 누적되므로 `orphans-<category-slug>-YYYY-MM.html` 월별 페이지로 분할. 카테고리별 index에서 월을 고를 수 있게 함. Cluster로 승격된 orphan은 해당 월 페이지에서 제거.
3. **Weekly page cluster card 단순화**: 멤버 논문을 **centroid 기준 cosine sim 상위 3편만** 미리보기로 표시 ("대표 논문"). 클릭 시 cluster 전용 page로 이동하여 전체 이력 열람. 출판일 기반이 아니라 centroid 대표성 기반이므로 "this week's new N"은 표시하지 않음 (성장 여부는 sparkline과 `+N` badge로만).
4. **Back-fill UI 표기 불필요**: cluster evolution block에 back-fill 이벤트를 별도로 구분해서 보여주지 않는다. 사용자 입장에서는 "cluster가 이번 주에 성장했다"는 사실만 중요하므로 extended 이벤트와 동일하게 취급. 내부 `events` 로그에는 `type: backfill`로 남기되 UI에서는 구분 없음.

---

## 권장 순서

Phase 1 → 2 → 3 → 4 → 5 → 6. Phase 1과 2는 독립적이라 병렬 가능하지만, 통합 테스트는 Phase 3 완료 후 일괄 수행 권장.
