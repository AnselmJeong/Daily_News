다음 반복에서 얹을 것 (organic_cluster.md §6의 5번 이후): drift-gated rename, merge/split/dormant/revive (daily-news consolidate 서브커맨드), cluster-<uid>.html 상세 페이지, decision 로그.



# Organic Cluster — 설계안

매일의 HDBSCAN 결과를 버리지 않고, **category 단위로 persistent한 cluster**를 누적·변형시키는 레이어를 얹는다. 새 논문이 들어오면 "과거 cluster에 흡수되는지 / 새 cluster를 만드는지"를 먼저 판정하고, 흡수될 때마다 centroid가 조금씩 이동하며 cluster의 성격이 유기적으로 변하도록 한다. 사용자는 이 변형 과정(확장·분화·통합·개명·휴면·부활)을 매일의 리포트에서 직접 관찰할 수 있어야 한다.

Scope: 모든 matching·merge·split은 **같은 category 폴더 안에서만** 수행한다. 다른 category 간에는 절대 섞지 않는다.

---

## 1. 현재 상태 (baseline)

- `cluster_category()` — category별로 today + 최근 14일 embedding을 모아 HDBSCAN ([cli.py:471](src/daily_news/cli.py:471))
- `save_theme_snapshot()` — 그 날의 cluster들을 centroid와 함께 `.cache/themes_history.jsonl`에 append ([cli.py:763](src/daily_news/cli.py:763))
- `score_rising()` — 14일 baseline 대비 Poisson 잔차로 rising/new/active 라벨 ([cli.py:747](src/daily_news/cli.py:747))

문제: cluster_id는 매일 재생성되고, 14일 창 밖으로 벗어난 논문은 같은 주제라도 새 cluster로 보인다. 테마의 "일대기"가 없다.

---

## 2. 도입할 개념 — Living Cluster

기존의 "daily cluster"(일회성)와 구분해서, category마다 **영속 저장소**에 사는 cluster 레코드를 둔다.

```
.cache/living_clusters/
  <category slug>/
    <cluster_uid>.json        # cluster 한 건의 전체 lineage
  registry.json               # category → [uid, ...] 인덱스 (빠른 로드용)
```

### 2.1 Cluster record schema

```json
{
  "uid": "cogcomp-0007",
  "category": "Cognitive and Computational Neuroscience",
  "created_at": "2026-03-02T10:41:00",
  "updated_at": "2026-04-19T10:41:00",
  "status": "active",          // active | dormant | merged_into:<uid> | split

  "centroid": [384 floats],    // 현재 centroid
  "centroid_at_last_name": [...], // 최근에 이름 지었을 때의 centroid (drift 측정용)

  "members": [
    {"fname": "...", "added": "2026-03-02T...", "event_id": 0},
    ...
  ],

  "theme_name": "Interoception and predictive self-models",
  "theme_summary": "...",
  "keywords": ["interoception", "predictive coding", ...],

  "name_history": [
    {"at": "2026-03-02", "name": "Interoception in autism"},
    {"at": "2026-04-10", "name": "Interoception and predictive self-models",
     "reason": "drift", "drift_cos": 0.18}
  ],

  "events": [
    {"at": "2026-03-02", "type": "born", "seed_files": [...], "from": null},
    {"at": "2026-03-15", "type": "extended", "added_files": [...],
     "n_before": 3, "n_after": 5, "centroid_shift_cos": 0.04},
    {"at": "2026-04-10", "type": "renamed",
     "from": "Interoception in autism", "to": "...", "drift_cos": 0.18},
    {"at": "2026-04-18", "type": "merged_with", "other": "cogcomp-0011",
     "into": "cogcomp-0007"}
  ]
}
```

핵심은 **centroid + members + events 타임라인**. 이 셋만 있으면 cluster의 일생을 재생할 수 있다.

---

## 3. 매일 파이프라인 — 새로운 흐름

현재: `harvest → extract → embed → cluster_per_category → name → rising → snapshot → render`

변경 후 (category별로, 순서대로):

### Step A. **Load registry** — 이 category의 living cluster들 로드

### Step B. **Match today's papers to existing clusters**
오늘 도착한 논문 각각에 대해:
1. 해당 category의 모든 active cluster centroid와 cosine similarity 계산
2. `sim ≥ τ_join` (예: 0.60) → 그 cluster에 **흡수** 후보
3. 그 중 최고값만 남긴다 (한 논문은 하나의 cluster에만 속함)
4. `τ_candidate ≤ sim < τ_join` → "경계" — Step C의 HDBSCAN에서 다시 본다
5. `sim < τ_candidate` → "orphan" — Step C로

흡수된 논문은 `extended` 이벤트로 기록, centroid는 incremental update:
`centroid ← (centroid * n + new_emb) / (n + 1)`

### Step C. **Cluster the orphans** — 기존과 동일한 HDBSCAN
- 입력: 흡수되지 못한 오늘 논문 + 최근 14일 orphan들
- 결과 cluster 중 오늘 논문 ≥ 1건인 덩어리에 대해:
  - **기존 registry의 어떤 cluster와도 centroid-sim이 낮다** → 새 cluster UID 발급 → `born` 이벤트
  - 만약 HDBSCAN이 묶은 덩어리의 centroid가 이미 있는 dormant cluster와 `sim ≥ τ_revive`로 매치되면 → `revived` 이벤트로 되살림 (휴면 깨우기)

### Step D. **Detect structural changes** — registry 전체 재검토
- **Merge**: category 안 두 active cluster의 centroid-sim ≥ `τ_merge` (예: 0.85) → 작은 쪽을 큰 쪽으로 흡수, `merged_with` / `merged_into` 이벤트
- **Split**: 한 cluster 안에서 silhouette로 sub-structure가 뚜렷할 때 (sub ≥ 3건, sub-silhouette ≥ θ) → 2개로 쪼개고 둘 다 lineage(`parent`)를 남김
- **Dormant**: 마지막 `extended`/`born` 이후 N일 (예: 30일) 새 멤버 없음 → status `dormant`

Merge/split은 매일이 아니라 `--consolidate` 서브커맨드로 주기적(주 1회 등)으로만 돌려도 됨. 시작 단계에선 꺼두고 merge만.

### Step E. **Drift-gated renaming**
- 각 cluster의 `centroid`와 `centroid_at_last_name` 의 cosine distance가 `δ_drift` (예: 0.15) 초과이거나, 멤버 수가 이름 지을 당시보다 50% 이상 늘었다면 → LLM 재호출해서 이름 갱신
- 이름이 실제로 바뀌면 `renamed` 이벤트 + `name_history` push, `centroid_at_last_name` 갱신
- 그 외엔 기존 이름 유지 (LLM 호출 비용 절감)

### Step F. **Compute rising**
기존 점수와 병렬로, living cluster 기반의 새 지표도 계산:
- `growth_rate_7d` = 최근 7일에 추가된 멤버 수 / cluster 총 멤버 수
- `is_new_theme` = `born` 이벤트가 오늘인 cluster
- `is_reviving` = `revived` 이벤트가 오늘인 cluster

### Step G. **Persist**
- 각 cluster의 json 파일 rewrite (atomic: tmpfile + rename)
- `themes_history.jsonl`에도 기존대로 오늘치 snapshot 추가 — **cluster_id 자리에 living uid를 써서** 과거 daily 리포트와 lineage가 서로 연결되게 한다

### Step H. **Render**
daily HTML 상단에 "🌱 Cluster evolution today" 섹션 추가.

---

## 4. 사용자 가시성 — HTML 렌더링에서 보여줘야 할 것

### 4.1 daily-YYYY-MM-DD.html 최상단 highlight에 신설
- 🌿 **Born** (n개): "오늘 새로 형성된 주제"
- ➕ **Extended** (n개): "기존 주제에 논문이 더해짐" — "Interoception and predictive self-models +2 → 총 9편"
- 🔀 **Renamed** (n개): "before → after" + drift 값
- 💤→✨ **Revived** (n개): "3주 만에 돌아옴"
- 🔗 **Merged** / ✂️ **Split** (있을 때만)

### 4.2 각 category 섹션의 cluster 카드
- 카드 우상단에 **뱃지**: `NEW`, `+2 today`, `renamed`, `revived`
- 카드 하단에 **timeline 스파크라인**: 주차별 추가 논문 수(10주) — cluster가 꾸준히 크는지 한 번에 보임
- "이름 변천" 펼침: `name_history`를 옅게 타임스탬프와 함께

### 4.3 rollup index.html
- category마다 active living cluster 상위 N개 = 그 폴더의 "현재 지도"
- 각 cluster를 클릭하면 전용 페이지 `cluster-<uid>.html`
  - 전체 멤버 목록(시간순)
  - 이벤트 로그 (born / extended / renamed / merged / revived)
  - centroid drift 그래프 (선택)

---

## 5. 파라미터 초안 (실험으로 조정)

| 이름 | 기본값 | 의미 |
|------|-------|------|
| `τ_join` | 0.60 | 이 이상이면 기존 cluster에 흡수 |
| `τ_candidate` | 0.45 | 이 미만이면 완전 orphan |
| `τ_revive` | 0.65 | dormant cluster 부활 임계 |
| `τ_merge` | 0.85 | 두 cluster 통합 임계 |
| `δ_drift` | 0.15 | cos distance. 초과 시 재명명 |
| `split_min_size` | 3 | sub-cluster 최소 크기 |
| `split_silhouette` | 0.5 | sub-cluster 분리 임계 |
| `dormancy_days` | 30 | 휴면 판정 |
| `name_growth_ratio` | 1.5 | 멤버 1.5배 되면 재명명 고려 |

384-d MiniLM 기준 cosine similarity 분포를 먼저 실측해보고 값을 잡는 게 안전. 최초 며칠은 `--dry-run` 모드로 결정만 로그에 남기고 registry는 건드리지 않는 옵션이 필요.

---

## 6. 구현 단계 (제안 순서)

1. **Schema + storage 레이어**
   - `living_cluster.py` 신설: `LivingCluster` dataclass, load/save/atomic-write
   - registry 인덱스 (category → uid list)
2. **Bootstrap 마이그레이션** — 기존 `themes_history.jsonl` 최근 14일 snapshot을 읽어 living cluster 초기 상태 생성 (한 번만 실행)
3. **Matcher** — Step B: embedding vs registry centroids, `τ_join` 흡수 로직
4. **Orphan clusterer** — Step C: 기존 HDBSCAN을 orphan에만 적용, born/revived 판정
5. **Drift renamer** — Step E: 조건부 LLM 재호출, `name_history` 관리
6. **Snapshot 확장** — Step G: `themes_history.jsonl`의 `cluster_id`를 living uid로 통일, 이벤트도 같이 기록
7. **Renderer 확장** — Step H + §4: daily highlight, 카드 뱃지, sparkline
8. **(나중에) Consolidator** — Step D: merge / split / dormant는 별도 서브커맨드 (`daily-news consolidate`)로 분리, 주 1회 cron
9. **Cluster 상세 페이지** `cluster-<uid>.html` — timeline + members
10. **관측/튜닝** — 판정 로그(`.cache/cluster_decisions.jsonl`)에 각 논문이 어느 cluster로 갔고 왜 그랬는지 남겨서 사후 검증

단계 1~4까지만 돼도 "오늘 논문이 과거 cluster에 흡수되어 확장됐다"는 이야기를 할 수 있다. 5~7은 사용자 가시성, 8 이후는 유기체로서의 완성도.

---

## 7. Edge cases / 조심할 것

- **카테고리 이동**: 사용자가 PDF를 다른 폴더로 옮기면, 원 cluster에서 그 멤버를 제거하고 새 카테고리에서 matching을 다시 돌려야 한다. `index.json`과 registry를 하루 한 번 reconcile하는 단계 필요.
- **파일 삭제**: 멤버가 실제 디스크에 없으면 cluster에서 빼되 lineage에는 `removed` 이벤트를 남긴다.
- **centroid incremental drift**: 평균 업데이트는 노이즈를 누적시킬 수 있음. 주기적으로 실제 멤버 embedding으로 centroid를 재계산하는 "rebuild" 패스가 안전.
- **멤버 1명짜리 cluster**: `born`은 최소 2건 orphan이 HDBSCAN으로 묶일 때만. 1건은 계속 orphan 상태로 14일 창 안에 남김.
- **재현성**: LLM 호출이 낄 때마다 이름이 달라질 수 있으므로, `renamed` 판단은 반드시 drift/성장 조건을 먼저 통과해야 호출까지 간다.
- **비용**: Step B는 category당 O(today × clusters) — 작음. Step D의 pairwise merge 검사만 cluster 수에 제곱이므로 consolidate 전용에 둔다.

---

## 8. 오늘 당장 손댈 수 있는 최소 변경 (MVP)

가장 작은 만족스러운 단위:

1. living cluster json 저장소 + bootstrap
2. Step B (join) + Step C (born) — merge/split/rename은 일단 생략
3. daily HTML에 "Extended / Born" 두 줄만 추가
4. cluster 카드 제목 옆에 `NEW` / `+N today` 뱃지

이 정도만 해도 "과거 cluster에 오늘 논문이 붙었다"는 핵심 서사가 사용자 눈에 보인다. Rename·merge·split·revive는 이후 반복에서 얹는다.
