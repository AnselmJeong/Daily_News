# Daily News — 뉴로사이언스 논문 데일리 브리프

`Articles/` 폴더에 새로 추가되는 PDF들을 모아 매일 한 장의 HTML 다이제스트로
만들어 주는 파이프라인. 단순한 요약을 넘어, **카테고리별로 영속되는 "Living
Cluster"** 레이어를 유지해서 한 번 형성된 주제가 어떻게 확장·개명·휴면·부활·병합
되는지를 시간에 따라 추적한다.

- 실행 명령: `daily-news`
- 매일 산출: `_news/daily-YYYY-MM-DD.html`
- 롤업: `_news/index.html` (최근 30일)
- 클러스터 상세 페이지: `_news/cluster-<uid>.html`
- 영속 상태: `_news/.cache/living_clusters/<category>/<uid>.json`

---

## 목차

1. [개요](#1-개요)
2. [디렉터리 구조](#2-디렉터리-구조)
3. [설치](#3-설치)
4. [Ollama 모델 설정](#4-ollama-모델-설정)
5. [매일 사용하기](#5-매일-사용하기)
6. [자동화](#6-자동화)
7. [Organic Cluster 운영](#7-organic-cluster-운영)
8. [산출 HTML 해석](#8-산출-html-해석)
9. [파라미터 튜닝](#9-파라미터-튜닝)
10. [트러블슈팅](#10-트러블슈팅)
11. [개발자 노트](#11-개발자-노트)

---

## 1. 개요

### 파이프라인 단계

```
harvest → extract → embed → (join | born) → name → rename → rising → persist → render
```

| 단계 | 내용 |
|------|------|
| **Harvest** | `index.json`의 오늘자 엔트리 + 루트에 있는 미분류 PDF의 `mtime`. |
| **Extract** | PyMuPDF로 처음 3페이지에서 title + abstract 추출. `abstracts.jsonl`에 캐시(파일당 1회). |
| **Embed** | `sentence-transformers/all-MiniLM-L6-v2`(384-d, normalized). `embeddings.npz`에 캐시. |
| **Join** | 오늘 논문 각각을 카테고리 내 기존 Living Cluster centroid와 cosine 매칭. sim ≥ τ_join(0.60)이면 흡수(= `extended`). dormant LC는 τ_revive(0.65)로 부활. |
| **Born** | 흡수되지 못한 orphan들에 대해 HDBSCAN을 최근 14일 문맥과 함께 돌림. ≥ 2편이 묶이면 새 Living Cluster 생성(= `born`). |
| **Name** | 각 클러스터에 Ollama(기본: `minimax-m2.7:cloud`)로 한국어 theme name + summary + keywords 부여. |
| **Rename** | 기존 LC의 centroid가 `centroid_at_last_name` 대비 cosine distance δ_drift(0.15) 넘거나 멤버 수가 1.5배가 됐으면 LLM 재호출로 개명. `name_history` 기록. |
| **Rising** | 14일 baseline 대비 Poisson 잔차로 `rising`/`active`/`new` 라벨. |
| **Persist** | 터치된 LC json 파일 atomic 저장 + registry index 갱신. |
| **Render** | `daily-YYYY-MM-DD.html`, `cluster-<uid>.html`, `index.html`. |

### 운영 루프

- **매일**: `daily-news` — 위 전체 파이프라인.
- **주 1회(권장)**: `daily-news --consolidate` — 전역 housekeeping.
  - 같은 카테고리 내 두 active LC centroid-sim ≥ τ_merge(0.85) → merge.
  - 마지막 활동이 30일 지난 LC → `dormant`.

---

## 2. 디렉터리 구조

```
Articles/                              # ← ARTICLES_ROOT
├── index.json                         # 카테고리 → 파일명 매핑
├── <Category 1>/
│   └── *.pdf
├── <Category 2>/
│   └── *.pdf
├── _uncategorized/                    # 아직 분류 안 된 PDF들
└── _news/                             # 산출물 + 캐시
    ├── index.html                     # 롤업 페이지
    ├── daily-YYYY-MM-DD.html          # 매일 생성
    ├── cluster-<slug>-0001.html       # LC별 상세 페이지
    ├── weekly-YYYY-W##.html           # (구 포맷, 유지만)
    └── .cache/
        ├── abstracts.jsonl            # 파일별 title/abstract 캐시
        ├── embeddings.npz             # 384-d 벡터 캐시
        ├── themes_history.jsonl       # 일별 cluster snapshot
        ├── daily_summaries.jsonl      # 일별 헤드라인 + 통계
        ├── run_log.jsonl              # 실행 메타
        ├── run.log                    # stdout/stderr
        ├── cluster_decisions.jsonl    # join/rename/merge 판정 로그 ← NEW
        └── living_clusters/           # 영속 클러스터 저장소 ← NEW
            ├── registry.json          # category → [uid, ...] 인덱스
            └── <category-slug>/
                └── <uid>.json         # Living Cluster 전체 lineage
```

### Living Cluster 파일 스키마

`living_clusters/<slug>/<slug>-NNNN.json`:

```json
{
  "uid": "cognitive-and-computational-neuroscience-0007",
  "category": "Cognitive and Computational Neuroscience",
  "created_at": "2026-03-02T10:41:00",
  "updated_at": "2026-04-19T10:41:00",
  "status": "active",
  "theme_name": "Interoception and predictive self-models",
  "theme_summary": "…",
  "keywords": ["interoception", "predictive coding", "autism"],
  "centroid": [384 floats],
  "centroid_at_last_name": [384 floats],
  "members": [
    { "fname": "2026 - … .pdf", "added": "2026-03-02" }
  ],
  "name_history": [
    { "at": "2026-03-02", "name": "Interoception in autism" },
    { "at": "2026-04-10", "name": "Interoception and predictive self-models",
      "reason": "drift", "drift_cos": 0.182 }
  ],
  "events": [
    { "at": "2026-03-02", "type": "born", "seed_files": […] },
    { "at": "2026-03-15", "type": "extended", "added_files": […],
      "n_before": 3, "n_after": 5, "centroid_shift_cos": 0.041 },
    { "at": "2026-04-10", "type": "renamed", "from": "…", "to": "…" }
  ]
}
```

---

## 3. 설치

이 프로젝트는 `uv`로 관리된다.

### 3.1 도구 설치

```bash
# 프로젝트 루트(pyproject.toml 있는 곳)에서
uv tool install -e .

# 또는 경로를 직접 지정
uv tool install -e /Volumes/Aquatope/_DEV_/Daily_News
```

`-e`(editable) 덕분에 `src/daily_news/*.py`를 수정하면 재설치 없이 즉시 반영된다.
성공하면 PATH에 `daily-news` 커맨드가 노출된다.

```bash
which daily-news     # ~/.local/bin/daily-news
daily-news --help
```

### 3.2 재설치 / 제거

```bash
# 의존성 재빌드 (pyproject.toml 수정 후)
uv tool install --force -e .

# 완전 제거
uv tool uninstall daily-news
```

### 3.3 최초 모델 다운로드

첫 실행 시 sentence-transformers가 `all-MiniLM-L6-v2`(~80 MB)를
`~/.cache/huggingface`에 자동 다운로드한다.

---

## 4. Ollama 모델 설정

기본값: `minimax-m2.7:cloud` @ `http://localhost:11434`. 필요 시 환경변수로 덮어쓴다.

```bash
# 이미 받아져 있는지
ollama list | grep minimax

# 없으면
ollama pull minimax-m2.7:cloud

# 다른 모델을 쓰고 싶다면
export OLLAMA_MODEL="llama3.2:3b"
export OLLAMA_URL="http://localhost:11434"
```

LLM을 아예 건너뛰려면 `--no-llm` — 오프라인 디버깅용으로 제목/카테고리 기반의
보수적 라벨만 붙인다. 이 fallback 라벨은 실제 theme/keyword로 취급하지 않으며,
rename 단계도 자동으로 건너뛰게 된다.

---

## 5. 매일 사용하기

### 5.1 기본 실행

```bash
cd /Volumes/Aquatope/_DEV_/Daily_News/articles
uv run daily-news
# 또는 PATH에 설치됐다면
daily-news
```

실행 시 `index.json`이 있는 디렉터리를 articles root로 자동 감지한다. 명시하려면:

```bash
daily-news --articles-root /path/to/Articles
ARTICLES_ROOT=/path/to/Articles daily-news
```

우선순위: `--articles-root` > `$ARTICLES_ROOT` > CWD.

### 5.2 자주 쓰는 플래그

| 플래그 | 용도 |
|--------|------|
| `--date YYYY-MM-DD` | 특정 날짜 다이제스트 재생성. idempotent(살짝 — 아래 주의). |
| `--from YYYY-MM-DD` / `--to YYYY-MM-DD` | 기간 처리. `--from`은 포함 시작일, `--to`는 포함 종료일. 예: `--from 2026-04-01 --to 2026-04-19`. |
| `--since-hours N` | 최근 N시간에 추가된 PDF만 처리(날짜 경계 우회). |
| `--no-llm` | Ollama 건너뛰기. 제목/카테고리 기반 fallback만 사용하고, 실제 theme/keyword 섹션과 rename은 건너뜀. |
| `--consolidate` | 데일리 파이프라인 대신 housekeeping만. 아래 §7.4 참조. |
| `--dry-run` | `--consolidate`와 함께. 판정 로그만 남기고 디스크는 변경 안 함. |
| `--verbose`, `-v` | DEBUG 레벨 로그. |

### 5.3 재실행 안전성

- Abstracts·embeddings는 파일명 기준 캐시 → 같은 PDF는 한 번만 처리.
- `--from`은 기간 시작일부터 `--to`(없으면 오늘)까지를 하나의 작업 창으로 묶는다.
  `--date`, `--since-hours`와는 동시에 쓸 수 없다.
- 같은 날짜로 재실행할 때 `living_clusters/`에 같은 fname을 중복 추가하지 않도록
  `apply_living_updates`에서 `seen = {m.fname for m in lc.members}` 필터가 있음.
- `themes_history.jsonl`은 append-only라서 같은 날짜로 재실행하면 중복 스냅샷이 쌓임.
  정확한 rebuild가 필요하면 해당 날짜 라인을 수동 삭제 후 `--date` 지정.

---

## 6. 자동화

### 6.1 매일 저녁 실행 — launchd (macOS)

`bin/com.anselm.articles.dailynews.plist`의 `ABSOLUTE_PATH_TO_ARTICLES`를 실제 경로로
치환한 뒤:

```bash
cp bin/com.anselm.articles.dailynews.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.anselm.articles.dailynews.plist
launchctl start com.anselm.articles.dailynews    # 스모크 테스트
tail -f _news/.cache/run.log
```

제거:

```bash
launchctl unload ~/Library/LaunchAgents/com.anselm.articles.dailynews.plist
```

### 6.2 PDF 추가 감지 — fswatch

```bash
brew install fswatch
./bin/watch_articles.sh &
```

- 전체 `Articles/` 트리를 지켜보다가 `.pdf` add/change 이벤트 발생 후
  120초 침묵이면 1회 실행.
- 한 번에 30개를 drag-in 해도 run이 1번만 돎.
- 쿨다운: `COOLDOWN=300 ./bin/watch_articles.sh`.

### 6.3 주간 housekeeping — crontab 예시

```cron
# 매주 일요일 03:00에 consolidate
0 3 * * 0 cd /Volumes/Aquatope/_DEV_/Daily_News/articles && /opt/homebrew/bin/uv run daily-news --consolidate >> _news/.cache/consolidate.log 2>&1
```

---

## 7. Organic Cluster 운영

### 7.1 개념

| 용어 | 의미 |
|------|------|
| **Daily Cluster** | 매일 HDBSCAN이 만든 일회성 묶음(기존 방식). |
| **Living Cluster(LC)** | 카테고리마다 영속되는 클러스터. centroid·members·events·name_history 보유. |
| **lineage** | 오늘의 daily cluster가 LC에 대해 취한 관계: `born` / `extended` / `fresh`. |
| **status** | LC의 현재 상태: `active` / `dormant` / `merged_into:<uid>`. |

### 7.2 이벤트 타입

| 타입 | 발생 조건 | 언제 |
|------|-----------|------|
| `born` | 오늘 orphan들이 HDBSCAN으로 ≥ 2편 묶였고 기존 LC와 유사도 낮음 | 매일 |
| `extended` | 오늘 논문이 기존 active LC에 sim ≥ 0.60으로 흡수 | 매일 |
| `revived` | 오늘 논문이 dormant LC에 sim ≥ 0.65로 흡수 | 매일 |
| `renamed` | LC centroid 드리프트 ≥ 0.15 **또는** 멤버 수 × 1.5 이상, LLM이 실제로 다른 이름을 냄 | 매일 |
| `merged_with` / `merged_into` | 같은 카테고리 내 두 active LC centroid-sim ≥ 0.85 | `--consolidate` |
| `dormant` | 마지막 활동 후 30일 경과 | `--consolidate` |
| `bootstrapped` | 최초 실행 시 `themes_history.jsonl`에서 seed | 1회 |

### 7.3 최초 실행 시 — Bootstrap

`living_clusters/`가 비어 있으면 `themes_history.jsonl`의 최근 14일 스냅샷을 읽어
카테고리별로 초기 LC를 자동 생성한다. `events`에 `{"type": "bootstrapped"}` 한 건이
남는다. 이후 실행부터는 정상적인 match/grow 사이클이 돈다.

처음부터 맨땅에서 시작하고 싶다면:

```bash
rm -rf _news/.cache/living_clusters/
# themes_history.jsonl도 비우려면:
> _news/.cache/themes_history.jsonl
```

### 7.4 Consolidate — 주간 housekeeping

```bash
# 먼저 dry-run으로 어떤 판단이 내려질지 확인
daily-news --consolidate --dry-run --verbose

# 괜찮으면 실제로 적용
daily-news --consolidate
```

결과:

- Merge: "두 개의 유사한 주제가 하나로 합쳐졌다" — `cluster_decisions.jsonl`에
  `{"kind":"merge_check", "a":"…","b":"…","sim":0.87,"outcome":"merge:a<-b"}`.
- Dormant: "한 달간 새 논문 없는 주제는 잠시 접어둔다" — 삭제가 아니라 대기 상태.
  다음 매칭에서 τ_revive(0.65)로 깨울 수 있음.
- Dry-run은 디스크에 아무것도 쓰지 않고 로그만 남김.

### 7.5 수동 개입

- **특정 LC를 잠재우고 싶다**: 파일의 `"status": "active"` → `"dormant"`로 수정.
- **두 LC를 강제로 합치고 싶다**: 작은 쪽의 `members`를 큰 쪽에 붙여넣고 centroid를
  재계산하는 편집, 그리고 작은 쪽 status를 `"merged_into:<uid>"`로. 그보다는
  `τ_merge`를 0.80까지 낮추고 `--consolidate` 돌리는 게 쉬움.
- **이름을 수동으로 고정**: `theme_name`을 수정하고 `centroid_at_last_name`을
  `centroid`와 똑같이 맞춰두면 다음 drift rename이 당분간 안 발동함.

---

## 8. 산출 HTML 해석

### 8.1 `daily-YYYY-MM-DD.html`

- **Hero** — 오늘의 through-line 헤드라인 + lede, 통계.
- **Top Themes** — 오늘 논문 수와 최근 baseline 논문 수를 합산해 가장 큰 주제 4개 카드.
- **Cluster Evolution** ← 새 섹션. 🌿 Born / ➕ Extended / 🔀 Renamed / ✨ Revived
  아이템. 각 항목 클릭 시 해당 LC 상세 페이지로 이동.
- **Today's Reading** — 카테고리별 paper 카드. cluster-head 우상단에
  `NEW` / `+N today` / `RENAMED` / `REVIVED` 뱃지, 그리고 **10주 sparkline**이
  달려 있음 → 해당 LC가 최근 꾸준히 크는지 한 번에 보임.

### 8.2 `cluster-<uid>.html`

LC 한 건의 전체 일대기:

- 26주 cadence sparkline
- Name history (모든 개명 이력, drift 이유 포함)
- Event timeline (born / extended / renamed / merged / dormant / revived)
- Members (추가일 역순)

### 8.3 `index.html`

최근 30일 daily 호의 아카이브. 각 호의 대표 테마가 옆에 붙음.

---

## 9. 파라미터 튜닝

전부 `src/daily_news/living_cluster.py` 상단에 상수로 정의:

| 이름 | 기본값 | 역할 |
|------|-------|------|
| `TAU_JOIN` | 0.60 | active LC 흡수 임계 |
| `TAU_CANDIDATE` | 0.45 | (예약됨) 경계 구간 하한 |
| `TAU_REVIVE` | 0.65 | dormant LC 부활 임계 |
| `TAU_MERGE` | 0.85 | consolidate 시 두 LC 병합 임계 |
| `DELTA_DRIFT` | 0.15 | centroid cos distance, 초과 시 재명명 고려 |
| `NAME_GROWTH_RATIO` | 1.5 | 멤버 수가 이 배수 넘으면 재명명 고려 |
| `DORMANCY_DAYS` | 30 | 활동 없는 일 수, 초과 시 dormant |
| `BORN_MIN_SIZE` | 2 | HDBSCAN 덩어리가 이 이상이어야 LC 생성 |

### 실측으로 임계 잡기

`cluster_decisions.jsonl`에는 매 논문의 best-match sim이 기록된다.

```bash
# 오늘 join 판정의 best_sim 분포 보기
grep '"kind":"join_check"' _news/.cache/cluster_decisions.jsonl \
  | python3 -c 'import sys,json; import statistics as S; \
     xs=[json.loads(l).get("best_sim") for l in sys.stdin]; \
     xs=[x for x in xs if x is not None]; \
     print(f"n={len(xs)} median={S.median(xs):.3f} p90={sorted(xs)[int(len(xs)*0.9)]:.3f}")'
```

- `orphan` 결과의 best_sim이 대부분 0.5–0.6대에 분포한다면 τ_join을 0.55로 낮춰 흡수
  비율을 높일 수 있음.
- 반대로 `joined:*` 중 오답이 많이 보이면 τ_join을 0.65로 올리기.
- merge_check에서 sim 0.75–0.85 구간이 수두룩하면 τ_merge를 0.82까지 낮춰 병합을
  적극적으로.

---

## 10. 트러블슈팅

### `IsADirectoryError: [Errno 21] Is a directory: '.'`
→ 과거 버그. `configure_paths`의 `global` 선언이 누락돼 경로가 `.`(cwd)로 남는
문제. 현재 버전에서 수정됨. 최신 소스로 업데이트.

### `ModuleNotFoundError: pymupdf`
```bash
uv tool install --force -e .   # 의존성 재빌드
```
일부 배포판에서는 `fitz`로 import되기도 — 코드가 둘 다 시도함.

### Theme name이 항상 키워드 폴백
- Ollama가 안 켜져 있음: `ollama serve`
- 모델 이름 오타: `ollama list` 확인
- `-v`로 돌려서 `ollama cluster failed…` 로그 확인

### "no new articles"
`run_log.jsonl`에 `"note": "no new articles"`가 남음.
- PDF의 `mtime`이 시간대 경계 넘어서 엉킨 경우: `--since-hours 24`로 우회.
- `index.json`의 `added` 타임스탬프를 봐서 오늘자 엔트리가 있는지 확인.

### 같은 날짜로 재실행했더니 Extended가 0개
정상. 같은 파일명은 이미 LC 멤버에 있어서 중복 흡수되지 않음. `themes_history.jsonl`
스냅샷만 이중으로 쌓이는데, 그건 무해함.

### Cluster가 너무 쉽게 merge됨 / 너무 안 함
`cluster_decisions.jsonl`의 `merge_check`를 열어 실제 sim 분포 확인 → τ_merge 조정
→ 다음 주 consolidate 때 반영.

### Bootstrap이 너무 많은 LC를 만들었다
`themes_history.jsonl`의 최근 14일 × 카테고리 × cluster_id 조합이 그대로 seed가 됨.
원하지 않는 LC는 `living_clusters/<slug>/<uid>.json`을 수동 삭제하거나 status를
`dormant`로 바꾸고 다음 consolidate에서 정리.

### Abstract 추출 품질 낮음
`abstracts.jsonl`에서 `"abstract_method": "nomatch"` 또는 `"empty"` 비율 확인. 대부분
스캔본 PDF — OCR은 현재 미구현.

### 클러스터 상세 페이지가 안 열림 (404)
render 순서 문제. 최신 코드는 `render_daily_html`보다 뒤에 `render_cluster_detail_pages`가
있어야 정상. 수동 확인:
```bash
ls _news/cluster-*.html | wc -l   # 0보다 커야 함
```

---

## 11. 개발자 노트

### 11.1 핵심 파일

- `src/daily_news/cli.py` — 파이프라인 오케스트레이션과 렌더링 전부.
- `src/daily_news/living_cluster.py` — LC dataclass, 스토리지, cosine 연산, 이벤트
  헬퍼, bootstrap.
- `organic_cluster.md` — 설계 문서(시스템 철학, 향후 계획).

### 11.2 확장 포인트

- **Split 감지**: `living_cluster.py`에 silhouette 기반 split을 추가하고
  `run_consolidate`에 호출. LC record에 `parent` 필드를 둬서 lineage 유지.
- **Cross-category reconcile**: 사용자가 PDF를 다른 폴더로 옮길 때 원 LC에서
  멤버를 빼고 새 카테고리에서 match를 다시 돌리는 reconcile pass. `index.json`을
  기준으로 하루 1회.
- **더 나은 Rising 지표**: 현재는 14일 baseline Poisson 잔차. LC의 7일 growth
  rate와 age를 결합한 Bayesian 버전으로 교체 가능.
- **Decision log 시각화**: `cluster_decisions.jsonl`을 읽어 임계 튜닝용 대시보드
  HTML을 생성하는 서브커맨드(`--audit`).

### 11.3 캐시 정리

```bash
# 전부 새로 태우기 (처음부터 재구축)
rm -rf _news/.cache/
daily-news --date 2026-04-19     # 혹은 여러 날짜 순회
```

경고: embeddings 재계산은 논문 수가 많으면 오래 걸림.

### 11.4 LC 스키마 진화

필드를 추가할 때는 `LivingCluster.from_json`에서 `obj.get(new_field, default)`로
받아야 옛 파일들이 깨지지 않음. 기존 파일을 한 번씩 load→save 하면 새 필드가 기록됨.

### 11.5 테스트

현재 정식 테스트 스위트는 없음. 개발 중에는:

```bash
# 파싱/import 체크
python3 -c "import ast; ast.parse(open('src/daily_news/cli.py').read())"
python3 -c "import sys; sys.path.insert(0,'src'); from daily_news import cli, living_cluster"

# consolidate dry-run
uv run daily-news --consolidate --dry-run --verbose

# 특정 날짜 rebuild
uv run daily-news --date 2026-04-19 --verbose

# 2026-04-01부터 2026-04-19까지 추가분을 한 번에 처리
uv run daily-news --from 2026-04-01 --to 2026-04-19 --verbose
```

---

## 라이선스

MIT. 세부 사항은 `pyproject.toml` 참조.
