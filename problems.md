확인 결과: **현재 코드는 organic_cluster.md의 방향을 “부분적으로” 구현해 두었지만, 지금 `_news`에 생성된 2026-04-01~04-19 brief는 그 구조로 축적되고 있지 않습니다.**  
즉, 앞으로의 weekly brief가 과거 cluster의 성장/탄생/개명/부활을 제대로 참조한다고 보기에는 아직 부족합니다.

현재 구현되어 있는 것:

- [living_cluster.py](/Volumes/Aquatope/_DEV_/Daily_News/src/daily_news/living_cluster.py)에 `LivingCluster` 저장 구조가 있음
- 기존 LC centroid와 새 논문 embedding을 비교해서 `extended` 처리하는 join 로직이 있음
- orphan 논문들이 2편 이상 묶이면 `born` LC를 만들 수 있음
- centroid drift 또는 멤버 수 1.5배 성장 시 rename 검토 로직이 있음
- `daily-news --consolidate`로 merge/dormant 처리하는 주간 housekeeping도 있음
- daily HTML에 cluster evolution / cluster detail page를 렌더링하는 연결도 있음

하지만 실제 현재 상태는 이렇습니다.

```text
articles/_news/.cache/living_clusters/ 없음
themes_history.jsonl: 156 rows
dates: 2026-04-19 하나뿐
lineage: fresh 156개
living_uid: null 156개
extended/born/renamed/revived: 0개
cluster_decisions: join_check 156개 전부 orphan
```

핵심 원인은 세 가지입니다.

1. **현재 04-01~04-19 ingest가 “하나의 04-19 brief”로만 저장됨**  
   CLI의 `--from 2026-04-01 --to 2026-04-19`는 19일치 weekly-like window를 하나의 target date `2026-04-19`로 처리합니다. 그래서 `_news`에는 `daily-2026-04-19.html`만 있고, 04-01부터 04-18까지의 별도 temporal snapshots가 없습니다.

2. **초기 Living Cluster가 만들어지지 않음**  
   bootstrap은 `themes_history.jsonl`의 “target date 이전” 기록에서 LC를 seed합니다. 그런데 첫 실행에서는 이전 날짜 기록이 없고, `before_date=target_date` 때문에 같은 2026-04-19 기록은 bootstrap에서 제외됩니다. 따라서 첫 run에서는 기존 LC가 0개입니다.

3. **orphan들이 모두 singleton이 되어 `born` 조건을 통과하지 못함**  
   `BORN_MIN_SIZE = 2`라서 최소 2편이 HDBSCAN으로 묶여야 새 LC가 생깁니다. 그런데 현재 기록을 보면 `n_today=1`인 singleton cluster만 생성되었습니다. 그래서 `born`도 안 되고, LC 저장소도 안 생깁니다.

그래서 답은: **개념적으로는 가능하도록 뼈대가 있지만, 현재 산출물 기준으로는 아직 organic cluster가 작동하지 않습니다.**  
지금 상태로 다음 주에 새 brief를 만들어도, 기존 LC가 없기 때문에 과거 cluster에 “붙는” 것이 아니라 다시 orphan/fresh로 시작할 가능성이 큽니다. 새 논문들이 우연히 2편 이상 HDBSCAN으로 묶일 때만 그때부터 LC가 생깁니다.

제가 보기엔 다음 보완이 필요합니다.

1. **초기 seed 생성 기능 추가**  
   현재 2026-04-01~04-19 brief를 첫 기준선으로 삼아 `living_clusters/`를 강제로 초기화하는 명령이 필요합니다. 예: `daily-news --init-living --from-history --date 2026-04-19`.

2. **weekly run을 first-class로 다루기**  
   지금은 이름만 daily이고 `--from/--to`가 range ingest를 하긴 하지만, 저장 모델은 target date 하나입니다. weekly brief라면 `weekly-2026-04-01_2026-04-19.html` 같은 issue identity와 snapshot date/window를 명확히 분리하는 편이 좋습니다.

3. **born 생성 기준 튜닝**  
   현재 HDBSCAN이 너무 보수적으로 singleton을 많이 만들고 있습니다. `04-01~04-19`의 26편이 전부 1편짜리 theme으로 저장된 것은 organic cluster의 출발점으로 부적절합니다. category별로 HDBSCAN threshold를 조정하거나, fallback agglomerative grouping을 더 적극적으로 써야 합니다.

4. **rising theme을 LC 기반으로 바꾸기**  
   지금 `score_rising()`은 여전히 daily cluster의 `n_today`와 `n_recent` 기준입니다. organic_cluster.md의 의도인 “LC가 시간에 따라 성장하는가”를 보려면 `members.added` 기반 주간 growth rate, 최근 N주 증가량, 누적 크기 대비 최근 증가율을 별도로 계산해야 합니다.

5. **중복 snapshot 방지**  
   `themes_history.jsonl`은 같은 2026-04-19 run을 반복할 때 계속 append되어 지금 156행이 되었습니다. weekly 운영에서는 같은 issue date/window 재실행 시 기존 snapshot을 replace하거나 run id를 구분해야 합니다.

정리하면, **organic_cluster.md의 큰 설계는 코드에 상당히 들어와 있지만, 현재 데이터와 실행 흐름은 아직 그 설계를 실제로 활성화하지 못하고 있습니다.**  
가장 먼저 해야 할 일은 `2026-04-01~04-19` 묶음을 “초기 living cluster seed”로 제대로 생성하고, 다음 weekly run이 그 seed에 join/extend/rename을 걸 수 있게 만드는 것입니다.