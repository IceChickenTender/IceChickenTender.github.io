---
title: "[ElasticSearch] 6. ElasticSearch Hybrid Retrieval과 RRF 이해하기"
categories:
  - ElasticSearch
tags:
  - ElasticSearch
  
toc: true
toc_sticky: true
toc_label: "ElasticSearch Hybrid Retrieval과 RRF 이해하기"
---

# 개요

Elasticsearch에서 검색을 구현할 때 Lexical Search와 Vector Search는 서로 다른 강점을 가집니다.

**Lexical Search**는 사용자의 Query와 Document에 실제로 존재하는 Term을 기준으로 검색하는 방식입니다. Elasticsearch에서는 `match`와 같은 Full-text Query와 BM25를 조합하여 관련도 순위를 계산할 수 있습니다.

반면 **Dense Vector Search**는 Query와 Document를 Vector로 표현한 뒤 Vector 간 Similarity를 이용하여 의미적으로 가까운 Document를 검색합니다.

두 검색 방식은 서로 다른 장단점을 가지기 때문에 하나만 사용하는 것보다 두 결과를 결합하는 **Hybrid Retrieval**을 사용할 수 있습니다.

# 1. Hybrid Retreiverl 이란?

**Hybrid Retrieval**은 서로 다른 복수의 Retrieval 방식에서 얻은 검색 결과를 결합하여 하나의 최종 Ranking을 만드는 검색 전략입니다.

대표적인 조합이 다음과 같습니다.

```text
Lexical Search
    +
Dense Vector Search
    ↓
Hybrid Retrieval
    ↓
Final Ranking
```

## Lexical Search의 강점

BM25 기반 Lexical Search는 실제 Term Matching에 강합니다.

예를 들어 다음과 같은 검색어가 있다고 하겠습니다.

```text
HBM3E
Samsung
RTX 5090
```

제품명, 회사명, 코드와 같이 정확한 문자열 자체가 중요한 검색에서는 Lexical Search가 강점을 가집니다.

하지만 Query와 의미는 같더라도 서로 다른 단어가 사용되면 관련 Document를 놓칠 수 있습니다.

## Dense Vector Search의 강점

Dense Vector Search는 Embedding Model을 사용해 Query와 Document의 의미를 Vector 공간에 표현한 뒤 의미적으로 가까운 Document를 검색합니다.

따라서 Query와 정확한 Term이 일치하지 않아도 의미가 유사한 Document를 찾을 수 있습니다.

반면 정확한 제품명, 코드, 고유명사 등의 Lexical Signal이 중요한 경우에는 원하는 결과보다 의미적으로만 비슷한 Document가 높은 순위에 나타날 수도 있습니다.

따라서 Hybrid Retrieval은 두 검색 방식의 장점을 함께 활용하는 것을 목적으로 합니다.

```text
BM25
→ 정확한 Term Matching

Dense Vector Search
→ Semantic Similarity

Hybrid Retrieval
→ 두 신호를 함께 활용
```

Elasticsearch의 Hybrid Search 역시 Full-text Search와 Vector Search를 결합해 하나의 Ranked List를 만드는 방식입니다.

---

# 2. ElasticSearch Retriever

**Retriever**는 Elasticsearch Search API에서 상위 Document를 검색하는 방법을 표현하는 추상화입니다.

여러 Retriever를 조합하면 검색 과정을 Tree 형태로 구성할 수 있습니다.

이번 학습에서는 다음 세 가지 Retriever가 중요합니다.

## standard retriever

`standard` Retriever는 일반적인 Elasticsearch Query DSL을 실행해 상위 Document를 반환하는 Retriever입니다.

즉 `standard`는 BM25 전용 Retriever가 아닙니다.

```text
standard
└── Query DSL
    ├── match
    ├── multi_match
    ├── term
    ├── bool
    └── range
```

이번 실습에서는 다음과 같이 사용했습니다.

```text
standard Retriever
    ↓
match Query
    ↓
Lexical Search
    ↓
BM25 Ranking
```

## knn retriever

`knn` Retriever는 `dense_vector` Field를 대상으로 k-nearest neighbor Search를 수행하는 Retriever입니다.

Query Vector와 가까운 Document Vector를 찾아 Vector Similarity 기반 Ranking을 생성합니다.

## rrf retriever

`rrf` Retriever는 여러 Child Retriever에서 얻은 Ranking을 **Reciprocal Rank Fusion(RRF)** 알고리즘으로 결합하는 Compound Retriever입니다.

예를 들면 다음과 같습니다.

```text
             ┌─ standard → BM25 Ranking
User Query ──┤
             └─ knn      → Vector Ranking
                       ↓
                      RRF
                       ↓
                Final Ranking
```

---

# 3. BM25 Score와 Vector Search Score를 그대로 더하기 어려운 이유

Hybrid Retrieval을 구현한다고 해서 두 `_score`를 단순히 더하면 되는 것은 아닙니다.

예를 들어 다음과 같은 결과가 있다고 하겠습니다.

```text
BM25
A = 8.2
B = 5.1

Vector
A = 0.78
B = 0.95
```

BM25와 Vector Search는 Score 계산 방법이 다릅니다.

BM25는 Term Frequency, Document Frequency, Document Length 등의 정보를 이용합니다.

반면 Vector Search는 Cosine Similarity, Dot Product, L2 Distance 등의 Vector Similarity를 사용합니다.

따라서 다음 두 값이 같다고 해서 같은 의미를 가지는 것은 아닙니다.

```text
BM25 Score = 0.8
Vector Score = 0.8
```

즉 두 Score는 **서로 다른 계산 방법과 Scale을 가지기 때문에 Raw Score를 직접 비교하거나 단순 합산하기 어렵습니다.**

이를 해결하는 방법 중 하나가 RRF입니다.

---

# 4. RRF란?

**Reciprocal Rank Fusion(RRF)**은 여러 Retrieval 결과의 Raw Score 대신 **Rank, 즉 순위**를 이용해 하나의 Ranking으로 결합하는 알고리즘입니다.

각 Retriever에서 Document가 받은 순위를 이용하여 다음과 같은 형태로 점수를 계산합니다.

```text
RRF Score
=
Σ 1 / (rank_constant + rank)
```

Document가 여러 Retrieval 결과에 존재한다면 각 Ranking에서 얻은 값을 모두 더합니다.

예를 들어 다음 결과가 있다고 하겠습니다.

```text
BM25 Ranking

1. A
2. B
3. C


Vector Ranking

1. C
2. A
3. D
```

`rank_constant = 1`이라면 A는 다음과 같습니다.

```text
BM25 Rank 1
→ 1 / (1 + 1)
→ 1/2

Vector Rank 2
→ 1 / (1 + 2)
→ 1/3

A RRF Score
= 1/2 + 1/3
= 5/6
```

C는 다음과 같습니다.

```text
BM25 Rank 3
→ 1/4

Vector Rank 1
→ 1/2

C RRF Score
= 3/4
```

따라서 최종적으로는:

```text
A > C
```

가 됩니다.

RRF의 중요한 특징은 **BM25 Score와 Vector Score 자체를 사용하지 않는다는 점**입니다.

따라서 서로 다른 Score Scale을 맞추기 위한 별도의 Score Normalization 없이 결과를 결합할 수 있습니다.

## 4.1 rank_constant

**`rank_constant`**는 RRF에서 Rank 간 영향력 차이를 조절하는 값입니다.

```text
1 / (rank_constant + rank)
```

`rank_constant`가 작으면 상위 Rank의 영향력이 상대적으로 커집니다.

반대로 값이 커질수록 Rank 간 차이가 완만해집니다.

이번 실습에서는 먼저 계산을 쉽게 확인하기 위해:

```text
rank_constant = 1
```

을 사용했습니다.

이후 Elasticsearch의 기본값인:

```text
rank_constant = 60
```

도 적용했습니다.

예를 들어 Doc1과 Doc3의 결과는 다음과 같이 변화했습니다.

```text
rank_constant = 1

Doc1 = 1.000000
Doc3 = 0.583333
```

```text
rank_constant = 60

Doc1 = 0.032787
Doc3 = 0.032002
```

Ranking은 동일했지만 Rank 간 Score 차이가 훨씬 완만해졌습니다.

## 4.2 rank_window_size

**`rank_window_size`**는 각 Child Retriever에서 RRF 계산에 참여시킬 상위 Document의 범위입니다.

예를 들어:

```text
rank_window_size = 50
```

이라면 각 Retriever에서 상위 최대 50개의 결과가 RRF 계산 대상으로 사용됩니다.

---

# 5. 실제 Hybrid Retrieval 실습

실습에서는 다음 6개 Document를 사용했습니다.

| Doc | Content | Vector |
|---|---|---|
| Doc1 | Samsung semiconductor HBM earnings increased | `[1, 0, 0]` |
| Doc2 | Samsung semiconductor market outlook | `[0.9, 0.1, 0]` |
| Doc3 | SK hynix HBM semiconductor earnings report | `[0.8, 0.2, 0]` |
| Doc4 | Hyundai automobile sales report | `[0.1, 0.9, 0]` |
| Doc5 | Samsung semiconductor annual report | `[0.7, 0.3, 0.1]` |
| Doc6 | Battery and energy storage market | `[0.1, 0.2, 0.9]` |

이 Vector들은 실제 Embedding Model에서 생성된 것이 아니라 **Vector Search와 RRF 동작을 확인하기 위해 임의로 구성한 실습용 Vector**입니다.

검색 의도는 다음과 같이 구성했습니다.

```text
Lexical Query
→ "HBM earnings"

Query Vector
→ [1.0, 0.0, 0.0]
```

## 5.1 BM25 Search 결과

`standard` Retriever와 `match` Query를 사용했습니다.

```json
{
  "size": 6,
  "retriever": {
    "standard": {
      "query": {
        "match": {
          "content": "HBM earnings"
        }
      }
    }
  }
}
```

결과는 다음과 같았습니다.

```text
1. Doc1 → _score = 2.0007746
2. Doc3 → _score = 1.8437374
```

Doc1과 Doc3에는 모두 `HBM`, `earnings` Term이 존재합니다.

반면 Doc2와 Doc5 등은 해당 Term을 포함하지 않기 때문에 BM25 결과에는 나타나지 않았습니다.

## 5.2 Vector Search 결과

동일한 검색 의도를 Vector Search에서는 다음 Query Vector로 표현했습니다.

```text
[1.0, 0.0, 0.0]
```

결과는 다음과 같았습니다.

```text
1. Doc1 → 1.000000
2. Doc2 → 0.99694186
3. Doc3 → 0.98507124
4. Doc5 → 0.95566120
5. Doc4 → 0.55521580
6. Doc6 → 0.55391640
```

BM25와 비교하면 차이가 명확합니다.

```text
BM25

1. Doc1
2. Doc3
```

```text
Vector

1. Doc1
2. Doc2
3. Doc3
4. Doc5
5. Doc4
6. Doc6
```

BM25에서는 Term이 일치하지 않아 검색되지 않았던 Doc2와 Doc5도 Vector Search에서는 높은 순위에 위치했습니다.

## 5.3 RRF 적용

Elasticsearch에는 Native `rrf` Retriever가 존재합니다.

하지만 실습 환경에서 실행했을 때 다음 오류가 발생했습니다.

```text
current license is non-compliant for
[Reciprocal Rank Fusion (RRF)]
```

현재 Elastic의 Self-managed Subscription 기준으로 `standard`, `knn` Retriever와 Vector Search는 Basic에서 사용할 수 있지만, Native `rrf` Retriever와 RRF Hybrid Search 기능은 Enterprise 기능으로 구분되어 있습니다.

따라서 이번 실습에서는 Elasticsearch에서 BM25와 Vector Search를 각각 수행한 뒤 Python Backend에서 RRF를 직접 구현했습니다.

구조는 다음과 같습니다.

```text
Elasticsearch
│
├── BM25 Search
│      ↓
│   BM25 Ranking
│
└── Vector Search
       ↓
    Vector Ranking
       │
       ▼
 Python Backend
       │
       ▼
      RRF
       │
       ▼
 Final Ranking
```

RRF 구현은 다음과 같습니다.

```python
from collections import defaultdict


def rrf(rankings, rank_constant=60):
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (rank_constant + rank)

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
```

RRF를 적용한 결과는 다음과 같습니다. `rank_constant = 1`일 때 RRF Score는 다음과 같습니다.

```text
Doc1
= 1/2 + 1/2
= 1.000000

Doc3
= 1/3 + 1/4
= 0.583333

Doc2
= 1/3
= 0.333333

Doc5
= 1/5
= 0.200000

Doc4
= 1/6
= 0.166667

Doc6
= 1/7
= 0.142857
```

따라서 최종 결과는 다음과 같습니다. 여기서 중요한 것은 **Doc3과 Doc2의 순위 변화**입니다.

```text
1. Doc1
2. Doc3
3. Doc2
4. Doc5
5. Doc4
6. Doc6
```

다음은 local ElasticSearch와 연결하고, 결과를 받아오면서 RRF를 계산하도록 구현한 전체 코드입니다. 

코드 실행을 위해선 pip로 "elasticsearch" 설치가 필요하며, 문서 검색 결과를 받아오는 elasticsearch 버전에 맞게 설치가 되어야 합니다. 저는 9.4.2 버전의 elaticsearch를 사용하고 있으므로 `pip install "elasticsearch>=9,<10"` 라는 설치 커맨드로 설치를 진행했습니다.

```python
from collections import defaultdict
from elasticsearch import Elasticsearch


# ---------------------------------------
# 1. Elasticsearch 연결
# ---------------------------------------

# 보안을 비활성화한 로컬 환경인 경우
es = Elasticsearch("http://localhost:9200")

INDEX_NAME = "vector-search-test"


# ---------------------------------------
# 2. BM25 검색
# ---------------------------------------

def search_bm25():
    response = es.search(
        index=INDEX_NAME,
        size=6,
        query={
            "match": {
                "content": "HBM earnings"
            }
        }
    )

    return response["hits"]["hits"]


# ---------------------------------------
# 3. Vector 검색
# ---------------------------------------

def search_vector():
    response = es.search(
        index=INDEX_NAME,
        size=6,
        query={
            "knn": {
                "field": "embedding",
                "query_vector": [1.0, 0.0, 0.0],
                "k": 6,
                "num_candidates": 6
            }
        }
    )

    return response["hits"]["hits"]


# ---------------------------------------
# 4. RRF
# ---------------------------------------

def rrf(rankings, rank_constant=1):
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (rank_constant + rank)

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


# ---------------------------------------
# 5. Elasticsearch 검색 실행
# ---------------------------------------

bm25_hits = search_bm25()
vector_hits = search_vector()


# Elasticsearch 검색 결과를
# RRF가 사용할 문서 ID Ranking으로 변환
bm25_ranking = [hit["_id"] for hit in bm25_hits]
vector_ranking = [hit["_id"] for hit in vector_hits]


# 문서 제목 출력용
documents = {}

for hit in bm25_hits + vector_hits:
    documents[hit["_id"]] = hit["_source"]


# ---------------------------------------
# 6. 개별 Retrieval 결과 출력
# ---------------------------------------

print("=== BM25 Ranking ===")

for rank, hit in enumerate(bm25_hits, start=1):
    print(
        f"Rank {rank}: "
        f"Doc{hit['_id']} | "
        f"{hit['_source']['title']} | "
        f"_score={hit['_score']}"
    )


print("\n=== Vector Ranking ===")

for rank, hit in enumerate(vector_hits, start=1):
    print(
        f"Rank {rank}: "
        f"Doc{hit['_id']} | "
        f"{hit['_source']['title']} | "
        f"_score={hit['_score']}"
    )


# ---------------------------------------
# 7. RRF 실행
# ---------------------------------------

rrf_result = rrf(
    rankings=[
        bm25_ranking,
        vector_ranking
    ],
    rank_constant=60
)


print("\n=== RRF Hybrid Ranking ===")

for rank, (doc_id, score) in enumerate(rrf_result, start=1):
    print(
        f"Rank {rank}: "
        f"Doc{doc_id} | "
        f"{documents[doc_id]['title']} | "
        f"RRF Score={score:.6f}"
    )
```

---

# 6. 정리

이번에 학습한 Hybrid Retrieval의 전체 흐름은 다음과 같습니다.

```text
                    User Query
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   Lexical Retrieval             Semantic Retrieval
          │                             │
 standard Retriever              Embedding Model
          │                             │
       Query DSL                  Query Vector
          │                             │
        BM25                       kNN Search
          │                             │
          ▼                             ▼
   BM25 Ranking                  Vector Ranking
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
                        RRF
                         │
       Rank 기반 RRF Score 계산
                         │
                         ▼
                  Final Ranking
```

정리하면 Hybrid Retrieval은 BM25와 Dense Vector Search처럼 서로 다른 Retrieval의 장점을 함께 활용하는 검색 전략입니다.

BM25는 정확한 Term Matching에 강하고 Dense Vector Search는 의미적 유사성 검색에 강합니다.

RRF는 두 방식의 Raw Score가 아니라 Ranking을 이용하기 때문에 서로 다른 Score Scale을 직접 Normalization하지 않고도 결과를 결합할 수 있습니다.

다만 Ranking만 사용하기 때문에 각 Retriever 내부의 Score 차이 정보는 손실됩니다.

따라서 실제 시스템에서는 검색 데이터와 목표에 따라 RRF, Linear Combination, Reranking 등의 방법을 선택하거나 조합할 필요가 있습니다.

# 참조

- Elastic Docs — Hybrid search [Hybrid search](https://www.elastic.co/docs/solutions/search/hybrid-search?utm_source=chatgpt.com)
- Elasticsearch Reference — Retrievers [Retrievers](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers?utm_source=chatgpt.com)
- Elasticsearch Reference — Standard retriever [Standard retriever](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/standard-retriever?utm_source=chatgpt.com)
- Elasticsearch Reference — kNN retriever [kNN retriever](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/knn-retriever?utm_source=chatgpt.com)
- Elasticsearch Reference — RRF retriever [RRF retriever](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever?utm_source=chatgpt.com)
- Elasticsearch Reference — Reciprocal rank fusion [Reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion?utm_source=chatgpt.com)
- Elastic — Subscriptions [Elastic Stack subscriptions](https://www.elastic.co/subscriptions?utm_source=chatgpt.com)