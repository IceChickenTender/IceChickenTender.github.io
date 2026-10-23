---
title: "[ElasticSearch] 5. ElasticSearch Vector Search 이해하기: dense_vector부터 HNSW와 kNN까지"
categories:
  - ElasticSearch
tags:
  - ElasticSearch
  
toc: true
toc_sticky: true
toc_label: "ElasticSearch Vector Search 이해하기: dense_vector부터 HNSW와 kNN까지"
---

# 개요

이전까지 Elasticsearch의 Analyzer, Inverted Index, BM25, Query DSL과 Metadata Filtering을 학습했습니다.

BM25 기반 검색은 Query와 Document가 공유하는 **Term**을 중심으로 관련도를 계산합니다. 하지만 실제 검색에서는 사용하는 단어가 달라도 의미가 비슷한 문서를 찾고 싶은 경우가 있습니다.

예를 들어:

```text
Query
자동차 매출 감소

Document
차량 판매 실적 하락
```

은 의미적으로 유사하지만 동일한 Term은 많지 않습니다.

이러한 한계를 보완하기 위해 사용할 수 있는 방식이 **Vector Search**입니다.

---

# 1. Lexical Search와 Vector Search

**Lexical Search**는 Query와 Document의 Term Matching을 기반으로 검색하는 방식입니다. 여기서 BM25는 Lexical Search 자체가 아니라 **Term 정보를 이용해 Document의 관련도를 계산하는 Scoring 방식**입니다.

반면 **Dense Vector Search**는 Text를 Embedding Vector로 표현하고 Query Vector와 Document Vector의 유사도를 기준으로 검색합니다.

즉 두 방식의 핵심 차이는 다음과 같습니다.

```text
Lexical Search
→ Term Matching 기반 검색

Dense Vector Search
→ Embedding Vector의 유사성 기반 검색
```

---

# 2. Embedding과 `dense_vector`

## Embedding이란?

**Embedding은 Text와 같은 데이터를 모델이 다룰 수 있는 고정 길이의 수치 Vector로 표현한 값**입니다.

예를 들어:

```text
"반도체 시장 전망"

↓ Embedding Model

[0.13, -0.42, 0.81, ...]
```

처럼 표현할 수 있습니다.

Embedding Model은 의미적으로 유사한 입력이 Vector 공간에서도 가까운 위치에 존재하도록 학습됩니다.

따라서 Text 자체를 직접 비교하는 대신 Vector 간의 거리를 비교하여 의미적 유사성을 검색할 수 있습니다.

## `dense_vector`란?

Elasticsearch의 **`dense_vector`는 Dense Vector 값을 저장하고 Vector Search에 사용할 수 있는 Field Type**입니다.

예:

```json
"embedding": {
  "type": "dense_vector",
  "dims": 3,
  "similarity": "cosine"
}
```

여기서 역할을 구분하면:

```text
Embedding Model
→ Text를 Vector로 변환

dense_vector
→ Vector 저장
→ Vector Index 구성
→ Vector Search에 사용
```

입니다.

`dims`는 Vector의 차원 수를 의미합니다.

```text
[0.1, 0.2, 0.3]

→ dims = 3
```

실제 Embedding Model을 사용할 때는 모델의 출력 차원과 `dense_vector`의 `dims`가 맞아야 합니다.

---

# 3. Similarity Metric

Similarity Metric은 **두 Vector가 얼마나 비슷하거나 가까운지를 계산하는 기준**입니다.

대표적으로:

```text
Cosine Similarity
Dot Product
L2 Distance
```

등이 있습니다.

이번 실습에서는 **Cosine Similarity**를 사용했습니다.

Cosine Similarity는 두 Vector의 크기보다는 **두 Vector가 가리키는 방향, 즉 두 Vector 사이의 각도가 얼마나 비슷한지**를 측정합니다.

---

# 4. kNN이란?

**kNN(k-Nearest Neighbors)은 Query Vector와 가장 가까운 k개의 Vector를 찾는 문제**입니다.

예를 들어:

```text
k = 3
```

이면 Query와 가장 가까운 Document Vector 3개를 찾는 것이 목적입니다.

여기서 중요한 것은:

```text
kNN = Exact Search
```

가 아니라는 점입니다.

kNN 문제를 해결하는 방법에는 크게 두 가지가 있습니다.

```text
kNN
├─ Exact kNN
└─ Approximate kNN
```

---

# 5. Exact kNN

**Exact kNN은 검색 대상이 되는 모든 Document Vector와 Query Vector의 Similarity를 계산하여 실제로 가장 가까운 k개를 찾는 방식**입니다.

```text
Query Vector
 ↓
Doc1과 비교
Doc2와 비교
Doc3과 비교
...
모든 Vector와 비교
 ↓
Top-k
```

모든 대상을 직접 비교하므로 정확한 Top-k를 얻을 수 있습니다.

하지만 Vector가 수백만 개, 수천만 개가 되면 매 Query마다 모든 Vector와 유사도를 계산해야 하므로 비용이 증가합니다.

따라서 Exact kNN은 작은 Dataset이나 Metadata Filter로 검색 범위가 크게 줄어든 경우에 적합할 수 있습니다.

---

# 6. Approximate kNN과 ANN

**ANN(Approximate Nearest Neighbor)은 모든 Vector를 직접 비교하지 않고 가까울 가능성이 높은 Vector들을 효율적으로 탐색하여 Top-k를 찾는 방식**입니다.

```text
Exact

모든 Vector 비교
→ 실제 Top-k
→ 정확하지만 비용 증가


Approximate

유망한 영역 탐색
→ 후보 검색
→ Top-k
→ 빠르지만 일부 실제 Top-k를 놓칠 수 있음
```

따라서 ANN에서는 **Recall과 Latency 사이의 Trade-off**가 발생합니다.

## Recall

Vector Search에서 ANN의 Recall은 보통:

> Exact Search가 찾은 실제 Top-k 중 ANN도 얼마나 찾아냈는가

를 의미합니다.

예를 들어 Exact Top-5가:

```text
A B C D E
```

이고 ANN 결과가:

```text
A B C D F
```

라면 실제 Top-5 중 4개를 찾은 것입니다.

ANN은 알고리즘에 오류가 있어서 결과를 놓치는 것이 아니라, **검색 속도를 위해 전체 Vector 공간을 탐색하지 않는 근사 방식이기 때문에 실제 nearest neighbor를 놓칠 가능성이 존재하는 것**입니다.

---

# 7. HNSW

Elasticsearch에서는 Approximate kNN을 위해 대표적으로 **HNSW(Hierarchical Navigable Small World)** Graph를 사용합니다.

HNSW는 **Vector들을 여러 Layer의 Graph로 구성하고 Graph의 Edge를 따라가면서 Query와 가까운 Vector 영역을 빠르게 탐색하는 ANN 알고리즘**입니다.

## 7.1 Layer 구조

HNSW의 가장 아래인 **Layer 0에는 모든 Vector Node가 존재**합니다.

하지만 모든 Node가 서로 연결되는 Complete Graph는 아닙니다.

각 Node는 제한된 수의 Neighbor와만 연결됩니다.

```text
      B
     / \
A ─ C   D
    |   |
    E ─ F
```

상위 Layer에는 Layer 0의 Node 중 일부만 존재합니다.

```text
Layer 2

A -------- F


Layer 1

A --- C -- F --- H


Layer 0

A B C D E F G H I J
```

어떤 Node가 어느 Layer까지 존재할지는 확률적으로 결정됩니다.

예를 들어:

```text
A → maxLevel 2
B → maxLevel 0
C → maxLevel 1
```

이라면:

```text
A → Layer 2, 1, 0

B → Layer 0

C → Layer 1, 0
```

에 존재합니다.

따라서:

```text
Layer 2 ⊂ Layer 1 ⊂ Layer 0
```

의 관계가 만들어집니다.

---

## 7.2 HNSW에 여러 Layer가 필요한 이유

Layer 0만으로도 Graph Search 자체는 가능합니다.

하지만 Layer 0에는 모든 Vector가 있기 때문에 먼 영역까지 이동하려면 많은 Neighbor를 거쳐야 할 수 있습니다.

HNSW의 상위 Layer는 Node 수가 적기 때문에 Vector 공간을 빠르게 이동하기 위한 **Shortcut Graph** 역할을 합니다.

```text
상위 Layer
→ Node 수 적음
→ 넓은 범위를 빠르게 Navigation

하위 Layer
→ Node 수 많음
→ 가까운 Vector를 세밀하게 탐색
```

검색 흐름은 다음과 같습니다.

```text
Top Layer
   ↓
빠르게 Query와 가까운 영역 탐색
   ↓
다음 Layer
   ↓
검색 범위 축소
   ↓
...
   ↓
Layer 0
   ↓
최종 Neighbor 후보 탐색
```

즉 HNSW Search는 최종적으로 Layer 0까지 내려오지만, 상위 Layer를 이용해 **Layer 0에서 탐색을 시작할 위치를 빠르게 좋은 영역으로 이동시키는 것**이 핵심입니다.

---

## 7.3 Entry Point

**Entry Point는 HNSW 탐색을 시작하기 위한 Top Layer의 시작 Node**입니다.

Entry Point는 Query가 들어왔을 때 Query와 가장 가까운 Node를 찾아 결정하는 것이 아닙니다.

Graph를 만드는 Indexing 과정에서 이미 결정됩니다.

처음 Graph에 들어간 Node가 Entry Point가 되며, 이후 기존 최고 Layer보다 더 높은 새로운 Top Layer를 생성하는 Node가 들어오면 해당 Node가 새로운 Entry Point가 될 수 있습니다.

```text
기존 Top Layer = 2
Entry Point = A

새 X
maxLevel = 2
→ Entry Point 유지

새 Y
maxLevel = 4
→ 새로운 Top Layer 생성
→ Y가 새로운 Entry Point
```

따라서 Entry Point가 Query와 가까울 것이라는 보장은 없습니다.

Entry Point의 목적은 **정답 근처에서 시작하는 것**이 아니라, HNSW Graph 탐색을 시작할 하나의 출발점을 제공하는 것입니다.

---

## 7.4 HNSW Graph는 어떻게 만들어지는가?

새로운 Vector X가 들어오면 먼저 X의 최대 Layer가 확률적으로 정해집니다.

예를 들어:

```text
X
maxLevel = 2
```

이면 X는:

```text
Layer 2
Layer 1
Layer 0
```

에 존재합니다.

기존 HNSW Graph의 최고 Layer가 Layer 4라면 X는 Layer 4와 3에는 추가되지 않습니다.

하지만 X와 가까운 영역을 빠르게 찾기 위해 기존 Entry Point에서 Layer 4부터 탐색을 시작합니다.

```text
Layer 4
→ Navigation

Layer 3
→ Navigation

Layer 2
→ X의 Neighbor 후보 탐색 및 연결

Layer 1
→ Neighbor 후보 탐색 및 연결

Layer 0
→ Neighbor 후보 탐색 및 연결
```

즉 X가 존재하지 않는 상위 Layer는 **실제 연결을 만드는 곳이 아니라 X가 들어갈 지역을 빠르게 찾기 위한 Navigation 용도**입니다.

---

## 7.5 Neighbor Selection과 Edge

새로운 Vector가 추가될 때 HNSW는 가까운 Candidate를 탐색하고 그중 일부를 Neighbor로 선택하여 Edge를 만듭니다.

단순히 가장 가까운 Vector를 무제한으로 연결하지 않습니다.

한 Node가 가질 수 있는 연결 수에는 제한이 있으며 Elasticsearch에서는 `m` Parameter가 이 Graph 연결 수를 제어합니다.

또한 새로운 Edge를 추가하면서 기존 Node의 연결 한도를 초과하면 **기존 Edge가 제거될 수도 있습니다.**

다만 단순히 가장 유사도가 낮은 Edge부터 제거하는 것은 아닙니다.

HNSW는 가까운 Node와의 연결뿐 아니라 **Graph의 서로 다른 영역으로 이동하는 데 유용한 다양한 연결**을 유지하도록 Neighbor Selection Heuristic을 사용합니다.

즉:

```text
가까운 Neighbor
+
Graph Navigation에 유용한 Neighbor
```

를 고려하여 Graph를 구성합니다.

---

## 7.6 `k`와 `num_candidates`

Approximate kNN 검색에서 중요한 두 Parameter가 있습니다.

### `k`

**최종적으로 찾고 싶은 nearest neighbor의 수**입니다.

```text
k = 3

→ Query와 가장 가까운 Document 3개
```

### `num_candidates`

**Approximate Search에서 최종 k개를 고르기 위해 고려할 후보의 수**입니다.

예:

```text
k = 3
num_candidates = 100
```

이면 개념적으로:

```text
HNSW 탐색
 ↓
유망 Candidate 탐색
 ↓
최종 Top 3 선택
```

의 구조입니다.

일반적으로:

```text
num_candidates ↑
→ Recall 개선 가능
→ Latency 증가
```

라는 Trade-off가 있습니다.

`k`와 `num_candidates`는 같은 값이 아닙니다.

---

# 8. Vector Search의 `_score`

Vector Search에서도 `_score`는 **Ranking Score**입니다.

BM25와 Vector Search 모두 `_score`를 Ranking에 사용하지만 계산 방식이 다릅니다.

```text
BM25 _score

TF
IDF
Document Length
...
↓
Ranking Score
```

```text
Vector _score

Query Vector
Document Vector
Similarity Metric
↓
Ranking Score
```

따라서 BM25 `_score`와 Vector `_score`의 숫자를 직접 비교하면 안 됩니다.

---

# 9. Vector Search 실습

이번 실습에서는 실제 Embedding Model을 사용하지 않고 사람이 검색 결과를 쉽게 예상할 수 있도록 3차원 Vector를 직접 만들었습니다.

## 9.1 Index 생성 및 Mapping 설정

실습을 위한 Index를 따로 만들었고, Mapping은 다음과 같이 구성했습니다.

```http
PUT vector-search-test
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "index.mapping.exclude_source_vectors": false
  },
  "mappings": {
    "properties": {
      "title": {
        "type": "text"
      },
      "content": {
        "type": "text"
      },
      "company": {
        "type": "keyword"
      },
      "document_type": {
        "type": "keyword"
      },
      "published_at": {
        "type": "date"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 3,
        "index": true,
        "similarity": "cosine",
        "index_options": {
          "type": "hnsw",
          "m": 16,
          "ef_construction": 100,
          "flat_index_threshold": 0
        }
      }
    }
  }
}
```

## 9.2 실습 Document 저장

실습 테스트를 위한 Document를 다음과 같이 Index에 저장했습니다.

```http
POST vector-search-test/_bulk?refresh=true
{"index":{"_id":"1"}}
{"title":"Samsung HBM Earnings","content":"Samsung semiconductor HBM earnings increased","company":"samsung","document_type":"annual_report","published_at":"2026-03-15","embedding":[1.0,0.0,0.0]}
{"index":{"_id":"2"}}
{"title":"Samsung Semiconductor Outlook","content":"Samsung semiconductor market outlook","company":"samsung","document_type":"news","published_at":"2026-06-01","embedding":[0.9,0.1,0.0]}
{"index":{"_id":"3"}}
{"title":"SK Hynix HBM Report","content":"SK hynix HBM semiconductor earnings report","company":"sk_hynix","document_type":"annual_report","published_at":"2026-03-20","embedding":[0.8,0.2,0.0]}
{"index":{"_id":"4"}}
{"title":"Hyundai Automobile Report","content":"Hyundai automobile sales report","company":"hyundai","document_type":"annual_report","published_at":"2026-04-10","embedding":[0.1,0.9,0.0]}
{"index":{"_id":"5"}}
{"title":"Samsung Old Semiconductor Report","content":"Samsung semiconductor annual report","company":"samsung","document_type":"annual_report","published_at":"2024-12-20","embedding":[0.7,0.3,0.1]}
{"index":{"_id":"6"}}
{"title":"Battery Market Report","content":"Battery and energy storage market","company":"lg_energy_solution","document_type":"annual_report","published_at":"2026-05-10","embedding":[0.1,0.2,0.9]}
```

아래 명령을 실행해 Document들이 잘 저장되었는지 확인할 수 있습니다.

```http
GET vector-search-test/_search
{
  "size": 10,
  "query": {
    "match_all": {}
  }
}
```

아래와 같이 출력된다면 정상적으로 저장된 것입니다.

```json
{
  "took": 1,
  "timed_out": false,
  "_shards": {
    "total": 1,
    "successful": 1,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 6,
      "relation": "eq"
    },
    "max_score": 1,
    "hits": [
      {
        "_index": "vector-search-test",
        "_id": "1",
        "_score": 1,
        "_source": {
          "title": "Samsung HBM Earnings",
          "content": "Samsung semiconductor HBM earnings increased",
          "company": "samsung",
          "document_type": "annual_report",
          "published_at": "2026-03-15",
          "embedding": [
            1,
            0,
            0
          ]
        }
      },
      {
        "_index": "vector-search-test",
        "_id": "2",
        "_score": 1,
        "_source": {
          "title": "Samsung Semiconductor Outlook",
          "content": "Samsung semiconductor market outlook",
          "company": "samsung",
          "document_type": "news",
          "published_at": "2026-06-01",
          "embedding": [
            0.9,
            0.1,
            0
          ]
        }
      },
      {
        "_index": "vector-search-test",
        "_id": "3",
        "_score": 1,
        "_source": {
          "title": "SK Hynix HBM Report",
          "content": "SK hynix HBM semiconductor earnings report",
          "company": "sk_hynix",
          "document_type": "annual_report",
          "published_at": "2026-03-20",
          "embedding": [
            0.8,
            0.2,
            0
          ]
        }
      },
      {
        "_index": "vector-search-test",
        "_id": "4",
        "_score": 1,
        "_source": {
          "title": "Hyundai Automobile Report",
          "content": "Hyundai automobile sales report",
          "company": "hyundai",
          "document_type": "annual_report",
          "published_at": "2026-04-10",
          "embedding": [
            0.1,
            0.9,
            0
          ]
        }
      },
      {
        "_index": "vector-search-test",
        "_id": "5",
        "_score": 1,
        "_source": {
          "title": "Samsung Old Semiconductor Report",
          "content": "Samsung semiconductor annual report",
          "company": "samsung",
          "document_type": "annual_report",
          "published_at": "2024-12-20",
          "embedding": [
            0.7,
            0.3,
            0.1
          ]
        }
      },
      {
        "_index": "vector-search-test",
        "_id": "6",
        "_score": 1,
        "_source": {
          "title": "Battery Market Report",
          "content": "Battery and energy storage market",
          "company": "lg_energy_solution",
          "document_type": "annual_report",
          "published_at": "2026-05-10",
          "embedding": [
            0.1,
            0.2,
            0.9
          ]
        }
      }
    ]
  }
}
```

테스트 Vector는 다음과 같습니다.

| Document | Vector |
|---|---|
| Doc1 | `[1.0, 0.0, 0.0]` |
| Doc2 | `[0.9, 0.1, 0.0]` |
| Doc3 | `[0.8, 0.2, 0.0]` |
| Doc4 | `[0.1, 0.9, 0.0]` |
| Doc5 | `[0.7, 0.3, 0.1]` |
| Doc6 | `[0.1, 0.2, 0.9]` |

Query Vector는:

```text
[1.0, 0.0, 0.0]
```

으로 설정했습니다.

## 9.3 Approximate kNN 실습

다음 Query를 실행했습니다.

```http
GET vector-search-test/_search
{
  "size": 3,
  "query": {
    "knn": {
      "field": "embedding",
      "query_vector": [1.0, 0.0, 0.0],
      "k": 3,
      "num_candidates": 6
    }
  }
}
```

검색 결과는 예상한 대로:

```text
1. Doc1
   _score = 1.0

2. Doc2
   _score = 0.99694186

3. Doc3
   _score = 0.98507124
```

였습니다.

Query Vector와 방향이 완전히 같은 Doc1이 가장 높은 Score를 받았고, 이후 방향이 가장 가까운 순서대로 Doc2와 Doc3이 검색되었습니다.

## 9.4 Exact kNN 실습

Exact Search에서는 HNSW Graph의 일부 후보를 탐색하는 대신 검색 조건을 만족하는 모든 Document Vector의 Similarity를 계산합니다.

실습한 Elasticsearch 환경에서는 `script_score`를 이용했습니다.

```http
GET vector-search-test/_search
{
  "size": 3,
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "(cosineSimilarity(params.query_vector, 'embedding') + 1.0) / 2.0",
        "params": {
          "query_vector": [1.0, 0.0, 0.0]
        }
      }
    }
  }
}
```

결과는:

```text
Doc1 → 1.0
Doc2 → 0.99694186
Doc3 → 0.98507124
```

로 Approximate Search와 동일했습니다.

하지만 이것이 ANN과 Exact Search가 항상 같은 결과를 반환한다는 뜻은 아닙니다.

이번 실습은 Document가 6개뿐이고 ANN에서도 `num_candidates=6`을 사용했기 때문에 실제 nearest neighbor를 놓칠 가능성이 매우 작은 환경이었습니다.

대규모 Dataset에서는 ANN이 Vector 공간의 일부만 근사적으로 탐색하기 때문에 Exact Top-k 중 일부를 놓칠 수 있습니다.

### `size`와 `k`의 차이

Exact Search에서:

```text
hits.total.value = 6
size = 3
```

이었습니다.

이는:

```text
6개 Document
→ 모두 Matching 및 Scoring 대상

size = 3
→ Score 상위 3개만 Response로 반환
```

을 의미합니다.

따라서:

```text
size ≠ k
```

입니다.

`size`는 Search API가 반환할 hit의 수이고, `k`는 kNN Search에서 찾고자 하는 nearest neighbor의 수입니다.

## 9.5 Vector Search와 Metadata Filter

Vector Search에서도 Metadata Filtering을 사용할 수 있습니다.

이번 실습에서는:

```text
Query Vector
[1,0,0]

company = samsung
published_at >= 2025-01-01
```

조건을 사용했습니다.

```http
GET vector-search-test/_search
{
  "size": 2,
  "query": {
    "knn": {
      "field": "embedding",
      "query_vector": [1.0, 0.0, 0.0],
      "k": 2,
      "num_candidates": 6,
      "filter": {
        "bool": {
          "must": [
            {
              "term": {
                "company": "samsung"
              }
            },
            {
              "range": {
                "published_at": {
                  "gte": "2025-01-01"
                }
              }
            }
          ]
        }
      }
    }
  }
}
```

결과는:

```text
Doc1
Doc2
```

였습니다.

Vector 기준으로 원래 3위였던 Doc3은 Query와 매우 유사하지만:

```text
company = sk_hynix
```

였기 때문에 결과에서 제외되었습니다.

즉 역할은 다음과 같이 나뉩니다.

```text
Metadata Filter
→ 어떤 Document가 검색 결과 후보가 될 수 있는가?

Vector Similarity
→ 조건을 만족하는 Document 중
   Query와 얼마나 유사한가?
```

---

# 10. Pre-filter

`knn.filter`는 **Approximate kNN Search 도중 검색 가능한 Document를 제한하는 Pre-filter**입니다.

즉 먼저 Top-k를 뽑은 다음 조건에 맞지 않는 결과를 제거하는 Post-filter와 다릅니다.

개념적으로:

```text
Pre-filter

Filter 조건을 만족할 수 있는 문서
+
kNN Search
↓
조건을 만족하는 Top-k
```

입니다.

반대로 Post-filter라면:

```text
먼저 Vector Top-k
↓
Filter
↓
일부 결과 제거
```

가 되어 최종 결과가 k개보다 적어질 수도 있습니다.

다만 Pre-filter를:

```text
전체 문서를 먼저 완전히 Filtering
→ 그 결과만 HNSW 검색
```

이라는 두 단계의 물리적인 과정으로 단순화하면 안 됩니다.

Elasticsearch는 Approximate kNN 탐색 중 Filter 조건을 함께 적용합니다.

또한 Filter 조건을 만족하는 Vector 수가 매우 적다면 HNSW를 계속 탐색하는 것보다 해당 Vector를 직접 Scoring하는 것이 더 효율적이므로 내부적으로 brute-force 방식으로 전환될 수도 있습니다.

---

# 11. 정리

## 11.1 전체 Vector Retrieval 흐름

지금까지 학습한 내용을 하나의 흐름으로 연결하면 다음과 같습니다.

### Indexing

```text
Document Text
      ↓
Embedding Model
      ↓
Document Vector
      ↓
dense_vector
      ↓
HNSW Index 구성
```

### Search

```text
User Query
    ↓
Embedding Model
    ↓
Query Vector
    ↓
Metadata Filter
    ↓
HNSW Approximate Search
    ↓
num_candidates 범위의 후보 탐색
    ↓
Vector Similarity 계산
    ↓
_score
    ↓
Top-k
```

Exact Search를 선택한다면:

```text
Query Vector
    ↓
조건을 만족하는 모든 Document Vector
    ↓
Similarity 계산
    ↓
_score
    ↓
Top-k
```

가 됩니다.

## 11.2 학습 내용 정리

Vector Search에서 가장 중요한 개념은 다음과 같이 정리할 수 있습니다.

```text
Embedding Model
→ Text를 Vector로 변환

dense_vector
→ Vector 저장 및 Vector Search에 사용

Similarity Metric
→ Vector가 얼마나 가까운지 계산

kNN
→ 가장 가까운 k개를 찾는 문제

Exact kNN
→ 모든 Matching Vector 비교

Approximate kNN
→ 일부 Vector 공간을 효율적으로 탐색

HNSW
→ Elasticsearch에서 사용하는 대표적인 ANN Graph 구조

Layer 0
→ 모든 Vector 존재, 최종 세밀한 탐색

Upper Layer
→ 일부 Vector만 존재, 빠른 Navigation

Entry Point
→ Top Layer 검색 시작점

num_candidates
→ ANN의 후보 탐색 범위 조절

k
→ 최종 nearest neighbor 수

Metadata Filter
→ 검색 가능한 Document 조건 제한

_score
→ Vector Similarity 기반 Ranking Score
```

결국 Lexical Search와 Vector Search는 다음과 같이 구분할 수 있습니다.

```text
Lexical Retrieval

Term
↓
Inverted Index
↓
BM25
↓
Ranking
```

```text
Dense Retrieval

Embedding
↓
dense_vector
↓
HNSW / Vector Search
↓
Similarity
↓
Ranking
```

두 방식은 서로 다른 장단점을 가지고 있으며, 다음 학습에서는 이 둘을 함께 사용하는 **Hybrid Retrieval**과 검색 결과를 결합하는 **RRF(Reciprocal Rank Fusion)**를 살펴봅니다.

---

# 참조

- [Elastic Docs — Dense vector field type](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector)
- [Elastic Docs — kNN search in Elasticsearch](https://www.elastic.co/docs/solutions/search/vector/knn)
- [Elastic Docs — Exact kNN search](https://www.elastic.co/docs/solutions/search/vector/knn/exact-knn)
- [Elastic Docs — kNN query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-knn-query)
- [Elastic Docs — Filter approximate kNN results](https://www.elastic.co/docs/solutions/search/vector/knn/filtered-knn-search)
- [Elastic Docs — Script score query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-script-score-query)
- [Elasticsearch Labs — HNSW graph: How to improve Elasticsearch performance](https://www.elastic.co/search-labs/blog/hnsw-graph)