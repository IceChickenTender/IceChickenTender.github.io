---
title: "[ElasticSearch] ElasticSearch 실전 명령어 모음: Index부터 Query DSL, Vector Search 까지"
categories:
  - ElasticSearch
tags:
  - ElasticSearch
  
toc: true
toc_sticky: true
toc_label: "ElasticSearch 실전 명령어 모음: Index부터 Query DSL, Vector Search 까지"
---

# 개요

Elasticsearch를 프로젝트에서 사용하다 보면 Mapping이나 Query DSL의 개념 자체보다 다음과 같은 문법이 갑자기 기억나지 않는 경우가 많습니다.

- Index를 어떻게 만들었지?
- Mapping은 어떻게 확인했지?
- match와 term 문법이 뭐였지?
- Metadata Filter는 어디에 넣었지?
- kNN Search는 어떻게 작성했지?

Elasticsearch 명령어는 매일 반복해서 사용하는 문법이라기보다 Index 생성, Document 적재, 검색 Query 작성, 검색 결과 확인 등의 작업이 필요할 때 간헐적으로 사용하는 경우가 많습니다.

따라서 이 글은 Elasticsearch 개념을 설명하기 위한 글이 아니라 **필요할 때 바로 찾아 사용할 수 있는 개인용 Command Reference**를 목적으로 작성했습니다.

## 환경

- Elasticsearch 9.4.2
- Kibana 9.4.2
- Windows Local
- Basic License
- Kibana Dev Tools
- Python Elasticsearch Client 9.x

Elasticsearch는 버전에 따라 API와 지원 기능이 달라질 수 있습니다.

따라서 이 글에서 명령어를 먼저 확인하되, 실제 프로젝트 적용 시 현재 사용 중인 Elasticsearch 버전의 공식 문서도 함께 확인하는 것을 권장합니다.

---

# 1. 가장 기본적인 REST API 형태

Kibana Dev Tools에서는 다음 형태로 Elasticsearch REST API를 사용합니다.

```text
HTTP_METHOD /경로
{
  JSON Body
}
```

주로 사용하는 HTTP Method는 다음과 같습니다.

```text
GET
→ 조회 / 검색

PUT
→ 명확한 대상 생성 또는 설정

POST
→ 검색 / 수정 / 자동 ID 생성 등

DELETE
→ 삭제
```

예:

```http
GET my-index/_search
{
  "query": {
    "match_all": {}
  }
}
```

---

# 2. ElasticSearch 상태 확인

## 2.1 ElasticSearch 연결 확인

```http
GET /
```

Elasticsearch Version, Cluster 이름 등의 기본 정보를 확인합니다.

## 2.2 Cluster Health 확인

```http
GET /_cluster/health
```

주요 상태:

```text
green
→ Primary / Replica Shard 정상

yellow
→ Primary는 정상이나 일부 Replica가 할당되지 않음

red
→ 일부 Primary Shard가 할당되지 않음
```

Local Single Node 환경에서는 Replica가 존재하면 `yellow`가 나타날 수 있습니다.

## 2.3 Index 목록 확인

```http
GET /_cat/indices?v
```

대표적으로 확인할 수 있는 정보:

```text
health
status
index
docs.count
store.size
```

사람이 직접 빠르게 Index 상태를 확인할 때 사용합니다.

---

# 3. Index

## 3.1 Index 생성

가장 단순한 형태입니다.

```http
PUT my-index
```

## 3.2 Mapping과 함께 Index 생성

실제 프로젝트에서는 명시적으로 Mapping을 정의하는 것이 좋습니다.

```http
PUT financial-documents
{
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
      }
    }
  }
}
```

용도:

```text
text
→ Full-text Search

keyword
→ Exact Match / Filter / Sort / Aggregation

date
→ 날짜 검색 / Range Filter
```

## 3.3 Index 정보 확인

```http
GET financial-documents
```

Index의 Mapping과 Settings 등을 함께 확인할 수 있습니다.

## 3.4 Mapping 확인

```http
GET financial-documents/_mapping
```

Field Type이 기억나지 않을 때 자주 사용합니다.

## 3.5 Settings 확인

```http
GET financial-documents/_settings
```

Shard 등의 Index Settings를 확인합니다.

## 3.6 기존 Index에 Field 추가

```http
PUT financial-documents/_mapping
{
  "properties": {
    "sector": {
      "type": "keyword"
    }
  }
}
```

Mapping Update로 새로운 Field를 추가할 수 있습니다.

기존 Field의 Type을 임의로 변경하는 용도로 사용하면 안 됩니다. Field Mapping 변경이 필요한 경우 일반적으로 새 Mapping을 가진 Index를 만들고 Reindex하는 방식을 고려합니다.

## 3.7 Index 삭제

```http
DELETE financial-documents
```

주의:

```text
Index 삭제
=
Document + Mapping + Index Metadata 삭제
```

복구가 필요한 데이터에 무심코 실행하지 않습니다.

---

# 4. Document

## 4.1 ID를 지정하여 Document 생성

```http
PUT financial-documents/_doc/1
{
  "title": "Samsung HBM Earnings",
  "content": "Samsung semiconductor HBM earnings increased",
  "company": "samsung",
  "document_type": "annual_report",
  "published_at": "2026-03-15"
}
```

`1`을 Document `_id`로 사용합니다.

## 4.2 Elasticsearch가 ID를 자동 생성하도록 Document 생성

```http
POST financial-documents/_doc
{
  "title": "Samsung HBM Earnings",
  "content": "Samsung semiconductor HBM earnings increased",
  "company": "samsung"
}
```

Elasticsearch가 `_id`를 자동 생성합니다.

## 4.3 Document 조회

```http
GET financial-documents/_doc/1
```

## 4.4 `_source`만 조회

```http
GET financial-documents/_source/1
```

Document의 실제 JSON 데이터만 보고 싶을 때 사용합니다.

## 4.5 Document 일부 수정

```http
POST financial-documents/_update/1
{
  "doc": {
    "company": "samsung_electronics"
  }
}
```

전체 Document를 다시 작성하지 않고 일부 Field를 수정할 때 사용합니다.

## 4.6 Document 삭제

```http
DELETE financial-documents/_doc/1
```

## 4.7 Document 개수 확인

```http
GET financial-documents/_count
```

조건을 적용할 수도 있습니다.

```http
GET financial-documents/_count
{
  "query": {
    "term": {
      "company": "samsung"
    }
  }
}
```

---

# 5. Bulk API

여러 Document를 한 번에 Indexing할 때 사용합니다.

```http
POST _bulk
{"index":{"_index":"financial-documents","_id":"1"}}
{"title":"Samsung HBM Earnings","company":"samsung"}
{"index":{"_index":"financial-documents","_id":"2"}}
{"title":"SK Hynix HBM Report","company":"sk_hynix"}
```

Bulk API에서는 다음과 같이 **Action Line과 Document Line을 줄 단위로 구분**합니다.

```text
Action
Document
Action
Document
...
```

대량 Document를 하나씩 API 호출하는 것보다 Bulk API 또는 Python Bulk Helper를 사용하는 것이 적절합니다.

---

# 6. Analyzer 확인

Text가 실제로 어떤 Token으로 분석되는지 확인할 때 `_analyze` API를 사용합니다.

## Standard Analyzer

```http
POST _analyze
{
  "analyzer": "standard",
  "text": "Samsung HBM Earnings"
}
```

---

## 특정 Index의 Analyzer 사용

```http
POST financial-documents/_analyze
{
  "field": "content",
  "text": "Samsung HBM Earnings"
}
```

해당 Field에 Mapping된 Analyzer 기준으로 분석 결과를 확인할 수 있습니다.

Analyzer 문제를 디버깅할 때 매우 유용합니다.

---

# 7. 전체 Document 검색

```http
GET financial-documents/_search
{
  "query": {
    "match_all": {}
  }
}
```

기본적으로 모든 Document에 Matching됩니다.

---

# 8. match Query

## 용도

`text` Field에서 Full-text Search를 수행합니다.

검색어도 Analyzer를 거쳐 Term으로 변환됩니다.

```http
GET financial-documents/_search
{
  "query": {
    "match": {
      "content": "HBM earnings"
    }
  }
}
```

기본 Operator는 `OR`입니다.

즉:

```text
HBM OR earnings
```

형태로 Matching될 수 있습니다.

## 모든 Term을 요구하고 싶은 경우

```http
GET financial-documents/_search
{
  "query": {
    "match": {
      "content": {
        "query": "HBM earnings",
        "operator": "and"
      }
    }
  }
}
```

```text
HBM AND earnings
```

에 해당하는 검색입니다.

---

# 9. term Query

## 용도

Indexed Term을 기준으로 정확하게 Matching합니다.

검색 문자열에 Full-text Analysis를 적용하지 않습니다.

Metadata 검색에 자주 사용합니다.

```http
GET financial-documents/_search
{
  "query": {
    "term": {
      "company": "samsung"
    }
  }
}
```

대표적인 사용 대상:

```text
company
document_type
status
category
ID
```

즉 `keyword` Field와 함께 사용하는 경우가 많습니다.

주의:

```text
match
→ Full-text Search

term
→ Exact Term Matching
```

`text` Field에 `term` Query를 무조건 사용하지 않습니다.

---

# 10. range Query

날짜나 숫자의 범위를 검색합니다.

## 날짜

```http
GET financial-documents/_search
{
  "query": {
    "range": {
      "published_at": {
        "gte": "2026-01-01",
        "lte": "2026-12-31"
      }
    }
  }
}
```

주요 연산자:

```text
gt
→ >

gte
→ >=

lt
→ <

lte
→ <=
```

---

# 11. bool Query

여러 Query 조건을 조합할 때 사용합니다.

기본 구조:

```http
GET financial-documents/_search
{
  "query": {
    "bool": {
      "must": [],
      "should": [],
      "filter": [],
      "must_not": []
    }
  }
}
```

## must

반드시 Matching해야 하며 Query Context이므로 관련도 Score 계산에도 참여합니다.

```http
GET financial-documents/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "content": "semiconductor"
          }
        },
        {
          "match": {
            "content": "earnings"
          }
        }
      ]
    }
  }
}
```

주의:

```text
must 안에 match 2개
```

와

```text
match 하나에 "semiconductor earnings"
```

는 동일하지 않습니다.

후자의 `match`는 기본 Operator가 `OR`이기 때문입니다.

## filter

반드시 조건을 만족해야 하지만 `_score`에는 영향을 주지 않습니다.

Metadata 조건에 주로 사용합니다.

```http
GET financial-documents/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "content": "HBM earnings"
          }
        }
      ],
      "filter": [
        {
          "term": {
            "company": "samsung"
          }
        },
        {
          "term": {
            "document_type": "annual_report"
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
```

## should

선택적인 Relevance Signal을 추가합니다.

```http
GET financial-documents/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "content": "semiconductor"
          }
        }
      ],
      "should": [
        {
          "match": {
            "content": "HBM"
          }
        }
      ]
    }
  }
}
```

`HBM`까지 Matching되면 Score가 높아질 수 있습니다.

## minimum_should_match

특정 개수 이상의 `should` 조건을 반드시 만족하게 할 수 있습니다.

```http
GET financial-documents/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "content": "semiconductor"
          }
        }
      ],
      "should": [
        {
          "match": {
            "content": "HBM"
          }
        },
        {
          "match": {
            "content": "earnings"
          }
        }
      ],
      "minimum_should_match": 1
    }
  }
}
```

## must_not

조건에 Matching되는 Document를 제외합니다.

```http
GET financial-documents/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "content": "semiconductor"
          }
        }
      ],
      "must_not": [
        {
          "term": {
            "document_type": "news"
          }
        }
      ]
    }
  }
}
```

`must_not`은 Filter Context에서 동작하며 Score에 영향을 주지 않습니다.

---

# 12. 반환 결과 개수 제한

```http
GET financial-documents/_search
{
  "size": 5,
  "query": {
    "match": {
      "content": "HBM"
    }
  }
}
```

`size`는 **Search Response에서 반환할 Hit 수**입니다.

Vector Search의 `k`와 다른 개념입니다.

---

# 13. 특정 Field만 반환하기

검색 결과에서 불필요한 Field를 제외하고 싶은 경우 `_source` Filtering을 사용할 수 있습니다.

```http
GET financial-documents/_search
{
  "_source": [
    "title",
    "company",
    "published_at"
  ],
  "query": {
    "match": {
      "content": "HBM"
    }
  }
}
```

결과에서 필요한 Metadata만 확인하고 싶을 때 편리합니다.

---

# 14. BM25 Score 분석: `_explain`

특정 Document가 왜 해당 Score를 받았는지 확인할 때 사용합니다.

```http
GET financial-documents/_explain/1
{
  "query": {
    "match": {
      "content": "HBM"
    }
  }
}
```

다음과 같은 BM25 요소를 확인할 때 유용합니다.

```text
freq
idf
k1
b
dl
avgdl
```

검색 결과의 Score가 예상과 다를 때 사용할 수 있습니다.

---

# 15. Dense Vector Mapping

Vector Search를 사용하기 위해 `dense_vector` Field를 정의합니다.

예:

```http
PUT financial-vector-index
{
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
      "published_at": {
        "type": "date"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```

`dims`는 사용하는 Embedding Model의 Vector Dimension과 일치해야 합니다.

예:

```text
Model Vector Dimension = 768

↓

dense_vector dims = 768
```

---

# 16. Vector를 포함한 Document 저장

```http
PUT financial-vector-index/_doc/1
{
  "title": "Samsung HBM Earnings",
  "content": "Samsung semiconductor HBM earnings increased",
  "company": "samsung",
  "published_at": "2026-03-15",
  "embedding": [
    0.123,
    0.456,
    0.789
  ]
}
```

실제 프로젝트에서는 Embedding Model이 생성한 전체 차원의 Vector를 저장합니다.

---

# 17. Approximate kNN Search

실습에서 사용한 형태입니다.

```http
GET financial-vector-index/_search
{
  "size": 5,
  "query": {
    "knn": {
      "field": "embedding",
      "query_vector": [
        0.123,
        0.456,
        0.789
      ],
      "k": 5,
      "num_candidates": 50
    }
  }
}
```

각 값의 의미:

```text
field
→ 검색할 dense_vector Field

query_vector
→ User Query의 Embedding Vector

k
→ 최종적으로 찾으려는 nearest neighbors 수

num_candidates
→ ANN 검색 과정에서 고려할 Candidate 범위

size
→ Search Response에서 반환할 Hit 수
```

핵심:

```text
k ≠ num_candidates ≠ size
```

---

# 18. kNN + Metadata Filter

RAG 프로젝트에서 특히 자주 사용할 가능성이 높은 형태입니다.

```http
GET financial-vector-index/_search
{
  "size": 5,
  "query": {
    "knn": {
      "field": "embedding",
      "query_vector": [
        0.123,
        0.456,
        0.789
      ],
      "k": 5,
      "num_candidates": 50,
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

개념적으로:

```text
Vector Similarity Search

조건:
company = samsung
published_at >= 2025-01-01
```

를 함께 수행합니다.

---

# 19. Exact Vector Search

현재 실습 환경인 Elasticsearch 9.4.2에서는 `script_score`를 사용하여 Exact Search를 수행했습니다.

```http
GET financial-vector-index/_search
{
  "size": 5,
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "(cosineSimilarity(params.query_vector, 'embedding') + 1.0) / 2.0",
        "params": {
          "query_vector": [
            0.123,
            0.456,
            0.789
          ]
        }
      }
    }
  }
}
```

이 방식은 Matching된 Document Vector를 직접 비교하는 Brute-force 방식입니다.

따라서 대규모 검색에서는 ANN보다 비용이 커질 수 있습니다.

실전에서는 주로 다음 용도로 생각할 수 있습니다.

```text
작은 데이터
Exact 결과 확인
ANN Recall 검증
제한된 Candidate 재계산
```

참고로 최신 Elasticsearch 버전에는 별도의 `dense_vector` Exact Query가 추가되어 있으므로, 다른 버전에서 사용할 때는 최신 공식 문서를 확인합니다.

---

# 20. Python ElasticSearch Client 설치

Elasticsearch 9.x 환경에서는 9.x Python Client를 사용할 수 있습니다.

```bash
pip install "elasticsearch>=9,<10"
```

---

# 21. Python에서 Elasticsearch 연결

## Security를 비활성화한 Local 환경

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

print(es.info())
```

---

## HTTPS + 인증 환경

```python
import os

from elasticsearch import Elasticsearch


es = Elasticsearch(
    "https://localhost:9200",
    ca_certs=r"C:\path\to\http_ca.crt",
    basic_auth=(
        "elastic",
        os.environ["ELASTIC_PASSWORD"]
    )
)
```

비밀번호를 Source Code에 직접 작성하지 않고 환경변수 등을 이용합니다.

---

# 22. Python에서 Document Indexing

```python
es.index(
    index="financial-documents",
    id="1",
    document={
        "title": "Samsung HBM Earnings",
        "content": "Samsung semiconductor HBM earnings increased",
        "company": "samsung",
        "published_at": "2026-03-15"
    }
)
```

---

# 23. Python Bulk Indexing

```python
from elasticsearch import helpers


documents = [
    {
        "_index": "financial-documents",
        "_id": "1",
        "title": "Samsung HBM Earnings",
        "company": "samsung"
    },
    {
        "_index": "financial-documents",
        "_id": "2",
        "title": "SK Hynix HBM Report",
        "company": "sk_hynix"
    }
]


helpers.bulk(
    es,
    documents
)
```

대량 데이터를 Indexing할 때 사용합니다.

---

# 24. Python에서 BM25 Search

```python
response = es.search(
    index="financial-documents",
    size=5,
    query={
        "match": {
            "content": "HBM earnings"
        }
    }
)

hits = response["hits"]["hits"]

for hit in hits:
    print(
        hit["_id"],
        hit["_score"],
        hit["_source"]
    )
```

Kibana Dev Tools의 JSON Query DSL 구조를 Python Dictionary 형태로 거의 그대로 사용할 수 있습니다.

---

# 25. Python에서 Metadata Filter

```python
response = es.search(
    index="financial-documents",
    size=5,
    query={
        "bool": {
            "must": [
                {
                    "match": {
                        "content": "HBM earnings"
                    }
                }
            ],
            "filter": [
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
)
```

---

# 26. Python에서 Vector Search

```python
response = es.search(
    index="financial-vector-index",
    size=5,
    query={
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": 5,
            "num_candidates": 50
        }
    }
)
```

`query_vector`에는 Embedding Model이 User Query로부터 생성한 Vector를 전달합니다.

---

# 27. Python에서 RRF 직접 구현

현재 실습 환경에서는 Elasticsearch Native RRF Retriever를 Basic License에서 실행할 수 없었기 때문에 Application Layer에서 직접 구현했습니다.

```python
from collections import defaultdict


def rrf(rankings, rank_constant=60):
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(
            ranking,
            start=1
        ):
            scores[doc_id] += (
                1 /
                (rank_constant + rank)
            )

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
```

사용:

```python
bm25_ranking = [
    hit["_id"]
    for hit in bm25_hits
]

vector_ranking = [
    hit["_id"]
    for hit in vector_hits
]


result = rrf(
    rankings=[
        bm25_ranking,
        vector_ranking
    ],
    rank_constant=60
)
```

전체 구조:

```text
User Query
    │
    ├───────────────┐
    │               │
    ▼               ▼
BM25 Search     Embedding Model
    │               │
    │          Query Vector
    │               │
    │          Vector Search
    │               │
    ▼               ▼
BM25 Ranking   Vector Ranking
    │               │
    └───────┬───────┘
            ▼
      Python Backend
            │
           RRF
            │
            ▼
      Final Ranking
```

---

# 28. Native RRF Retriever 참고

Elasticsearch에는 Native RRF Retriever도 존재합니다.

개념적인 구조는 다음과 같습니다.

```json
{
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": {
              "match": {
                "content": "HBM earnings"
              }
            }
          }
        },
        {
          "knn": {
            "field": "embedding",
            "query_vector": [
              0.123,
              0.456,
              0.789
            ],
            "k": 10,
            "num_candidates": 50
          }
        }
      ],
      "rank_constant": 60,
      "rank_window_size": 50
    }
  }
}
```

다만 현재 학습 환경의 Basic License에서는 실행 시 다음 오류가 발생했습니다.

```text
current license is non-compliant for
[Reciprocal Rank Fusion (RRF)]
```

따라서 프로젝트에서는 필요에 따라 Python Backend에서 직접 RRF를 구현할 수 있습니다.

라이선스 정책은 변경될 수 있으므로 실제 사용 시 최신 Elastic Subscription 정보를 확인합니다.

---

# 29. 문제 발생 시 확인 순서

검색 결과가 예상과 다를 때는 다음 순서로 확인합니다.

```text
1. Mapping 확인
   GET index/_mapping

2. Analyzer 확인
   POST index/_analyze

3. Query 자체 확인
   match / term / bool / filter

4. 검색 결과 _score 확인

5. BM25 원인이 궁금하면
   _explain 확인

6. Vector Search라면
   dims / similarity 확인

7. k / num_candidates 확인

8. Metadata Filter 조건 확인
```

즉 Query만 계속 수정하기보다:

```text
Mapping
→ Analysis
→ Query
→ Score
```

순서로 문제를 좁히는 것이 좋습니다.

---

# 마무리

이 글은 Elasticsearch 공식 문서를 대체하기 위한 자료가 아니라 **프로젝트 중 빠르게 기억을 복구하기 위한 개인용 Reference**입니다.

Elasticsearch를 사용하다 문법이 기억나지 않을 때 먼저 이 글에서 구조를 확인하고, 버전이나 세부 Parameter가 중요한 경우 공식 문서를 추가로 확인하는 방식으로 사용합니다.

# 참조

- [Elasticsearch REST APIs](https://www.elastic.co/docs/reference/elasticsearch/rest-apis)
- [Create an index API](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-indices-create)
- [Delete index API](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-indices-delete)
- [Update mapping API](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-indices-put-mapping)
- [CAT indices API](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-cat-indices)
- [Cluster health API](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-cluster-health)
- [Query DSL](https://www.elastic.co/docs/reference/query-languages/querydsl)
- [Match query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-query)
- [Term-level queries](https://www.elastic.co/docs/reference/query-languages/query-dsl/term-level-queries)
- [Range query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-range-query)
- [Query and filter context](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-filter-context)
- [Explain API](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-explain)
- [Dense vector field type](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector)
- [Script score query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-script-score-query)
- [Elasticsearch Python Client](https://www.elastic.co/docs/reference/elasticsearch/clients/python)
- [Python Client - Connecting](https://www.elastic.co/docs/reference/elasticsearch/clients/python/connecting)
- [Python Client - Getting started](https://www.elastic.co/docs/reference/elasticsearch/clients/python/getting-started)