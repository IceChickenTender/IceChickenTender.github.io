---
title: "[ElasticSearch] 3. ElasticSearch의 BM25와 검색 결과의 관련도 계산"
categories:
  - ElasticSearch
tags:
  - ElasticSearch
  
toc: true
toc_sticky: true
toc_label: "ElasticSearch의 BM25와 검색 결과의 관련도 계산"
---

# 개요

이전 학습에서는 Elasticsearch가 Analyzer를 통해 Text를 검색 가능한 Term으로 만들고, Inverted Index를 이용해 해당 Term이 포함된 Document를 빠르게 찾는 과정을 살펴보았습니다.

하지만 Inverted Index만으로는 한 가지 문제가 남습니다.

```text
Query: "삼성전자 반도체"

          ↓

Inverted Index

          ↓

Doc1
Doc2
Doc3
```

세 Document가 모두 검색어와 Matching된다면 **어떤 Document를 가장 먼저 보여줘야 할까요?**

이 문제를 해결하기 위해 Elasticsearch는 각 Document와 Query의 관련성을 계산하는 **Relevance Scoring**을 수행하며, 기본적으로 **BM25**를 사용합니다.

---

# 1. Inverted Index와 BM25의 역할

먼저 두 개념의 역할을 구분해야 합니다.

```text
Inverted Index
→ Query의 Term이 포함된 Document를 빠르게 찾습니다.

BM25
→ 검색된 Document가 Query와 얼마나 관련 있는지 점수를 계산합니다.
```

즉 Inverted Index가 **검색 후보를 찾는 구조**라면, BM25는 **검색 결과의 순위를 결정하기 위한 Scoring 방식**입니다.

Elasticsearch는 계산된 관련도를 `_score`에 저장하고 일반적으로 `_score`가 높은 Document를 더 위에 배치합니다.

---

# 2. TF(Term Frequency)

가장 단순하게 생각하면 특정 검색어가 문서에 많이 등장할수록 해당 문서와 관련성이 높다고 볼 수 있습니다.

예를 들어 Query가 `"반도체"`라고 했을 때:

```text
Doc1: 반도체 시장 전망
Doc2: 반도체 반도체 반도체 시장 전망
```

Doc2가 반도체에 관한 내용을 더 많이 다루고 있을 가능성이 있습니다.

이때 사용하는 것이 **TF(Term Frequency)**입니다.

> **TF는 하나의 Document 안에서 특정 Term이 등장한 횟수입니다.**

하지만 단순히 등장 횟수가 많다고 점수를 계속 같은 비율로 증가시키는 것은 문제가 있습니다.

---

# 3. TF Saturation

다음 두 변화는 같은 의미를 가지고 있을까요?

```text
"반도체" 1회 → 2회

"반도체" 99회 → 100회
```

두 경우 모두 등장 횟수는 1 증가했지만 검색 관점에서 주는 정보는 동일하지 않습니다.

처음 몇 번의 등장은 해당 문서가 특정 주제를 다루고 있다는 강한 신호가 될 수 있지만 이미 수십 번 등장한 이후 한 번 더 등장하는 것은 상대적으로 적은 정보를 제공합니다.

BM25는 이를 **TF Saturation**으로 반영합니다.

```text
TF 증가
   ↓
Score 증가

하지만

TF가 높아질수록
추가 등장에 따른 Score 증가폭은 감소
```

BM25의 `k1` Parameter가 이 TF Saturation의 정도를 조절합니다.

- `k1`이 작을수록 TF의 영향이 빠르게 포화됩니다.
- `k1`이 클수록 Term 반복 횟수의 영향이 더 오래 유지됩니다.

Elasticsearch의 기본 `k1`은 `1.2`이지만 값을 암기하는 것보다 **TF 증가 효과가 무한정 선형적으로 증가하지 않는다는 점**을 이해하는 것이 중요합니다.

---

# 4. DF와 IDF

Term의 등장 횟수뿐 아니라 **Term 자체가 얼마나 흔한지도** 중요합니다.

예를 들어 100개의 Document가 있을 때:

```text
"기업"
→ 90개 Document에 존재

"HBM3E"
→ 2개 Document에 존재
```

`기업`이라는 Term은 많은 Document에서 등장하기 때문에 특정 문서를 구별하는 정보가 적습니다.

반면 `HBM3E`는 매우 적은 Document에만 존재하기 때문에 특정 주제를 구별하는 강한 정보가 될 수 있습니다.

여기서 사용하는 개념이 **DF(Document Frequency)**와 **IDF(Inverse Document Frequency)**입니다.

## DF

> **특정 Term을 포함하고 있는 Document의 수**

입니다.

주의할 점은 **전체 Term 등장 횟수가 아니라 Document 수**라는 것입니다.

예를 들어:

```text
Doc1: 반도체 반도체 반도체
Doc2: 반도체
Doc3: 자동차
```

라면:

```text
TF("반도체", Doc1) = 3
DF("반도체") = 2
```

입니다.

## IDF

IDF는 DF를 이용하여 **전체 Document에서 희귀한 Term에 더 높은 중요도를 부여하기 위한 값**입니다.

```text
DF가 큼
→ 흔한 Term
→ IDF 낮음

DF가 작음
→ 희귀한 Term
→ IDF 높음
```

IDF는 단순히 `1 / DF`를 계산하는 것은 아닙니다.

Lucene의 BM25에서는 다음과 같은 형태를 사용합니다.

```text
                N - DF + 0.5
IDF = log(1 + ───────────────)
                  DF + 0.5
```

여기서 `N`은 전체 Document 수, `DF`는 해당 Term을 포함한 Document 수입니다.

수식을 암기할 필요는 없으며 **희귀한 Term일수록 검색 결과를 구분하는 데 더 큰 영향을 준다**는 것이 핵심입니다.

---

# 5. Document Length Normalization

TF가 같더라도 Document의 길이가 다르면 의미가 달라질 수 있습니다.

```text
Doc A
전체 10단어
"반도체" 2회

Doc B
전체 10,000단어
"반도체" 2회
```

두 Document 모두 TF는 2입니다.

하지만 짧은 Document에서 두 번 등장했다면 해당 Document가 반도체를 중심으로 작성되었을 가능성이 더 높은 반면, 매우 긴 Document에서는 일부 내용에서만 언급되었을 가능성이 있습니다.

BM25는 따라서 현재 Field의 길이를 전체 Document의 평균 Field Length와 비교하여 TF를 정규화합니다.

이것을 **Document Length Normalization**이라고 합니다.

BM25의 `b` Parameter가 Length Normalization의 적용 정도를 조절합니다.

```text
b = 0
→ Document Length를 고려하지 않음

b가 커짐
→ Document Length의 영향을 더 많이 반영
```

Elasticsearch의 기본값은 `0.75`입니다.

---

# 6. BM25의 전체 구조

BM25의 대표적인 구조를 단순화하면 다음과 같습니다.

```text
                     TF × (k1 + 1)
Score ≈ IDF × ──────────────────────────────
                 TF + k1 × (1 - b + b × DL / AVGDL)
```

각 요소의 의미는 다음과 같습니다.

```text
TF
→ 현재 Document에서 Term이 등장한 횟수

IDF
→ 전체 Document에서 Term이 얼마나 희귀한지

DL
→ 현재 Field의 길이

AVGDL
→ 전체 Document의 평균 Field 길이

k1
→ TF Saturation 조절

b
→ Document Length Normalization 조절
```

수식 자체보다 각각의 요소가 **어떤 문제를 해결하기 위해 존재하는지** 이해하는 것이 중요합니다.

---

# 7. ElasticSearch의 `_score`

Elasticsearch에서 Query를 실행하면 검색 결과에 `_score`가 포함됩니다.

```json
{
  "_id": "1",
  "_score": 0.22523367,
  "_source": {
    "content": "..."
  }
}
```

`_score`는 해당 Query에 대해 Document가 얼마나 관련 있다고 계산되었는지를 나타내는 Relevance Score입니다.

일반적으로:

```text
_score가 높음
→ 해당 Query에 더 높은 관련성을 가진 것으로 평가
```

합니다.

다만 다음처럼 해석하면 안 됩니다.

```text
_score = 2
_score = 1

→ 첫 번째 Document가 정확히 두 배 관련 있다. (X)
```

`_score`는 관련도 확률이나 퍼센트가 아니라 **같은 검색 Context에서 Document의 상대적인 Ranking을 결정하기 위한 값**으로 이해하는 것이 적절합니다.

---

# 8. 실습 1: TF와 TF Saturation 확인

Document Length를 동일하게 유지하고 `semiconductor`의 등장 횟수만 변경했습니다.

```text
Doc1
semiconductor market outlook growth

Doc2
semiconductor semiconductor outlook growth

Doc3
semiconductor semiconductor semiconductor growth
```

## 8.1 Index 생성

```http
DELETE bm25-tf-test

PUT bm25-tf-test
{
  "settings": {
    "number_of_shards": 1
  },
  "mappings": {
    "properties": {
      "content": {
        "type": "text"
      }
    }
  }
}
```

## 8.2 Document 저장

```http
POST bm25-tf-test/_doc/1
{
  "content": "semiconductor market outlook growth"
}

POST bm25-tf-test/_doc/2
{
  "content": "semiconductor semiconductor outlook growth"
}

POST bm25-tf-test/_doc/3
{
  "content": "semiconductor semiconductor semiconductor growth"
}

POST bm25-tf-test/_refresh
```

## 8.3 검색

```http
GET bm25-tf-test/_search
{
  "query": {
    "match": {
      "content": "semiconductor"
    }
  }
}
```

## 8.4 결과

실제 결과는 다음과 같았습니다.

| Document | TF | `_score` |
|---|---:|---:|
| Doc1 | 1 | 0.13353139 |
| Doc2 | 2 | 0.18360566 |
| Doc3 | 3 | 0.20983505 |

TF가 증가하면서 `_score`도 증가했지만 증가폭은 점점 작아졌습니다.

```text
TF 1 → 2
Score 증가폭이 큼

TF 2 → 3
Score는 증가하지만 증가폭 감소
```

이를 통해 **Term Frequency가 증가한다고 Score가 선형적으로 증가하지 않는 TF Saturation**을 직접 확인할 수 있었습니다.

---

# 9. 실습 2: Document Length Normalization 확인

이번에는 `semiconductor`의 TF를 모두 2로 유지하고 Document Length만 변경했습니다.

## 9.1 Index 생성

```http
DELETE bm25-length-test

PUT bm25-length-test
{
  "settings": {
    "number_of_shards": 1
  },
  "mappings": {
    "properties": {
      "content": {
        "type": "text"
      }
    }
  }
}
```

## 9.2 Document 저장

```http
POST bm25-length-test/_doc/1
{
  "content": "semiconductor semiconductor market growth"
}

POST bm25-length-test/_doc/2
{
  "content": "semiconductor semiconductor market growth company business investment technology economy outlook"
}

POST bm25-length-test/_doc/3
{
  "content": "semiconductor semiconductor market growth company business investment technology economy outlook revenue profit report industry global domestic strategy product demand supply forecast"
}

POST bm25-length-test/_refresh
```

## 9.3 검색

```http
GET bm25-length-test/_search
{
  "query": {
    "match": {
      "content": "semiconductor"
    }
  }
}
```

## 9.4 결과

실제 검색 결과는 다음과 같았습니다.

| Document | TF | Field Length | `_score` |
|---|---:|---:|---:|
| Doc1 | 2 | 4 | 0.22523367 |
| Doc2 | 2 | 10 | 0.19129148 |
| Doc3 | 2 | 21 | 0.14988218 |

TF는 모두 동일하지만 Field가 길어질수록 `_score`가 감소했습니다.

이를 통해 BM25가 단순히 Term의 등장 횟수만 사용하는 것이 아니라 **Document Length를 함께 고려해 TF를 정규화한다는 것**을 확인할 수 있었습니다.

---

# 10. 실습 3: IDF 확인

## 10.1 Index 생성

```http
DELETE bm25-idf-test

PUT bm25-idf-test
{
  "settings": {
    "number_of_shards": 1
  },
  "mappings": {
    "properties": {
      "content": {
        "type": "text"
      }
    }
  }
}
```

## 10.2 Document 저장

```http
POST bm25-idf-test/_doc/1
{
  "content": "market semiconductor hbm3e"
}

POST bm25-idf-test/_doc/2
{
  "content": "market semiconductor memory"
}

POST bm25-idf-test/_doc/3
{
  "content": "market semiconductor investment"
}

POST bm25-idf-test/_doc/4
{
  "content": "market semiconductor company"
}

POST bm25-idf-test/_refresh
```

## 10.3 검색

```http
GET bm25-idf-test/_search
{
  "query": {
    "match": {
      "content": "market hbm3e"
    }
  }
}
```

## 10.4 결과

| Document | `_score` |
|---|---:|
| Doc1 | 1.3093333 |
| Doc2 | 0.10536051 |
| Doc3 | 0.10536051 |
| Doc4 | 0.10536051 |

`market`은 모든 Document에 존재하지만 `hbm3e`는 Doc1에만 존재합니다.

따라서 `hbm3e`가 훨씬 높은 IDF를 가지며, 해당 Term을 포함하는 Doc1의 Score에 크게 기여한 것을 확인할 수 있었습니다.

---

# 11. 실습 4: `_explain`으로 BM25 계산 확인

Elasticsearch의 `_explain` API를 이용하면 특정 Document가 왜 해당 `_score`를 받았는지 확인할 수 있습니다.

```http
GET bm25-tf-test/_explain/3
{
  "query": {
    "match": {
      "content": "semiconductor"
    }
  }
}
```

실제 결과에서는 다음 값을 확인할 수 있었습니다.

```text
freq  = 3
k1    = 1.2
b     = 0.75
dl    = 4
avgdl = 4
idf   = 0.13353139
```

특히 BM25의 TF 부분은 다음 값으로 계산되었습니다.

```text
freq = 3

        ↓

k1, b, dl, avgdl 적용

        ↓

tf = 0.71428573
```

즉 `freq=3`이라는 raw Term Frequency가 그대로 Score에 들어가는 것이 아니라 **BM25의 TF Saturation과 Document Length Normalization을 거쳐 변환된 값**이 사용됩니다.

또한 TF 실험에서는:

```text
Doc1 freq = 1
Doc3 freq = 3
```

을 확인했고, Document Length 실험에서는:

```text
Doc1 dl = 4
Doc3 dl = 21
```

을 확인했습니다.

따라서 `_score`의 차이가 단순한 결과값이 아니라 실제 BM25의 TF와 Document Length 계산에 의해 만들어졌다는 것을 확인할 수 있었습니다.

---

# 정리

BM25를 이해할 때 수식 자체를 암기하는 것보다 다음 질문에 답할 수 있는 것이 중요합니다.

- 특정 Term이 문서에서 많이 등장하면 왜 관련성이 높아질 수 있는가?
- 그렇다고 Term이 100번 등장하면 1번 등장한 문서보다 왜 100배 중요하다고 볼 수 없는가?
- 모든 Document에 등장하는 흔한 Term보다 희귀한 Term이 왜 더 중요한가?
- 동일한 TF라도 Document Length에 따라 왜 중요도가 달라질 수 있는가?

BM25는 이러한 문제를 다음 요소를 이용해 해결합니다.

```text
TF
→ Document 내부의 Term 빈도

IDF
→ Corpus 전체에서 Term의 희귀성

TF Saturation
→ 반복 횟수의 영향이 과도하게 커지는 것을 방지

Document Length Normalization
→ 긴 Document가 단순히 길다는 이유로 유리해지는 것을 보정

k1
→ TF Saturation 조절

b
→ Length Normalization 조절
```

이를 통해 Elasticsearch는 Inverted Index로 찾은 Document들에 `_score`를 부여하고 Query와 더 관련성이 높은 Document를 상위에 배치합니다.

다음 학습에서는 실제 Elasticsearch Query를 다루면서 `match`, `term`, `bool`, `must`, `should`, `filter`가 지금까지 학습한 **Analyzer, Inverted Index, BM25**와 어떻게 연결되는지 살펴봅니다.

---

# 참조

- [Elastic Docs — Similarity settings](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)
- [Elastic Docs — Similarity mapping parameter](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/similarity)
- [Elastic Docs — Ranking and reranking](https://www.elastic.co/docs/solutions/search/ranking)
- [Elastic Docs — Query and filter context](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-filter-context)
- [Elastic Docs — Explain a document match result](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-explain)
- [Apache Lucene — BM25Similarity](https://lucene.apache.org/core/10_2_2/core/org/apache/lucene/search/similarities/BM25Similarity.html)