---
title: "[ElasticSearch] 4. ElasticSearch Query DSL과 Metadata Filtering"
categories:
  - ElasticSearch
tags:
  - ElasticSearch
  
toc: true
toc_sticky: true
toc_label: "ElasticSearch Query DSL과 Metadata Filtering"
---

# 개요

이전 학습에서는 Elasticsearch가 Text를 Analyzer로 분석해 Term을 만들고, Inverted Index를 통해 관련 Document를 찾은 뒤 BM25를 이용해 관련도 점수인 `_score`를 계산하는 과정을 살펴보았습니다.

이번에는 이러한 검색 기능을 실제로 사용하기 위한 **Query DSL**과, Full-Text Search에 구조화된 조건을 결합하는 **Metadata Filtering**을 정리합니다.

---

# 1. Query DSL이란?

**Query DSL(Query Domain Specific Language)**은 Elasticsearch에서 **어떤 Field를 어떤 조건과 방식으로 검색할 것인지 JSON 형태로 표현하는 검색 언어**입니다.

Elasticsearch의 Query는 크게 다음 두 종류로 볼 수 있습니다.

- **Leaf Query**: 특정 Field에서 값을 직접 검색하는 Query입니다. `match`, `term`, `range` 등이 이에 해당합니다.
- **Compound Query**: 여러 Query를 조합하거나 동작 방식을 변경하는 Query입니다. 대표적으로 `bool` Query가 있습니다.

예를 들어 다음은 `content` Field에서 `"semiconductor"`를 검색하는 가장 기본적인 Query입니다.

```http
GET query-dsl-test/_search
{
  "query": {
    "match": {
      "content": "semiconductor"
    }
  }
}
```

Query DSL은 검색 결과가 나온 뒤 적용하는 기능이 아니라,

> **Elasticsearch에게 어떤 검색을 수행할 것인지 지시하는 검색 요청 자체**

라고 이해할 수 있습니다.

---

# 2. Query Context와 Filter Context

Elasticsearch의 Query clause는 **어떤 Context에서 실행되는가**에 따라 Score 계산 여부가 달라집니다.

## Query Context

**Query Context는 “이 Document가 Query와 얼마나 잘 맞는가?”를 판단하는 검색 영역**입니다.

Document가 검색 조건을 만족하는지 확인할 뿐만 아니라 관련도를 계산하고 `_score`에 반영합니다.

```text
Query Context
→ Matching 여부 판단
→ Relevance Score 계산
→ Ranking에 사용
```

## Filter Context

**Filter Context는 “이 Document가 조건을 만족하는가?”만 판단하는 검색 영역**입니다.

관련도가 얼마나 높은지는 계산하지 않습니다.

```text
Filter Context
→ 조건 만족 여부 판단
→ Yes / No
→ Score 계산 X
```

예를 들어:

```text
company = samsung
published_at >= 2025-01-01
```

과 같은 조건은 삼성전자와 얼마나 관련 있는지를 평가할 필요가 없습니다.

조건을 만족하는지만 판단하면 됩니다.

---

# 3. 무엇이 Query Context이고 무엇이 Filter Context인가?

중요한 점은 **`match`, `term`, `range` 같은 Query 종류 자체가 Query Context 또는 Filter Context로 고정되는 것이 아니라는 것입니다.**

Query가 **어느 위치에서 실행되는지**에 따라 Context가 결정됩니다.

`bool` Query를 기준으로 정리하면 다음과 같습니다.

| 위치 | Context | Score 계산 |
|---|---|---|
| 최상위 `query` | Query Context | O |
| `bool.must` | Query Context | O |
| `bool.should` | Query Context | O |
| `bool.filter` | Filter Context | X |
| `bool.must_not` | Filter Context | X |

예를 들어 같은 `term` Query라도:

```json
{
  "query": {
    "term": {
      "company": "samsung"
    }
  }
}
```

처럼 최상위 `query`에서 사용하면 **Query Context**에서 실행됩니다.

반면:

```json
{
  "query": {
    "bool": {
      "filter": [
        {
          "term": {
            "company": "samsung"
          }
        }
      ]
    }
  }
}
```

처럼 `filter` 안에서 사용하면 **Filter Context**에서 실행됩니다.

따라서:

```text
match = 항상 Query Context
term = 항상 Filter Context
```

라고 이해하면 안 됩니다.

더 정확하게는:

```text
match / term / range
→ 어떤 검색을 수행할 것인가를 결정하는 Query 종류

must / should / filter / must_not
→ bool 안에서 Query가 어떤 방식으로 실행될지를 결정
```

이라고 이해하는 것이 좋습니다.

---

# 4. `match` Query

**`match` Query는 입력된 Text를 Analyzer로 분석한 뒤 검색하는 대표적인 Full-Text Query**입니다.

```http
GET query-dsl-test/_search
{
  "query": {
    "match": {
      "title": "Semiconductor"
    }
  }
}
```

예를 들어 `title` Field가 기본 `standard` analyzer를 사용하고 있다면:

```text
Semiconductor
      ↓
Analyzer
      ↓
semiconductor
```

와 같이 검색 Query도 분석됩니다.

검색 흐름은 다음과 같습니다.

```text
Query Text
   ↓
Analyzer
   ↓
Query Term
   ↓
Inverted Index
   ↓
Matching
   ↓
BM25
   ↓
_score
```

## 여러 Term을 검색할 때

`match` Query에 다음과 같이 여러 단어가 들어간 경우:

```text
"semiconductor earnings"
```

Analyzer가 이를 여러 Term으로 나눈 뒤 Boolean Query를 구성합니다.

기본 `operator`는 `OR`입니다.

즉 기본적으로:

```text
semiconductor OR earnings
```

처럼 검색됩니다.

두 Term을 모두 필수로 만들고 싶다면:

```json
{
  "match": {
    "content": {
      "query": "semiconductor earnings",
      "operator": "and"
    }
  }
}
```

처럼 설정할 수 있습니다.

---

# 5. `term` Query

**`term` Query는 입력 Text를 Full-Text Analyzer로 분석하지 않고 Field에 저장된 정확한 Term을 검색하는 Query**입니다.

```http
GET query-dsl-test/_search
{
  "query": {
    "term": {
      "company": "samsung"
    }
  }
}
```

따라서 다음과 같은 구조화된 값의 검색에 주로 사용합니다.

```text
company
document_type
status
product_id
```

특히 `keyword` Field와 잘 어울립니다.

다만 `keyword` Field에 별도의 `normalizer`가 설정되어 있다면 그 정규화 규칙이 적용될 수 있으므로, `term`을 단순히 “입력 문자열을 무조건 그대로 비교한다”고 이해하기보다는 **정확한 Indexed Term을 대상으로 하는 Query**라고 이해하는 것이 더 정확합니다.

---

# 6. `match`와 `term`의 차이

가장 중요한 차이는 **검색 입력을 분석하는가**입니다.

```text
match
→ Query Text를 Analyzer로 분석
→ Full-Text Search에 적합

term
→ Full-Text 분석 없이 정확한 Term 검색
→ 구조화된 값 검색에 적합
```

`text` Field의 값은 Indexing할 때 Analyzer에 의해 변경될 수 있기 때문에 일반적으로 `text` Field에는 `match`를 사용하는 것이 적합합니다.

---

# 7. `bool` Query

**`bool` Query는 여러 Query를 Boolean 논리로 조합하기 위한 Compound Query**입니다.

주요 clause는 다음 네 가지입니다.

```text
bool
├─ must
├─ should
├─ filter
└─ must_not
```

각각은 조건의 필수 여부뿐 아니라 **Score 계산에 참여하는가**에서도 차이가 있습니다.

## 7.1 `must`

**`must`는 내부 Query가 반드시 Matching되어야 하며 Query Context에서 실행되는 clause**입니다.

따라서 내부 Query의 Score가 최종 `_score`에 반영될 수 있습니다.

```json
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
```

두 Query를 모두 만족해야 검색됩니다.

```text
must
→ 반드시 만족
→ Query Context
→ Scoring 참여
```

## 7.2 `filter`

**`filter`는 조건을 반드시 만족해야 하지만 Relevance Score 계산에는 참여하지 않는 clause**입니다.

```json
"filter": [
  {
    "term": {
      "company": "samsung"
    }
  }
]
```

여기서는:

```text
company가 samsung인가?
```

만 확인하면 됩니다.

```text
filter
→ 반드시 만족
→ Filter Context
→ Score 계산 X
```

따라서 Metadata처럼 정확한 조건을 적용하는 데 적합합니다.

## 7.3 `should`

**`should`는 Matching되었을 때 Document의 Score를 높일 수 있는 조건**입니다.

예를 들어:

```json
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
```

이면:

```text
semiconductor
→ 반드시 필요

HBM
→ 없어도 검색 가능
→ 존재하면 Score 상승
```

합니다.

`bool` Query에 `must`나 `filter`가 존재한다면 기본적으로 `should`의 `minimum_should_match`는 `0`입니다.

## 7.4 `minimum_should_match`

**`minimum_should_match`는 여러 `should` clause 중 최소 몇 개를 만족해야 검색 결과에 포함할지를 지정하는 옵션**입니다.

```json
"minimum_should_match": 1
```

이면 최소 하나의 `should` Query를 만족해야 합니다.

예를 들어:

```text
must:
semiconductor

should:
HBM

minimum_should_match = 1
```

이라면 `semiconductor`뿐 아니라 `HBM`도 Matching되어야 합니다.

## 7.5 `must_not`

**`must_not`은 해당 조건을 만족하는 Document를 검색 결과에서 제외하는 clause**입니다.

```json
"must_not": [
  {
    "term": {
      "document_type": "news"
    }
  }
]
```

`must_not`은 Filter Context에서 동작하므로 Score 계산에는 참여하지 않습니다.

## 7.5 `bool` clause 정리

| Clause | 조건 | Context | Score |
|---|---|---|---|
| `must` | 반드시 만족 | Query Context | O |
| `should` | 기본적으로 선택적 | Query Context | O |
| `filter` | 반드시 만족 | Filter Context | X |
| `must_not` | Matching 시 제외 | Filter Context | X |

---

# 8. Metadata와 Metadata Filtering

**Metadata는 Document 본문의 실제 내용이 아니라 Document의 속성이나 분류 정보를 나타내는 구조화된 데이터**입니다.

금융 문서라면 다음과 같은 값이 Metadata가 될 수 있습니다.

```text
company
document_type
published_at
year
quarter
source
page
```

**Metadata Filtering은 이러한 Metadata 조건을 이용해 검색 대상 Document의 범위를 제한하는 방법**입니다.

예를 들어:

> Samsung의 2025년 이후 사업보고서 중 semiconductor earnings와 관련된 내용을 검색

한다면 다음처럼 분리할 수 있습니다.

```text
Relevance Search
────────────────────
semiconductor earnings

Metadata Filtering
────────────────────
company = samsung
document_type = annual_report
published_at >= 2025-01-01
```

검색 관련도와 구조적 조건을 분리할 수 있다는 것이 핵심입니다.

---

# 9. 실습

## 9.1 실습 환경 구성

다음 Mapping을 사용했습니다.

```http
PUT query-dsl-test
{
  "settings": {
    "number_of_shards": 1
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
      }
    }
  }
}
```

검색 목적에 따라 Field Type을 다음처럼 설계했습니다.

```text
title / content
→ text
→ Full-Text Search

company / document_type
→ keyword
→ Exact Matching / Filtering

published_at
→ date
→ 날짜 범위 검색
```

실습용 Document는 다음과 같이 구성했습니다.

```http
POST query-dsl-test/_doc/1
{
  "title": "Semiconductor Earnings Report",
  "content": "Samsung semiconductor earnings increased due to strong HBM demand",
  "company": "samsung",
  "document_type": "annual_report",
  "published_at": "2026-03-15"
}

POST query-dsl-test/_doc/2
{
  "title": "Semiconductor Market Outlook",
  "content": "Samsung semiconductor market demand is expected to increase",
  "company": "samsung",
  "document_type": "news",
  "published_at": "2026-06-01"
}

POST query-dsl-test/_doc/3
{
  "title": "Memory Earnings Report",
  "content": "SK hynix semiconductor earnings improved with strong memory demand",
  "company": "sk_hynix",
  "document_type": "annual_report",
  "published_at": "2026-03-20"
}

POST query-dsl-test/_doc/4
{
  "title": "Automotive Business Report",
  "content": "Hyundai automotive earnings increased with global vehicle sales",
  "company": "hyundai",
  "document_type": "annual_report",
  "published_at": "2026-02-10"
}

POST query-dsl-test/_doc/5
{
  "title": "HBM Semiconductor Earnings Analysis",
  "content": "Samsung HBM semiconductor earnings earnings earnings improved strongly",
  "company": "samsung",
  "document_type": "annual_report",
  "published_at": "2024-12-20"
}
```

## 9.2 실습 1: `match`와 `term` 비교

먼저 다음 Query를 실행했습니다.

```http
GET query-dsl-test/_search
{
  "query": {
    "match": {
      "title": "Semiconductor"
    }
  }
}
```

검색 결과는 **3개의 Document**였습니다.

다음으로 동일한 값을 `term`으로 검색했습니다.

```http
GET query-dsl-test/_search
{
  "query": {
    "term": {
      "title": "Semiconductor"
    }
  }
}
```

이번에는 **검색 결과가 0개**였습니다.

다시 소문자로 검색했습니다.

```http
GET query-dsl-test/_search
{
  "query": {
    "term": {
      "title": "semiconductor"
    }
  }
}
```

이번에는 다시 **3개의 Document가 검색**되었습니다.

`title`은 `text` Field이며 별도의 Analyzer를 지정하지 않았기 때문에 기본 `standard` analyzer가 사용됩니다.

Indexing 시:

```text
Semiconductor
      ↓
standard analyzer
      ↓
semiconductor
```

로 저장됩니다.

`match`는 Query에도 Analyzer를 적용하므로:

```text
Semiconductor
      ↓
semiconductor
```

가 되어 정상적으로 검색됩니다.

반면 `term "Semiconductor"`는 동일한 Full-Text 분석을 거치지 않으므로 Indexed Term인 `semiconductor`와 Matching되지 않았습니다.

## 9.3 실습 2: `keyword` Field와 `term`

이번에는 `keyword`인 `company` Field를 검색했습니다.

```http
GET query-dsl-test/_search
{
  "query": {
    "term": {
      "company": "samsung"
    }
  }
}
```

결과는 Samsung Document 3개였습니다.

반면:

```http
GET query-dsl-test/_search
{
  "query": {
    "term": {
      "company": "Samsung"
    }
  }
}
```

으로 검색했을 때는 결과가 없었습니다.

이번 Mapping의 `company` Field에는 별도의 normalizer를 설정하지 않았기 때문에 `samsung`과 `Samsung`은 서로 다른 Term입니다.

따라서 `keyword + term`이 **정확한 Metadata Matching**에 사용된다는 것을 확인했습니다.

## 9.4 실습 3: `must`

다음 Query를 실행했습니다.

```http
GET query-dsl-test/_search
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

두 조건을 모두 만족하는 Document 3개가 검색되었습니다.

그중 `earnings`가 여러 번 등장하는 Document의 `_score`가 가장 높았습니다.

```text
Doc5 → 0.75013
Doc1 → 0.55902886
Doc3 → 0.55902886
```

즉 `must`에 있는 두 Query가 모두 필수 조건으로 작동하면서 각각 Scoring에도 참여한 것을 확인했습니다.

## 9.5 실습 4: `must` + `filter`

다음으로 Full-Text Search에 Metadata 조건을 추가했습니다.

```http
GET query-dsl-test/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "content": "semiconductor earnings"
          }
        }
      ],
      "filter": [
        {
          "term": {
            "company": "samsung"
          }
        }
      ]
    }
  }
}
```

결과는 Samsung Document 3개였습니다.

```text
Doc5 → 0.75013
Doc1 → 0.55902886
Doc2 → 0.2933976
```

`company=samsung`은 검색 대상을 제한했지만 Score에는 추가되지 않았습니다.

또한 `match "semiconductor earnings"`의 기본 operator는 `OR`이므로 `earnings`가 없는 Doc2도 `semiconductor`에 Matching되어 검색되었습니다.

## 9.6 실습 5: Metadata 조건을 `must`에 넣어보기

이번에는 동일한 `company=samsung`을 `filter`가 아니라 `must`에 넣었습니다.

```http
GET query-dsl-test/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "content": "semiconductor earnings"
          }
        },
        {
          "term": {
            "company": "samsung"
          }
        }
      ]
    }
  }
}
```

검색된 Document는 동일했지만 `_score`가 증가했습니다.

```text
filter 사용
Doc5 → 0.75013
Doc1 → 0.55902886
Doc2 → 0.2933976

must 사용
Doc5 → 1.2891264
Doc1 → 1.0980253
Doc2 → 0.83239406
```

즉:

```text
filter
→ 결과 제한
→ Score 영향 X

must
→ 결과 제한
→ company=samsung Query의 Score도 최종 Score에 추가
```

된 것을 확인할 수 있었습니다.

이 실습을 통해 **`term` Query 자체가 Filter인 것이 아니라 어느 Context에 배치되는지가 중요하다**는 것을 확인했습니다.

## 9.7 실습 6: `should`

다음 Query에서는 `semiconductor`를 필수 조건으로 두고 `HBM`을 선택적인 추가 조건으로 지정했습니다.

```http
GET query-dsl-test/_search
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

총 4개의 Document가 검색되었습니다.

HBM을 포함하는 Document는:

```text
Doc5 → 1.1862597
Doc1 → 1.1301275
```

로 상위에 위치했고, HBM이 없는 Document는:

```text
Doc2 → 0.2933976
Doc3 → 0.27951443
```

로 더 낮은 Score를 받았습니다.

즉:

```text
semiconductor
→ 반드시 필요

HBM
→ 없어도 검색 가능
→ 존재하면 Score 증가
```

라는 `should`의 역할을 확인했습니다.

## 9.8 실습 7: `minimum_should_match`

앞 Query에 다음 옵션을 추가했습니다.

```json
"minimum_should_match": 1
```

전체 Query는 다음과 같습니다.

```http
GET query-dsl-test/_search
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
      ],
      "minimum_should_match": 1
    }
  }
}
```

검색 결과는 기존 4개에서 **2개로 감소**했습니다.

```text
Doc5
Doc1
```

두 Document 모두 `semiconductor`와 `HBM`을 가지고 있습니다.

이번 예제에서는 `should`가 하나뿐이므로:

```text
minimum_should_match = 1
```

을 적용하면서 해당 `should`가 사실상 필수 조건이 된 것입니다.

## 9.9 실습 8: Metadata Filtering

이번에는 실제 금융 검색과 비슷한 조건을 구성했습니다.

> Samsung의 2025년 이후 annual report 중 semiconductor earnings와 관련된 Document 검색

```http
GET query-dsl-test/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "content": "semiconductor earnings"
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

조건을 모두 만족한 Document는 **Doc1 하나**였습니다.

```text
company
= samsung

document_type
= annual_report

published_at
>= 2025-01-01

content
= semiconductor earnings 관련
```

이 실습을 통해 Full-Text Relevance Search와 Metadata Filtering을 하나의 Query 안에서 조합할 수 있음을 확인했습니다.

## 9.10 실습 9: `must_not`

마지막으로 뉴스 Document를 제외했습니다.

```http
GET query-dsl-test/_search
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

검색 결과에서 `document_type=news`인 Doc2가 제거되고:

```text
Doc5
Doc1
Doc3
```

만 남았습니다.

`must_not`은 Filter Context에서 실행되므로 제외 조건 자체는 `_score`에 영향을 주지 않습니다.

---

# 10. 실습에서 확인한 핵심 내용

이번 실습을 통해 다음 내용을 직접 확인했습니다.

| 실습 | 확인한 내용 |
|---|---|
| `match` vs `term` | Analyzer 적용 여부에 따라 검색 결과가 달라짐 |
| `keyword` + `term` | 구조화된 정확한 값 검색 |
| `must` | 필수 조건이면서 Score 계산 참여 |
| `filter` | 필수 조건이지만 Score 계산에는 참여하지 않음 |
| `must` vs `filter` | 같은 Document가 검색되어도 `_score`가 달라질 수 있음 |
| `should` | 선택 조건 Matching 시 Ranking 상승 |
| `minimum_should_match` | `should` 중 최소 Matching 개수를 강제 |
| Metadata Filtering | Full-Text Search와 구조적 조건 결합 |
| `must_not` | 특정 조건의 Document 제외 |

---

# 11. 정리

## 11.1 전체 검색 흐름

Indexing 단계:

```text
Document
   ↓
Mapping
   ↓
Analyzer
   ↓
Term
   ↓
Inverted Index
```

검색 단계:

```text
사용자 검색 요청
        ↓
     Query DSL
        ↓
      bool
   ┌────┴─────────────┐
   │                  │
must / should      filter / must_not
   │                  │
Query Context      Filter Context
   │                  │
match 등           term / range 등
   │                  │
Analyzer          조건 만족 여부
   │
Query Term
   │
Inverted Index
   │
BM25
   │
_score
   │
Ranking
```

여기서 `term`이나 `match`가 반드시 한쪽 Context에 속하는 것은 아닙니다.

위 그림은 **일반적으로 많이 사용하는 구성**을 나타낸 것입니다.

Context는 Query의 종류가 아니라 **Query가 배치된 위치**에 의해 결정됩니다.

## 11.2 학습 내용 정리

이번 학습에서 가장 먼저 구분해야 할 것은 **Query 종류와 Context는 서로 다른 개념**이라는 점입니다.

```text
match / term / range
→ 어떤 방식으로 검색할 것인가?

Query Context / Filter Context
→ 검색 결과의 Score를 계산할 것인가?

must / should / filter / must_not
→ bool Query에서 조건들을 어떻게 조합할 것인가?
```

Full-Text Search에서는 일반적으로:

```text
match
→ Analyzer
→ Inverted Index
→ BM25
→ _score
```

흐름을 사용하고, Metadata 조건은:

```text
term / range
→ filter
→ 조건 만족 여부 판단
```

형태로 사용하는 것이 자연스럽습니다.

결국 Elasticsearch 검색은 다음 두 가지 문제를 분리해서 처리한다고 볼 수 있습니다.

```text
이 Document가 Query와 얼마나 관련 있는가?
→ Query Context
→ Relevance Ranking

이 Document가 필요한 조건을 만족하는가?
→ Filter Context
→ Metadata Filtering
```

이 구조는 이후 학습할 **Vector Search와 Hybrid Retrieval에서도 그대로 중요한 기반**이 됩니다.

---

# 참조

- [Elastic Docs — Query DSL reference](https://www.elastic.co/docs/reference/query-languages/querydsl)
- [Elastic Docs — Query and filter context](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-filter-context)
- [Elastic Docs — Full text queries](https://www.elastic.co/docs/reference/query-languages/query-dsl/full-text-queries)
- [Elastic Docs — Match query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-query)
- [Elastic Docs — Term-level queries](https://www.elastic.co/docs/reference/query-languages/query-dsl/term-level-queries)
- [Elastic Docs — Term query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-term-query)
- [Elastic Docs — Boolean query](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-bool-query)
- [Elastic Docs — minimum_should_match parameter](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-minimum-should-match)