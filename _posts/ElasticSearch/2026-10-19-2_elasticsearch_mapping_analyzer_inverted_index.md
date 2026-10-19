---
title: "[ElasticSearch] 2. ElasticSearch 검색 구조 이해하기: Mapping부터 Analyzer와 Inverted Index까지"
categories:
  - ElasticSearch
tags:
  - ElasticSearch
  
toc: true
toc_sticky: true
toc_label: "ElasticSearch 검색 구조 이해하기: Mapping부터 Analyzer와 Inverted Index까지"
---

# 개요

Elasticsearch는 Document를 검색에 적합한 형태로 인덱싱하고 빠르게 검색할 수 있도록 설계된 검색엔진입니다.

Elasticsearch의 기본적인 검색 흐름은 다음과 같이 이해할 수 있습니다.

```text
Index
→ Document / Field
→ Mapping
→ Analyzer
→ Token / Term
→ Inverted Index
→ Full-Text Search
```

이번 글에서는 검색 결과의 순위를 계산하는 BM25 이전까지, **Document가 어떻게 검색 가능한 구조로 만들어지는지**를 정리합니다.

---

# 1. Index, Document, Field

## Index

Index는 서로 관련된 Document를 저장하고 검색하기 위한 논리적인 단위입니다.

Index에는 Document뿐 아니라 해당 Document를 어떻게 다룰지를 결정하는 Mapping과 여러 Settings를 정의할 수 있습니다.

## Document

Document는 Elasticsearch에 저장되는 하나의 데이터 단위이며 JSON 형태로 표현됩니다.

```json
{
  "company": "삼성전자",
  "title": "2026년 반도체 사업 전망",
  "content": "삼성전자는 반도체 사업 투자를 확대하고 있다.",
  "published_at": "2026-10-01"
}
```

## Field

Document를 구성하는 각각의 데이터 항목을 Field라고 합니다.

위 예에서는 다음이 Field입니다.

```text
company
title
content
published_at
```

각 Field에는 별도의 Field Type을 지정할 수 있습니다.

```text
company      → keyword
title        → text
content      → text
published_at → date
```

즉 **Field와 Field Type은 서로 다른 개념**입니다.

---

# 2. Mapping이 중요한 이유

Mapping은 Elasticsearch가 **Document와 각각의 Field를 어떻게 저장하고 인덱싱할지를 정의하는 구조**입니다.

단순히 관계형 DB의 Schema처럼

> 이 Field에는 문자열을 넣고, 이 Field에는 날짜를 넣는다.

정도로 이해하면 부족합니다.

Elasticsearch에서는 Mapping에 따라 **같은 데이터라도 검색 방식 자체가 달라질 수 있기 때문입니다.**

예를 들어 다음 두 Field는 모두 문자열입니다.

```text
company = "삼성전자"

content = "삼성전자는 반도체 사업 투자를 확대하고 있다."
```

하지만 검색 목적은 다릅니다.

```text
company
→ "삼성전자"라는 하나의 값을 기준으로 Filtering

content
→ "삼성전자", "반도체", "투자" 등의 Text Query로 검색
```

따라서 Mapping을 다음과 같이 설계할 수 있습니다.

```json
{
  "mappings": {
    "properties": {
      "company": {
        "type": "keyword"
      },
      "title": {
        "type": "text"
      },
      "content": {
        "type": "text"
      },
      "published_at": {
        "type": "date"
      }
    }
  }
}
```

이처럼 Mapping은 **원본 데이터의 종류뿐 아니라 앞으로 그 데이터를 어떻게 검색할지를 고려하여 설계해야 합니다.**

---

# 3. Mapping의 기본 구조

Mapping은 크게 보면 다음과 같은 구조로 이해할 수 있습니다.

```text
mappings
 └─ properties
      ├─ field A
      │    ├─ type
      │    └─ 기타 Mapping Parameter
      │
      ├─ field B
      │    ├─ type
      │    └─ 기타 Mapping Parameter
      │
      └─ ...
```

## `properties`

`properties`에는 Document에서 사용할 Field들을 정의합니다.

```json
{
  "mappings": {
    "properties": {
      "company": {
        "type": "keyword"
      },
      "content": {
        "type": "text"
      }
    }
  }
}
```

`properties` 안에서는 단순 Field뿐 아니라 `object`, `nested`와 같이 내부에 또 다른 Field를 가지는 구조도 정의할 수 있습니다.

중첩 구조의 경우 추후에 필요할 때 따로 다루도록 하겠습니다.

## `type`

각 Field가 어떤 데이터를 가지며 어떤 검색 방식으로 사용될지 결정하는 중요한 요소입니다.

대표적으로:

```text
text
keyword
date
integer / long / double
boolean
object
dense_vector
```

등이 있습니다.

특히 문자열은 목적에 따라 `text` 또는 `keyword`로 나뉩니다.

---

# 4. Field Type 외에도 Mapping에서 설정할 수 있는 것

Mapping은 단순히 `type`만 지정하는 구조가 아닙니다.

Field Type에 따라 여러 Mapping Parameter를 사용할 수 있습니다. Elasticsearch는 `analyzer`, `search_analyzer`, `fields`, `index`, `doc_values`, `normalizer` 등 다양한 Mapping Parameter를 제공합니다.

예를 들어 `text` Field에는 어떤 Analyzer를 사용할지 지정할 수 있습니다.

```json
{
  "mappings": {
    "properties": {
      "content": {
        "type": "text",
        "analyzer": "standard"
      }
    }
  }
}
```

즉 Mapping은 Elasticsearch에게 단순히

```text
content는 문자열이다.
```

라고 알려주는 것이 아니라,

```text
content는 Full-Text Search 대상이며
이 Analyzer를 이용해 분석해서
검색 가능한 Term을 만들어라.
```

라는 정보까지 줄 수 있습니다.

이 때문에 Mapping은 Elasticsearch 검색 설계에서 중요합니다.

---

# 5. Dynamic Mapping과 Explicit Mapping

## Dynamic Mapping

Elasticsearch는 새로운 Document가 들어왔을 때 Field의 값을 보고 적절한 Field Type을 자동으로 추론할 수 있습니다.

예를 들어:

```json
{
  "count": 10
}
```

과 같은 데이터가 들어오면 Elasticsearch가 해당 Field의 타입을 자동으로 추론할 수 있습니다.

이를 **Dynamic Mapping**이라고 합니다.

Dynamic Mapping은 빠르게 데이터를 넣어 확인할 때는 편리하지만, Elasticsearch가 추론한 Field Type이 실제 검색 목적과 항상 일치하는 것은 아닙니다.

## Explicit Mapping

사용자가 직접 Field와 Field Type을 정의하는 방식입니다.

```json
{
  "mappings": {
    "properties": {
      "company": {
        "type": "keyword"
      },
      "content": {
        "type": "text"
      }
    }
  }
}
```

실제 검색 시스템에서는 데이터의 의미와 검색 방식을 개발자가 더 잘 알고 있기 때문에, 중요한 Field는 Explicit Mapping을 통해 직접 정의하는 것이 좋습니다. Elastic 역시 구체적인 use case에 맞게 indexing을 제어하기 위해 Explicit Mapping을 사용하는 것을 권장합니다.

따라서 실제 프로젝트에서는:

> Elasticsearch가 알아서 Type을 정하게 한다.

보다는

> **각 Field를 어떤 방식으로 검색할 것인지 먼저 결정하고 Mapping을 설계한다.**

라는 접근이 중요합니다.

---

# 6. `text`와  `keyword`

## `text`

Full-Text Search가 필요한 비정형 Text에 사용합니다.

```text
"삼성전자 2026년 반도체 사업 전망"
```

이러한 Text는 Analyzer를 거쳐 여러 검색 가능한 Term으로 변환됩니다.

## `keyword`

구조화된 값을 하나의 값으로 검색하거나 Filtering, Sorting 등에 사용할 때 적합합니다.

```text
"삼성전자"
"사업보고서"
"완료"
```

따라서 같은 문자열이라도 **어떤 방식으로 검색할 것인지에 따라 Mapping이 달라집니다.**

---

# 7. Analyzer

Analyzer는 Text를 검색에 사용할 Token으로 변환하는 Text Analysis Pipeline입니다.

기본 구조는 다음과 같습니다.

```text
Original Text
    ↓
Character Filter
    ↓
Tokenizer
    ↓
Token Filter
    ↓
Tokens
```

## Character Filter

Tokenizer가 동작하기 전에 원본 문자열 자체를 전처리합니다.

## Tokenizer

Text를 여러 Token으로 분리합니다.

```text
Samsung develops semiconductor

↓

Samsung
develops
semiconductor
```

## Token Filter

이미 생성된 Token을 변경하거나 제거합니다.

```text
QUICK
→ quick
```

같은 lowercase 처리나 불용어 제거 등이 여기에 해당합니다.

---

# 8. Metadata Extraction과 Analyzer는 다르다

원본 금융 데이터가 단순한 Text로만 존재한다면 Elasticsearch에 넣기 전에 별도의 데이터 가공이 필요할 수 있습니다.

예를 들어:

```text
삼성전자는 2026년 3분기 반도체 사업에서 영업이익이 증가했다.
```

에서 다음 Metadata를 추출할 수 있습니다.

```json
{
  "company": "삼성전자",
  "year": 2026,
  "quarter": 3
}
```

Rule, Regex, Parser, NER 등을 이용하여 이러한 정보를 만드는 것은 **Metadata Extraction**입니다.

반면 Analyzer의 목적은 다릅니다.

```text
Metadata Extraction
→ Document의 구조화된 속성을 생성

Analyzer
→ Text를 Full-Text Search에 사용할 검색 단위로 변환
```

따라서 외부 형태소 분석기나 NER 시스템을 이용하여 Metadata를 추출하더라도 Elasticsearch Analyzer와 반드시 역할이 중복되는 것은 아닙니다.

---

# 9. `standard analyzer`와 `standard tokenizer`

`standard tokenizer`는 Text를 Token으로 분리합니다.

```text
THE Quick FoX
```

↓

```text
THE
Quick
FoX
```

반면 `standard analyzer`는 Standard Tokenizer와 Lowercase Token Filter 등이 결합된 Analyzer입니다.

따라서:

```text
THE Quick FoX
```

↓

```text
the
quick
fox
```

가 됩니다.

---

# 10. Token과 Term

Token은 Analyzer가 Text를 분석하는 과정에서 만드는 결과이며 문자열 외에도 Position, Offset 등의 정보를 가질 수 있습니다.

Term은 그 분석 결과를 바탕으로 **Inverted Index에서 실제 Indexing과 Matching에 사용되는 검색 단위**라고 이해할 수 있습니다.

---

# 11. Inverted Index

다음과 같은 Document가 있다고 가정합니다.

```text
Doc1: Samsung develops semiconductor technology
Doc2: Samsung develops smartphones
Doc3: SK Hynix develops semiconductor memory
```

일반적인 데이터 구조는:

```text
Document → Terms
```

이지만 검색엔진은 이를 뒤집어:

```text
samsung       → Doc1, Doc2
semiconductor → Doc1, Doc3
smartphones   → Doc2
```

와 같은 **Term → Documents** 구조를 만듭니다.

이것이 Inverted Index입니다.

사용자가 `semiconductor`를 검색하면 모든 Document의 본문을 하나씩 검사하지 않고 해당 Term과 연결된 Document를 바로 찾을 수 있습니다.

---

# 12. Index-time Analysis와 Search-time Analysis

문서를 Indexing할 때뿐 아니라 사용자의 Query도 Analyzer를 거칩니다.

```text
Document
→ Index Analyzer
→ Indexed Term

Query
→ Search Analyzer
→ Query Term
```

예를 들어 Document를 Indexing할 때:

```text
Coffee
→ coffee
```

로 변환했다면 Query의 `Coffee` 역시 호환되는 형태인 `coffee`로 분석되어야 합니다.

Index Analyzer와 Search Analyzer가 반드시 동일해야 하는 것은 아니지만 **서로 호환되는 Term을 만들어야 정상적인 Matching이 가능합니다.**

---

# 13. `_analyze` API로 Analyzer 동작 확인

Elasticsearch의 `_analyze` API를 이용하면 Analyzer가 실제로 어떤 Token을 생성하는지 확인할 수 있습니다.

## Standard Analyzer

```http
POST _analyze
{
  "analyzer": "standard",
  "text": "The 2 QUICK Brown-Foxes jumped over the lazy dog's bone."
}
```

`standard analyzer`는 Standard Tokenizer와 Lowercase Token Filter 등을 사용하므로 `QUICK`은 `quick`으로 변환됩니다. 다음은 위 요청을 실행했을 때의 결과입니다.

```json
{
  "tokens": [
    {
      "token": "the",
      "start_offset": 0,
      "end_offset": 3,
      "type": "<ALPHANUM>",
      "position": 0
    },
    {
      "token": "2",
      "start_offset": 4,
      "end_offset": 5,
      "type": "<NUM>",
      "position": 1
    },
    {
      "token": "quick",
      "start_offset": 6,
      "end_offset": 11,
      "type": "<ALPHANUM>",
      "position": 2
    },
    {
      "token": "brown",
      "start_offset": 12,
      "end_offset": 17,
      "type": "<ALPHANUM>",
      "position": 3
    },
    {
      "token": "foxes",
      "start_offset": 18,
      "end_offset": 23,
      "type": "<ALPHANUM>",
      "position": 4
    },
    {
      "token": "jumped",
      "start_offset": 24,
      "end_offset": 30,
      "type": "<ALPHANUM>",
      "position": 5
    },
    {
      "token": "over",
      "start_offset": 31,
      "end_offset": 35,
      "type": "<ALPHANUM>",
      "position": 6
    },
    {
      "token": "the",
      "start_offset": 36,
      "end_offset": 39,
      "type": "<ALPHANUM>",
      "position": 7
    },
    {
      "token": "lazy",
      "start_offset": 40,
      "end_offset": 44,
      "type": "<ALPHANUM>",
      "position": 8
    },
    {
      "token": "dog's",
      "start_offset": 45,
      "end_offset": 50,
      "type": "<ALPHANUM>",
      "position": 9
    },
    {
      "token": "bone",
      "start_offset": 51,
      "end_offset": 55,
      "type": "<ALPHANUM>",
      "position": 10
    }
  ]
}
```

## Tokenizer와 Token Filter 역할 비교

Tokenizer만 사용하면 대소문자는 유지됩니다.

```http
POST _analyze
{
  "tokenizer": "standard",
  "text": "THE Quick FoX JUMPs"
}
```

```json
{
  "tokens": [
    {
      "token": "THE",
      "start_offset": 0,
      "end_offset": 3,
      "type": "<ALPHANUM>",
      "position": 0
    },
    {
      "token": "Quick",
      "start_offset": 4,
      "end_offset": 9,
      "type": "<ALPHANUM>",
      "position": 1
    },
    {
      "token": "FoX",
      "start_offset": 10,
      "end_offset": 13,
      "type": "<ALPHANUM>",
      "position": 2
    },
    {
      "token": "JUMPs",
      "start_offset": 14,
      "end_offset": 19,
      "type": "<ALPHANUM>",
      "position": 3
    }
  ]
}
```

여기에 `lowercase` Token Filter를 추가하면 다음과 같이 변합니다.

```http
POST _analyze
{
  "tokenizer": "standard",
  "filter": [
    "lowercase"
  ],
  "text": "THE Quick FoX JUMPs"
}
```

결과를 보면 모든 Token의 값들이 소문자로 변경된 것을 확인할 수 있습니다.

```json
{
  "tokens": [
    {
      "token": "the",
      "start_offset": 0,
      "end_offset": 3,
      "type": "<ALPHANUM>",
      "position": 0
    },
    {
      "token": "quick",
      "start_offset": 4,
      "end_offset": 9,
      "type": "<ALPHANUM>",
      "position": 1
    },
    {
      "token": "fox",
      "start_offset": 10,
      "end_offset": 13,
      "type": "<ALPHANUM>",
      "position": 2
    },
    {
      "token": "jumps",
      "start_offset": 14,
      "end_offset": 19,
      "type": "<ALPHANUM>",
      "position": 3
    }
  ]
}
```

이를 통해 다음 차이를 확인할 수 있습니다.

```text
Tokenizer
→ Text를 Token으로 분리

Token Filter
→ 생성된 Token을 후처리
```

## Stop Token Filter

```http
POST _analyze
{
  "tokenizer": "standard",
  "filter": [
    "lowercase",
    "stop"
  ],
  "text": "The quick fox jumps over the lazy dog"
}
```

이 경우 `the`와 같은 Stop Word가 Tokenizer 이후 Token Filter 단계에서 제거됩니다.

이번 실습을 통해 `standard tokenizer` 자체가 소문자 변환을 수행하는 것이 아니라, `standard analyzer` 안에 포함된 Lowercase Token Filter가 해당 역할을 수행한다는 점을 확인할 수 있었습니다.

```json
{
  "tokens": [
    {
      "token": "quick",
      "start_offset": 4,
      "end_offset": 9,
      "type": "<ALPHANUM>",
      "position": 1
    },
    {
      "token": "fox",
      "start_offset": 10,
      "end_offset": 13,
      "type": "<ALPHANUM>",
      "position": 2
    },
    {
      "token": "jumps",
      "start_offset": 14,
      "end_offset": 19,
      "type": "<ALPHANUM>",
      "position": 3
    },
    {
      "token": "over",
      "start_offset": 20,
      "end_offset": 24,
      "type": "<ALPHANUM>",
      "position": 4
    },
    {
      "token": "lazy",
      "start_offset": 29,
      "end_offset": 33,
      "type": "<ALPHANUM>",
      "position": 6
    },
    {
      "token": "dog",
      "start_offset": 34,
      "end_offset": 37,
      "type": "<ALPHANUM>",
      "position": 7
    }
  ]
}
```

# 14. 전체 검색 흐름

현재까지 학습한 내용을 연결하면 다음과 같습니다.

```text
Document
   ↓
Mapping
   ↓
Field Type 결정
   ↓
text Field
   ↓
Index Analyzer
   ↓
Token / Term
   ↓
Inverted Index
```

사용자 Query가 들어오면:

```text
Query
   ↓
Search Analyzer
   ↓
Query Term
   ↓
Inverted Index 조회
   ↓
관련 Document 탐색
```

여기까지가 **검색 가능한 Document를 찾는 과정**입니다.

그다음에는:

> 여러 Document 중 어떤 Document가 Query와 가장 관련성이 높은가?

라는 문제가 남습니다.

이 문제는 이후 TF, IDF, BM25와 같은 **Relevance Scoring**에서 다룹니다.

---

# 정리

이번 학습에서 가장 중요한 점은 Elasticsearch의 검색 구조가 Mapping부터 시작된다는 것입니다.

```text
원본 데이터
   ↓
Document 설계
   ↓
Mapping
   ├─ Field
   ├─ Field Type
   └─ 검색/Indexing 관련 Parameter
   ↓
Analyzer
   ↓
Token / Term
   ↓
Inverted Index
   ↓
Full-Text Search
```

특히 기억해야 할 내용은 다음과 같습니다.

- Mapping은 단순한 Schema가 아니라 **검색을 위한 Field 구조와 Indexing 방식을 설계하는 역할**을 합니다.
- Dynamic Mapping도 가능하지만 실제 검색 시스템에서는 중요한 Field를 Explicit Mapping으로 설계하는 것이 좋습니다.
- 같은 문자열이라도 검색 목적에 따라 `text`와 `keyword`를 다르게 사용할 수 있습니다.
- Metadata Extraction과 Analyzer는 목적이 다릅니다.
- Analyzer는 Text를 검색 가능한 Token과 Term으로 변환합니다.
- Inverted Index는 `Term → Documents` 구조를 이용해 관련 Document를 빠르게 찾습니다.
- Query 역시 Analyzer를 거쳐 Indexed Term과 호환되는 형태로 만들어져야 합니다.
- Inverted Index는 Document를 찾는 역할을 하며, 검색 결과의 순위를 결정하는 것은 이후 학습할 BM25의 역할입니다.

# 참조

- [Elastic Docs — Mapping in Elasticsearch](https://www.elastic.co/docs/manage-data/data-store/mapping)
- [Elastic Docs — Explicit mapping](https://www.elastic.co/docs/manage-data/data-store/mapping/explicit-mapping)
- [Elastic Docs — Dynamic mapping](https://www.elastic.co/docs/manage-data/data-store/mapping/dynamic-field-mapping)
- [Elastic Docs — Field data types](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/field-data-types)
- [Elastic Docs — Mapping parameters](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/mapping-parameters)
- [Elastic Docs — Properties](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/properties)
- [Elastic Docs — Text field type](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/text)
- [Elastic Docs — Keyword type family](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/keyword)
- [Elastic Docs — Text analysis](https://www.elastic.co/docs/manage-data/data-store/text-analysis)
- [Elastic Docs — Anatomy of an analyzer](https://www.elastic.co/docs/manage-data/data-store/text-analysis/anatomy-of-an-analyzer)
- [Elastic Docs — Standard analyzer](https://www.elastic.co/docs/reference/text-analysis/analysis-standard-analyzer)
- [Elastic Docs — Search analyzer](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/search-analyzer)
- [Elastic Docs — Analyze API](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-indices-analyze)
- [Apache Lucene — Index package](https://lucene.apache.org/core/10_1_0/core/org/apache/lucene/index/package-summary.html)