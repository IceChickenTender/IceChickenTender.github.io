---
title: "[LLM/RAG] LangChain - 1. LangChain 시작하기"
categories:
  - LLM/RAG
  - LangChain
tags:
  - LLM/RAG
  - LangChain

use_math: true  
toc: true
toc_sticky: true
toc_label: "LangChain 시작, LangChain 과 LLMChain"
---

LLM/RAG 에 대해서 제대로 공부하기 위해 기초적인 Langchain 에 대해서 공부해보고자 합니다. 그리고 공부하는 과정에서 추후에 제가 참고하기 위해 블로그로 기록해 놓고자 합니다. 이번 포스트를 시작으로 LangChain의 주요 구성 요소들을 세세하게 다뤄보고, 최종적으로 LangChain을 이용한 RAG 시스템을 만들어보고자 합니다.

LangChain 공부는 wiki docs 에 있는 `판다스 스튜디오`님이 작성하신 `랭체인(LangChain) 입문부터 응용까지` 를 참고하여 진행하도록 하겠습니다. 또한 실습 코드들은 구글 코랩에 작성되는 코드들이니 이를 참고해 주시기 바랍니다.

# 1. LangChain 개요

## 1.1 개요

랭체인(LangChain)은 **해리슨 체이스(Harrison Chase)**가 2022년 10월에 시작한 오픈 소스 프로젝트입니다.  
당시 그는 머신러닝 스타트업 **로버스트 인텔리전스(Robust Intelligence)**에서 근무하며, 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발이 데이터 연결, 체인 구성, 외부 API 통합 등 많은 복잡한 작업을 요구한다는 한계를 직접 경험했습니다.  
이를 해결하고자, **챗봇·질의응답(QA)·자동 요약** 등 다양한 LLM 애플리케이션을 쉽고 빠르게 구축할 수 있는 개발 프레임워크를 설계했습니다.

---

## 1.2 성장 과정
- **2023년 4월**  
  랭체인은 법인으로 전환하고 **세쿼이아캐피털(Sequoia Capital)** 등 주요 벤처캐피털로부터 투자를 유치하며 빠르게 성장했습니다.
- **2024년 1월**  
  최초의 안정적(stable) 버전인 **v0.1.0**을 출시하였으며, Chains·Agents·Tools·RAG 통합 등 LLM 개발에 필수적인 기능을 체계적으로 지원하기 시작했습니다.

---

## 1.3 버전 안정화와 개선

이전 버전까지는 잦은 업데이트와 그로 인한 버그, 코드 오류로 인해 개발자들이 불편을 겪는 경우가 많았습니다.  
그러나 **v0.1.0** 이후에는 이러한 문제가 상당 부분 개선되었으며, 현재는 보다 **안정적이고 개발자 친화적인 환경**을 제공합니다.

---

## 1.4 주요 특징
- **Chains** : LLM 호출을 여러 단계로 연결해 복잡한 작업을 구성
- **Agents** : 동적으로 어떤 도구(함수, API, DB)를 호출할지 결정
- **Tools** : 검색, 계산, API 호출 등 외부 리소스와 연동
- **RAG 지원** : 검색 기반의 프롬프트 구성으로 LLM 응답 품질 향상
- **다양한 통합성** : OpenAI, Anthropic, Hugging Face 등 주요 LLM과 연동 가능

---

## 1.5 정리
랭체인은 LLM 애플리케이션 개발의 복잡성을 크게 줄여주는 프레임워크로,  
**빠른 프로토타이핑 → 안정적인 서비스 구현**까지 전 과정을 지원합니다.  
기업과 개인 개발자 모두에게 유용하며, RAG 기반 서비스나 에이전트 시스템 구축에도 널리 활용되고 있습니다.

---

## 1.6 LangChain 프레임워크의 구성

랭체인(LangChain) 프레임워크는 LLM 애플리케이션 개발에 도움이 되는 여러 구성 요소로 이루어져 있습니다. 특히 개발자들이 다양한 LLM 작업을 신속하게 구축하고 배포할 수 있도록 설계되었습니다. 랭체인의 주요 구성 요소는 다음과 같습니다.

1. **랭체인 라이브러리(LangChain Libraries)** : 파이썬과 자바스크립트 라이브러리를 포함하며, 다양한 컴포넌트의 인터페이스와 통합, 이 컴포넌트들을 체인과 에이전트로 결합할 수 있는 기본 런타임, 그리고 체인과 에이전트의 사용 가능한 구현이 가능합니다.

2. **랭체인 템플릿(LangChain Templates)**: 다양한 작업을 위해 쉽게 배포할 수 있는 참조 아키텍처 모음입니다. 이 템플릿은 개발자들이 특정 작업에 맞춰 빠르게 애플리케이션을 구축할 수 있도록 돕습니다.

3. **랭서브(LangServe)**: 랭체인 체인을 REST API 로 배포할 수 있게 해주는 라이브러리입니다. 이를 통해 개발자들은 자신의 애플리케이션을 외부 시스템과 쉽게 통합할 수 있습니다.

4. **랭스미스(LangSmith)**: 개발자 플랫폼으로, LLM 프레임워크에서 구축된 체인을 디버깅, 테스트, 평가, 모니터링할 수 있으며, 랭체인과의 원활한 통합을 지원합니다.

우선 제가 참고한 wiki docs 에서는 랭체인 라이브러리(LangChain Libraries) 위주로 다루기 때문에 저 또한 랭체인 라이브러리 위주로 진행하고 추후에 다른 프레임워크들에 대해서도 알아보도록 하겠습니다.

---

## 1.7 필수 라이브러리 설치

사용할 랭체인 버전은 v0.3 이상 버전을 사용할 예정입니다.

---

### 1.7.1 랭체인 설치

구글 코랩에서 다음과 같이 작성 후 실행해 줍니다.

```python
!pip install langchain>=0.3.0
```

---

### 1.7.2 OpenAI 관련 패키지 설치

OpenAI 모델을 사용할 때 필요한 의존성 라이브러리를 설치하는 방법입니다.   
구글 코랩에서 다음과 같이 작성 후 실행해 줍니다.

langchain-openai 설치
```python
!pip install langchain-openai>=0.2.0
```

tiktoken 설치
```python
!pip install tiktoken
```

---

# 2. LLM 체인(LLMChain) 만들기

## 2.1 LLMChain 이란?

`LLMChain` 은 **프롬프트 → LLM 호출 → 출력 반환**의 가장 기본적인 파이프라인을 캡슐화한 클래스입니다. 즉 "프롬프트를 만들고 그걸 모델에 넣어 한 번 추론하는" 템플릿화된 단일 스텝 체인입니다. 핵심 구성은 다음과 같습니다.

---

## 2.2 기본 LLM 체인의 구성 요소

1. 프롬프트(Prompt): 사용자 또는 시스템에서 제공하는 입력으로, LLM 에게 특정 작업을 수행하도록 요청하는 지시문입니다. 프롬프트는 질문, 명령, 문장 시작 부분 등 다양한 형태를 취할 수 있으며, LLM 의 응답을 유도하는 데 중요한 역할을 합니다.

2. LLM(Large Language Model): GPT, Gemini 등 대규모 언어 모델로, 대량의 텍스트 데이터에서 학습하여 언어를 이해하고 생성할 수 있는 인공지능 시스템입니다. LLM 은 프롬프트를 바탕으로 적절한 응답을 생성하거나, 주어진 작업을 수행하는데 사용됩니다.

---

## 2.2.1 일반적인 작동 방식

1. 프롬프트 생성: 사용자의 요구 사항이나 특정 작업을 정의하는 프롬프트를 생성합니다. 이 프롬프트는 LLM 에게 전달되기 전에, 작업의 목적과 맥락을 명확히 전달하기 위해 최적화될 수 있습니다.

2. LLM 처리: LLM 은 제공된 프롬프트를 분석하고, 학습된 지식을 바탕으로 적절한 응답을 생성합니다. 이 과정에서 LLM 은 내부적으로 다양한 언어 패턴과 내외부 지식을 활용하여, 요청된 작업을 수행하거나 정보를 제공합니다.

3. 응답 반환: LLM 에 의해 생성된 응답은 최종 사용자에게 필요한 형태로 변환되어 제공됩니다. 이 응답은 직접적인 답변, 생성된 텍스트, 요약된 정보 등 다양한 형태를 취할 수 있습니다.

---

## 2.2.2 실습 예제

### 2.2.2.1 LLM 으로부터 답변 받기

OpenAI 의 ChatOpenAI 함수를 사용하면 OpenAI 의 여러 모델에 API 로 접근할 수 있습니다. 다음 예제는 랭체인에서 gpt-4o-mini 모델을 사용하여 LLM 모델 인스턴스를 생성하고, "지구의 자전 주기는?" 라는 프롬프트를 LLM 모델에 전달하는 과정을 보여줍니다. 실행 결과로 "지구의 자전 주기는?" 에 대한 답변을 반환합니다.

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = 'API_KEY'

llm = ChatOpenAI(model="gpt-4o-mini")

llm.invoke("지구의 자전 주기는?")
```

```
실행 결과

AIMessage(content="지구의 자전 주기는 약 24시간입니다. 정확히는 23시간 56분 4초로, 이를 '항성일'이라고 합니다. 그러나 우리가 일반적으로 사용하는 24시간은 태양이 하늘에서 동일한 위치에 돌아오는 시간을 기준으로 한 '태양일'입니다.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 68, 'prompt_tokens': 15, 'total_tokens': 83, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_34a54ae93c', 'id': 'chatcmpl-C4kgwz8xFLUk3WrJtrlcxvKaRM2Ew', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='run--f1453437-9460-4eab-a7bf-2f7113f935d8-0', usage_metadata={'input_tokens': 15, 'output_tokens': 68, 'total_tokens': 83, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

---

### 2.2.2.2 프롬프트 템플릿 적용

이번에는 간단한 프롬프트 템플릿을 적용해 보도록 하겠습니다. 더 자세한 내용은 추후에 다룰 예정입니다. 예제에서는 `langchain_core` 의 `prompts` 모듈에서 `ChatPromptTemplate` 클래스를 사용하여 천문학 전문가로서 질문에 답변하는 형식의 프롬프트 템플릿을 생성합니다. 이 템플릿을 사용하면, 입력으로 주어진 질문 `{input}` 에 대해 천문학 전문가의 관점에서 답변을 생성하는 질문 프롬프트를 만들 수 있습니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")

llm = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | llm

chain.invoke({"input": "지구의 자전 주기는?"})
```

실행 결과를 보면 이전에는 "지구의 자전 주기는 약 24시간입니다. 정확히는 23시간 56분 4초로, 이를 '항성일'이라고 합니다. 그러나 우리가 일반적으로 사용하는 24시간은 태양이 하늘에서 동일한 위치에 돌아오는 시간을 기준으로 한 '태양일'입니다." 라고 한 답변과는 다른 답변을 받아온 것을 확인할 수 있습니다.

```
실행결과

AIMessage(content='지구의 자전 주기는 약 24시간입니다. 정확히 말하자면, 지구가 한 바퀴 자전하는 데 걸리는 시간은 약 23시간 56분 4초로, 이를 태양일로 환산하면 약 24시간이 됩니다. 태양일은 지구가 태양을 기준으로 하루를 완성하는 데 걸리는 시간을 의미합니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 86, 'prompt_tokens': 29, 'total_tokens': 115, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_560af6e559', 'id': 'chatcmpl-C4krNA5uwKQbjjG95Pqv7Hk4ddx7G', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='run--17a488f7-0861-4d34-b9d4-8b469aa6422d-0', usage_metadata={'input_tokens': 29, 'output_tokens': 86, 'total_tokens': 115, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

또한 `Parser` 를 적용하여 특정 출력 형태로 LLM 의 결과를 받아볼 수 있습니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer te question. <Question>: {input}")

llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

chain.invoke({"input": "지구의 자전 주기는?"})
```

그럼 이전 출력들과는 다르게 단순히 LLM 의 답변만을 출력합니다.

```
실행결과

지구의 자전 주기는 약 24시간입니다. 더 정확하게는 평균적으로 23시간 56분 4초로, 이를 '항성일'이라고 합니다. 그러나 우리가 일상적으로 사용하는 24시간은 태양일로, 태양이 같은 위치에 다시 나타나기까지의 시간을 기준으로 합니다. 태양일은 항성일보다 약 4분 정도 더 긴 이유는 지구가 태양 주위를 공전하고 있기 때문입니다.
```

## 2.3 멀티 체인(Multi-Chain)

멀티 체인(Multi-Chain) 은 여러 개의 체인을 연결해 복합적인 구조로 작용하는 것을 말합니다. 이러한 구조는 각기 다른 목적을 가진 여러 체인을 조합하여, 입력 데이터를 다양한 방식으로 처리하고 최종적인 결과를 도출할 수 있도록 합니다. 복잡한 데이터 처리, 의사 결정, AI 기반 작업 흐름을 설계할 때 특히 유용합니다.

### 2.3.1 순차적인 연결 체인

다음 예제를 통해서 2개의 체인(chain1, chain2) 를 정의하고, 순차적으로 체인을 연결하여 수행하는 작업을 해보겠습니다.   

첫 번째 체인(chain1) 은 한국어 단어를 영어로 번역하는 작업을 수행합니다. `ChatPromptTemplate.from_template` 를 사용하여 프롬프트를 정의하고, `ChatOpenAI` 인스턴스로 GPT 모델을 사용하며, `StrOutputParser()` 를 통해 모델의 출력을 문자열로 파싱합니다.   

chain1 은 입력받은 "미래" 라는 단어를 영어로 번역하라는 요청을 gpt-4o-mini 모델에 전달하고, 모델이 생성한 번역 결과를 출력합니다. 체인을 실행하는 invoke 메소드는 입력으로 딕셔너리(dictionary) 객체를 받으며, 이 딕셔너리의 키는 프롬프트 템플릿에서 정의된 변수명({korean_word})와 일치해야 합니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English")
prompt2 = ChatPromptTemplate.from_template("explain {english_word} using oxford dictionary to me in Korean.")

llm = ChatOpenAI(model="gpt-4o-mini")

chain1 = prompt1 | llm | StrOutputParser()

chain1.invoke({"korean_word": "미래"})
```

실행 결과로 wiki docs 에서는 "future" 라는 한 단어만 출력된다고 했지만 저는 아래와 같이 출력되는 것을 확인했습니다.

```
실행 결과

The Korean word "미래" translates to "future" in English.
```

chain1 에서 출력한 값을 입력값으로 받아서, 이 번역된 단어를 english_word 변수에 저장합니다. 다음으로, 이 변수를 사용해 두 번째 체인(chain2) 의 입력으로 제공하고, 영어 단어의 뜻을 한국어로 설명하는 작업을 수행합니다. 최종 출력은 문자열로 출력되도록 합니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English")
prompt2 = ChatPromptTemplate.from_template("explain {english_word} using oxford dictionary to me in Korean.")

llm = ChatOpenAI(model="gpt-4o-mini")

chain1 = prompt1 | llm | StrOutputParser()

chain2 = (
    {"english_word":chain1}
    | prompt2
    | llm
    | StrOutputParser()
)

chain2.invoke({"korean_word":"미래"})
```

wiki docs 에서와는 결과가 조금 다릅니다. 아마도 wiki docs 가 작성될 때의 `gpt-4o-mini` 학습과 제가 블로그를 작성할 때의 `gpt-4o-mini` 학습이 조금 다를 수도 있고 아니면 또 다른 원인이 있을 것 같은데 그래도 얻고자 하는 결과는 얻을 수 있었습니다.

```
실행 결과

"미래"라는 한국어 단어는 영어로 "future"로 번역됩니다. 옥스포드 사전(Oxford Dictionary)에 따르면, "future"는 다음과 같이 정의됩니다:

1. **명사**:
   - 아직 오지 않은 시간, 특히 현재 시점 이후의 시간.
   - 어떤 일이나 사건이 일어날 것으로 예상되는 시간.

이와 같이 "미래"는 우리가 앞으로 경험하게 될 시간이나 상황을 의미합니다.
```

---

## 2.4 체인을 실행하는 방법

## 2.4.1 LangChain 의 "Runnable" 프로토콜

LangChain 의 `Runnable` 프로토콜은 사용자가 사용자 정의 체인을 쉽게 생성하고 관리할 수 있도록 설계된 핵심적인 개념입니다. 이 프로토콜을 통해, 개발자는 일관된 인터페이스를 사용하여 다양한 타입의 컴포넌트를 조합하고, 복잡한 데이터 처리 파이프라인을 구성할 수 있습니다. `Runnable` 프로토콜은 다음과 같은 주요 메소드를 제공합니다.

- invoke: 주어진 입력에 대해 체인을 호출하고, 결과를 반환합니다. 이 메소드는 단일 입력에 대해 동기적으로 작동합니다.
- batch: 입력 리스트에 대해 체인을 호출하고, 각 입력에 대한 결과를 리스트로 반환합니다. 이 메소드는 여러 입력데 해새 동기적으로 작동하며, 효율적인 배치 처리를 가능하게 합니다.
- stream: 입력에 대해 체인을 호출하고, 결과의 조각들을 스트리밍합니다. 이는 대용량 데이터 처리나 실시간 데이터 처리에 유용합니다.
- 비동기 버전: ainvoke, abatch, astream 등의 메소드는 각각의 동기 버전에 대한 비동기 실행을 지원합니다. 이를 통해 비동기 프로그래밍 패러다임을 사용하여 더 높은 처리 성능과 효율을 달성할 수 있습니다.

각 컴포넌트는 입력 및 출력 유형이 명확하게 정의되어 있으며, `Runnable` 프로토콜을 구현함으로써, 이러한 컴포넌트들은 입력과 출력 스키마를 검사할 수 있습니다. 이는 개발자가 타입 안정성을 보장하고, 예상치 못한 오류를 방지할 수 있도록 도와줍니다.

LangChain 을 사용하여 커스텀 체인을 생성하는 과정은 다음과 같습니다

1. 필요한 컴포넌트들을 정의하고, 각각 `Runnable` 인터페이스를 구현합니다.
2. 컴포넌트들을 조합하여 사용자 정의 체인을 생성합니다.
3. 생성된 체인을 사용하여 데이터 처리 작업을 수행합니다. 이때, `invoke`, `batch`, `stream` 메소드를 사용하여 원하는 방식으로 데이터를 처리할 수 있습니다.

LangChain 의 `Runnable` 프로토콜을 사용하면, 개발자는 보다 유연하고 확장 가능한 방식으로 데이터 처리 작업을 설계하고 구현할 수 있으며, 복잡한 언어 처리 작업을 보다 쉽게 관리할 수 있습니다.

invoke 사용 예제

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 컴포넌트 정의

prompt = ChatPromptTemplate.from_template("지구과학에서 {topic}에 대해 간단히 설명해 주세요")
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 2. 체인 생성

chain = prompt | model | output_parser

# 3. invoke 메소드 사용
result = chain.invoke({"topic": "지구 자전"})
print("invoke 결과:", result)

```

```
실행 결과

invoke 결과: 지구 자전은 지구가 자신의 축을 중심으로 회전하는 과정을 말합니다. 이 회전은 약 24시간, 즉 하루에 한 번 이루어지며, 지구 자전으로 인해 낮과 밤이 발생합니다. 자전의 방향은 서쪽에서 동쪽으로 향하고 있으며, 이로 인해 태양은 동쪽에서 떠서 서쪽으로 지는 것처럼 보입니다.

지구의 자전은 또한 지구의 기후와 날씨에 영향을 미치며, 지구의 자전축은 약 23.5도의 경사를 가지고 있어 계절 변화에도 중요한 역할을 합니다. 고대부터 오늘날까지 지구 자전은 천문학적 현상과 다양한 자연현상을 이해하는 데 중요한 기초가 되고 있습니다.
```

```python

# batch 메소드 사용
topics = ["지구 공전", "화산 활동", "대륙 이동"]
results = chain.batch([{"topic":t} for t in topics])

for topic, result in zip(topics, results):
  print(f"{topic} 설명: {result[:50]}...") # 결과의 처음 50자만 출력

```

```
실행 결과

지구 공전 설명: 지구 공전은 지구가 태양 주위를 타원형 궤도로 돌면서 한 바퀴 도는 과정을 의미합니다. 이...
화산 활동 설명: 화산 활동은 지구 내부의 마그마가 지표로 분출되는 과정으로, 대개 지진 활동과 관련이 있습...
대륙 이동 설명: 대륙 이동은 지구의 대륙들이 서로 다른 방향으로 이동하는 과정을 설명하는 이론으로, 주로 ...
```

```python
# stream 메소드 사용

stream = chain.stream({"topic":"지진"})
print("stream 결과:")

for chunk in stream:
  print(chunk, end="", flush=True)
print()
```

```
실행 결과

stream 결과:
지진은 지구의 지각에서 발생하는 갑작스러운 진동이나 충격으로, 주로 지각 변동이나 지하에서의 압력이 변화할 때 발생합니다. 지진의 주요 원인은 다음과 같습니다.

1. **단층 활동**: 지각에 있는 두 지점이 서로 움직일 때 발생하는데, 이 과정에서 저장된 에너지가 갑작스럽게 방출되며 지진이 발생합니다.

2. **화산 활동**: 화산이 폭발하거나 마그마가 이동할 때 발생할 수 있습니다.

3. **인간 활동**: 광산 채굴, 대형 건축물 건설, 댐 건설 등 인위적인 활동도 지진을 유발할 수 있습니다.

지진의 강도는 리히터 규모나 모멘트 규모로 측정되며, 진원의 깊이와 거리에 따라 지상에서 느껴지는 강도도 달라집니다. 강한 지진은 건물이나 인프라에 심각한 피해를 줄 수 있으며, tsunamis와 같은 2차 재해를 유발할 수도 있습니다. 지진의 발생 및 예측을 위해 지진계와 같은 다양한 기상 장비가 사용됩니다.
```

```python
import nest_asyncio
import asyncio

# nest_asyncio 적용 (구글 코랩 등 주피터 노트북에서 실행 필요)
nest_asyncio.apply()

# 비동기 메소드 사용(async/await 구문 필요)

async def run_async():
  result = await chain.ainvoke({"topic":"해류"})
  print("ainvoke 결과:", result[:50], "...")

asyncio.run(run_async())
```

```
실행 결과

ainvoke 결과: 해류는 바다의 물이 일정한 방향으로 흐르는 현상을 말합니다. 해류는 여러 가지 요인에 의해 ...
```

---

## 2.5 콜백 및 스트리밍 고급제어

### 2.5.1 CallbackHandler 시스템

LangChain 문서에 따르면, 콜백 시스템은 LLM 애플리케이션의 다양한 단계를 모니터링할 수 있게 해주는 강력한 기능입니다. 이는 로깅, 모니터링, 스트리밍 및 기타 작업에 유용합니다.

#### 2.5.1.1 기본 콜백 핸들러 구현 분석

##### 2.5.1.1.1 AstronomyCallbackHandler 클래스 구조

```python
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
from typing import Dict, List, Any
import time
from datetime import datetime

class AstronomyCallbackHandler(BaseCallbackHandler):
    """천문학 체인 전용 콜백 핸들러"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = None
        self.chain_steps = []
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

```

핵심 초기화 요소
- `BaseCallbackHandler` 를 상속받아 표준 인터페이스를 구현합니다.
- `verbose` 모드를 통해 출력 상세도를 제어합니다.
- 실행 시간 추적을 위한 `start_time`을 관리합니다.
- 체인 실행 단계를 기록하는 `chain_steps` 리스트를 유지합니다.
- 토큰 사용량을 추적하는 딕셔너리를 포함합니다

---

##### 2.5.1.1.2 핵심 콜백 메서드 구현

```python
def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
    """체인 시작 시 호출"""
    self.start_time = time.time()
    self.chain_steps = []

    if self.verbose:
        print(f"🚀 [체인 시작] {datetime.now().strftime('%H:%M:%S')}")
        print(f"📝 입력: {inputs}")
        print("-" * 50)

```

체인 시작 이벤트 처리
- 실행 시작 시간을 기록합니다.
- 체인 단계 리스트를 초기화합니다.
- 사용자 친화적인 로깅 형식으로 입력 정보를 출력합니다.

```python
def on_llm_end(self, response: LLMResult, **kwargs) -> None:
    """LLM 호출 완료 시 호출"""
    self.chain_steps.append("LLM 호출 완료")

    # 토큰 사용량 업데이트
    if hasattr(response, 'llm_output') and response.llm_output:
        usage = response.llm_output.get('token_usage', {})
        self.token_usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
        self.token_usage['completion_tokens'] += usage.get('completion_tokens', 0)
        self.token_usage['total_tokens'] += usage.get('total_tokens', 0)

```

LLM 완료 이벤트 처리
- 체인 단계에 완료 상태를 추가합니다.
- 응답 객체에서 토큰 사용량 정보를 추출하여 누적합니다.
- 안전한 딕셔너리 접근을 통해 예외 상황을 처리합니다.

---

#### 2.5.1.2 멀티 콜백 시스템 구현

##### 2.5.1.2.1 성능 측정 전용 콜백

```python
class PerformanceCallbackHandler(BaseCallbackHandler):
    """성능 측정 전용 콜백"""

    def __init__(self):
        self.performance_data = {}
        self.current_step = None
        self.step_start_time = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.current_step = 'llm_processing'
        self.step_start_time = time.time()

    def on_llm_end(self, response, **kwargs):
        if self.current_step and self.step_start_time:
            duration = time.time() - self.step_start_time
            self.performance_data[self.current_step] = duration

```

성능 메트릭 수집
- 각 처리 단계별 실행 시간을 정밀하게 측정합니다.
- LLM 처리와 전체 체인 실행 시간을 분리하여 추적합니다.
- 오버헤드 계산을 위한 데이터를 수집합니다.

---

##### 2.5.1.2.2 로깅 전용 콜백

```python
class LoggingCallbackHandler(BaseCallbackHandler):
    """로깅 전용 콜백"""

    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.logs = []

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')

```

로깅 시스템 특징
- 타임스탬프가 포함된 구조화된 로그 형식을 제공합니다.
- 메모리 내 로그와 파일 로그를 동시에 지원합니다.
- 안전한 파일 쓰기를 위한 컨텍스트 매니저를 사용합니다.

---

#### 2.5.1.3 멀티 콜백 활용 패턴

##### 2.5.1.3.1 콜백 조합 사용

```python
performance_callback = PerformanceCallbackHandler()
logging_callback = LoggingCallbackHandler()

# 여러 콜백을 함께 사용
callbacks = [performance_callback, logging_callback]

# 테스트 실행
test_topics = ["블랙홀", "중성자별", "적색거성"]

for topic in test_topics:
    result = chain.invoke(
        {"topic": topic}, 
        config={"callbacks": callbacks}
    )

```

멀티 콜백 장점
- 각 콜백이 서로 다른 관점에서 체인 실행을 모니터링합니다.
- 성능, 로깅, 오류 처리를 독립적으로 관리할 수 있습니다.
- 모듈화된 구조로 재사용성이 높습니다.

---

##### 2.5.1.3.2 실시간 성능 분석

```python
# 성능 리포트 출력
perf_report = performance_callback.get_report()
print("성능 지표:")
for metric, value in perf_report.items():
    print(f"  - {metric}: {value}")

```

성능 분석 결과
- LLM 처리 시간과 전체 실행 시간을 분리하여 측정합니다.
- 오버헤드를 계산하여 최적화 포인트를 식별합니다.
- 실시간으로 성능 메트릭을 확인할 수 있습니다.

### 2.5.2 LangChain 콜백 이벤트 체계

LangChain 문서에서 정의한 주요 콜백 이벤트들은 다음과 같습니다.

|이벤트|트리거 시점|연관메소드|
|:---:|:---------:|:-------:|
|Chain start|체인 시작시 | `on_chain_start`|
|Chain end  |체인 종료 시| `on_chain_end`  |
|LLM start  |LLM 시작 시 | `on_llm_start` |
|LLM end    |LLM 종료 시 | `on_llm_end`   |
|LLM error  |LLM 오류 시 | `on_llm_error` |

### 2.5.3 실제 활용 시나리오

#### 2.5.3.1 프로덕션 모니터링

- API 호출 빈도와 응답 시간 추적
- 토큰 사용량 기반 비용 계산
- 오류율 모니터링 및 알림 시스템

#### 2.5.3.2 디버깅 및 최적화

- 체인 실행 과정의 상세 추적
- 병목 구간 식별 및 성능 튜닝
- A/B 테스트를 위한 메트릭 수집

#### 2.5.3.3 사용자 경험 개선

- 실시간 진행 상황 표시
- 스트리밍 기반 응답 제공
- 오류 발생 시 사용자 친화적 메시지 제공

이러한 CallbackHandler 시스템은 LangChain 애플리케이션의 투명성과 제어성을 크게 향상시키며, 프로덕션 환경에서의 안정적인 운영을 가능하게 합니다.

---

# 마치며

wiki docs 의 "랭체인(LangChain) 입문부터 응용까지" 의 LangChain 기초 부분에 있는 것들을 포스트에 작성하면서 LangChain 의 기초에 대한 공부를 진행해 보았습니다. 직접 따라해보면서 LangChain 이 대략적으로 무엇인지 알게 되었고, 또 wiki docs 에 있는 코드들 중에는 실행이 안되거나, 설명이 부족하거나, 실행결과가 다른 것들이 많았습니다. 그래서 기존 작성된 코드에 직접 수정을 하다보니 코드가 눈과 손에 자연스레 익숙해지기도 했습니다. 이제 다음부턴 LangChain 을 이용한 RAG 에 대한 공부를 진행하고자 합니다. 

# 참조

- 판다스 스튜디오 - 랭체인(LangChain) 입문부터 응용까지(https://wikidocs.net/book/14473)
- 테디노트 - LangChain 한국어 튜토리얼KR(https://wikidocs.net/book/14314)