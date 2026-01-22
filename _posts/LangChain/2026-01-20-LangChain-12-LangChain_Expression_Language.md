---
title: "[LangChain] Langchain - 12. LangChain Expression Language에 대해서 알아보자"
categories:
  - LangChain
tags:
  - LangChain
  
use_math: true  
toc: true
toc_sticky: true
toc_label: "LangChain Expression Language에 대해서 알아보자"
---

# 머리말

여태까지 LangChain을 이용해 문서로드->청킹->문서 임베딩->벡터 DB 적재->검색기를 이용한 쿼리와 유사한 문서 추출->쿼리와 함께 쿼리와 유사한 문서를 LLM에게 제공->고품질의 답변 과정의 기초적인 RAG 시스템에 대해서만 다뤄보았습니다. 이제 기초를 넘어 심화 과정을 진행하고자 합니다. 심화 과정에 앞서 LangChain에서 제공하는 LCEL(LangChain Expression Language)에 대해서 미리 알고 가면 좋을 것 같아 LCEL을 먼저 다뤄보고자 합니다.

# 1. 개요

LangChain Expression Language(LCEL)은 LangChain 라이브러리에서 제공하는 선언적 방식의 인터페이스로, 복잡한 LLM(Large Language Model) 애플리케이션을 구축하고 실행하기 위한 도구입니다. LCEL은 LLM, 프롬프트, 검색기, 메모리 등 다양한 컴포넌트를 조합하여 강력하고 유연한 AI 시스템을 만들 수 있게 해줍니다.

## 1.1 LCEL 주요 특징

1. 선언적 구문: 복잡한 로직을 간결하고 읽기 쉬운 방식으로 표현할 수 있습니다.
2. 모듈성: 다양한 컴포넌트를 쉽게 조합하고 재사용할 수 있습니다.
3. 유연성: 다양한 유형의 LLM 애플리케이션을 구축할 수 있습니다.
4. 확장성: 사용자 정의 컴포넌트를 쉽게 통합할 수 있습니다.
5. 최적화: 실행 시 자동으로 최적화를 수행합니다.

## 1.2 LCEL 기본 구성 요소

1. Runnable: 모든 LCEL 컴포넌트의 기본 클래스입니다.
2. Chain: 여러 Runnable을 순차적으로 실행합니다.
3. RunnableMap: 여러 Runnable을 병렬로 실행합니다.
4. RunnableSequence: Runnable의 시퀀스를 정의합니다.
5. RunnableLambda: 사용자 정의 함수를 Runnable로 래핑합니다.

## 1.3 고급기능

1. 병렬처리: RunnableMap을 사용하여 여러 작업을 동시에 실행할 수 있습니다.
2. 조건부 실행: RunnableBranch를 사용하여 조건에 따라 다른 경로를 실행할 수 있습니다.
3. 재시도 및 폴백: 실패 시 자동으로 재시도하거나 대체 경로를 실행할 수 있습니다.
4. 스트리밍: 대규모 데이터를 효율적으로 처리할 수 있습니다.

## 1.4 장단점

장점

1. 코드 가독성: 복잡한 로직을 명확하고 간결하게 표현할 수 있습니다.
2. 유지보수성: 모듈화된 구조로 인해 유지보수가 용이합니다.
3. 성능: 자동 최적화를 통해 효율적인 실행이 가능합니다.
4. 확장성: 새로운 컴포넌트를 쉽게 추가하고 통합할 수 있습니다.

단점

1. 학습 곡선: 새로운 패러다임에 익숙해지는 데 시간이 필요할 수 있습니다.
2. 디버깅: 복잡한 체인의 경우 디버깅이 어려울 수 있습니다.
3. 성능 오버헤드: 매우 간단한 작업의 경우 오버헤드가 발생할 수 있습니다.

## 1.5 활용 사례

LCEL은 다음과 같은 다양한 LLM 애플리케이션 구축에 사용될 수 있습니다.

- 대화형 AI 시스템
- 문서 요약 및 분석 도구
- 질의응답 시스템
- 데이터 추출 및 변환 파이프라인
- 다국어 번역 서비스

# 2. RunnablePassthrough

RunnablePassthrough는 데이터를 전달하는 역할을 합니다. 이 클래스는 `invoke()` 메서드를 통해 입력된 데이터를 그대로 반환합니다. 이는 데이터를 변경하지 않고 파이프라인의 다음 단계로 전달하는 데 사용될 수 있습니다. 

RunnablePassthrough는 다음과 같은 시나리오에서 유용할 수 있습니다.

- 데이터를 변환하거나 수정할 필요가 없는 경우
- 파이프라인의 특정 단계를 건너뛰어야 하는 경우
- 디버깅 또는 테스트 목적으로 데이터 흐름을 모니터링해야 하는 경우

RunnablePassthrough는 Runnable 인터페이스를 구현하므로, 다른 Runnable 객체와 함께 파이프라인에서 사용될 수 있습니다.

## 2.1 데이터 전달하기(RunnablePassthrough)

RunnablePassthrough는 입력을 변경하지 않고 그대로 전달하거나 추가 키를 더하여 전달할 수 있습니다.
일반적으로 RunnableParallel과 함께 사용되어 데이터를 맵의 새로운 키에 할당하는 데 활용됩니다. RunnablePassthrough()를 단독으로 호출하면 단순히 입력을 받아 그대로 전달합니다.

assign과 함께 호출된 RunnablePassthrough(RunnablePassthrough.assign(...))는 입력을 받아 assign 함수에 전달된 추가 인자를 더합니다.

그렇다면 예제 코드로 한 번 알아보도록 하겠습니다. 아래 예제 코드는 RunnablePassthrough를 잘 보여주는 데이터의 원본을 유지하는 "그대로 통과시키기" 예제입니다.

예제 실행 전에 예제 실행에 필요한 라이브러리 설치부터 진행해 줍니다.

```bin
!pip install langchain-core
```

아래 예시에서는 RunnablePassthrough의 기능을 이용해 입력을 그대로 original_question에 주입시켜주는 예제입니다.

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# 1. 입력된 텍스트에서 키워드만 뽑아내는 가상의 함수
def extract_keyword(text):
    # 실제로는 NLP 모델이 들어갈 자리입니다.
    return text.split()[-1] 

# 2. 프롬프트 구성
prompt = ChatPromptTemplate.from_template(
    "사용자 질문: {original_question}\n추출된 키워드: {keyword}\n위 정보를 바탕으로 답변하세요."
)

# 3. 체인 구성
# RunnablePassthrough()는 입력받은 '질문' 그 자체를 'original_question' 키에 그대로 매핑합니다.
chain = (
    {
        "original_question": RunnablePassthrough(), 
        "keyword": extract_keyword
    }
    | prompt
)

# 실행
result = chain.invoke("오늘 서울의 날씨는 어때?")
print(result.messages[0].content)
```

```
Output:
사용자 질문: 오늘 서울의 날씨는 어때?
추출된 키워드: 어때?
위 정보를 바탕으로 답변하세요.
```

다음 예제는 RunnablePassthrough의 assign 함수를 이용한 중간에 값을 주입하는 예제입니다. RunnablePassthrough()를 이용해 원본 입력 값인 "안녕"을 유지합니다. 그리고 중간에 RunnablePassthrough.assign()을 이용해 question에 "오늘 날씨 어때?"를 주입하는 예제입니다.

```python
# 1. 언어를 감지하는 가상의 체인 (결과로 {'language': 'Korean'} 등을 반환한다고 가정)
detect_language_chain = {"language": lambda x: "Korean",
                         "original_text": RunnablePassthrough() # 원본 보존
                         } 

# 2. 최종 프롬프트 체인
prompt = ChatPromptTemplate.from_template("{original_text}, {language}로 답변해줘. 질문은 이거야: {question}")
final_chain = prompt

# 3. 전체 연결
full_pipeline = (
    detect_language_chain 
    | RunnablePassthrough.assign(question=lambda x: "오늘 날씨 어때?") # 여기서 고정값 주입
    | final_chain
)

result = full_pipeline.invoke("안녕")
print(result.messages[0].content)
```

# 3. RunnableLambda

`RunnableLambda`는 사용자 정의 함수를 실행할 수 있는 기능을 제공합니다. 이를 통해 개발자는 자신만의 함수를 정의하고, 해당 함수를 RunnableLambda를 사용하여 실행할 수 있습니다. 예를 들어, 데이터 전처리, 계산, 또는 외부 API와의 상호 작용과 같은 작업을 수행하는 함수를 정의하고 실행할 수 있습니다.

## 3.1 사용자 정의 함수(RunnableLambda)를 실행하는 방법

사용자 정의 함수를 RunnableLambda로 래핑하여 활용할 수 있는데, 여기서 주의할 점은 **사용자 정의 함수가 받을 수 있는 인자는 1개 뿐이라는 점** 입니다.

만약 여러 인수를 받는 함수로 구현하고 싶다면, 단일 입력을 받아들이고 이를 여러 인수로 풀어내는 래퍼를 작성해야 합니다.

아래는 RunnableLambda를 사용한 예제입니다. chain에서 인자 a를 받을 때는 입력된 input_1의 글자수 길이를 받도록 RunnableLambda에 length_function을 적용하였고, 인자 b를 받을 때는 input_1과 input_2의 글자 수를 곱한 값을 받도록 RunnableLambda에 multiple_length_function 함수를 적용하였습니다.

다음은 예제 실행에 필요한 라이브러리입니다.

```bin
!pip install langchain-core langchain-openai langchain-classic langchain-community
```

```python
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def length_function(text): # 텍스트의 길이를 반환하는 함수
    return len(text)

def _multiple_length_function(text1, text2): # 두 텍스트의 길이를 곱하는 함수
    return len(text1) * len(text2)

def multiple_length_function(_dict,): # 2개 인자를 받는 함수로 연결하는 wrapper 함수
    return _multiple_length_function(_dict["text1"], _dict["text2"])

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template("what is {a} + {b}?")

# ChatOpenAI 모델 초기화
model = ChatOpenAI()

# 프롬프트와 모델을 연결하여 체인 생성
chain1 = prompt | model

# 체인 구성
chain = (
    {
        "a": itemgetter("input1") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("input_1"), "text2": itemgetter("input_2")}
        | RunnableLambda(multiple_length_function)
    }
    | prompt
    | model
    | StrOutputParser()
)
```

```python
# 주어진 인자들로 체인을 실행합니다.
chain.invoke({"input_1": "bar", "input_2": "gah"})
```

결과를 보면 3 + 9 = 12 가 출력되는 것을 확인할 수 있습니다. 

```
Output:
'3 + 9 = 12'
```

## 3.2 RunnableConfig 인자로 활용

RunnableLambda는 선택적으로 RunnableConfig를 수용할 수 있습니다. 여기서 RunnableConfig란? RunnableConfig는 체인이 실행될 때 흐르는 **설정 정보 꾸러미**입니다. 여기에는 다음과 같은 정보들이 담깁니다.

- tags: LangSmith에서 검색하기 위한 태그
- metadata: 실행 단위에 대한 추가 정보
- callbacks: 토큰 사용량 계산이나 로그 기록을 위한 도구
- recursion_limit: 재귀 호출 제한

이를 통해 콜백, 태그 및 기타 구성 정보를 중첩된 실행에 전달할 수 있습니다. 그러면 RunnableConfig를 사용한 예제 코드를 보도록 하겠습니다.

아래 예제는 상위에서 정의한 실행 환경(Config)을 하위의 파생 작업들에게 어떻게 전달 시키는가를 보여주는 예제입니다.

아래 코드의 흐름을 보면 우선 오류를 유도합니다. `input="{foo:: bar}`는 잘못된 JSON 형식으로 `json.loads(text)`에서 무조건 에러가 발생합니다. 에러가 발생하면 `except`구문으로 들어갑니다. 여기서 LLM(fixing_chain)을 호출해 "이 JSON 좀 고쳐줘"라고 요청합니다. 이때 `invoke`의 두 번째 인자로 `config`를 그대로 전달합니다. 이 덕분에 `get_openai_callback()`은 원본 실행뿐만 아니라, 수정하기 위해 호출한 LLM의 토큰 사용량까지 합산해서 계산할 수 있게 됩니다. 최대 3번까지 수정을 시도하며, 성공하면 파싱된 JSON 객체를 반환합니다.

아래 코드의 가장 중요한 부분은 `text = fixing_chain.invoke({"input": text, "error": e}, config)` 입니다. config를 인자로 넘겨주는 이유는 `parse_or_fix` 함수는 `RunnableLambda`로 감싸져 있습니다. 이 함수가 호출될 때, LangChain은 자동으로 현재 실행 중인 설정(`config`)을 두 번째 인자로 넣어줍니다. `fixing_chain.invoke(...)`를 호출할 때 `config`를 빼먹으면, 외부에서 설정한 `tags`("my-tag")나 `callbacks`(토큰 계산기)가 내부의 `fixing_chain`에는 적용되지 않습니다. LangSmith 같은 도구를 쓸 때, `config`를 넘겨줘야만 상위 체인과 하위 체인이 하나의 작업으로 묶여서 보입니다. 그러지 않으면 별개의 작업으로 인식되어 디버깅이 어려워집니다.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
import json

def parse_or_fix(text: str, config: RunnableConfig):
    # 다음 텍스트를 수정하는 프롬프트 템플릿을 생성합니다.
    fixing_chain = (
        ChatPromptTemplate.from_template(
            "Fix the following text:\n\ntext\n{input}\n\nError: {error}"
            " Don't narrate, just respond with the fixed data."
        )
        | ChatOpenAI()
        | StrOutputParser()
    )

    # 최대 3번 시도합니다.
    for _ in range(3):
        try:
            # JSON 형식으로 텍스트를 파싱합니다.
            return json.loads(text)
        except Exception as e:
            # 파싱 중 오류가 발생하면 수정 체인을 호출하여 텍스트를 수정합니다.
            text = fixing_chain.invoke({"input": text, "error": e}, config)
            print(f"config: {config}")
    # 파싱에 실패한 경우 "Failed to parse" 문자열을 반환합니다.
    return "Failed to parse"
```

```python
from langchain_classic.callbacks import get_openai_callback

with get_openai_callback() as cb:
    # RunnableLambda를 사용하여 parse_or_fix 함수를 호출합니다.
    output = RunnableLambda(parse_or_fix).invoke(
        input="{foo:: bar}",
        config={"tags": ["my-tag"], "callbacks": [cb]},
    )
    # 수정한 결과를 출력합니다.
    print(f"\n\n수정한결과:\n{output}")
```

결과를 보면 config 정보가 출력되며 수정된 JSON이 출력되는 것을 확인할 수 있습니다.

```
Output:
config: {'tags': ['my-tag'], 'metadata': {}, 'callbacks': <langchain_core.callbacks.manager.CallbackManager object at 0x7ae36c357f20>, 'recursion_limit': 25, 'configurable': {}}


수정한결과:
{'foo': 'bar'}
```

# 4. RunnableParallel

RunnableParallel의 정의와 존재 의의로 3줄로 요약하면 다음과 같습니다.

1. 하나의 입력을 여러 경로로 복제하여 병렬로 실행함으로써 전체 체인의 응답 속도(Latency)를 최적화하는 도구입니다.
2. 여러 작업의 결과를 각각의 키(Key)에 매핑하여 딕셔너리 형태로 묶어주는 **데이터 구조화 및 변환**의 핵심 역할을 수행합니다.
3. 원본 데이터와 가공 데이터를 동시에 다음 단계로 전달할 수 있게 하여, 복잡한 파이프라인 내에서 데이터 유실을 방지하고 흐름을 제어합니다.

그렇다면 간단한 예제를 통해 RunnableParallel에 대해서 좀 더 구체적으로 알아보도록 하겠습니다. 아래 예제 코드는 하나의 입력값(도시 이름)을 받아 두 가지 서로 다른 정보(날씨, 추천 명소)를 동시에 가져온 뒤, 원본 데이터와 함께 하나로 묶어주는 과정을 보여줍니다.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 1. 데이터를 가공하는 가상의 함수들(실제로는 API 호출이나 DB 조회가 들어가는 자리입니다.)
def get_weather(city: str) -> str:
    # 도시별 날씨를 가져오는 로직 시뮬레이션
    return f"{city}의 현재 날씨는 '맑음'입니다."

def get_tourist_spot(city: str) -> str:
    # 도시별 명소를 가져오는 로직 시뮬레이션
    return f"{city}의 추천 명소는 '남산타워'입니다.'"

# 2. RunnableParallel 구성
# 입력된 'city' 문자열이 세 가지 경로로 동시에 전달됩니다.
parallel_chain = RunnableParallel(
    weather=get_weather, # 경로 1: 날씨 정보 생성
    spot=get_tourist_spot, # 경로 2: 명소 정보 생성
    original_city=RunnablePassthrough() # 경로 3: 원본 입력 보존
)

# 3. 체인 실행
# 입력값 "서울"이 위에서 정의한 3개 경로로 각가 전달되어 병렬 처리됩니다.
result = parallel_chain.invoke("서울")

print("--- 실행 결과 ---")
print(result)
```

```
Output:
--- 실행 결과 ---
{'weather': "서울의 현재 날씨는 '맑음'입니다.", 'spot': "서울의 추천 명소는 '남산타워'입니다.'", 'original_city': '서울'}
```
# 4. 동적 속성 지정(configurable_fields, configurable_alternatives)

이번엔 chian 호출 시 다양한 옵션을 동적으로 설정할 수 있는 방법에 대해서 알아보도록 하겠습니다. 다음의 두 가지 방식으로 동적 구성을 할 수 있습니다.

- configurable_fields 메서드: 이 메서드를 통해 실행 가능한 객체의 특정 필드를 구성할 수 있습니다.
- configurable_alternatives: 이 메서드를 사용하면 런타임 중에 설정할 수 있는 특정 실행 가능한 객체에 대한 대안을 나열할 수 있습니다.

## 4.1 configurable_fields

configurable_fields는 런타임(실행 시점)에 체인의 특정 구성 요소를 동적으로 변경할 수 있게 해주는 강력한 도구입니다. configurable_fields를 사용하면 코드의 재사용성과 제어의 분리가 가능해집니다. 다음은 configurable_fields를 사용해야 하는 이유입니다.

- 상태 비저장(Stateless) 설계: 하나의 체인 객체를 만들어 두고, 실행 시점에만 설정을 주입함으로써 멀티테넌트(Multi-tenant) 환경(사용자마다 설정이 다른 환경)에서 안전하게 공유할 수 있습니다.
- A/B 테스트 용이성: 동일한 로직에서 파라미터만 살짝 바꾼 여러 버전의 체인을 만들어 성능을 비교하기에 최적입니다.
- UI와의 결합: 웹 서비스의 슬라이더나 드롭다운 메뉴값을 체인에 직접 전달하기 매우 깔끔한 구조를 제공합니다.

ChatOpenAI를 사용할 때, 우리는 model_name과 같은 설정을 조정할 수 있습니다. model_name은 GPT의 버전을 명시할 때 사용하는 속성입니다.

만약 고정된 model_name이 아닌 동적으로 모델을 지정하고 싶을 때는 다음과 같이 ConfiguralbeField를 활용하여 동적으로 설정할 수 있는 속성 값으로 변환할 수 있습니다. 그럼 예제를 통해 한 번 알아보도록 하겠습니다.

configurable_fields 메서드를 사용하여 model_name 속성을 동적 구성 가능한 필드로 지정합니다.

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0).configurable_fields(
    model_name = ConfigurableField(
        id="gpt_version",
        name="Version of GPT",
        # model_name의 설명을 설정합니다.
        description="Official model name of GPTs. ex) gpt-4o, gpt-4o-mini",
    )
)
```
model.invoke() 호출시 config={"configurable": {"키": "값"}} 형식으로 동적 지정할 수 있습니다.

```python
model.invoke(
    "대한민국의 수도는 어디야?",
    # gpt_version을 gpt-3.5-turbo로 설정합니다.
    config={"configurable": {"gpt_version": "gpt-3.5-turbo"}}
).__dict__
```

```
Output:
{'content': '대한민국의 수도는 서울이야.',
 'additional_kwargs': {'refusal': None},
 'response_metadata': {'token_usage': {'completion_tokens': 16,
   'prompt_tokens': 22,
   'total_tokens': 38,
   'completion_tokens_details': {'accepted_prediction_tokens': 0,
    'audio_tokens': 0,
    'reasoning_tokens': 0,
    'rejected_prediction_tokens': 0},
   'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
  'model_provider': 'openai',
  'model_name': 'gpt-3.5-turbo-0125',
  'system_fingerprint': None,
  'id': 'chatcmpl-D04BCyldejZpluCL5F2Y2QaIZXD6Y',
  'service_tier': 'default',
  'finish_reason': 'stop',
  'logprobs': None},
 'type': 'ai',
 'name': None,
 'id': 'lc_run--019bdb28-b9a0-73f0-8f3b-41bfce5eb660-0',
 'tool_calls': [],
 'invalid_tool_calls': [],
 'usage_metadata': {'input_tokens': 22,
  'output_tokens': 16,
  'total_tokens': 38,
  'input_token_details': {'audio': 0, 'cache_read': 0},
  'output_token_details': {'audio': 0, 'reasoning': 0}}}
```

이번에는 gpt-4o-mini 모델을 사용해 보도록 하겠습니다.

```python
model.invoke(
    # gpt_version을 gpt-4o-mini로 설정합니다.
    "대한민국의 수도는 어디야?",
    config={"configurable": {"gpt_version": "gpt-4o-mini"}},
).__dict__
```

```
Output:
{'content': '대한민국의 수도는 서울입니다.',
 'additional_kwargs': {'refusal': None},
 'response_metadata': {'token_usage': {'completion_tokens': 8,
   'prompt_tokens': 15,
   'total_tokens': 23,
   'completion_tokens_details': {'accepted_prediction_tokens': 0,
    'audio_tokens': 0,
    'reasoning_tokens': 0,
    'rejected_prediction_tokens': 0},
   'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
  'model_provider': 'openai',
  'model_name': 'gpt-4o-mini-2024-07-18',
  'system_fingerprint': 'fp_c4585b5b9c',
  'id': 'chatcmpl-D04DF9rLAF7UT5bo8CcZgVbCOJlju',
  'service_tier': 'default',
  'finish_reason': 'stop',
  'logprobs': None},
 'type': 'ai',
 'name': None,
 'id': 'lc_run--019bdb2a-a694-7760-914a-d78ea6240551-0',
 'tool_calls': [],
 'invalid_tool_calls': [],
 'usage_metadata': {'input_tokens': 15,
  'output_tokens': 8,
  'total_tokens': 23,
  'input_token_details': {'audio': 0, 'cache_read': 0},
  'output_token_details': {'audio': 0, 'reasoning': 0}}}
```

model 객체의 with_config() 메서드를 사용하여 configurable 매개변수를 설정할 수도 있습니다. 이전과 동작하는 방식은 동일합니다.

```python
model.with_config(configurable={"gpt_version": "gpt-4o-mini"}).invoke(
    "대한민국의 수도는 어디야?"
).__dict__
```

모델 버전 외에도 실무에서 가장 빈번하게 활용되는 세 가지 추가적인 핵심 예제를 더 소개하도록 하겠습니다.

가장 대표적인 활용 사례로 LLM의 하이퍼 파라미터 중 하나인 `temperature`을 조절하는 것입니다. temperatrue는 사용자의 요청 성격에 따라 창의적인 답변이 필요할 때와 정밀한 답변이 필요할 때 조절하는 값입니다. 아래는 예제 코드입니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableField

model = ChatOpenAI(model="gpt-4o-mini").configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperatrue",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)

# 창의적인 글쓰기 모드 (temperature=0.9)
creative_chain = model.with_config(configurable={"llm_temperature":0.9})

# 정밀한 사실 확인 모드(temperature=0)
precise_chain = model.with_config(configurable={"llm_temperature":0})
```

두 번째로는 검색기의 검색 개수 설정입니다. 이는 RAG 시스템을 구축할 때 매우 유용합니다. 질문의 복잡도나 사용자의 설정에 따라 참조할 문서의 개수(`k`)를 동적으로 조절할 수 있습니다.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import ConfigurableField

# 가상의 벡터 저장소 검색기 설정
retriever = FAISS.from_texts(["..."], OpenAIEmbeddings()).as_retriever()

# search_kwargs 내의 k 값을 설정 가능하도록 변경
configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs of the retriever",
    )
)

# 좁고 깊게 검색 (k=3)
deep_chain = configurable_retriever.with_config(configurable={"search_kwargs": {"k":3}})

# 넓게 훑기 (k=10)
broad_chain = configurable_retriever.with_config(configurable={"search_kwargs":{"k":10}})
```

마지막으로 검색 전략의 동적 변경(Search Type)입니다. 단순 유사도 검색(`similarity`)을 할지, 아니면 결과의 다양성을 보장하는 **MMR(Maximum Marginal Relevance)** 방식을 사용할지 런타임에 결정할 수 있습니다.

```python
configurable_retriever = retriever.configurable_fields(
    search_type=ConfigurableField(
        id="retriever_search_type", 
        name="Retriever Search Type",
        description="The type of search to perform",
    )
)

# 일반적인 검색 방식 사용
similarity_chain = configurable_retriever.with_config(
    configurable={"retriever_search_type": "similarity"}
)

# 결과의 중복을 피하고 다양성을 높이는 방식 사용
mmr_chain = configurable_retriever.with_config(
    configurable={"retriever_search_type": "mmr"}
)
```

## 4.2 Configurable Alternatives

`configurable_alternatives`는 LCEL에서 제공하는 메서드로, 체인을 구성하는 특정 Runnable 객체(LLM, 프롬프트, 리트리버 등)를 실행 시점에 다른 대안(alternatives) 객체로 완전히 교체할 수 있게 해주는 기능입니다. 단순히 모델의 파라미터(temperature, 최대 토큰 등)를 바꾸는 것을 넘어, "OpenAI 모델을 쓸 것인가, 아니면 Anthropic 모델을 쓸 것인가?" 또는 "단순 프롬프트를 쓸 것인가, 아니면 예시가 포함된 Few-shot 프롬프트를 쓸 것인가?"와 같은 구조적 선택을 가능하게 합니다.

configurable_alternatives를 사용하는 이유는 소프트웨어 엔지니어링과 NLP 프로젝트 관리 측면에서 다음과 같은 이점을 제공합니다.

1. 모델 벤더 독립성 및 결합도 감소(Decoupling)
    특정 LLM 제공 업체에 종속되지 않는 아키텍처를 설계할 수 있습니다. 예를 들어, OpenAI 서버에 장애가 발생했을 때 코드를 수정하고 다시 배포할 필요 없이, 설정값 하나만 바꿔서 바로 Anthropic이나 로컬의 Llama 모델로 전환할 수 있습니다.

2. 신속한 A/B 테스트 및 실험
    NLP 연구나 서비스 고도화 과정에서 어떤 프롬프트나 모델이 더 나은 성능을 내는지 비교하는 것은 필수입니다. `configurable_alternatives`를 사용하면 동일한 데이터 파이프라인 안에서 모델이나 리트리버 전략만 바꿔가며 성능을 즉각적으로 벤치마킹할 수 있습니다.

3. 개발 및 운영 환경의 분리
    - 개발 환경: 비용 절감을 위해 로컬에서 구동되는 Ollama 모델 사용
    - 운영 환경: 신뢰도가 높은 클라우드 API 모델 사용 환경 변수에 따라 체인의 구성 요소를 동적으로 선택하도록 설계할 수 있습니다.

### 4.2.1 LLM 객체의 대안(alternatives) 설정 방법

LLM을 활용하여 이를 수행하는 방법을 살펴보겠습니다. 아래 예제 코드에서는 기본적으로 google의 gemini를 사용하지만 configurable_alternatives를 사용해 옵션으로 다른 LLM 모델도 사용할 수 있습니다.

```python
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

llm = ChatGoogleGenerativeAI(
    temperature=0, model="gemini-2.5-flash"
    ).configurable_alternatives(
        # 이 필드에 id를 부여합니다.
        # 최종 실행 가능한 객체를 구성할 때, 이 id를 사용하여 이 필드를 구성할 수 있습니다.
        ConfigurableField(id="llm"),

        # 기본 키를 설정합니다.
        # 이 키를 지정하면 위에서 초기화된 기본 LLM(Gemini)이 사용됩니다.
        default_key="gemini",

        # 'openai'라는 이름의 새 옵션을 추가하며, 이는 'ChatOpenAI()'와 동일합니다.

        openai=ChatOpenAI(model="gpt-4o-mini"),

        # 'gpt4'라는 이름의 새 옵션을 추가하며, 이는 'ChatOpenaI(model="gpt-4")'와 동일합니다.
        gpt4o=ChatOpenAI(model="gpt-4o"),

        # 아래에 여 많은 구성 옵션을 추가할 수 있습니다.
    )

prompt = PromptTemplate.from_template("{topic}에 대해 간단히 설명해 주세요.")
chain = prompt | llm
```

chain.invoke() 메서드는 기본 LLM인 gemini을 활용한 체인을 호출합니다.

```python
# Gemini를 기본으로 호출합니다.

chain.invoke({"topic": "뉴진스"}).__dict__
```

```
Output:
{'content': "뉴진스(NewJeans)는 2022년 어도어(ADOR)에서 데뷔한 5인조 K팝 걸그룹입니다.\n\n**주요 특징:**\n\n1.  **음악 스타일:** '이지 리스닝(Easy Listening)'을 지향하며, 편안하고 세련된 R&B 기반의 팝 사운드가 특징입니다. 듣기 편안하면서도 중독성 있는 멜로디로 큰 사랑을 받고 있습니다.\n2.  **콘셉트:** 청량하고 자연스러운 매력과 Y2K(2000년대 초반) 감성을 현대적으로 재해석한 콘셉트로 독보적인 분위기를 구축했습니다. 멤버들의 꾸밈없는 비주얼과 스타일링도 큰 주목을 받습니다.\n3.  **주요 히트곡:** 데뷔 초부터 'Hype Boy', 'Attention'으로 큰 인기를 얻었으며, 이후 'Ditto', 'OMG', 'Super Shy', 'ETA' 등 발표하는 곡마다 음원 차트를 휩쓸며 국내외에서 신드롬급 인기를 구가하고 있습니다.\n\n간단히 말해, 뉴진스는 **'이지 리스닝 음악과 Y2K 감성을 현대적으로 재해석한 자연스러운 콘셉트'**로 K팝 씬에 신선한 바람을 불어넣으며 빠르게 최정상급 인기를 얻은 걸그룹입니다.",
 'additional_kwargs': {},
 'response_metadata': {'finish_reason': 'STOP',
  'model_name': 'gemini-2.5-flash',
  'safety_ratings': [],
  'model_provider': 'google_genai'},
 'type': 'ai',
 'name': None,
 'id': 'lc_run--019bdf7c-a061-7261-b740-d96b47bbc1b0-0',
 'tool_calls': [],
 'invalid_tool_calls': [],
 'usage_metadata': {'input_tokens': 12,
  'output_tokens': 1266,
  'total_tokens': 1278,
  'input_token_details': {'cache_read': 0},
  'output_token_details': {'reasoning': 959}}}
```

chain.with_config(configurable={"llm":"모델"})를 사용하여 사용할 llm으로 다른 모델을 지정할 수 있습니다.

```python
# 체인의 설정을 변경하여 호출합니다.
chain.with_config(configurable={"llm":"openai"}).invoke({"topic":"뉴진스"}).__dict__
```

```
Output:

{'content': '뉴진스(NewJeans)는 2022년 8월 1일에 데뷔한 대한민국의 걸그룹으로, 하이브의 자회사인 ADOR에서 기획한 그룹입니다. 뉴진스는 독창적이고 세련된 음악 스타일과 패션 감각으로 주목받고 있으며, 4세대 K-pop 대표 그룹 중 하나로 떠오르고 있습니다. \n\n그룹 이름인 "뉴진스"는 \'새로운 진(Jean) 스타일\', 즉 시대와 세대를 초월하는 음악과 패션을 의미합니다. 데뷔곡인 "Attention"과 후속곡 "Hype Boy"는 큰 사랑을 받았으며, 이들의 음악과 비디오는 큰 인기를 끌고 있습니다. 뉴진스는 멤버들의 개성과 다양한 매력을 바탕으로 팬들과의 소통을 중요시하며, 글로벌 사랑을 받고 있는 그룹입니다.',
 'additional_kwargs': {'refusal': None},
 'response_metadata': {'token_usage': {'completion_tokens': 196,
   'prompt_tokens': 19,
   'total_tokens': 215,
   'completion_tokens_details': {'accepted_prediction_tokens': 0,
    'audio_tokens': 0,
    'reasoning_tokens': 0,
    'rejected_prediction_tokens': 0},
   'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
  'model_provider': 'openai',
  'model_name': 'gpt-4o-mini-2024-07-18',
  'system_fingerprint': 'fp_c4585b5b9c',
  'id': 'chatcmpl-D0NDgezVAfy3StdDSzpdGCgJk0G4p',
  'service_tier': 'default',
  'finish_reason': 'stop',
  'logprobs': None},
 'type': 'ai',
 'name': None,
 'id': 'lc_run--019bdf85-849e-7af1-8846-a78f40bc5b05-0',
 'tool_calls': [],
 'invalid_tool_calls': [],
 'usage_metadata': {'input_tokens': 19,
  'output_tokens': 196,
  'total_tokens': 215,
  'input_token_details': {'audio': 0, 'cache_read': 0},
  'output_token_details': {'audio': 0, 'reasoning': 0}}}
```

### 4.2.2 프롬프트의 대안 설정 방법

프롬프트도 이전의 LLM 대안 설정 방법과 유사한 작업을 수행할 수 있습니다.

```python
# 언어 모델을 초기화하고 temperature을 0으로 설정합니다.

llm = ChatOpenAI(temperature=0)

prompt = PromptTemplate.from_template(
    "{country}의 수도는 어디야?" # 기본 프롬프트 템플릿
).configurable_alternatives(
    # 이 필드에 id를 부여합니다.
    ConfigurableField(id="prompt"),

    # 기본 키를 설정합니다.
    default_key = "capital",

    # 'area'이라는 새로운 옵션을 추가합니다.
    area = PromptTemplate.from_template("{country}의 면적은 얼마야?"),

    # 'population'이라는 새로운 옵션을 추가합니다.
    population=PromptTemplate.from_template("{country}의 인구는 얼마야?"),

    # 'eng'이라는 새로운 옵션을 추가합니다.
    eng=PromptTemplate.from_template("{input}을 영어로 번역해 주세요."),

    # 아래에 더 많은 옵션을 추가할 수 있습니다.
)

chain = prompt | llm
```

아무런 설정 변경이 없다면 기본 프롬프트가 입력됩니다.

```python
# config 변경 없이 체인을 호출합니다.
chain.invoke({"country":"대한민국"})
```

```
Output:
AIMessage(content='대한민국의 수도는 서울이야.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 16, 'prompt_tokens': 22, 'total_tokens': 38, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-D0NXDCnbwnuhAbXQLlyazzIDb77Vw', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019bdf97-fbc1-7d21-af67-0e46f3670032-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 22, 'output_tokens': 16, 'total_tokens': 38, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

with_config로 다른 프롬프트를 호출합니다.

```python
# with_config로 체인의 설정을 변경하여 호출합니다.
chain.with_config(configurable={"prompt":"area"}).invoke({"country": "대한민국"})
```

```
Output:
AIMessage(content='대한민국의 총 면적은 약 100,363km² 입니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 25, 'prompt_tokens': 24, 'total_tokens': 49, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-D0NY8Ggi0rYC0u3l4HNolwiG1NpWH', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019bdf98-ddd5-7c02-9d47-0f17777de1d6-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 24, 'output_tokens': 25, 'total_tokens': 49, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

# 5. RunnableWithMessageHistory

RunnablewithMessageHistory는 LCEL 체인에 대화 기록(Chat Message History) 관리 로직을 주입하는 전용 래퍼(Wrapper)입니다.

LLM API(OpenAI, Anthropic 등) 자체는 과거의 대화를 기억하지 못하는 'Stateless' 특성을 가집니다. 이를 해결하기 위해 개발자는 매번 이전 대화 내용을 프롬프트에 포함해서 보내야 하는데, RunnableWithMessageHistory는 이 과정을 자동화하여 **특정 세션 ID에 해당하는 대화 기록을 불러오고, 실행 후 새로운 대화를 다시 저장하는 일**을 대신 처리해 줍니다.

RunnableWithMessageHistory가 중요한 이유는 다음과 같습니다.

1. LLM의 한계 극복(Stateless to Stateful)
    LLM은 개별 요청(Request)을 독립적인 사건으로 취급합니다. RunnableWithMessageHistory는 외부 저장소(Redis, Postgre, in-memory 등)와 체인을 연결하여, 서비스가 사용자별로 고유한 맥락을 유지할 수 있게 만드는 `기억장치` 역할을 수행합니다.

2. 관심사의 분리(Separation of Concerns)
    비즈니스 로직(데이터 가공, 프롬프트 설계)과 상태 관리 로직(데이터 베이스 읽기/쓰기)을 완벽하게 분리합니다. 개발자는 대화 기록을 어떻게 저장하고 불러올지 고민할 필요 없이, 오직 체인의 실행 로직에만 집중할 수 있습니다.

RunnableWithMessageHistory 사용 이유는 다음과 같습니다.

1. 수동 관리의 번거로움 제거
    LCEL 이전에는 개발자가 직접 리스트를 만들고 HumaMessage, AIMessage를 추가하며 관리해야 했습니다. 이 컴포넌트를 사용하면 코드 단 몇 줄로 이 복잡한 흐름을 선언적으로 정의할 수 있습니다.

2. 멀티 세션 지원(Multi-Session Management)
    동일한 체인을 사용하더라도 `session_id`만 다르게 전달하면 각기 다른 사용자의 대화 기록이 섞이지 않고 독립적으로 관리됩니다. 이는 실제 상용 채팅 서비스를 구축할 때 필수적인 기능입니다.

3. 유연한 백엔드 교체
    학습 단계에서는 간단한 `InMemoryChatMessageHistory`를 사용하다가, 실제 서비스 배포 시에는 코드의 핵심 로직을 건드리지 않고 `PostgreChatMessageHistory`나 `RedisChatMessageHistory`로 데이터 베이스만 쉽게 교체할 수 있습니다.

4. 자동화된 프롬프트 주입
    프롬프트 템플릿 내에 대화 기록이 들어갈 자리(PlaceHolder)만 지정해두면, 실행 시점에 해당 세션의 데이터를 자동으로 채워줍니다.

RunnableWithMessageHistory의 실제 활용 예시는 다음과 같습니다.

- 대화형 챗봇 개발: 사용자와의 대화 내역을 기반으로 챗봇의 응답을 조정할 수 있습니다.
- 복잡한 데이터 처리: 데이터 처리 과정에서 이전 단계의 결과를 참조하여 다음 단계의 로직을 결정할 수 있습니다.
- 상태 관리가 필요한 애플리케이션: 사용자의 이전 선택을 기억하고 그에 따라 다음 화면이나 정보를 제공할 수 있습니다.

RunnableWithMessageHistory의 설정 매개변수는 다음과 같습니다. 

- runnable
- BaseChatMessageHistory 이거나 상속받은 객체 ex) ChatMessageHistory
- input_message_key: chain을 invoke() 할때 사용자 쿼리 입력으로 지정하는 key
- history_messages_key: 대화 기록으로 지정하는 key

## 5.1 휘발성 대화기록: 인메모리(In-Memory)

아래는 채팅 기록이 메모리에 저장되는 간단한 예시입니다. `input_messages_key`는 최신 입력 메시지로 처리될 키를 지정하고, `history_messages_key`는 이전 메시지를 추가할 키를 지정합니다.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 {ability}에 능숙한 어시스턴트입니다. 20자 이내로 응답하세요",
        ),
        # 대화 기록을 변수로 사용, history가 MessageHistory의 key가 됨
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"), # 사용자 입력을 변수로 사용
    ]
)

runnable = prompt | model

store = {} # 세션 기록을 저장할 딕셔너리

# 세션 ID를 기반으로 세션 기록을 가져오는 함수

def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    print(session_ids)
    if session_ids not in store: # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 stroe에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]

with_message_history = (
    RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_message_key="input",
        history_messages_key="history",
        
                
    )
)
```

그러면 `with_message_history.invoke()`로 질문을 해보도록 하겠습니다.

```python
with_message_history.invoke(
    # 수학 관련 질문 "코사인의 의미는 무엇인가요?"를 입력으로 전달합니다.
    {"ability": "math", "input": "What does cosine mean?"},

    # 설정 정보로 세션 ID "abc123"을 전달합니다.
    config={"configurable": {"session_id": "abc123"}}
)
```

```
Output:
abc123
AIMessage(content='Cosine represents the ratio of the adjacent side to the hypotenuse in a right triangle.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 47, 'total_tokens': 66, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-D0OuT6jnrTeukYq23Cf3WED0r9HPw', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019bdfe8-a2c6-7282-b6cf-d267932af5ca-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 47, 'output_tokens': 19, 'total_tokens': 66, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

같은 `session_id`를 입력하면 이전 대화 스레드의 내용을 가져오기 때문에 이어서 대화가 가능합니다.

```python
# 메시지 기록을 포함하여 호출합니다.
with_message_history.invoke(
    {"ability": "math", "input": "이전의 내용을 한글로 답변해 주세요."},
    
    config={"configurable": {"session_id": "abc123"}},
)
```

```
Output:
abc123
AIMessage(content='코사인은 직각삼각형에서 인접 변의 길이와 빗변의 길이의 비율을 나타냅니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 47, 'prompt_tokens': 91, 'total_tokens': 138, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-D0PAXaoAkJnYMShRQj7ZL1hvulbtF', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019bdff7-d649-7422-a8b5-cdbc09f73b7a-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 91, 'output_tokens': 47, 'total_tokens': 138, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

하지만 다른 `session_id`를 지정하면 대화기록이 없기 때문에 답변을 제대로 수행하지 못합니다.

```python
# 새로운 session_id로 인해 이전 대화 내용을 기억하지 않습니다.
with_message_history.invoke(
    # 수학 능력과 입력 메시지를 전달합니다.
    {"ability": "math", "input": "이전의 내용을 한글로 답변해 주세요."},

    # 새로운 session_id를 설정합니다.
    config={"configurable": {"session_id": "def234"}}
)
```

```
Output:
def234
AIMessage(content='수학이 잘 됩니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 59, 'total_tokens': 68, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-D0PDzTGtmUBFFLbC7bw7lpqp0cKJs', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019bdffb-1df0-7c12-9f77-2b5bc34d9640-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 59, 'output_tokens': 9, 'total_tokens': 68, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

메시지 기록을 추적하는 데 사용되는 구성 매개변수는 `ConfigurableFieldSpec` 객체의 리스트를 `history_factory_config` 매개변수로 전달하여 사용자 정의할 수 있습니다.

이렇게 history_factory_config를 새로 설정하게 되면 기존 `session_id`설정을 덮어쓰게 됩니다.

아래 예시는 `user_id`와 `conversation_id`라는 두 가지 매개변수를 사용합니다.

```python
from langchain_core.runnables import ConfigurableFieldSpec

store = {}

def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_message_key="input",
    history_messages_key="history",
    history_factory_config=[ # 기존의 "session_id" 설정을 대체하게 됩니다.
        ConfigurableFieldSpec(
            id="user_id", # get_session_history 함수의 첫 번째 인자로 사용됩니다.
            annotation=str,
            name="User ID",
            description="사용자의 고유 식별자입니다.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="대화의 고유 식별자입니다.",
            default="",
            is_shared=True,
        ),
    ],
)
```

```python
with_message_history.invoke(
    {"ability": "math", "input": "Hello"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}},
)
```

```
Output:
AIMessage(content='안녕하세요. 무엇을 도와드릴까요?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 43, 'total_tokens': 64, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-D0PLcCdKBm8xZwyWWZmi6BjkkCHeQ', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019be002-543d-70d3-bbbc-9fce5f98699b-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 43, 'output_tokens': 21, 'total_tokens': 64, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
```

# 6. 사용자 정의 제네레이터(generator)

제네레이터 함수를 LCEL 파이프라인에서 사용할 수 있습니다. 이러한 제네레이터의 시그니처는 `Iterator[Input] -> Iterator[Output]` 이어야 합니다. 비동기 제네레이터의 경우에는 `AsyncIterator[Input] -> AsyncIterator[Output]` 입니다. 이는 다음과 같은 용도로 유용합니다.

- 사용자 정의 출력 파서 구현
- 이전 단계의 출력을 수정하면서 스트리밍 기능 유지

다음 예제는 LLM의 결과를 스트리밍 하면서 LLM의 결과를 generator를 적용하여 중간 중간에 수정하는 코드입니다.

```python
import re
from typing import Iterator
from langchain_core.runnables import RunnableLambda

def regex_token_filter(input_stream: Iterator[str]) -> Iterator[str]:
    # 1. 필터링할 패턴 정의 (단어 경계 \b 활용)
    # 비속어나 기밀 키워드를 정규표현식으로 구성
    bad_words_pattern = re.compile(r'\b(비밀|기밀|private|confidential)\b', re.IGNORECASE)
    
    buffer = "" # 단어 분절을 막기 위한 버퍼
    
    for chunk in input_stream:
        buffer += chunk
        
        # 버퍼에 패턴이 있는지 검사
        # 여기서는 간단하게 설명하기 위해 전체 버퍼를 검사하지만, 
        # 실무에서는 성능을 위해 일정 길이 이상의 버퍼만 검사 후 내보냅니다.
        if bad_words_pattern.search(buffer):
            # 패턴에 매칭되는 부분을 [FILTERED]로 치환
            new_content = bad_words_pattern.sub("[FILTERED]", buffer)
            
            # 필터링된 결과물 중 아직 내보내지 않은 부분만 yield
            # (이 예제는 개념 이해를 돕기 위해 단순화된 로직입니다)
            yield new_content
            buffer = "" # 처리 후 버퍼 비움
        else:
            # 매칭되는 게 없다면 안전한 범위 내에서 데이터 배출
            # 단어 중간일 수 있으므로 마지막 공백 이전까지만 내보내는 등의 전략이 가능합니다.
            if len(buffer) > 10: # 버퍼가 일정 이상 쌓이면 배출
                yield buffer
                buffer = ""
    
    # 마지막에 남은 버퍼 처리
    if buffer:
        yield bad_words_pattern.sub("[FILTERED]", buffer)

#체인에 적용 시
chain = prompt | model | StrOutputParser() | RunnableLambda(regex_token_filter)

print("--- 스트리밍 시작 ---")
for chunk in chain.stream({"topic": "회사의 비밀 보안 정책"}):
    print(chunk, end="", flush=True)
```

출력을 보면 중간 중간 `[FILTERED]`로 값이 바뀐 것을 확인할 수 있습니다.

```
Output:
--- 스트리밍 시작 ---
회사의 [FILTERED] 보안 정책은 기업의 [FILTERED] 정보를 보호하고, 정보 유출을 방지하기 위해 설정된 규칙 및 절차를 포함합니다. 이러한 정책은 회사의 무형 자산을 지키고, 법적 요구사항을 충족시키며, 고객 및 파트너의 신뢰를 유지하는 데 중요한 역할을 합니다. 일반적으로 보안 정책은 다음과 같은 요소를 포함합니다.

1. **정보 분류**: 정보의 중요도와 민감도를 기준으로 [FILTERED], 내부 사용, 공개 정보 등으로 분류합니다. 각 정보 유형에 따라 접근 권한과 보호 수준이 다르게 설정됩니다.

2. **접근 제어**: 정보에 대한 접근을 허가받은 사용자로 제한합니다. 이를 위해 사용자 인증, 권한 관리, 암호 정책 등이 포함됩니다.

3. **정보 취급 절차**: [FILTERED] 정보의 생성, 저장, 전달, 폐기와 관련된 절차를 규정합니다. 예를 들어, 기밀 문서를 안전하게 저장하거나 전송하는 방법이 명시됩니다.

4. **교육 및 인식**: 직원들에게 보안 정책과 그 중요성에 대한 교육을 제공하여, 정보 보안에 대한 인식을 높입니다. 정기적인 보안 교육 프로그램이 포함될 수 있습니다.

5. **모니터링 및 감사**: 정보 보안 정책이 제대로 시행되고 있는지 감시하고, 주기적으로 감사하여 보안 수준을 평가합니다. 이를 통해 보안 취약점을 조기에 발견하고 수정할 수 있습니다.

6. **사고 대응 계획**: 정보 유출이나 보안 사고 발생 시 대응 절차를 마련합니다. 사고 발생 시의 통보, 조사, 복구 및 재발 방지 조치가 포함됩니다.

7. **법적 및 규제 준수**: 관련 법률, 규제, 산업 표준을 준수하며, 필요한 경우 외부 감사나 인증을 받습니다.

회사의 [FILTERED] 보안 정책은 정기적으로 검토되고 업데이트되어야 하며, 최신 보안 위협에 대응할 수 있는 효과적인 체계를 유지해야 합니다. 이를 통해 기업은 자신과 고객의 정보를 안전하게 지킬 수 있습니다.
```

다음은 위의 코드를 비동기식으로 처리한 코드입니다. 실제로 실행을 해보면 한 번에 출력되던 위의 코드와는 달리 스트리밍 되듯 출력되는 것을 확인할 수 있습니다.

```python
import re
import asyncio
from typing import AsyncIterator
from langchain_core.runnables import RunnableLambda

# 1. 비동기 사용자 정의 제네레이터 함수 정의
# AsyncIterator[str]를 입력받아 처리 후 AsyncIterator[str]를 반환합니다.
async def async_regex_token_filter(input_stream: AsyncIterator[str]) -> AsyncIterator[str]:
    # 필터링할 패턴 정의
    bad_words_pattern = re.compile(r'\b(비밀|기밀|private|confidential)\b', re.IGNORECASE)
    buffer = ""
    
    # 비동기적으로 스트림을 하나씩 읽어옵니다.
    async for chunk in input_stream:
        buffer += chunk
        
        # 특정 조건(예: 공백 발견)에서 검사를 수행하거나 
        # 버퍼가 일정 크기 이상일 때 검사하여 성능과 지연시간을 조율합니다.
        if bad_words_pattern.search(buffer):
            processed_content = bad_words_pattern.sub("[FILTERED]", buffer)
            yield processed_content
            buffer = ""
        else:
            # 단어 잘림 방지를 위해 버퍼가 일정량 이상 쌓였을 때만 배출
            if len(buffer) > 15:
                yield buffer
                buffer = ""
                
    # 남아있는 버퍼 처리
    if buffer:
        yield bad_words_pattern.sub("[FILTERED]", buffer)

# 2. 체인 구성 (모델 스트리밍 설정 필요)
model = ChatOpenAI(model="gpt-4o-mini", streaming=True)
chain = prompt | model | StrOutputParser() | async_regex_token_filter

async for chunk in chain.astream({"topic": "회사의 비밀 보안 정책"}):
    print(chunk, end="", flush=True)
```

```
Output:
회사의 [FILTERED] 보안 정책은 기업이 내부 정보와 데이터의 안전을 보호하기 위해 수립한 규정 및 절차를 의미합니다. 이러한 정책은 [FILTERED] 정보의 정의, 접근 권한, 정보 보호 방법, 위반 시 처벌 등을 포함합니다. 일반적으로 [FILTERED] 보안 정책의 주요 요소는 다음과 같습니다.

1. **정보의 분류**: [FILTERED] 정보, 내부 정보, 공개 정보 등으로 데이터를 분류하고, 각 분류별로 보호 수준을 설정합니다.

2. **접근 권한 관리**: 정보를 접근할 수 있는 직원을 제한하며, 역할에 따라 필요한 최소한의 정보만 접근할 수 있도록 합니다.

3. **[FILTERED] 유지 계약**: 직원, 계약자, 협력사 등이 [FILTERED] 정보를 외부에 유출하지 않도록 [FILTERED] 유지 계약(NDA)을 체결하게 합니다.

4. **데이터 암호화**: 중요한 데이터는 저장이나 전송 시 암호화하여 제3자가 접근할 수 없도록 합니다.

5. **모니터링 및 감시**: 정보 접근 및 사용을 모니터링하여 비정상적인 활동을 감지하고, 정기적인 감사를 통해 정책 준수를 확인합니다.

6. **사고 대응 절차**: 정보 유출이나 보안 사고 발생 시 대응 절차를 마련하여 신속히 문제를 해결합니다.

7. **교육 및 훈련**: 직원들에게 [FILTERED] 보안 정책과 관련된 교육을 실시하여 보안 의식을 높이고, 실제 상황에서의 대응 방법을 익히도록 합니다.

8. **정기적인 검토 및 업데이트**: 보안 정책이 최신의 위협에 대응할 수 있도록 정기적으로 검토하고, 필요에 따라 업데이트합니다.

[FILTERED] 보안 정책은 기업의 자산을 보호하고, 신뢰성을 유지하는 데 중요한 역할을 합니다. 이 정책을 효과적으로 시행하려면 모든 직원이 정책을 이해하고 준수하는 것이 필수적입니다.
```

# 마치며

LCEL에 대해서 알아보았습니다. LCEL에 대해 알아보면서 느낀 것은 LCEL에 있는 것들의 대부분이 실무에 사용되는 코드들이기 때문에 막상 공부를 한다고 해서 머릿속에 바로 바로 들어오기 보단 이런 것들이 있구나 하는 것이 대부분이었습니다. 아마도 추후에 좀 더 고차원적인 RAG 시스템을 구축하면서 많이 사용할 것 같고 그때 좀 더 익숙해질 것 같은 느낌이 듭니다. 이번엔 간단히 LCEL이 어떤 것인지, 어떤 것들인지에 대해서 알아보고 추후에 참고할 수 있게 블로그에 정리하는 느낌으로 포스팅을 진행했습니다.

긴 글 읽어주셔서 감사드리며, 본문 내용 중에 잘못된 내용이나 오타, 궁금하신 사항이 있으신 경우 댓글 달아주시기 바랍니다.

# 참조

- 테디노트 - LangChain 한국어 튜토리얼KR(https://wikidocs.net/book/14314)