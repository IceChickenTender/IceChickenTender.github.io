---
title: "[LLM/RAG] LangChain - 2. LangChain의 Prompt와 PromptTemplate"
categories:
  - LLM/RAG
  - LangChain
tags:
  - LLM/RAG
  - LangChain
  - PromptTemplate
  
use_math: true  
toc: true
toc_sticky: true
toc_label: "LangChain의 Prompt와 PromptTemplate에 대해"
---

# 1. 프롬프트(Prompt)

프롬프트는 사용자와 언어 모델 간의 대화에서 질문이나 요청의 형태로 제시되는 입력문입니다. 이는 모델이 어떤 유형의 응답을 제공할지 결정하는데 중요한 역할을 합니다. 프롬프트의 필요성은 다음과 같습니다.

1. **문맥(Context) 설정** : 프롬프트는 언어 모델이 특정 문맥에서 작동하도록 설정하는 역할을 합니다. 이를 통해 모델은 제공된 정보를 바탕으로 보다 정확하고 관련성 높은 답변을 생성할 수 있습니다.

2. **정보 통합** : 여러 문서에서 검색된 정보는 서로 다른 관점이나 내용을 포함할 수 있습니다. 프롬프트 단계에서 정보를 통합하고, 모델이 이를 효율적으로 활용할 수 있는 형식으로 조정합니다.

3. **응답 품질 향상**: 질문에 대한 모델의 응답 품질은 프롬프트의 구성에 크게 의존합니다. 잘 구성된 프롬프트는 모델이 보다 정확하고 유용한 정보를 제공하게 돕습니다.

이번 항목에서는 랭체인에서 프롬프트를 구성할 때 사용하는 대표적인 도구인 프롬프트 템플릿을 다룹니다.

## 1.1 프롬프트 작성 원칙

모델이 최대한 정확하고 유용한 정보를 제공할 수 있도록 효과적인 프롬프트를 작성하는 것이 매우 중요합니다. 좋은 프롬프트를 만들기 위해서 다음과 같은 원칙을 고려합니다.

1. 명확성과 구체성
  - 질문은 명확하고 구체적이어야 합니다. 모호한 질문은 LLM 모델의 혼란을 초래할 수 있기 때문입니다.
  - 예시: "다음 주 주식 시장에 영향을 줄 수 있는 예정된 이벤트들은 무엇일까요" 는 "주식 시장에 대해 알려주세요." 보다 더 구체적이고 명확한 질문입니다. 

2. 배경 정보를 포함
  - 모델이 문맥을 이해할 수 있도록 필요한 배경 정보를 제공하는 것이 좋습니다. 이는 환각(hallucination)이 발생할 위험을 낮추고, 관련성 높은 응답을 생성하는데 도움을 줍니다.
  - 예시: "2020년 미국 대선의 결과를 바탕으로 현재 정치 상황에 대한 분석을 해주세요"

3. 간결함
  - 핵심 정보에 초점을 맞추고, 불필요한 정보는 배제합니다. 프롬프트가 길어지면 모델이 덜 중요한 부분에 집중하거나 모델이 답변을 내뱉을 때에 상당한 영향을 받는 문제가 발생할 수 있습니다.

4. 열린 질문 사용
  - 열린 질문을 통해 모델이 자세하고 풍부한 답변을 제공하도록 유도합니다. 단순한 "예" 또는 "아니오"로 대답할 수 있는 질문 보다는 더 많은 정보를 제공하는 질문이 좋습니다.
  - 예시: "신재생에너지에 대한 최신 연구 동향은 무엇인가요?"

5. 명확한 목표 설정
  - 얻고자 하는 정보나 결과의 유형을 정확하게 정의합니다. 이는 모델이 명확한 지침에 따라 응답을 생성하도록 돕습니다.
  - 예시: "AI 윤리에 대한 문제점과 해결 방안을 요약하여 설명해주세요."

6. 언어와 문체
  - 대화의 맥락에 적합한 언어와 문체를 선택합니다. 이는 모델이 상황에 맞는 표현을 선택하는데 도움이 됩니다.
  - 예시: 공식적인 보고서를 요청하는 경우, "XX 보고서에 대한 전문적인 요약을 부탁드립니다."와 같이 정중한 문체를 사용합니다.

---

## 1.2 프롬프트 템플릿(PromptTemplate)

그럼 이제부터 실습과 함께 LangChain에서 사용되는 PromptTemplate 종류들에 대해서 알아보도록 하겠습니다. 실습 전에 먼저 pip를 이용해 필요한 라이브러리 설치를 먼저 진행해 주시기 바랍니다.

```python
# python을 바로 실행시키는 환경이면 아래 명령어 실행
pip install -qU langchain langchain_google_genai

# Jupyter Notebook 환경이면 아래 명령어 실행
!pip install -qU langchain langchain_google_genai
```

PromptTemplate은 단일 문장 또는 간단한 명령을 입력하여 단일 문장 또는 간단한 응답을 생성하는데 사용되는 프롬프트를 구성할 수 있는 문자열 템플릿이며, 가장 기본적인 템플릿입니다.

### 1.2.1 구성 요소

PromptTemplate 객체를 생성할 때 정의해야 하는 주요 파라미터들은 다음과 같습니다.

1. template(str)
  - 정의: LLM에 보낼 실제 프롬프트의 청사진입니다.
  - 상세: 사용자의 입력값이나 외부 데이터가 들어갈 자리를 변수 처리하여 작성된 문자열입니다.

    ```python
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import PromptTemplate

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    # template 정의
    template = "서울의 수도는 어디인가요?"

    # from_template 메서드를 이용하여 PromptTemplate 객체 생성
    prompt = PromptTemplate.from_template(template)
    print(prompt)
    ```

    ```
    Output:
    input_variables=[] input_types={} partial_variables={} template='서울의 수도는 어디인가요?'
    ```

2. input_variables(List[str])
  - 정의: template 문자열 안에 포함된 변수들의 이름을 정의한 리스트입니다.
  - 역할: 템플릿 포맷팅 시, 이 리스트에 정의된 변수들에 매핑되는 값이 반드시 들어와야 합니다. 일종의 '인터페이스' 역할을 합니다.

    ```python
    # template 정의
    template = "{country}의 수도는 어디인가요?"

    # from_template 메소드를 이용하여 PromptTemplate 객체 생성
    prompt = PromptTemplate.from_template(
        template = template,
    )

    print(f"prompt : {prompt}")

    # prompt 생성
    prompt.format(country="대한민국")
    ```

    ```
    Output:
    prompt : input_variables=['country'] input_types={} partial_variables={} template='{country}의 수도는 어디인가요?'
    대한민국의 수도는 어디인가요?
    ```

3. partial_variables(Dict[str, Any], Optional)
  - 정의: 템플릿의 변수 중 미리 값을 채워두고 싶은 변수들을 정의하는 딕셔너리입니다.
  - 사용 이유:
    - 사용자 입력 시점마다 바뀌는 값이 아니라, 환경 설정(날짜, 사용자 ID, 페르소나 설정 등)처럼 고정되거나 백그라운드에서 자동으로 주입되어야 하는 값들을 처리할 때 씁니다.
    - 이것을 쓰면 클라이언트 코드에서 매번 date 같은 인자를 넘겨주지 않아도 되어 코드가 깔끔해집니다.

    ```python
    # template 정의
    template = "{country1}과 {country2}의 수도는 각각 어디인가요?"

    # PromptTemplate 객체를 활용하여 prompt_template 생성
    prompt = PromptTemplate(
        template = template,
        partial_variables={"country2":"미국"}
    )

    prompt
    print(prompt.format(country1="대한민국"))

    prompt_partial = prompt.partial(country2="캐나다")
    print(prompt_partial)

    print(prompt_partial.format(country1="대한민국"))

    chain = prompt_partial | llm
    print(chain.invoke("대한민국").content)

    print(chain.invoke({"country1":"대한민국", "country2":"호주"}).content)
    ```

    ```
    Output:
    대한민국과 미국의 수도는 각각 어디인가요?
    input_variables=['country1'] input_types={} partial_variables={'country2': '미국', 'count2': '캐나다'} template='{country1}과 {country2}의 수도는 각각 어디인가요?'
    대한민국과 미국의 수도는 각각 어디인가요?
    대한민국의 수도는 **서울**이고, 미국의 수도는 **워싱턴 D.C.**입니다.
    대한민국의 수도는 **서울**이고, 호주의 수도는 **캔버라**입니다.
    ```

4. template_format(str, Default:'f-string')
  - 정의: 템플릿 내의 변수를 어떤 문법으로 파싱할지 결정합니다.
  - 옵션:
    - 'f-string': 파이썬의 f-string처럼 {variable} 형태로 변수를 감쌉니다. 가장 직관적이고 빠릅니다.
    - 'jinja2': Python의 Jinja2 템플릿 엔진 문법을 사용합니다.

5. validate_template(bool, Default:True)
  - 정의: 객체 생성 시, template 문자열 안에 있는 변수와 input_variables에 정의된 변수가 일치하는지 검사할지 여부입니다.
  - 역할: 개발 단계에서의 실수를 방지합니다. 템플릿에는 {name}이 있는데 input_variables에 name이 빠져 있다면 에러를 발생시킵니다.
---

### 1.2.2 문자열 템플릿

다음 예제는 `langchain_core.prompts` 모듈의 `PromptTemplate` 클래스를 사용하여, "name" 과 "age" 라는 두 개의 변수를 포함하는 프롬프트 템플릿을 정의하고 있습니다. 이 템플릿을 이용하여 실제 입력값을 해당 위치에 채워 넣어 완성된 프롬프트를 생성하는 과정을 보여줍니다.

1. `PromptTemplate.from_template` 메서드를 사용하여 문자열 템플릿으로부터 `PromptTemplate` 인스턴스를 생성합니다. 이때, `template_text` 변수에 정의된 템플릿 문자열이 사용됩니다.

2. 생성된 `PromptTemplate` 인스턴스의 `format` 메서드를 사용하여, 실제 "name" 과 "age" 값으로 템플릿에 채워서 프롬프트를 구성합니다. 여기서는 `name="홍길동"`, `age=30`으로 지정하여 호출합니다.

3. 결과적으로, `filled_prompt` 변수에는 "안녕하세요, 제 이름은 홍길동이고, 나이는 30살입니다." 라는 완성된 프롬프트 문자열이 저장됩니다.

```python
from langchain_core.prompts import PromptTemplate

# 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
template_text = "안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다."

# PromptTemplate 인스턴스를 생성
prompt_template = PromptTemplate.from_template(template_text)

# 템플릿에 값을 채워서 프롬프트를 완성
filled_prompt = prompt_template.format(name="홍길동", age=30)

filled_prompt

```

```
실행 결과

안녕하세요, 제 이름은 홍길동이고, 나이는 30살입니다.
```

---

### 1.2.3 프롬프트 템플릿 간의 결합

`PromptTemplate` 클래스는 문자열을 기반으로 프롬프트 템플릿을 생성하고, '+' 연산자를 사용하여 직접 결합할 수 있는 동작을 지원합니다. `PromptTemplate` 인스턴스 간의 직접적인 결합뿐만 아니라, 이들 인스턴스와 문자열로 이루어진 템플릿을 결합하여 새로운 `PromptTemplate` 인스턴스를 생성하는 것도 가능합니다.

- 문자열 + 문자열
- PromptTemplate + PromptTemplate
- PromptTemplate + 문자열

```python
from langchain_core.prompts import PromptTemplate

# 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
template_text = "안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다."

# PromptTemplate 인스턴스를 생성
prompt_template = PromptTemplate.from_template(template_text)

# # 템플릿에 값을 채워서 프롬프트를 완성
# filled_prompt = prompt_template.format(name="홍길동", age=30)

# 문자열 템플릿 결합 (PromptTemplate + PromptTemplate + 문자열)

combined_prompt = (
    prompt_template
    + PromptTemplate.from_template("\n\n아버지를 아버지라 부를 수 없습니다.")   
    +"\n\n{language}로 번역해주세요."
)

combined_prompt
```

```
실행 결과

PromptTemplate(input_variables=['age', 'language', 'name'], input_types={}, partial_variables={}, template='안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다.\n\n아버지를 아버지라 부를 수 없습니다.\n\n{language}로 번역해주세요.')
```

`format` 메소드를 사용하여 앞에서 생성한 템플릿의 매개변수에 입력 값을 지정합니다. LLM 에게 전달한 프롬프트가 완성되는데, 주어진 문장을 "영어로 번역해주세요." 라는 지시사항을 포함하고 있습니다.

```python
from langchain_core.prompts import PromptTemplate

# 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
template_text = "안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다."

# PromptTemplate 인스턴스를 생성
prompt_template = PromptTemplate.from_template(template_text)

# # 템플릿에 값을 채워서 프롬프트를 완성
# filled_prompt = prompt_template.format(name="홍길동", age=30)

# 문자열 템플릿 결합 (PromptTemplate + PromptTemplate + 문자열)

combined_prompt = (
    prompt_template
    + PromptTemplate.from_template("\n\n아버지를 아버지라 부를 수 없습니다.")   
    +"\n\n{language}로 번역해주세요."
)

combined_prompt.format(name="홍길동", age=30, language="영어")

```

```
실행 결과


안녕하세요, 제 이름은 홍길동이고, 나이는 30살입니다.

아버지를 아버지라 부를 수 없습니다.

영어로 번역해주세요.
```

`ChatOpenAI 인스턴스를 생성하여 프롬프트 텍스트를 전달하고, 모델의 출력을 StrOutputParser` 를 통해 문자열로 변환하는 LLM 체인을 구성합니다. `invoke` 메소드를 사용하여 파이프라인을 실행하고, 최종적으로 문자열 출력을 얻습니다. 모델의 응답은 프롬프트에 주어진 문장을 영어로 번역한 텍스트가 출력됩니다.

```python
from langchain_core.prompts import PromptTemplate

# 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
template_text = "안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다."

# PromptTemplate 인스턴스를 생성
prompt_template = PromptTemplate.from_template(template_text)

# # 템플릿에 값을 채워서 프롬프트를 완성
# filled_prompt = prompt_template.format(name="홍길동", age=30)

# 문자열 템플릿 결합 (PromptTemplate + PromptTemplate + 문자열)

combined_prompt = (
    prompt_template
    + PromptTemplate.from_template("\n\n아버지를 아버지라 부를 수 없습니다.")   
    +"\n\n{language}로 번역해주세요."
)

combined_prompt.format(name="홍길동", age=30, language="영어")

llm = ChatOpenAI(model="gpt-4o-mini")
chain = combined_prompt | llm | StrOutputParser()
chain.invoke({"age":30, "language":"영어", "name":"홍길동"})
```

```
실행 결과

Hello, my name is Hong Gil-dong, and I am 30 years old. I cannot call my father "father."
```

---

## 1.3 챗 프롬프트 템플릿(ChatPromptTemplate)

ChatPromptTemplate 은 대화형 상황에서 여러 메시지 입력을 기반으로 단일 메시지 응답을 생성하는데 사용됩니다. 이는 대화형 모델이나 챗봇 개발에 주로 사용됩니다. 입력은 여러 메시지를 원소로 갖는 리스트로 구성되며, 각 메시지는 역할(role)과 내용(content)로 구성됩니다.

---

### 1.3.1 ChatPromptTemplate 구성요소

1. messages(List[BaseMessagePromptTemplate])
  - 정의: 채팅 흐름을 구성하는 메시지 템플릿들의 리스트입니다. 가장 중요한 파라미터입니다 .
  - 구성 요소: 리스트 안에는 다음과 같은 객체들이 순서대로 들어갑니다.

    - SystemMessagePromptTemplate : AI의 페르소나나 지시사항(System Prompt)
    - HumanMessagePromptTemplate : 사용자의 입력이 들어갈 자리
    - AIMessagePromptTemplate : (Few-shot 예시를 들 때) AI의 예시 답변
    - MessagesPlaceholder : 대화 기록(Memory)이나 Agent의 중간 생각(Scratchpad)이 동적으로 삽입될 위치입니다.

2. input_variables(List[str], Optional)
  - 정의: 템플릿 전체(messages 리스트 내의 모든 템플릿 포함)에서 사용자가 입력해야 할 변수명의 리스트입니다.
  - 특징: PromptTemplate과 달리, ChatPromptTemplate은 messages 안의 변수들을 자동으로 스캔해서 추론해 줍니다. 따라서 명시적으로 적지 않아도 되는 경우가 많습니다.

3. partial_variables(Dict[str, Any], Optional)
  - 정의: PromptTemplate과 동일하게, 미리 고정시켜 둘 변수값입니다.
  - 활용:
    - 함수(Function) 바인딩: 현재 시간을 가져오는 함수나, 문서 검색 도구를 미리 바인딩할 때 유용합니다. 
    - 포맷팅 지시: 출력 포맷(JSON 등)에 대한 지시사항이 너무 길 때, 이를 별도 변수로 빼서 partial_variables로 주입하면 메인 프롬프트가 깔끔해집니다.

4. template_format(str, Default:'f-string')
  - 정의: 메시지 내부의 텍스트를 파싱하는 문법입니다.

---

### 1.3.2 2-튜플 형태의 메시지 리스트

`ChatPromptTemplate.from_messages` 메서드를 사용하여 메시지 리스트로부터 `ChatPromptTemplate` 인스턴스를 생성하는 방식은 대화형 프롬프트를 생성하는데 유용합니다. 이 메서드는 2-튜플 형태의 메시지 리스트를 입력 받아, 각 메시지의 역할(role)과 내용(content) 을 기반으로 프롬프트를 구성합니다.

다음 예시에서 `ChatPromptTemplate.from_messages` 메서드는 전달된 메시지들을 기반으로 프롬프트를 구성합니다. 그리고 `format_messages` 메서드는 사용자의 입력을 프롬프트에 동적으로 삽입하여, 최종적으로 대화형 상황을 반영한 메시지 리스트를 생성합니다. 시스템은 자신의 기능을 설명하고 사용자는 천문학 관련 질문을 합니다.

```python
# 2-튜플 형태의 메시지 목록으로 프롬프트 생성 (role, content)

from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
    ("user", "{user_input}"),
    
])

messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")

messages
```

```
실행 결과

[SystemMessage(content='이 시스템은 천문학 질문에 답변할 수 있습니다.', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='태양계에서 가장 큰 행성은 무엇인가요?', additional_kwargs={}, response_metadata={})]
```

chat_prompt, llm, StrOutputParser() 를 순차적인 파이프라인으로 연결하여 구성된 chain 을 사용합니다. invoke 메소드를 호출하면 사용자 입력을 받아 언어 모델에 전달하고, 모델의 응답을 처리하여 최종 문자열 결과를 반환하는 과정을 자동화하여 수행합니다. 사용자는 천문학 관련 질문에 대한 언어 모델의 응답을 얻을 수 있습니다.

```python
from langchain_core.output_parsers import StrOutputParser

chain = chat_prompt | llm | StrOutputParser()

chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})

```

---

### 1.3.3 MessagePromptTemplate 활용

다음 예제는 `SystemMessagePromptTemplate` 와 `HumanMessagePromptTemplate` 를 사용하여 천문학 질문에 답변할 수 있는 시스템에 대한 대화형 프롬프트를 생성합니다. `ChatPromptTemplate.from_messages` 메소드를 통해 시스템 메시지와 사용자 메시지 템플릿을 포함하는 챗 프롬프트를 구성합니다. 이후, `chat_prompt.format_messages` 메서드를 사용하여 사용자의 질문을 포함한 메시지 리스트를 동적으로 생성합니다.

```python
# MessagePromptTemplate 활용

from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다."),
        HumanMessagePromptTemplate.from_template("{user_input}"),
    ]
)

messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")

messages
```

```
실행 결과

[SystemMessage(content='이 시스템은 천문학 질문에 답변할 수 있습니다.', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='태양계에서 가장 큰 행성은 무엇인가요?', additional_kwargs={}, response_metadata={})]
```

이렇게 생성된 메시지 리스트는 대화형 인터페이스나 언어 모델과의 상호작용을 위한 입력으로 사용될 수 있습니다. 각 메시지는 role(메시지를 말하는 주체, 여기서는 system 또는 user)과 content(메시지 내용)속성을 포함합니다. 이 구조는 시스템과 사용자 간의 대화 흐름을 명확하게 표현하며, 언어 모델이 이를 기반으로 적절한 응답을 생성할 수 있도록 돕습니다.

---

## 1.4 Few-Shot Prompt

Few-shot 학습은 사전학습된 LLM 에 새로운 작업을 시킬 때, 모델을 다시 훈련시키지 않고 몇 개의 예시(질문과 그에 대한 정답)를 프롬프트에 함께 제공하여, 모델이 그 패턴을 보고 동일한 방식으로 답변을 생성하도록 유도하는 기법입니다. 이 방법은 특히 모델이 생소한 작업이나 특정 형식을 따르는 작업을 할 때 성능을 향상시킬 수 있지만, 항상 큰 성능 향상을 보장하는 것은 아니며, 예시의 품질과 상황에 따라 효과가 달라집니다.

Few-shot 학습을 활용함으로써, 언어 모델은 주어진 예제들을 참고하여 더 정확하고 일관된 응답을 생성할 수 있습니다. 이는 특히 특정 도메인이나 형식의 질문에 대해 모델의 성능을 향상시키는데 효과적입니다.

### 1.4.1 Few-Shot 예제 사용하기

#### 1.4.1.1 Few-shot 예제 포맷터 생성

먼저 Few-shot 예제를 포맷팅하기 위한 템플릿을 생성합니다. `PromptTemplate` 은 질문과 답변을 포함하는 간단한 구조를 가지고 있습니다. 이 템플릿은 각 예제를 일관된 형식으로 표현할 수 있게 해주어, 모델이 입력과 출력의 패턴을 쉽게 인식할 수 있도록 합니다.

```python
from langchain_core.prompts import PromptTemplate

example_prompt = PromptTemplate.from_template("질문: {question}\n{answer}")
```

---

#### 1.4.1.2 예제 시트 생성

다음은 모델이 참조할 수 있는 질문과 다변의 예제 세트를 생성합니다. 이 예제 세트는 다양한 주제(지구과학, 생물학, 수학)를 포함하고 있어, 모델이 여러 분야의 질문에 대응할 수 있도록 준비시킵니다. 각 답변은 간결하고 직접적이어서 모델이 유사한 스타일로 답변하도록 유도합니다.

```
examples = [
    {
        "question": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
        "answer": "지구 대기의 약 78%를 차지하는 질소입니다."
    },
    {
        "question": "광합성에 필요한 주요 요소들은 무엇인가요?",
        "answer": "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다."
    },
    {
        "question": "피타고라스 정리를 설명해주세요.",
        "answer": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것입니다."
    },
    {
        "question": "지구의 자전 주기는 얼마인가요?",
        "answer": "지구의 자전 주기는 약 24시간(정확히는 23시간 56분 4초)입니다."
    },
    {
        "question": "DNA의 기본 구조를 간단히 설명해주세요.",
        "answer": "DNA는 두 개의 폴리뉴클레오티드 사슬이 이중 나선 구조를 이루고 있습니다."
    },
    {
        "question": "원주율(π)의 정의는 무엇인가요?",
        "answer": "원주율(π)은 원의 지름에 대한 원의 둘레의 비율입니다."
    }
]

```

---

#### 1.4.1.3 FewShotPromptTemplate 생성

다음 코드는 Few-shot 프롬프트 템플릿을 생성하고, 새로운 질문에 대한 프롬프트를 생성합니다. `FewShotPromptTemplate` 은 예제들을 결합하고 새로운 입력을 추가하여 최종 프롬프트를 생성합니다. 이 방식은 모델에게 관련 예제들을 제공하면서 새로운 질문을 처리하도록 지시합니다.

```python
# FewShotPromptTemplate 예제

from langchain_core.prompts import FewShotPromptTemplate

from langchain_core.prompts import PromptTemplate

example_prompt = PromptTemplate.from_template("질문: {question}\n{answer}")

examples = [
    {
        "question": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
        "answer": "지구 대기의 약 78%를 차지하는 질소입니다."
    },
    {
        "question": "광합성에 필요한 주요 요소들은 무엇인가요?",
        "answer": "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다."
    },
    {
        "question": "피타고라스 정리를 설명해주세요.",
        "answer": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것입니다."
    },
    {
        "question": "지구의 자전 주기는 얼마인가요?",
        "answer": "지구의 자전 주기는 약 24시간(정확히는 23시간 56분 4초)입니다."
    },
    {
        "question": "DNA의 기본 구조를 간단히 설명해주세요.",
        "answer": "DNA는 두 개의 폴리뉴클레오티드 사슬이 이중 나선 구조를 이루고 있습니다."
    },
    {
        "question": "원주율(π)의 정의는 무엇인가요?",
        "answer": "원주율(π)은 원의 지름에 대한 원의 둘레의 비율입니다."
    }
]

# FewShotPromptTemplate 를 생성합니다.
prompt = FewShotPromptTemplate(
    examples = examples,              # 사용할 예제들
    example_prompt = example_prompt,  # 예제 포맷팅에 사용할 템플릿
    suffix="질문: {input}",           # 예제 뒤에 추가될 접미사
    input_variables=["input"],         # 입력 변수 지정
)

# 새로운 질문에 대한 프롬프트를 생성하고 출력합니다.

print(prompt.invoke({"input":"화성의 표면이 붉은 이유는 무엇인가요?"}))
```

```
실행 결과

질문: 지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?
지구 대기의 약 78%를 차지하는 질소입니다.

질문: 광합성에 필요한 주요 요소들은 무엇인가요?
광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다.

질문: 피타고라스 정리를 설명해주세요.
피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것입니다.

질문: 지구의 자전 주기는 얼마인가요?
지구의 자전 주기는 약 24시간(정확히는 23시간 56분 4초)입니다.

질문: DNA의 기본 구조를 간단히 설명해주세요.
DNA는 두 개의 폴리뉴클레오티드 사슬이 이중 나선 구조를 이루고 있습니다.

질문: 원주율(π)의 정의는 무엇인가요?
원주율(π)은 원의 지름에 대한 원의 둘레의 비율입니다.

질문: 화성의 표면이 붉은 이유는 무엇인가요?

```

---

#### 1.4.1.4 예제 선택기 사용하기

다음 코드는 의미적 유사성을 기반으로 가장 관련성 높은 예제를 선택합니다. `SemanticSimilarityExampleSelector`는 입력 질문과 가장 유사한 예제를 선택합니다. 이 방법은 더 관련성 높은 컨텍스트를 제공하여 모델의 응답 품질을 향상시킬 수 있습니다.

우선 예제를 실행하기 전에 `langchain_chroma` 설치 부터 진행을 해줍니다.

```python
!pip install langchain-chroma
```

설치가 완료된 후 아래 코드를 실행해봅니다.

```python
# 예제 선택기 사용

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# SemanticSimilarityExampleSelector 를 초기화합니다.

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=1
)

# 새로운 질문에 대해 가장 유사한 예제를 선택합니다.

question = "화성의 표면이 붉은 이유는 무엇인가요?"
selected_examples = example_selector.select_examples({"question":question})
print(f"입력과 가장 유사한 예제:{question}")

for example in selected_examples:
  print("\n")
  for k, v in example.items():
    print(f"{k}: {v}")
```

위 코드는 다음과 같이 Few-shot 학습에 사용하기 위한 예제들 중에서 입력 질문과 가장 유사한 예제를 선택하여 출력합니다. 이를 이용해 모델은 입력과 가장 관련성 높은 예제를 참조하여 응답을 생성할 수 있습니다.

```
실행 결과

입력과 가장 유사한 예제:화성의 표면이 붉은 이유는 무엇인가요?


answer: 지구 대기의 약 78%를 차지하는 질소입니다.
question: 지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?
```

---

### 1.4.2 채팅 모델에서 Few-Shot 예제 사용하기

#### 1.4.2.1 고정 예제 사용하기

가장 기본적인 Few-shot 프롬프팅 기법은 고정된 예제를 사용하는 것입니다.

```python
# 기본적인 Few-shot 프롬프팅 사용법

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# 예제 정의

examples = [
    {"input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "output": "질소입니다."},
    {"input": "광합성에 필요한 주요 요소들은 무엇인가요?", "output": "빛, 이산화탄소, 물입니다."},
]

# 예제 프롬프트 템플릿 정의
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# Few-shot 프롬프트 템플릿 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 최종 프롬프트 템플릿 생성
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 과학과 수학에 대해 잘 아는 교육자입니다."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# 모델과 체인 생성
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
chain = final_prompt | model

# 모델에 질문하기
result = chain.invoke({"input": "지구의 자전 주기는 얼마인가요?"})
print(result.content)
```

실행 결과는 다음과 같습니다. 초반에 단순히 gpt-4o-mini 에게 답변을 물을 때와 달리 우리가 적용한 예제와 같이 간결하고 필요한 정보만 답변 해주는 것을 확인할 수 있습니다.

```
실행 결과

Few-shot 적용 전 대답

지구의 자전 주기는 약 24시간입니다. 더 정확하게는 평균적으로 23시간 56분 4초로, 이를 '항성일'이라고 합니다. 그러나 우리가 일상적으로 사용하는 24시간은 태양일로, 태양이 같은 위치에 다시 나타나기까지의 시간을 기준으로 합니다. 태양일은 항성일보다 약 4분 정도 더 긴 이유는 지구가 태양 주위를 공전하고 있기 때문입니다.

Few-shot 적용 후 대답

지구의 자전 주기는 약 24시간, 즉 1일입니다. 정확히는 약 23시간 56분 4초입니다. 이 시간을 기준으로 하루가 구성됩니다.

```

---

#### 1.4.2.2 동적 Few-shot 프롬프팅

이 방법은 예제 선택기(ExampleSelector) 를 사용해서 입력에 따라 전체 예제 세트에서 가장 관련성 높은 예제만 선택하여 보여주는 방법입니다.

```python
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# 더 많은 예제 추가
examples = [
    {"input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "output": "질소입니다."},
    {"input": "광합성에 필요한 주요 요소들은 무엇인가요?", "output": "빛, 이산화탄소, 물입니다."},
    {"input": "피타고라스 정리를 설명해주세요.", "output": "직각삼각형에서 빗변의 제곱은 다른 두 변의 제곱의 합과 같습니다."},
    {"input": "DNA의 기본 구조를 간단히 설명해주세요.", "output": "DNA는 이중 나선 구조를 가진 핵산입니다."},
    {"input": "원주율(π)의 정의는 무엇인가요?", "output": "원의 둘레와 지름의 비율입니다."},
]

# 벡터 저장소 생성
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

# 예제 선택기 생성
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

# Few-shot 프롬프트 템플릿 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

# 최종 프롬프트 템플릿 생성
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 과학과 수학에 대해 잘 아는 교육자입니다."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# 모델과 체인 생성
chain = final_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# ===== 여기부터: 입력과 유사한 예시 출력 =====
query = "태양계에서 가장 큰 행성은 무엇인가요?"

# 1) 선택된 예시를 직접 확인
selected = example_selector.select_examples({"input": query})
print("[선택된 예시들]")
for i, ex in enumerate(selected, 1):
    print(f"{i}. Q: {ex['input']}\n   A: {ex['output']}\n")

# 2) 모델 호출
result = chain.invoke({"input": query})
print("[모델 답변]")
print(result.content)

```

wiki docs 에 있는 코드에 선택된 예시를 출력하는 코드를 추가하고 실행 결과를 출력해보았습니다. 모델 답변은 문제가 없지만 선택된 예시들을 보면 예제에서 중복으로 2개가 뽑히는 것을 확인할 수 있습니다.

```
[선택된 예시들]
1. Q: 지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?
   A: 질소입니다.

2. Q: 지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?
   A: 질소입니다.

[모델 답변]
태양계에서 가장 큰 행성은 목성(Jupiter)입니다. 목성은 지름이 약 142,984킬로미터로, 태양계의 다른 행성들보다 훨씬 큽니다.
```

아래는 예제들 중에서 중복을 제거하는 코드를 추가한 코드입니다.

```python
# 동적 Few-shot 프롬프팅

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
import uuid # uuid 모듈 임포트

# 더 많은 예제 추가
examples = [
    {"input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "output": "질소입니다."},
    {"input": "광합성에 필요한 주요 요소들은 무엇인가요?", "output": "빛, 이산화탄소, 물입니다."},
    {"input": "피타고라스 정리를 설명해주세요.", "output": "직각삼각형에서 빗변의 제곱은 다른 두 변의 제곱의 합과 같습니다."},
    {"input": "DNA의 기본 구조를 간단히 설명해주세요.", "output": "DNA는 이중 나선 구조를 가진 핵산입니다."},
    {"input": "원주율(π)의 정의는 무엇인가요?", "output": "원의 둘레와 지름의 비율입니다."},
]

# 벡터 저장소 생성

to_vectorize = [" ".join(example.values()) for example in examples]
ids = [f"ex-{i}" for i in range(len(examples))]  # 고정 id

embeddings = OpenAIEmbeddings()

# 매번 새 컬렉션 이름으로 만들면 중복 누적 방지
collection_name = f"fewshot-{uuid.uuid4().hex[:8]}"

vectorstore = Chroma.from_texts(
    texts=to_vectorize,
    embedding=embeddings,
    metadatas=examples,
    ids=ids,
    collection_name=collection_name,
    persist_directory=None,  # 메모리만 사용(중복 누적 방지)
)

# 예제 선택기 생성
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,  # 최종 예시 개수
)

# Few-shot 프롬프트 템플릿 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]

    ),
)

# 최종 프롬프트 템플릿 생성
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 과학과 수학에 대해 잘 아는 교육자입니다."),
        few_shot_prompt,
        ("human", "{input}"),
    ]

)

# 모델과 체인 생성
chain = final_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# ===== 여기부터: 입력과 유사한 예시 출력 =====
query = "태양계에서 가장 큰 행성은 무엇인가요?"

# 1) 선택된 예시를 직접 확인
selected = example_selector.select_examples({"input": query})
print("[선택된 예시들]")
for i, ex in enumerate(selected, 1):
    print(f"{i}. Q: {ex['input']}\n   A: {ex['output']}\n")

# 2) 모델 호출
result = chain.invoke({"input": query})
print("[모델 답변]")
print(result.content)
```

실행 결과는 아래와 같습니다. 모델이 선택된 예제와 같이 아주 간결하게 답변을 해주는 것을 확인할 수 있습니다.

```
실행 결과

[선택된 예시들]
1. Q: 지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?
   A: 질소입니다.

2. Q: 광합성에 필요한 주요 요소들은 무엇인가요?
   A: 빛, 이산화탄소, 물입니다.

[모델 답변]
태양계에서 가장 큰 행성은 목성(Jupiter)입니다.
```

전반적으로, Few-shot 학습 기법은 모델이 특정 답변 스타일을 학습하고 다양한 주제에 대해 일관된 형식의 응답을 생성하는 데 효과적임을 알 수 있습니다. 동적 예제 선택은 더 관련성 높은 컨텍스트를 제공하여 모델의 성능을 향상시킬 수 있지만, 예제 세트의 다양성과 임베딩 모델의 성능이 중요한 요소로 작용합니다. 특히나 마지막 코드에서 Few-shot 을 위한 예제에서 중복 추출이 되지 않도록 했을 때 확실히 Few-shot 이 적용되어 모델의 답변이 달라지는 것을 확인할 수 있었습니다.

## 1.5 LangChain Hub

LangChain Hub는 프롬프트(prompt), 체인(Chain), 에이전트(Agent)와 같은 LangChain의 핵심 구성 요소들을 탐색 공유, 그리고 버전관리 할 수 있는 중앙 집중형 저장소 플랫폼입니다.

### 1.5.1 Hub로부터 Prompt 받아오기

```python
from langchain import hub

# 가장 최신 버전의 프롬프트를 가져옵니다.
prompt = hub.pull("rlm/rag-prompt")

print(prompt)
```

```
Output:
input_variables=['context', 'question'] input_types={} partial_variables={} metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'} messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"), additional_kwargs={})]
```

# 마치며

LangChain에서 가장 중요하다고 볼 수 있는 prompt와 prompt를 확장성 있게 활용하기 위한 PromptTemplate의 여러 종류들에 대해서 알아보았습니다. 이번 포스트를 준비하면서 느낀 점은 LLM의 prompt engineering은 방대하고 명확하지 않고, PromptTemplate도 확장성이 너무 뛰어나 어떻게 해야 잘 사용할지 감이 잘 안잡히는 느낌을 받았습니다. 즉, 직접 많이 사용해 봐야지 어떻게 사용하는지 알 것 같습니다. 이번 포스트는 이론적인 내용만 정리하는 포스트이고, 차후에 LangChain을 이용한 토이 프로젝트를 직접 진행해 보면서 직접 만지다 보면 좀 더 감을 잡을 것 같습니다.

긴 글 읽어주셔서 감사드리며, 본문 내용 중에 잘못된 내용이나 오타, 궁금하신 사항이 있으시면 댓글 달아주시기 바랍니다.

# 참조

- 판다스 스튜디오 - 랭체인(LangChain) 입문부터 응용까지(https://wikidocs.net/book/14473)
- 테디노트 - LangChain 한국어 튜토리얼KR(https://wikidocs.net/book/14314)
