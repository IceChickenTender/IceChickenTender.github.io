---
title: "[LLM/RAG] LangChain - 4. LangChain의 Parser"
categories:
  - LangChain
  
tags:
  - LangChain
  
use_math: true  
toc: true
toc_sticky: true
toc_label: "LangChain의 Parser"
---

# 1. 출력 파서 개요

랭체인에서 출력 파서(Output Parser)는 모델의 출력을 처리하고, 그 결과를 원하는 형식으로 변환하는 역할을 합니다. 출력 파서는 모델에서 반환된 원시 텍스트를 분석하고, 특정 정보를 추출하거나, 출력을 특정 형식으로 재구성하는데 사용됩니다.

이처럼 모델과 애플리케이션 간의 인터페이스 역할을 하며, 모델의 출력을 더 가치 있고 사용하기 쉬운 형태로 변환하는 데 핵심적인 역할을 합니다. 이를 통해 개발자는 모델의 원시 출력을 직접 처리하는 복잡성을 줄이고, 애플리케이션의 특정 요구 사항에 맞게 출력을 빠르게 조정할 수 있습니다.

출력 파서의 주요 기능

- 출력 포맷 변경 : 모델의 출력을 사용자가 원하는 형식으로 변환합니다. 예를 들어, JSON 형식으로 반환된 데이터를 테이블 형식으로 변환할 수 있습니다.
- 정보 추출 : 원시 텍스트 출력에서 필요한 정보(예:날짜, 이름, 위치 등)를 추출합니다. 이를 통해 복잡한 텍스트 데이터에서 구조화된 정보를 얻을 수 있습니다.
- 결과 정제: 모델 출력에서 불필요한 정보를 제거하거나, 응답을 더 명확하게 만드는 등의 후처리 작업을 수행합니다.
- 조건부 로직 적용 : 출력 데이터를 기반으로 특정 조건에 따라 다른 처리를 수행합니다. 예를 들어, 모델의 응답에 따라 사용자에게 추가 질문을 하거나, 다른 모델을 호출할 수 있습니다.

출력 파서의 사용 사례
1. 자연어 처리(NLP) 애플리케이션 : 질문 답변 시스템에서 정확한 답변만을 추출하여 사용자에게 제공합니다.
2. 데이터 분석 : 대량의 텍스트 데이터에서 특정 패턴이나 통계 정보를 추출하여 분석 보고서를 생성합니다.
3. 챗봇 개발 : 대화형 모델의 출력을 분석하여 사용자의 의도를 파악하고, 적절한 대화 흐름을 유지합니다.
4. 콘텐츠 생성 : 생성된 콘텐츠에서 중요한 정보를 요약하거나, 특정 형식(예 블로그 포스트, 뉴스 기사)에 맞게 콘텐츠를 재구성합니다.

# 2. PydanticOutputParser

PydanticOutputParser는 사용자가 정의한 Pydantic 스키마를 기반으로 LLM의 출력을 파싱하고 검증(Validation)하는 핵심 컴포넌트입니다. LLM에게 원하는 JSON 포맷 지시사항을 자동으로 주입하고, 결과물을 단순 텍스트나 딕셔너리가 아닌 Python 객체로 즉시 변환합니다. 이를 통해 타입 안전성을 확보하고 후속 로직을 견고하게 구현할 수 있습니다.

PydanticOutputParser에는 주로 두 가지 핵심 메서드가 구현되어야 합니다.

- get_format_instructions(): 언어 모델이 출력해야 할 정보의 형식을 정의하는 지침(instruction)을 제공합니다. 예를 들면, 언어 모델이 출력해야 할 데이터의 필드와 그 형태를 설명하는 지침을 문자열로 반환할 수 있습니다. 이때 설정하는 지침(instruction)의 역할이 매우 중요합니다. 이 지침에 따라 언어 모델은 출력을 구조화하고, 이를 특정 데이터 모델에 맞게 변환할 수 있습니다.

- parse(): 언어 모델의 출력(문자열로 가정)을 받아들여 이를  특정 구조로 분석하고 변환합니다. Pydantic와 같은 도구를 사용하여, 입력된 문자열로 사전 정의된 스키마에 따라 검증하고, 해당 스키마를 따르는 데이터 구조로 변환합니다.

그럼 이제 예제를 통해 PydanticOutputParser에 대해서 자세히 알아보도록 하겠습니다. 다음 라이브러리 설치를 먼저 진행해 주세요

```bash
pip install langchain langchain-core langchain-community langchain-teddynote langchain-google-genai google-generativeai pydantic
```

추론에 사용할 LLM 선언을 먼저 진행해 줍니다.

```python
from langchain_teddynote.messages import stream_response
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
```

그리고 특정 데이터 유형에서 데이터 추출을 할건데 이번엔 회사 이메일 내용에서 필요한 정보들을 뽑는다고 가정하고 진행해 보도록 하겠습니다.

```python
email_conversation = """From: 김철수 (chulsoo.kim@bikecorporation.me)
To: 이은채 (eunchae@teddyinternational.me)
Subject: "ZENESIS" 자전거 유통 협력 및 미팅 일정 제안

안녕하세요, 이은채 대리님,

저는 바이크코퍼레이션의 김철수 상무입니다. 최근 보도자료를 통해 귀사의 신규 자전거 "ZENESIS"에 대해 알게 되었습니다. 바이크코퍼레이션은 자전거 제조 및 유통 분야에서 혁신과 품질을 선도하는 기업으로, 이 분야에서의 장기적인 경험과 전문성을 가지고 있습니다.

ZENESIS 모델에 대한 상세한 브로슈어를 요청드립니다. 특히 기술 사양, 배터리 성능, 그리고 디자인 측면에 대한 정보가 필요합니다. 이를 통해 저희가 제안할 유통 전략과 마케팅 계획을 보다 구체화할 수 있을 것입니다.

또한, 협력 가능성을 더 깊이 논의하기 위해 다음 주 화요일(1월 15일) 오전 10시에 미팅을 제안합니다. 귀사 사무실에서 만나 이야기를 나눌 수 있을까요?

감사합니다.

김철수
상무이사
바이크코퍼레이션
"""
```

다음은 출력 파서(parser)를 사용하지 않는 경우 예시입니다.

```python
# 출력 파서를 사용하지 않는 예시
from itertools import chain
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "다음의 이메일 내용중 중요한 내용을 추출해 주세요.\n\n{email_conversation}"
)

chain = prompt | llm

answer = chain.stream({"email_conversation":email_conversation})

output = stream_response(answer, return_output=True)
print(output)
```

LLM이 중요한 어느 정도 중요한 내용을 추려줬지만 LLM이 판단한 것과 사람이 필요로 하는 중요한 정보는 다를 수 있습니다.

```
Output:
다음은 이 이메일의 중요한 내용을 추출한 것입니다.

*   **발신자:** 바이크코퍼레이션 김철수 상무
*   **수신자:** 테디인터내셔널 이은채 대리
*   **핵심 목적:** 테디인터내셔널의 신규 자전거 "**ZENESIS**" 모델에 대한 **유통 협력**을 제안함.
*   **요청 사항:**
    *   "ZENESIS" 모델의 **상세 브로슈어** (특히 기술 사양, 배터리 성능, 디자인 정보 포함) 요청.
*   **제안 사항:**
    *   협력 가능성 논의를 위한 **미팅 제안**.
    *   **일시:** 1월 15일(화) 오전 10시
    *   **장소:** 테디인터내셔널 사무실다음은 이 이메일의 중요한 내용을 추출한 것입니다.

*   **발신자:** 바이크코퍼레이션 김철수 상무
*   **수신자:** 테디인터내셔널 이은채 대리
*   **핵심 목적:** 테디인터내셔널의 신규 자전거 "**ZENESIS**" 모델에 대한 **유통 협력**을 제안함.
*   **요청 사항:**
    *   "ZENESIS" 모델의 **상세 브로슈어** (특히 기술 사양, 배터리 성능, 디자인 정보 포함) 요청.
*   **제안 사항:**
    *   협력 가능성 논의를 위한 **미팅 제안**.
    *   **일시:** 1월 15일(화) 오전 10시
    *   **장소:** 테디인터내셔널 사무실
```

그럼 이제 Pydantic 스타일로 정의된 클래스를 사용하여 이메일의 정보를 파싱해 보겠습니다. Field 안의 `description`은 텍스트 형태의 답변에서 주요 정보를 추출하기 위한 설명입니다. LLM이 바로 이 설명을 보고 필요한 정보를 추출하게 됩니다. 그러므로 이 설명은 정확하고 명확해야 합니다.

```python
class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")

# PydanticOutputParser 생성
parser = PydanticOutputParser(pydantic_object=EmailSummary)

print(parser.get_format_instructions())
```

우리가 정의한 EmailSummary 클래스를 PydanticOutputParser에 pydantic_object에 넣어주고 get_format_instructions()를 출력해보면 아래와 같이 JSON 형태로 정리가 되어 있는 것을 확인할 수 있습니다.

```
Output:
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"person": {"description": "메일을 보낸 사람", "title": "Person", "type": "string"}, "email": {"description": "메일을 보낸 사람의 이메일 주소", "title": "Email", "type": "string"}, "subject": {"description": "메일 제목", "title": "Subject", "type": "string"}, "summary": {"description": "메일 본문을 요약한 텍스트", "title": "Summary", "type": "string"}, "date": {"description": "메일 본문에 언급된 미팅 날짜와 시간", "title": "Date", "type": "string"}}, "required": ["person", "email", "subject", "summary", "date"]}
```
```

그럼 이제 이메일에서 정보를 추출하기 위한 prompt를 만들어 보고 만들어진 prompt의 format 부분에 이전에 만들어 뒀던 PydanticOutputParser의 get_format_instructions() 메서드를 넣어주도록 하겠습니다.

```python
prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. Please answer the following questions in KOREAN.

QUESTION:
{question}

EMAIL CONVERSATION:
{email_conversation}

FORMAT:
{format}
"""
)

# format 에 PydanticOutputParser의 부분 포맷팅(partial) 추가
prompt = prompt.partial(format=parser.get_format_instructions())
```

chain을 생성해 답변 결과를 출력해 보도록 하겠습니다.

```python
# chain 생성
chain = prompt | llm

# chain을 실행하고 결과를 확인
response = chain.stream(
    {
        "email_conversation": email_conversation,
        "question": "이메일 내용중 주요 내용을 추출해 주세요.",
    }
)

# 결과는 JSON 형태로 출력됩니다.
output = stream_response(response, return_output=True)
```

이메일 내용 중 우리가 정의했던 필요한 내용들만 골라서 JSON 형태로 바꿔준 것을 확인할 수 있습니다.

````
Output:
```json
{
  "person": "김철수",
  "email": "chulsoo.kim@bikecorporation.me",
  "subject": "\"ZENESIS\" 자전거 유통 협력 및 미팅 일정 제안",
  "summary": "바이크코퍼레이션의 김철수 상무가 이은채 대리에게 신규 자전거 \"ZENESIS\"의 유통 협력을 제안하며, ZENESIS 모델의 상세 브로슈어(기술 사양, 배터리 성능, 디자인)를 요청했습니다. 또한, 협력 논의를 위해 1월 15일 화요일 오전 10시에 귀사 사무실에서 미팅을 제안했습니다.",
  "date": "1월 15일 화요일 오전 10시"
}
```
````

마지막으로 parser를 사용하여 결과를 파싱하고 EmailSummary 객체 형태로 출력되도록 합니다.

```python
# PydnaticOutputParser를 사용하여 결과를 파싱합니다.
structured_output = parser.parse(output)
print(structured_output)
```

```
Output:
person='김철수' email='chulsoo.kim@bikecorporation.me' subject='"ZENESIS" 자전거 유통 협력 및 미팅 일정 제안' summary='바이크코퍼레이션의 김철수 상무가 이은채 대리에게 신규 자전거 "ZENESIS"의 유통 협력을 제안하며, ZENESIS 모델의 상세 브로슈어(기술 사양, 배터리 성능, 디자인)를 요청했습니다. 또한, 협력 논의를 위해 1월 15일 화요일 오전 10시에 귀사 사무실에서 미팅을 제안했습니다.' date='1월 15일 화요일 오전 10시'
```

이제 chain에 parser를 함께 묶어서 요청을 보내보고 받은 결과를 출력해 보도록 하겠습니다.

```python
# parser가 추가된 체인 생성
chain = prompt | llm | parser

# chain을 실행하고 결과를 출력합니다.
response = chain.invoke(
    {
        "email_conversation": email_conversation,
        "question": "이메일 내용 중 주요 내용을 추출해 주세요"
    }
)

print(response)
```

```
Output:
person='김철수' email='chulsoo.kim@bikecorporation.me' subject='"ZENESIS" 자전거 유통 협력 및 미팅 일정 제안' summary='바이크코퍼레이션의 김철수 상무가 테디인터내셔널의 신규 자전거 "ZENESIS" 유통 협력에 관심을 표하며, 상세 브로슈어(기술 사양, 배터리 성능, 디자인)를 요청하고, 협력 논의를 위해 1월 15일 화요일 오전 10시에 미팅을 제안했습니다.' date='1월 15일 화요일 오전 10시'
```

# 3. JSON Parser

다음 예제는 랭체인의 `JsonOutputParser` 와 Pydantic 을 사용하여, 모델 출력을 JSON 형식으로 파싱하고 Pydantic 모델로 구조화하는 과정을 설명합니다. `JsonOutputParser`는 모델의 출력을 JSON으로 해석하고, 지정된 Pydantic 모델(`CuisineRecipe`)에 맞게 데이터를 구조화하여 제공합니다.

먼저 자료구조를 의미하는 `CuisineRecipe` 클래스를 Pydantic `BaseModel` 을 사용하여 정의합니다. `name` 필드는 요리의 이름을 나타내고, `recipe` 필드는 해당 요리를 만드는 레시피를 뜻합니다. 출력 파서로 `JsonOutputParser` 인스턴스를 생성하고, `pydantic_object` 매개변수로 `CuisineRecipe` 클래스를 전달하여, 모델 출력을 해당 Pydantic 모델로 파싱하도록 설정합니다. 그리고 `output_parser.get_format_instructions()` 메소드를 호출하여 모델에 전달한 포맷 지시사항을 얻습니다. 이 지시사항은 모델이 출력을 생성할 때 JSON 형식을 따르도록 안내하는 역할을 합니다.

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# 자료구조 정의 (pydantic)
class CuisineRecipe(BaseModel):
    name: str = Field(description="name of a Cuisine")
    recipe: str = Field(description="recipe to cook the Cuisine")

# 출력 파서 정의
output_parser = JsonOutputParser(pydantic_object=CuisineRecipe)

format_instructions = output_parser.get_format_instructions()

print(format_instructions)

```

```
실행 결과

The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
"""
{"properties": {"name": {"title": "Name", "description": "name of a Cuisine", "type": "string"}, "recipe": {"title": "Recipe", "description": "recipe to cook the Cuisine", "type": "string"}}, "required": ["name", "recipe"]}
"""
```

다음으로 모델에 입력으로 전달한 프롬프트를 구성하는 것입니다. `PromptTemplate`를 사용하여 사용자 질문(`query`)을 기반으로 한 프롬프트를 생성합니다. 프롬프트에는 사용자의 질문과 모델에 전달할 포맷 지시사항이 포함됩니다.

```python
# prompt 구성

prompt = PromptTemplate(
    template = "Answer the user query. \n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions":format_instructions},
    
)

print(prompt)
```

```
실행 결과

input_variables=['query'] input_types={} partial_variables={'format_instructions': 'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"name": {"title": "Name", "description": "name of a Cuisine", "type": "string"}, "recipe": {"title": "Recipe", "description": "recipe to cook the Cuisine", "type": "string"}}, "required": ["name", "recipe"]}\n```'} template='Answer the user query. \n{format_instructions}\n{query}\n'
```

마지막으로 체인을 구성하고 호출하여 모델의 출력을 파싱하여 반환합니다. 이 체인은 사용자의 질문을 받아 프롬프트를 생성하고, 생성된 프롬프트를 모델에 전달한 후, 모델의 출력을 JSON 형식으로 파싱하고 `CuisineRecipe` 객체로 변환하는 과정을 수행합니다. 체인을 호출하면 사용자의 질문에 대한 응답을 `CuisineRecipe` 형태로 받게 됩니다. 이는 모델이 "Bibimbap" 요리법에 대한 정보를 JSON 형식으로 제공하고, 이 정보가 `CuisineRecipe` 객체로 구조화되는 것을 의미합니다.

```python
chain = prompt | model | output_parser

chain.invoke({"query": "Let me know how to cook Bibimbap"})
```

```
실행 결과

{'name': 'Bibimbap',
 'recipe': '1. Cook rice according to package instructions. 2. Prepare the vegetables: julienne carrots, slice cucumber, and sauté spinach and shiitake mushrooms. 3. In a pan, fry an egg sunny side up. 4. In a large bowl, place the cooked rice and arrange the vegetables on top of the rice. 5. Add the fried egg in the center. 6. Drizzle with gochuj'}
```

# 4. StrOutputParser

StrOutputParser는 ChatModel로 부터 받아오는 AIMessage에서 답변 내용이 있는 content의 내용을 뽑아 str로 변환해 줍니다. 앞의 문장만 보면 단순히 AIMessage에서 content만 뽑아오는 기능이 왜 중요한거지 하는 의문이 드실 수 있습니다. StrOutputParser는 LLM을 이용한 실시간 스트리밍을 진행할 때 진가를 발휘합니다. LLM을 이용한 서비스를 실시간으로 운영할 때 StrOutputParser를 사용하지 않으면 AIMessage를 처리하는 코드가 따로 필요하며, 추가 후처리도 진행해야 합니다. 하지만 StrOutputParser를 사용하면 순수한 텍스트만 결과로 나오게 되므로 프론트엔드 처리가 훨씬 간편해집니다. StrOutputParser는 주로 세 가지 용도로 사용됩니다.

1. 단순 챗봇/Q&A: 메타데이터 없이 답변 텍스트만 사용자에게 보여줄 때.
2. LCEL 체인 연결: 체인의 앞 단계 출력을 다음 단계의 프롬프트 입력(str)으로 바로 넘겨줘야 할 때(객체를 넘기면 에러가 날 수 있기 때문입니다.)
3. 스트리밍(Streaming): 답변을 한 글자씩 실시간으로 화면에 뿌려줄 때, 메시지 청크(Chunk) 객체가 아닌 텍스트 조각을 받기 위해 필수적으로 사용됨

그럼 이제 예제를 통해 StrOutputParser에 대해 간략하게 알아보도록 하겠습니다. 우선 StrOutputParser를 사용하지 않을 경우를 보도록 하겠습니다.

```python
# StrOutputParser를 사용하지 않았을 때

# 체인 구성
chain = prompt | llm

# 실행
response = chain.invoke({"topic": "LangChain"})

# 결과 출력
print(f"타입: {type(response)}")
print(f"결과 : {response}")
```

reponse의 type은 AIMessage라고 나오며 response에서 필요한 데이터를 얻기 위해선 추가적으로 코드를 더 짜야 합니다.

```
Output:
타입: <class 'langchain_core.messages.ai.AIMessage'>
결과 : content='LangChain은 LLM(대규모 언어 모델) 기반 애플리케이션 개발을 돕는 오픈소스 **프레임워크**입니다.\n\n주요 목적은 LLM이 외부 데이터 소스나 다른 도구(API 등)와 상호작용할 수 있도록 **연결**하고, 복잡한 작업을 여러 단계로 나누어 **논리적인 흐름(체인)**으로 조율하는 것입니다.\n\n이를 통해 LLM의 한계(예: 최신 정보 부족, 특정 작업 수행 불가)를 보완하고, 더 강력하고 유연하며 맥락을 이해하는 애플리케이션을 쉽게 구축할 수 있게 해줍니다.\n\n**요약:** LLM이 외부 세계와 소통하고 복잡한 작업을 효율적으로 처리할 수 있도록 도와주는 연결 및 조율 도구입니다.' additional_kwargs={} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--b8490335-ae29-46ed-9429-94885e4ae987-0' usage_metadata={'input_tokens': 11, 'output_tokens': 1174, 'total_tokens': 1185, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 999}}
```

그럼 이제 StrOutputParser를 사용했을 때의 결과를 보도록 하겠습니다.

```python
from langchain_core.output_parsers import StrOutputParser

# 체인 구성
chain = prompt | llm | StrOutputParser()

response = chain.invoke({"topic":"LangChain"})

# 결과 출력
print(f"타입: {type(response)}")
print(f"결과값:\n{response}")
```

```
Output:
타입: <class 'str'>
결과값:
LangChain은 **대규모 언어 모델(LLM)을 기반으로 한 애플리케이션 개발을 돕는 오픈소스 프레임워크**입니다.

쉽게 말해, LLM을 단순히 질문하고 답변받는 것을 넘어, LLM이 **외부 데이터와 상호작용하고, 여러 단계를 거쳐 복잡한 작업을 수행하도록 만들어주는 도구 모음**이라고 생각할 수 있습니다.

주요 기능은 다음과 같습니다:

1.  **체인 (Chains):** 여러 LLM 호출이나 다른 구성 요소를 연결하여 복잡한 작업을 순차적으로 처리합니다.
2.  **에이전트 (Agents):** LLM이 어떤 도구(Tool)를 사용할지 스스로 결정하여 목표를 달성하도록 돕습니다.
3.  **도구 (Tools):** LLM이 외부 API, 데이터베이스, 검색 엔진 등과 연동하여 정보를 검색하거나 특정 작업을 수행하게 합니다.
4.  **메모리 (Memory):** 대화 기록을 기억하여 연속적인 대화나 이전 정보를 활용할 수 있게 합니다.
5.  **프롬프트 템플릿 (Prompt Templates):** 효과적인 프롬프트를 쉽게 생성하고 관리할 수 있도록 합니다.

요약하자면, LangChain은 LLM을 '생각하고 행동하며 기억하는' 지능적인 주체로 만들어, 실제 세상의 문제를 해결하는 데 필요한 모든 구성 요소를 제공하는 **'조립 키트' 또는 '운영체제'**와 같다고 볼 수 있습니다. 이를 통해 개발자들은 챗봇, 질의응답 시스템, 자동화된 워크플로우 등 다양한 LLM 기반 서비스를 더 쉽고 효율적으로 구축할 수 있습니다.
```

# 마치며

LangChain에서 대표적으로 사용되는 세 가지 출력 파서(parser)에 대해서 알아보았습니다. 포스트에서 다뤄본 세 가지 파서외에도 CSVOutputParser, PandasDataFrameOutPutParser 등이 있지만 이러한 파서들은 추후에 해당 파서들이 필요하다고 느껴질 때 따로 다뤄보도록 하겠습니다.

긴 글 읽어주셔서 감사드리며, 본문 내용 중에 잘못된 내용이나 오타, 궁금하신 점이 있으시면 댓글 달아주시기 바랍니다.

# 참조

- 판다스 스튜디오 - 랭체인(LangChain) 입문부터 응용까지(https://wikidocs.net/book/14473)
- 테디노트 - LangChain 한국어 튜토리얼KR(https://wikidocs.net/book/14314)
