---
title: "[LLM/RAG] LangChain - 5. LangChain의 Memory"
categories:
  - LLM/RAG
  - LangChain
tags:
  - LLM/RAG
  - LangChain
  
use_math: true  
toc: true
toc_sticky: true
toc_label: "LangChain Memory"
---

# 1. 메모리와 대화 관리

대화형 AI 애플리케이션에서 가장 중요한 기능 중 하나는 이전 대화 내용을 기억하고 맥락을 유지하는 것입니다. LangChain 은 다양한 메모리 관리 방식을 제공하여 상태를 가진(stateful) 대화형 애플리케이션을 구축할 수 있도록 도와줍니다. LangChain에서는 대화 버퍼 메모리(ConversationBufferMemory)와 같은 대화의 문맥을 유지하기 위한 메모리 기능을 가지는 클래스를 지원했었습니다. 하지만 현재는 RunnableWithMessageHistory 클래스가 개발되어 레거시 클래스로 취급받고 있습니다. 이번 포스트에서는 RunnableWithMessageHistory 클래스에 대해서만 집중적으로 다뤄보도록 하고 나머지 클래스는 추후에 필요하면 알아보도록 하겠습니다.

## 1.1 메모리의 필요성과 개념

1. 메모리가 없는 대화의 문제점

  일반적인 LLM 은 상태를 저장하지 않기 때문에, 각 요청은 독립적으로 처리됩니다. 이는 다음과 같은 문제를 야기합니다

  ```python
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(model="gpt-4o-mini")

  # 첫 번째 질문
  response1 = llm.invoke("안녕하세요, 제 이름은 민수입니다.")
  print("응답 1:", response1.content)

  # 두 번째 질문
  response2 = llm.invoke("제 이름이 뭐였죠?")
  print("응답 2:", response2.content)

  ```

  ```
  실행 결과

  응답 1: 안녕하세요 민수님! 만나서 반갑습니다. 어떤 도움이 필요하신가요?
  응답 2: 죄송하지만 이전 대화 내용을 기억하지 못합니다. 성함을 알려주시면 감사하겠습니다.
  ```

2. 메모리의 역할

  메모리 시스템은 다음과 같은 기능을 제공합니다.

  1. 대화 기록 저장 : 이전 메시지들을 저장하고 관리
  2. 컨텍스트 유지: 대화의 흐름과 맥락 보전
  3. 개인화 : 사용자별 대화 세션 관리
  4. 효율성 : 토큰 사용량 최적화를 위한 메모리 관리

## 1.2 RunnableWithMessageHistory

LangChain v0.3+ 에서 `RunnableWithMessageHistory` 를 사용한 메모리 기능 구현이 가능합니다. 대화형 AI 시스템에서 이전 대화 내용을 기억하고 활용하는 핵심 메커니즘에 대해서 알아보겠습니다. LangChain 문서에 따르면, v0.3 이후부터는 LangGraph 기반의 메모리 시스템을 권장하지만, `RunnableWithMessageHistory` 는 여전히 간단한 채팅 애플리케이션에서 효과적으로 사용할 수 있으며 향후에도 지원될 예정입니다.

### 1.2.1 RunnableWithMessageHistory 기본 사용법

1. 기본 컴포넌트 설정

  ```python
  from langchain_openai import ChatOpenAI
  from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
  from langchain_core.runnables.history import RunnableWithMessageHistory
  from langchain_core.chat_history import InMemoryChatMessageHistory

  prompt = ChatPromptTemplate.from_messages([
      ("system", "당신은 천문학 전문가입니다. 사용자와 친근한 대화를 나누며 천문학 질문에 답변해주세요."),
      MessagesPlaceholder(variable_name="history"),
      ("human", "{question}")
  ])

  ```

  핵심 구성 요소

  - `MessagesPlaceholder(variable_name="history")` : 대화 히스토리가 삽입될 위치를 지정합니다
  - 이 플레이스홀더를 통해 이전 대화 내용이 프롬프트에 동적으로 포함됩니다.
  - 시스템 메시지는 AI 의 역할과 성격을 정의합니다.

2. 메모리 저장소 구성

  ```python
  store = {}

  def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
      if session_id not in store:
          store[session_id] = InMemoryChatMessageHistory()
      return store[session_id]

  ```

  메모리 관리 메커니즘

  - `InMemoryChatMessageHistory` : 메모리 내에서 대화 기록을 저장합니다.
  - `session_id` 를 통해 서로 다른 사용자나 대화 세션을 구분합니다.
  - 각 세션마다 독립적인 대화 히스토리를 유지합니다.
  - 프로세스가 종료되면 데이터가 사라지는 인메모리 방식입니다.

3. 메모리 기능을 가진 체인 생성

  ```python

  chain_with_history = RunnableWithMessageHistory(
      chain,                           # 기본 체인 (prompt | llm)
      get_session_history,             # 세션 히스토리를 가져오는 함수
      input_messages_key="question",   # 사용자 입력 키
      history_messages_key="history",  # 대화 기록 키
  )

  ```

  매개변수 설명

  - `chain` : 기본 LLM 체인(프롬프트 + 모델)입니다.
  - `get_session_history` : 세션별 히스토리를 관리하는 함수입니다.
  - `input_messages_key` : 새로운 사용자 입력을 식별하는 키입니다.
  - `history_messages_key` : 프롬프트 템플릿의 `MessagesPlaceholder` 와 일치해야 합니다.

### 1.2.2 대화 실행과 메모리 작동 과정

대화 흐름 분석

```python
config = {"configurable": {"session_id": "astronomy_chat_1"}}

# 첫 번째 대화
response1 = chain_with_history.invoke(
    {"question": "안녕하세요, 저는 지구과학을 공부하는 학생입니다."},
    config=config
)

```

첫 번째 호출 시 내부 동작

  1. 새로운 `InMemoryChatMessageHistory` 인스턴스가 생성됩니다.
  2. 히스토리가 비어있으므로 시스템 메시지와 사용자 메시지만 모델에 전달됩니다.
  3. AI 응답이 생성되고 히스토리에 저장됩니다.

```python
# 두 번째 대화 - 컨텍스트 유지
response2 = chain_with_history.invoke(
    {"question": "태양계에서 가장 큰 행성은 무엇인가요?"},
    config=config
)
```

두 번째 호출 시 내부 동작

  1. 같은 `session_id` 로 기존 히스토리를 조회합니다.
  2. 이전 대화(사용자 메시지 + AI 응답)가 `history` 플레이스홀더에 삽입됩니다.
  3. 새로운 질문과 함께 전체 컨텍스트가 모델에 전달됩니다.

```python
# 세 번째 대화 - 참조 관계 이해
response3 = chain_with_history.invoke(
    {"question": "그 행성의 위성은 몇 개나 되나요?"},
    config=config
)

```

세 번째 호출 시 컨텍스트 활용

- "그 행성"이라는 표현이 이전 대화의 "목성"을 참조한다는 것을 AI 가 이해합니다.
- 모든 이전 대화 히스토리가 컨텍스트로 제공되어 자연스러운 대화가 가능합니다.

---

전체 코드는 다음과 같습니다. 코드 중간 중간 store를 출력하도록 해보았습니다.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 천문학 전문가입니다. 사용자와 친근한 대화를 나누며 천문학 질문에 답변해 주세요."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key = "question",
    history_messages_key = "history"
)

config = {"configurable":{"session_id":"astronomy_chat_1"}}

# 첫 번째 대화
response1 = chain_with_history.invoke(
    {"question": "안녕하세요, 저는 지구과학을 공부하는 학생입니다."},
    config=config
)

print(f"response1 : {response1}")
print(f"store : {store}")

# 두 번째 대화 - 컨텍스트 유지
response2 = chain_with_history.invoke(
    {"question": "태양계에서 가장 큰 행성은 무엇인가요?"},
    config=config
)

print(f"response2 : {response2}")
print(f"store : {store}")

# 세 번째 대화 - 참조 관계 이해
response3 = chain_with_history.invoke(
    {"question": "그 행성의 위성은 몇 개나 되나요?"},
    config=config
)

print(f"response3 : {response3}")
print(f"store : {store}")

```

실행 결과는 아래와 같습니다. store에는 대답이 하나씩 늘어감에 따라 이전 대화가 저장되는 것을 확인할 수 있었습니다.

```
실행 결과

response1 : 안녕하세요! 만나서 정말 반가워요. 지구과학을 공부하는 학생이시군요! 정말 멋진 분야를 공부하고 계시네요.

사실 천문학과 지구과학은 아주 가까운 '자매 학문' 같은 사이랍니다. 우리가 발 딛고 있는 이 지구라는 행성을 제대로 이해하려면, 지구가 속한 태양계와 더 넓은 우주를 함께 알아야 하니까요.

천문학에 대해 궁금한 점이 있다면 무엇이든 편하게 물어보세요. 예를 들어, 별의 탄생과 죽음에 대한 이야기나, 신비로운 블랙홀, 혹은 우리가 살고 있는 태양계의 다른 행성들에 대한 궁금증도 좋고요. 지구과학과 관련해서 화성이나 달의 지질 활동에 대해 궁금할 수도 있겠네요!

어떤 이야기부터 시작해볼까요? 편하게 질문해주세요! 😊
store : {'astronomy_chat_1': InMemoryChatMessageHistory(messages=[HumanMessage(content='안녕하세요, 저는 지구과학을 공부하는 학생입니다.', additional_kwargs={}, response_metadata={}), AIMessage(content="안녕하세요! 만나서 정말 반가워요. 지구과학을 공부하는 학생이시군요! 정말 멋진 분야를 공부하고 계시네요.\n\n사실 천문학과 지구과학은 아주 가까운 '자매 학문' 같은 사이랍니다. 우리가 발 딛고 있는 이 지구라는 행성을 제대로 이해하려면, 지구가 속한 태양계와 더 넓은 우주를 함께 알아야 하니까요.\n\n천문학에 대해 궁금한 점이 있다면 무엇이든 편하게 물어보세요. 예를 들어, 별의 탄생과 죽음에 대한 이야기나, 신비로운 블랙홀, 혹은 우리가 살고 있는 태양계의 다른 행성들에 대한 궁금증도 좋고요. 지구과학과 관련해서 화성이나 달의 지질 활동에 대해 궁금할 수도 있겠네요!\n\n어떤 이야기부터 시작해볼까요? 편하게 질문해주세요! 😊", additional_kwargs={}, response_metadata={})])}
response2 : 아주 좋은 질문이에요! 태양계의 '왕'이라고 할 수 있는 행성이죠. 😄

태양계에서 가장 큰 행성은 바로 **목성(Jupiter)**이랍니다! 🪐

목성의 크기가 어느 정도냐면, 정말 상상을 초월할 정도예요.

*   **크기 비교:** 태양계에 있는 다른 모든 행성들(수성, 금성, 지구, 화성, 토성, 천왕성, 해왕성)을 전부 합쳐도 목성 안에 쏙 들어갈 정도랍니다. 지름만 해도 지구의 약 11배나 되고요.
*   **질량:** 질량은 다른 모든 행성을 합친 것보다도 2.5배나 더 무거워요. 그야말로 태양계의 '거인'이죠.

목성은 지구처럼 단단한 땅으로 이루어진 암석형 행성이 아니라, 주로 수소와 헬륨 같은 가스로 이루어진 **'가스 거인(Gas Giant)'**이에요. 그래서 표면에는 아름다운 줄무늬가 보이는데, 이건 사실 엄청나게 빠른 속도로 움직이는 구름의 띠랍니다.

목성의 가장 큰 특징 중 하나는 바로 **'대적점(Great Red Spot)'**이라고 불리는 거대한 소용돌이예요. 이 붉은 점은 사실 지구보다도 더 큰, 수백 년 동안 계속되고 있는 거대한 폭풍이랍니다. 정말 놀랍지 않나요?

지구과학을 공부하는 학생으로서 한 가지 더 흥미로운 점을 알려드리자면, 목성은 거대한 중력 덕분에 태양계의 '큰형님' 또는 '수호자' 같은 역할을 하기도 해요. 목성이 강력한 중력으로 소행성이나 혜성들을 끌어당겨서, 지구를 포함한 안쪽 행성들이 충돌하는 것을 막아주는 방패 역할을 해주거든요.

혹시 목성에 대해 더 궁금한 점이 있으신가요? 아니면 다른 행성에 대한 이야기도 좋고요! 언제든지 편하게 물어보세요. 👍
store : {'astronomy_chat_1': InMemoryChatMessageHistory(messages=[HumanMessage(content='안녕하세요, 저는 지구과학을 공부하는 학생입니다.', additional_kwargs={}, response_metadata={}), AIMessage(content="안녕하세요! 만나서 정말 반가워요. 지구과학을 공부하는 학생이시군요! 정말 멋진 분야를 공부하고 계시네요.\n\n사실 천문학과 지구과학은 아주 가까운 '자매 학문' 같은 사이랍니다. 우리가 발 딛고 있는 이 지구라는 행성을 제대로 이해하려면, 지구가 속한 태양계와 더 넓은 우주를 함께 알아야 하니까요.\n\n천문학에 대해 궁금한 점이 있다면 무엇이든 편하게 물어보세요. 예를 들어, 별의 탄생과 죽음에 대한 이야기나, 신비로운 블랙홀, 혹은 우리가 살고 있는 태양계의 다른 행성들에 대한 궁금증도 좋고요. 지구과학과 관련해서 화성이나 달의 지질 활동에 대해 궁금할 수도 있겠네요!\n\n어떤 이야기부터 시작해볼까요? 편하게 질문해주세요! 😊", additional_kwargs={}, response_metadata={}), HumanMessage(content='태양계에서 가장 큰 행성은 무엇인가요?', additional_kwargs={}, response_metadata={}), AIMessage(content="아주 좋은 질문이에요! 태양계의 '왕'이라고 할 수 있는 행성이죠. 😄\n\n태양계에서 가장 큰 행성은 바로 **목성(Jupiter)**이랍니다! 🪐\n\n목성의 크기가 어느 정도냐면, 정말 상상을 초월할 정도예요.\n\n*   **크기 비교:** 태양계에 있는 다른 모든 행성들(수성, 금성, 지구, 화성, 토성, 천왕성, 해왕성)을 전부 합쳐도 목성 안에 쏙 들어갈 정도랍니다. 지름만 해도 지구의 약 11배나 되고요.\n*   **질량:** 질량은 다른 모든 행성을 합친 것보다도 2.5배나 더 무거워요. 그야말로 태양계의 '거인'이죠.\n\n목성은 지구처럼 단단한 땅으로 이루어진 암석형 행성이 아니라, 주로 수소와 헬륨 같은 가스로 이루어진 **'가스 거인(Gas Giant)'**이에요. 그래서 표면에는 아름다운 줄무늬가 보이는데, 이건 사실 엄청나게 빠른 속도로 움직이는 구름의 띠랍니다.\n\n목성의 가장 큰 특징 중 하나는 바로 **'대적점(Great Red Spot)'**이라고 불리는 거대한 소용돌이예요. 이 붉은 점은 사실 지구보다도 더 큰, 수백 년 동안 계속되고 있는 거대한 폭풍이랍니다. 정말 놀랍지 않나요?\n\n지구과학을 공부하는 학생으로서 한 가지 더 흥미로운 점을 알려드리자면, 목성은 거대한 중력 덕분에 태양계의 '큰형님' 또는 '수호자' 같은 역할을 하기도 해요. 목성이 강력한 중력으로 소행성이나 혜성들을 끌어당겨서, 지구를 포함한 안쪽 행성들이 충돌하는 것을 막아주는 방패 역할을 해주거든요.\n\n혹시 목성에 대해 더 궁금한 점이 있으신가요? 아니면 다른 행성에 대한 이야기도 좋고요! 언제든지 편하게 물어보세요. 👍", additional_kwargs={}, response_metadata={})])}
response3 : 와, 정말 좋은 질문을 해주셨네요! 목성의 위성 이야기는 천문학에서 아주 흥미로운 주제 중 하나거든요.

현재까지 공식적으로 확인된 목성의 위성은 **95개**입니다! 🪐

정말 많죠? 이 숫자는 태양계 행성들 중에서 가장 많은 숫자예요. 얼마 전까지는 토성이 1위였는데, 새로운 위성들이 계속 발견되면서 목성이 다시 1위 자리를 되찾았죠. 토성과 목성은 마치 '위성 개수 경쟁'을 하는 것 같아요. 😄

그런데 이 95개의 위성들이 모두 똑같이 중요하게 다뤄지는 건 아니에요. 이 중에서 특히 유명하고 중요한 **'4대 위성'**이 있답니다. 1610년에 갈릴레오 갈릴레이가 직접 만든 망원경으로 처음 발견해서 **'갈릴레이 위성'**이라고 불리죠. 이 위성들은 각각 아주 독특한 특징을 가지고 있어요.

1.  **이오 (Io) 🌋:** 태양계에서 가장 화산 활동이 활발한 천체예요! 표면 전체가 유황 화합물로 뒤덮여 있어서 마치 '피자'처럼 보이기도 한답니다. 목성의 강력한 중력 때문에 내부가 계속 주물러지면서 엄청난 열이 발생해 화산이 끊임없이 폭발하고 있어요.

2.  **유로파 (Europa) 🌊:** 꽁꽁 얼어붙은 얼음으로 표면이 덮여 있지만, 그 아래에는 지구의 바다를 모두 합친 것보다 더 많은 양의 **액체 상태 바다**가 있을 것으로 추정되는 곳이에요. 그래서 과학자들이 외계 생명체의 존재 가능성이 가장 높은 곳 중 하나로 꼽는 곳이랍니다.

3.  **가니메데 (Ganymede) ✨:** 태양계에서 가장 큰 위성이에요. 심지어 행성인 수성보다도 더 크답니다! 태양계 위성 중에서 유일하게 자체적인 자기장을 가지고 있기도 해요.

4.  **칼리스토 (Callisto) 💥:** 표면이 태양계에서 가장 오래되었고, 수많은 운석 충돌구(크레이터)로 뒤덮여 있어요. 아주 오랫동안 별다른 지질 활동 없이 조용히 지내온 위성이죠.

나머지 위성들은 대부분 크기가 훨씬 작고, 모양도 감자처럼 울퉁불퉁한 경우가 많아요. 이 작은 위성들 중 다수는 목성이 처음 만들어질 때 함께 태어난 게 아니라, 나중에 목성의 강력한 중력에 붙잡힌 소행성이나 혜성일 것으로 추측된답니다. 그래서 지금도 새로운 위성이 계속 발견되고 있는 거고요.

지구과학을 공부하시니, 특히 유로파에 관심이 가실 것 같아요. 지구의 심해처럼, 유로파의 바다에도 생명체가 존재할 가능성이 제기되고 있어서 과학자들이 정말 많은 기대를 하고 있거든요.

이 네 개의 갈릴레이 위성 중에서 특별히 더 궁금한 위성이 있나요? 아니면 또 다른 질문도 환영입니다
store : {'astronomy_chat_1': InMemoryChatMessageHistory(messages=[HumanMessage(content='안녕하세요, 저는 지구과학을 공부하는 학생입니다.', additional_kwargs={}, response_metadata={}), AIMessage(content="안녕하세요! 만나서 정말 반가워요. 지구과학을 공부하는 학생이시군요! 정말 멋진 분야를 공부하고 계시네요.\n\n사실 천문학과 지구과학은 아주 가까운 '자매 학문' 같은 사이랍니다. 우리가 발 딛고 있는 이 지구라는 행성을 제대로 이해하려면, 지구가 속한 태양계와 더 넓은 우주를 함께 알아야 하니까요.\n\n천문학에 대해 궁금한 점이 있다면 무엇이든 편하게 물어보세요. 예를 들어, 별의 탄생과 죽음에 대한 이야기나, 신비로운 블랙홀, 혹은 우리가 살고 있는 태양계의 다른 행성들에 대한 궁금증도 좋고요. 지구과학과 관련해서 화성이나 달의 지질 활동에 대해 궁금할 수도 있겠네요!\n\n어떤 이야기부터 시작해볼까요? 편하게 질문해주세요! 😊", additional_kwargs={}, response_metadata={}), HumanMessage(content='태양계에서 가장 큰 행성은 무엇인가요?', additional_kwargs={}, response_metadata={}), AIMessage(content="아주 좋은 질문이에요! 태양계의 '왕'이라고 할 수 있는 행성이죠. 😄\n\n태양계에서 가장 큰 행성은 바로 **목성(Jupiter)**이랍니다! 🪐\n\n목성의 크기가 어느 정도냐면, 정말 상상을 초월할 정도예요.\n\n*   **크기 비교:** 태양계에 있는 다른 모든 행성들(수성, 금성, 지구, 화성, 토성, 천왕성, 해왕성)을 전부 합쳐도 목성 안에 쏙 들어갈 정도랍니다. 지름만 해도 지구의 약 11배나 되고요.\n*   **질량:** 질량은 다른 모든 행성을 합친 것보다도 2.5배나 더 무거워요. 그야말로 태양계의 '거인'이죠.\n\n목성은 지구처럼 단단한 땅으로 이루어진 암석형 행성이 아니라, 주로 수소와 헬륨 같은 가스로 이루어진 **'가스 거인(Gas Giant)'**이에요. 그래서 표면에는 아름다운 줄무늬가 보이는데, 이건 사실 엄청나게 빠른 속도로 움직이는 구름의 띠랍니다.\n\n목성의 가장 큰 특징 중 하나는 바로 **'대적점(Great Red Spot)'**이라고 불리는 거대한 소용돌이예요. 이 붉은 점은 사실 지구보다도 더 큰, 수백 년 동안 계속되고 있는 거대한 폭풍이랍니다. 정말 놀랍지 않나요?\n\n지구과학을 공부하는 학생으로서 한 가지 더 흥미로운 점을 알려드리자면, 목성은 거대한 중력 덕분에 태양계의 '큰형님' 또는 '수호자' 같은 역할을 하기도 해요. 목성이 강력한 중력으로 소행성이나 혜성들을 끌어당겨서, 지구를 포함한 안쪽 행성들이 충돌하는 것을 막아주는 방패 역할을 해주거든요.\n\n혹시 목성에 대해 더 궁금한 점이 있으신가요? 아니면 다른 행성에 대한 이야기도 좋고요! 언제든지 편하게 물어보세요. 👍", additional_kwargs={}, response_metadata={}), HumanMessage(content='그 행성의 위성은 몇 개나 되나요?', additional_kwargs={}, response_metadata={}), AIMessage(content="와, 정말 좋은 질문을 해주셨네요! 목성의 위성 이야기는 천문학에서 아주 흥미로운 주제 중 하나거든요.\n\n현재까지 공식적으로 확인된 목성의 위성은 **95개**입니다! 🪐\n\n정말 많죠? 이 숫자는 태양계 행성들 중에서 가장 많은 숫자예요. 얼마 전까지는 토성이 1위였는데, 새로운 위성들이 계속 발견되면서 목성이 다시 1위 자리를 되찾았죠. 토성과 목성은 마치 '위성 개수 경쟁'을 하는 것 같아요. 😄\n\n그런데 이 95개의 위성들이 모두 똑같이 중요하게 다뤄지는 건 아니에요. 이 중에서 특히 유명하고 중요한 **'4대 위성'**이 있답니다. 1610년에 갈릴레오 갈릴레이가 직접 만든 망원경으로 처음 발견해서 **'갈릴레이 위성'**이라고 불리죠. 이 위성들은 각각 아주 독특한 특징을 가지고 있어요.\n\n1.  **이오 (Io) 🌋:** 태양계에서 가장 화산 활동이 활발한 천체예요! 표면 전체가 유황 화합물로 뒤덮여 있어서 마치 '피자'처럼 보이기도 한답니다. 목성의 강력한 중력 때문에 내부가 계속 주물러지면서 엄청난 열이 발생해 화산이 끊임없이 폭발하고 있어요.\n\n2.  **유로파 (Europa) 🌊:** 꽁꽁 얼어붙은 얼음으로 표면이 덮여 있지만, 그 아래에는 지구의 바다를 모두 합친 것보다 더 많은 양의 **액체 상태 바다**가 있을 것으로 추정되는 곳이에요. 그래서 과학자들이 외계 생명체의 존재 가능성이 가장 높은 곳 중 하나로 꼽는 곳이랍니다.\n\n3.  **가니메데 (Ganymede) ✨:** 태양계에서 가장 큰 위성이에요. 심지어 행성인 수성보다도 더 크답니다! 태양계 위성 중에서 유일하게 자체적인 자기장을 가지고 있기도 해요.\n\n4.  **칼리스토 (Callisto) 💥:** 표면이 태양계에서 가장 오래되었고, 수많은 운석 충돌구(크레이터)로 뒤덮여 있어요. 아주 오랫동안 별다른 지질 활동 없이 조용히 지내온 위성이죠.\n\n나머지 위성들은 대부분 크기가 훨씬 작고, 모양도 감자처럼 울퉁불퉁한 경우가 많아요. 이 작은 위성들 중 다수는 목성이 처음 만들어질 때 함께 태어난 게 아니라, 나중에 목성의 강력한 중력에 붙잡힌 소행성이나 혜성일 것으로 추측된답니다. 그래서 지금도 새로운 위성이 계속 발견되고 있는 거고요.\n\n지구과학을 공부하시니, 특히 유로파에 관심이 가실 것 같아요. 지구의 심해처럼, 유로파의 바다에도 생명체가 존재할 가능성이 제기되고 있어서 과학자들이 정말 많은 기대를 하고 있거든요.\n\n이 네 개의 갈릴레이 위성 중에서 특별히 더 궁금한 위성이 있나요? 아니면 또 다른 질문도 환영입니다", additional_kwargs={}, response_metadata={})])}
```

### 1.2.3 실제 메시지 구조 예시

첫 번째 호출 시 모델에 전달되는 메시지

```
[
    SystemMessage(content="당신은 천문학 전문가입니다..."),
    HumanMessage(content="안녕하세요, 저는 지구과학을 공부하는 학생입니다.")
]

```

세 번째 호출 시 모델에 전달되는 메시지

```
[
    SystemMessage(content="당신은 천문학 전문가입니다..."),
    HumanMessage(content="안녕하세요, 저는 지구과학을 공부하는 학생입니다."),
    AIMessage(content="안녕하세요! 지구과학을 공부하고 계시는군요..."),
    HumanMessage(content="태양계에서 가장 큰 행성은 무엇인가요?"),
    AIMessage(content="태양계에서 가장 큰 행성은 목성(Jupiter)입니다!..."),
    HumanMessage(content="그 행성의 위성은 몇 개나 되나요?")
]

```

### 1.2.4 주요 특징과 장점

1. 자동 컨텍스트 관리
   이전 모든 대화 내용이 자동으로 현재 프롬프트에 포함되어 연속적인 대화가 가능합니다.

2. 세션 기반 관리
  서로 다른 `seesion_id`를 사용하여 여러  사용자나 대화 스레드를 독립적으로 관리할 수 있습니다.

3. 간편한 구현
  복잡한 메모리 관리 로직 없이도 대화형 AI 를 구현할 수 있습니다.

---

### 1.2.5 프로덕션 환경에서의 고려사항

1. 영구 저장소 필요성
  `InMemoryChatMessageHistory` 대신 데이터베이스 기반 저장소를 사용해야 합니다.

2. 토큰 제한 관리
  대화가 길어질수록 토큰 사용량이 증가하므로 히스토리 트리밍이 필요합니다.

3. LangGraph 마이그레이션
  더 복잡한 기능이 필요한 경우 LangGraph 의 메모리 시스템으로 전환을 고려해야 합니다.

  이 구현 방식은 LangChain 에서 대화형 AI 의 메모리 기능을 이해하고 구축하는데 효과적인 방법입니다.

# 마치며

LangChain의 대화 관리 메모리 기능에 대해 알아보았습니다. 이번엔 이러한 기능이 있다는 것만 알아보았습니다만 추후에 토이 프로젝트로 벡터 DB나 일반 DB와 연동해 대화 메모리 기능을 적용을 진행해 보고자 합니다. 그리고 제가 필요하다고 느낀다면 RunnableWithMessageHistory 이 전의 클래스들을 한 번 정리해 보고자 합니다.

긴 글 읽어주셔서 감사드리며, 본문 내용 중에 잘못된 내용이나 오타, 궁금하신 사항이 있으실 경우 댓글 달아주시기 바랍니다.

# 참조

- 판다스 스튜디오 - 랭체인(LangChain) 입문부터 응용까지(https://wikidocs.net/book/14473)