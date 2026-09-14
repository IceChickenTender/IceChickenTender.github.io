---
title: "[PaperReview] ReAct: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS 논문 리뷰"
categories:
  - PaperReview
tags:
  - PaperReview

use_math: true
toc: true
toc_sticky: true
toc_label: "ReAct: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS 논문 리뷰"
---

# 1. Introduction: Context & Motivation

ReAct 논문에서 저자들은 Large Language Model(LLM)의 능력을 크게 두 가지 관점으로 나누어 바라봅니다.

첫 번째는 **Reasoning(추론)** 능력입니다. 대표적으로 Chain-of-Thought(CoT) Prompting과 같이 LLM이 최종 정답만 바로 출력하는 것이 아니라, 정답을 도출하기 위한 중간 추론 과정(Reasoning Trace)을 자연어로 생성하도록 하는 방법입니다.

두 번째는 **Acting(행동)** 능력입니다. 여기서 말하는 Acting은 단순히 텍스트를 생성하는 것이 아니라, LLM이 특정 Task를 해결하기 위해 외부 Environment와 상호작용할 수 있는 Action을 선택하거나 생성하는 것을 의미합니다. 예를 들어 Wikipedia에서 정보를 검색하거나, 웹페이지에서 버튼을 클릭하거나, 가상의 환경에서 특정 물체를 집거나 이동하는 행동 등이 이에 해당합니다.

ReAct 논문의 저자들은 당시의 연구들이 이러한 **Reasoning과 Acting을 주로 서로 독립적인 주제로 연구하고 있었다는 점**에 주목했습니다. 그리고 인간은 실제로 문제를 해결할 때 단순히 생각만 하거나 행동만 하는 것이 아니라, 생각과 행동을 반복하며 서로 보완한다는 점에서 아이디어를 얻었습니다.

## 1.1 LLM을 이용한 Reasoning과 Acting

ReAct 논문에서는 인간이 문제를 해결하는 과정을 하나의 예시로 설명합니다.

예를 들어 요리를 하는 상황을 생각해 보면, 사람은 단순히 정해진 행동을 순서대로 수행하는 것이 아니라 행동과 행동 사이에서 지속적으로 생각합니다.

- "재료를 모두 손질했으니 이제 물을 끓여야겠다."
- "소금이 없으니 간장과 후추를 대신 사용해야겠다."
- "반죽을 어떻게 만드는지 모르겠으니 인터넷에서 찾아봐야겠다."

이처럼 사람은 현재까지의 진행 상황을 파악하고, 계획을 수정하거나, 필요한 외부 정보를 찾아야 하는지를 판단하기 위해 자연어 형태의 사고 과정을 사용합니다.

반대로 행동 자체가 Reasoning을 도와주기도 합니다.

- 냉장고를 열어 어떤 재료가 있는지 확인한다.
- 요리책을 펼쳐 조리법을 확인한다.
- 인터넷에서 필요한 정보를 검색한다.

즉 사람의 문제 해결 과정은 다음처럼 단순히 한 방향으로만 진행되지 않습니다.

`Reasoning → Acting`

또는

`Acting → Reasoning`

중 하나가 아니라,

`Reasoning → Acting → 새로운 정보 획득 → Reasoning → Acting → ...`

처럼 Reasoning과 Acting이 서로 영향을 주는 반복적인 과정에 가깝습니다.

논문의 저자들은 이러한 **Reasoning과 Acting 사이의 긴밀한 Synergy(상호작용)**가 인간이 처음 접하는 상황이나 불확실한 정보가 존재하는 상황에서도 비교적 안정적으로 문제를 해결할 수 있게 하는 중요한 요소라고 설명합니다.

LLM에서도 이와 유사하게 두 종류의 연구가 이미 진행되고 있었습니다.

첫 번째는 CoT와 같은 **Reasoning 중심의 연구**이며, 두 번째는 LLM을 이용해 특정 Action이나 Action Plan을 생성하는 **Acting 중심의 연구**입니다.

ReAct는 이 두 흐름을 하나의 문제 해결 과정 안에서 결합하는 것을 목표로 합니다.

## 1.2 Reasoning-only 접근의 한계

LLM의 Reasoning 능력을 보여준 대표적인 방법이 **Chain-of-Thought(CoT) Prompting**입니다.

CoT는 LLM에게 단순히 정답만 출력하도록 하는 대신, 정답을 도출하기 위한 중간 Reasoning Trace를 생성하도록 합니다.

예를 들어 일반적인 Prompting이 다음과 같다면,

`Question → Answer`

CoT는 다음과 같은 형태를 가집니다.

`Question → Reasoning Trace → Answer`

이러한 방식은 산술 문제, 상식 추론, Symbolic Reasoning과 같이 여러 단계의 추론이 필요한 Task에서 LLM의 성능을 크게 향상시켰습니다.

하지만 ReAct 논문의 저자들은 CoT에 한 가지 중요한 한계가 있다고 지적합니다.

CoT에서 생성되는 Reasoning은 기본적으로 **LLM 내부의 표현과 지식만을 기반으로 진행되며 외부 세계와 직접 연결되어 있지 않습니다.**

논문에서는 이를 일종의 **"static black box"**라고 표현합니다.

예를 들어 CoT가 다음과 같은 Reasoning을 수행한다고 가정할 수 있습니다.

`Question → 내부 지식을 이용한 추론 → 추가 추론 → Answer`

문제는 중간에 모델이 잘못된 사실을 생성하더라도 이를 외부에서 확인할 방법이 없다는 것입니다.

초기의 잘못된 정보가 이후의 Reasoning에 계속 사용되면,

`잘못된 사실 → 잘못된 추론 → 추가적인 잘못된 추론 → 오답`

처럼 **Error Propagation(오류 전파)**이 발생할 수 있습니다.

또한 LLM이 실제로 존재하지 않는 사실을 만들어내는 **Hallucination(환각)** 역시 문제가 됩니다.

즉 CoT는 LLM에게 더 긴 Reasoning 과정을 생성하도록 만들어 주지만, 그 Reasoning에서 사용되는 정보가 실제 외부 세계와 일치하는지를 직접 확인할 수 있는 구조는 아닙니다.

ReAct의 저자들은 이 점 때문에 Reasoning 과정이 외부 Environment와 상호작용할 수 있어야 한다고 보았습니다.

## 1.3 Acting-only 접근의 한계

Reasoning 연구와 별개로, LLM을 이용하여 실제 Environment에서 Action을 수행하거나 Action Plan을 생성하려는 연구도 진행되고 있었습니다.

일반적인 구조는 다음과 같습니다.

`Environment Observation → LLM → Action 생성 → Controller가 Action 실행`

예를 들어 로봇 환경이라면,

- 특정 위치로 이동하기
- 물체 집기
- 물체 놓기

등의 Action을 LLM이 생성하도록 할 수 있습니다.

웹 환경이라면,

- 검색하기
- 특정 페이지 선택하기
- 버튼 클릭하기

등이 Action이 될 수 있습니다.

이러한 방법의 장점은 LLM이 단순히 내부 지식만 이용하는 것이 아니라 **실제 Environment와 상호작용할 수 있다는 점**입니다.

하지만 저자들은 기존 Acting 중심의 접근에도 한계가 있다고 지적합니다.

기존 방법들은 주로 현재 Observation을 바탕으로 **다음에 어떤 Action을 수행해야 하는지를 직접 예측하는 것**에 초점을 맞추고 있었습니다.

따라서 다음과 같은 고수준 Reasoning이 부족할 수 있습니다.

- 현재 Task의 목표를 여러 개의 Subgoal로 나누기
- 지금까지 어떤 Subgoal을 완료했는지 기억하기
- 기존 계획이 실패했을 때 새로운 계획으로 변경하기
- 다음에 어떤 정보를 찾아야 하는지 판단하기
- 현재 상태를 바탕으로 장기적인 Action Plan을 세우기

논문에서는 기존의 Acting 중심 접근들이 주로 Language Model을 이용하여 Domain-specific Action이나 Plan을 생성한 뒤 Controller를 통해 실행하는 형태이며, 일부 예외를 제외하면 **고수준 목표에 대한 추론이나 Working Memory를 위해 Language Model을 적극적으로 사용하지 않았다**고 설명합니다.

즉 Acting-only 방식에서는 다음과 같은 문제가 발생할 수 있습니다.

`Observation → Action → Observation → Action → ...`

각각의 Action은 그 순간의 Observation에 대해서는 그럴듯할 수 있지만, 전체 Task를 해결하기 위한 장기적인 계획이나 현재 진행 상태에 대한 추론이 부족하면 잘못된 Action을 반복하거나 중요한 Subgoal을 놓칠 수 있습니다.

ReAct의 저자들은 이러한 문제를 해결하기 위해 **Acting 과정 내부에 명시적인 Reasoning Trace를 추가할 필요가 있다고 판단했습니다.**

## 1.4 Reasoning과 Acting을 결합한 ReAct의 아이디어

ReAct는 이름 그대로 **Reasoning + Acting**을 결합한 방법입니다.

기존 Reasoning-only 방식이

`Question → Thought → Thought → Answer`

와 같은 구조였다면,

Acting-only 방식은

`Question → Action → Observation → Action → Observation → ...`

과 같은 형태를 가질 수 있습니다.

ReAct는 이 두 가지를 결합하여 다음과 같은 문제 해결 과정을 구성합니다.

`Question → Thought → Action → Observation → Thought → Action → Observation → ... → Answer`

여기서 중요한 점은 Thought와 Action이 단순히 독립적으로 존재하는 것이 아니라 서로 영향을 준다는 것입니다.

ReAct에서 **Reasoning이 Acting을 돕는 과정**은 다음과 같습니다.

- 현재 문제를 여러 개의 단계로 분해합니다.
- 다음에 어떤 Action을 수행해야 하는지 결정합니다.
- 현재까지 얻은 정보를 바탕으로 기존 계획을 수정합니다.
- 예상하지 못한 상황이 발생했을 때 다른 Action을 선택합니다.

논문에서는 이를 **"reason to act"**라고 설명합니다.

반대로 **Acting 역시 Reasoning을 돕습니다.**

LLM이 외부 Environment에 Action을 수행하면 새로운 Observation을 얻을 수 있습니다.

예를 들어 Wikipedia에서 특정 정보를 검색하면 기존에 LLM 내부에 없던 정보를 얻을 수 있고, 이 정보는 이후 Reasoning 과정에 다시 사용됩니다.

논문에서는 이를 **"act to reason"**이라고 설명합니다.

따라서 ReAct의 핵심 아이디어는 단순히 CoT 뒤에 검색 기능을 추가하는 것이 아니라,

**Reasoning을 통해 어떤 Action을 수행할지 결정하고, Action을 통해 얻은 새로운 정보를 다시 Reasoning에 반영하는 구조를 만드는 것**이라고 볼 수 있습니다.

이를 간단히 표현하면 다음과 같습니다.

`Reason to Act → Act → Observe → Act to Reason → Reason to Act → ...`

ReAct 논문에서는 이러한 구조를 Question Answering, Fact Verification, Text-based Game, Web Navigation과 같이 서로 다른 성격을 가진 Task에 적용합니다.

특히 HotpotQA와 FEVER에서는 Wikipedia API와 상호작용하여 외부 정보를 가져오고, ALFWorld와 WebShop에서는 실제 Environment에서 여러 Action을 수행하도록 하여 Reasoning과 Acting의 결합이 다양한 Task에서도 효과가 있는지를 확인합니다.

ReAct 논문의 핵심 질문을 한 문장으로 정리하면 다음과 같습니다.

> **LLM이 내부적으로 Reasoning만 하거나 외부 Environment에서 Acting만 하는 것이 아니라, Reasoning과 Acting을 하나의 문제 해결 과정 안에서 반복적으로 결합한다면 더 정확하고 안정적으로 문제를 해결할 수 있지 않을까?**

그리고 ReAct는 이 질문에 대해 **Thought, Action, Observation이 서로 연결된 Task-solving Trajectory**를 생성하도록 하는 방식으로 접근합니다.

# 2. Architecture & Objective

ReAct 논문을 처음 읽었을 때에는 ReAct를 단순히 `CoT + Acting` 형태의 Prompting Technique으로 이해했습니다. 하지만 논문의 Section 2를 자세히 보면 ReAct는 단순히 Prompt의 문장 형태만 변경하는 것보다 조금 더 큰 구조를 전제로 하고 있습니다.

ReAct는 기본적으로 **Agent가 Environment와 상호작용하며 Task를 해결하는 상황**에서 출발합니다.

Agent는 현재 Environment의 상태를 Observation으로 전달받고, 지금까지의 Context를 바탕으로 다음 Action을 결정합니다. ReAct는 이러한 기존 Agent의 Action Space에 자연어 형태의 Reasoning을 추가하여, Agent가 단순히 행동만 선택하는 것이 아니라 행동 사이에서 명시적인 Thought를 생성할 수 있도록 합니다.

즉 ReAct의 핵심은 새로운 Language Model Architecture를 만드는 것이 아니라,

**기존 Agent-Environment Interaction 안에서 LLM이 Thought와 Action을 함께 생성하도록 만드는 것**

이라고 볼 수 있습니다.

## 2.1 Agent와 Environment의 기본 구조

ReAct 논문에서는 먼저 일반적인 Agent의 문제 해결 과정을 다음과 같이 정의합니다.

시간 단계 $t$에서 Agent는 Environment로부터 Observation $o_t$를 전달받습니다.

$$
o_t \in O
$$

여기서 $O$는 Agent가 Environment로부터 받을 수 있는 Observation들의 집합입니다.

Agent는 현재까지의 Context $c_t$를 바탕으로 Policy $\pi$에 따라 다음 Action $a_t$를 선택합니다.

$$
a_t \in A
$$

$$
\pi(a_t \mid c_t)
$$

여기서 $A$는 Agent가 수행할 수 있는 Action Space이며, $\pi(a_t \mid c_t)$는 현재 Context $c_t$가 주어졌을 때 Action $a_t$를 선택하는 Policy를 의미합니다.

Agent가 현재까지 가지고 있는 Context는 다음과 같이 정의됩니다.

$$
c_t = (o_1, a_1, \dots, o_{t-1}, a_{t-1}, o_t)
$$

즉 Agent가 가지고 있는 Context에는 지금까지 수행한 모든 Action과, 그 Action의 결과로 Environment에서 전달받은 Observation이 포함됩니다.

이를 단순하게 표현하면 다음과 같습니다.

`Observation → Action → Observation → Action → ...`

예를 들어 Wikipedia 검색 Agent를 생각한다면,

`Question`   
`→ Search[Apple Remote]`   
`→ Wikipedia 검색 결과`   
`→ Search[Front Row]`   
`→ Wikipedia 검색 결과`   

와 같은 흐름이 될 수 있습니다.

문제는 현재까지의 Observation만 보고 다음 Action을 바로 생성하는 것이 항상 쉽지는 않다는 것입니다.

Agent는 다음 Action을 결정하기 위해 현재까지 무엇을 했는지 기억하고, 어떤 정보가 부족한지 판단하고, 현재의 계획을 수정해야 할 수도 있습니다.

하지만 Acting-only Agent에서는 이러한 과정이 명시적으로 드러나지 않습니다.

ReAct는 바로 이 문제를 해결하기 위해 Agent의 Action Space 자체를 확장합니다.

## 2.2 Action Space에 Language Space를 추가한다는 의미

ReAct의 핵심 아이디어는 기존 Agent의 Action Space $A$에 **Language Space $L$**을 추가하는 것입니다.

논문에서는 이를 다음과 같이 표현합니다.

$$
\hat{A} = A \cup L
$$

여기서

- $A$: 실제 Environment에 영향을 주는 Action들의 집합
- $L$: 자연어로 표현할 수 있는 Language Space
- $\hat{A}$: ReAct에서 확장된 전체 Action Space

를 의미합니다.

ReAct에서는 Language Space $L$에 속하는 Action을 **Thought 또는 Reasoning Trace**라고 부릅니다.

$$
\hat{a}_t \in L
$$

즉 ReAct에서는 다음 두 종류의 Action이 하나의 Action Space 안에 존재한다고 볼 수 있습니다.

1. **Environment Action**

   실제 Environment의 상태를 변경하거나 외부 정보를 가져오는 Action입니다.

   예를 들어 HotpotQA에서는 다음과 같은 Action이 존재합니다.

   `Search[Colorado orogeny]`   
   `Lookup[eastern sector]`   
   `Finish[1,800 to 7,000 ft]`

2. **Language Action**

   Environment의 상태를 직접 변경하지 않고, 이후의 Reasoning과 Acting을 돕기 위해 자연어 형태로 생성되는 Thought입니다.

   예를 들어 다음과 같습니다.

   `먼저 Colorado orogeny를 검색하여 eastern sector가 어느 지역까지 이어지는지 찾아야 한다.`

중요한 차이는 **Thought는 외부 Environment에 직접적인 영향을 주지 않는다는 것**입니다.

예를 들어,

`Thought: High Plains의 고도를 찾아야 한다.`

라고 생각했다고 해서 Wikipedia에서 자동으로 검색 결과가 발생하는 것은 아닙니다.

반면,

`Action: Search[High Plains]`

를 수행하면 실제 외부 Environment에 요청이 전달되고 그 결과로 Observation을 받을 수 있습니다.

따라서 Thought는 Environment를 변경하기 위한 Action이 아니라 **Agent 자신의 Context를 변경하기 위한 Language Action**이라고 이해할 수 있습니다.

Thought가 생성되면 Agent의 Context는 다음과 같이 갱신됩니다.

$$
c_{t+1} = (c_t, \hat{a}_t)
$$

즉 Agent가 생성한 Thought 역시 이후의 Action을 결정할 때 사용할 수 있는 Context의 일부가 됩니다.

이 구조 덕분에 Agent는 단순히

`Observation → Action`

을 반복하는 것이 아니라,

`Observation → Thought → Action`

과 같이 현재 상태를 먼저 Reasoning한 후 Action을 선택할 수 있습니다.

## 2.3 Thought, Action, Observation의 역할

ReAct를 이해하기 위해서는 **Thought, Action, Observation을 명확하게 구분할 필요가 있습니다.**

### Thought

Thought는 LLM이 현재 Context를 바탕으로 생성하는 자연어 형태의 Reasoning Trace입니다.

Thought의 역할은 단순히 "생각을 길게 적는 것"이 아닙니다.

논문에서는 Thought가 다음과 같은 역할을 수행할 수 있다고 설명합니다.

- Task의 목표를 여러 Subgoal로 분해
- 이후 수행할 Action Plan 생성
- 문제 해결에 필요한 Commonsense Knowledge 활용
- Observation에서 중요한 정보 추출
- 지금까지의 진행 상황 추적
- 기존 Action Plan 변경
- 예상하지 못한 상황에 대한 예외 처리

예를 들어 다음과 같은 Thought가 생성될 수 있습니다.

`Thought: Colorado orogeny의 eastern sector가 어느 지역까지 이어지는지 먼저 찾아야 한다.`

Observation을 받은 이후에는 Thought가 다음과 같이 변경될 수 있습니다.

`Thought: eastern sector가 High Plains까지 이어진다는 것을 확인했다. 이제 High Plains의 elevation range를 찾아야 한다.`

즉 Thought는 **현재까지 얻은 정보를 기반으로 다음 Action을 결정하기 위한 중간 Reasoning 과정**입니다.

### Action

Action은 LLM이 다음에 Environment에서 무엇을 수행해야 하는지를 결정한 결과입니다.

HotpotQA와 FEVER 실험에서는 다음 세 가지 Action이 사용됩니다.

`Search[entity]`

Wikipedia에서 해당 Entity를 검색합니다.

`Lookup[string]`

현재 검색된 Wikipedia 문서 안에서 특정 문자열이 포함된 문장을 찾습니다.

`Finish[answer]`

현재 문제 해결을 종료하고 최종 Answer를 반환합니다.

다른 Environment에서는 Action Space도 달라집니다.

예를 들어 ALFWorld에서는,

`go to fridge 1`   
`open fridge 1`   
`take lettuce 1`

같은 Action을 사용할 수 있으며,

WebShop에서는,

`search`   
`click`   
`Buy Now`

등의 Action을 사용할 수 있습니다.

따라서 ReAct의 Action은 특정한 하나의 행동을 의미하는 것이 아니라 **현재 Agent가 연결되어 있는 Environment에서 수행할 수 있는 행동**을 의미합니다.

### Observation

Observation은 Action이 실행된 결과로 Environment에서 반환되는 정보입니다.

여기서 가장 중요하게 보아야 할 점은, **Observation은 LLM이 생성하는 정보가 아니라 Environment가 반환하는 정보라는 점입니다.**

예를 들어 LLM이 다음 Action을 생성했다고 가정합니다.

`Action: Search[Colorado orogeny]`

그렇다고 LLM이 바로 다음 Wikipedia 검색 결과까지 스스로 생성하는 것이 아닙니다.

실제로 Search Action을 Environment에서 실행하고,

`Observation: The Colorado orogeny was an episode of mountain building ...`

과 같은 검색 결과를 얻어야 합니다.

따라서 세 요소의 생성 주체는 다음과 같이 구분할 수 있습니다.

`Thought → LLM`   
`Action → LLM`   
`Observation → Environment`

이 구분은 ReAct를 단순한 CoT Prompting과 구분하는 데 매우 중요합니다.

## 2.4 ReAct의 전체 실행 과정

ReAct 논문에서는 저자들이 자신들의 방법을 **Prompt-based Paradigm**이라고 설명합니다.

하지만 실제 ReAct의 동작 과정을 이해하려면 Prompt뿐만 아니라 **Agent와 Environment 사이의 반복적인 Interaction**까지 함께 생각해야 합니다.

예를 들어 새로운 Question이 다음과 같이 들어왔다고 가정해 보겠습니다.

`Question: Colorado orogeny의 eastern sector가 이어지는 지역의 elevation range는 얼마인가?`

먼저 LLM은 현재 Question과 Few-shot Example을 바탕으로 Thought와 Action을 생성합니다.

`Thought 1: 먼저 Colorado orogeny를 검색하여 eastern sector가 어디까지 이어지는지 찾아야 한다.`

`Action 1: Search[Colorado orogeny]`

그러면 Search Action이 Environment에서 실행되고 다음 Observation을 얻습니다.

`Observation 1: The Colorado orogeny was an episode of mountain building ...`

이 정보는 기존 Context에 추가됩니다.

다음 LLM Inference에서는 기존 Question만 전달되는 것이 아니라 지금까지의 Trajectory 전체가 Context로 사용됩니다.

`Question`   
`+ Thought 1`   
`+ Action 1`   
`+ Observation 1`

이를 바탕으로 LLM은 다음 Thought와 Action을 생성합니다.

`Thought 2: 현재 Observation에는 eastern sector 정보가 없다. eastern sector를 Lookup해야 한다.`   
`Action 2: Lookup[eastern sector]`

그리고 다시 Environment에서 Observation을 얻습니다.

`Observation 2: The eastern sector extends into the High Plains ...`

다음 Context는 다시 확장됩니다.

`Question`   
`+ Thought 1`   
`+ Action 1`   
`+ Observation 1`   
`+ Thought 2`   
`+ Action 2`   
`+ Observation 2`

그리고 이 Context를 이용해 다시 LLM을 호출합니다. 이 과정을 반복하면 전체적인 흐름은 다음과 같이 볼 수 있습니다.

`Question`   
`→ LLM`   
`→ Thought`   
`→ Action`   
`→ Environment`   
`→ Observation`   
`→ 기존 Context에 Observation 추가`   
`→ LLM 재호출`   
`→ Thought`   
`→ Action`   
`→ Environment`   
`→ Observation`   
`→ ...`   
`→ Finish[Answer]`

이를 Pseudo Code 형태로 표현하면 다음과 같이 이해할 수 있습니다.

```python
context = few_shot_examples + question

while True:

    output = LLM(context)

    thought, action = parse(output)

    context += thought
    context += action

    if action is Finish:
        return answer

    observation = environment.execute(action)

    context += observation
```

# 3. Deep Dive: ReAct는 무엇이 다른가?

ReAct 논문을 처음 읽으면서 가장 이해하기 어려웠던 부분 중 하나는 **ReAct와 Chain-of-Thought(CoT)의 관계**였습니다.

ReAct라는 이름 자체가 `Reasoning + Acting`이기 때문에 처음에는 자연스럽게 다음과 같이 생각했습니다.

`ReAct = CoT + 외부 Action` 즉 CoT와 같은 Prompting Technique인줄 알았습니다.

하지만 논문에서는 ReAct와 CoT를 서로 다른 방법으로 분리하여 비교하고 있으며, 심지어 ReAct와 CoT-SC를 다시 결합했을 때 가장 좋은 성능을 얻었다고 설명합니다.

처음에는 이미 ReAct 내부에 Thought라는 Reasoning 과정이 있는데 왜 다시 CoT와 결합할 수 있는지가 이해되지 않았습니다.

이를 이해하기 위해서는 **CoT의 Reasoning Trace와 ReAct의 Thought가 비슷한 역할을 수행하지만, 두 방법의 전체적인 문제 해결 구조는 서로 다르다**는 점을 먼저 이해할 필요가 있습니다.

## 3.1 ReAct vs. CoT(Chain-of-Thought)

Chain-of-Thought(CoT)는 LLM이 최종 Answer를 바로 생성하는 대신, 중간 Reasoning Trace를 먼저 생성하도록 유도하는 Prompting Technique입니다.

일반적인 Prompting이 다음과 같다면,

`Question → Answer`

CoT는 다음과 같은 형태를 가집니다.

`Question → Reasoning Trace → Answer`

중요한 점은 CoT에서는 문제 해결 과정 전체가 기본적으로 **LLM의 Generation 내부에서 진행될 수 있다는 것**입니다. 예를 들어,

```
Question: A와 B 중 어떤 것이 먼저 만들어졌는가?

Thought: A는 1844년에 만들어졌고 B는 1989년에 만들어졌다.
1844 < 1989이므로 A가 먼저 만들어졌다.

Answer: A
```

와 같이 Question을 입력받은 LLM이 Reasoning Trace와 Answer를 연속적으로 생성할 수 있습니다.

반면 ReAct는 다음과 같은 구조를 가집니다.

`Question → Thought → Action → Observation → Thought → Action → Observation → ...`

여기서 가장 중요한 차이는 **Observation의 존재**입니다.

ReAct의 Thought와 Action은 LLM이 생성할 수 있지만, Observation은 외부 Environment에서 전달받아야 합니다.

따라서 실제 실행 관점에서 두 방법을 비교하면 다음과 같습니다.

### CoT

```
Prompt
  ↓
LLM
  ↓
Reasoning Trace
  ↓
Answer
```

### ReAct

```
Prompt
  ↓
LLM
  ↓
Thought + Action
  ↓
Environment
  ↓
Observation
  ↓
LLM
  ↓
Thought + Action
  ↓
...
```

즉 CoT와 ReAct의 차이는 단순히 **Reasoning 문장을 생성하느냐 마느냐**의 차이가 아닙니다.

CoT는 주어진 Context 안에서 Reasoning을 계속 진행하는 반면, ReAct는 Reasoning 과정 중에 외부 Environment에 Action을 수행하고 그 결과로 새로운 Observation을 얻어 기존 Context를 확장할 수 있습니다.

논문에서는 이를 실험적으로 비교하기 위해 ReAct Trajectory에서 일부 요소를 제거하여 Baseline을 구성합니다.

- `Standard`: Thought, Action, Observation을 모두 제거
- `CoT`: Action과 Observation을 제거하여 Thought만 유지
- `Act`: Thought를 제거하여 Action과 Observation만 유지
- `ReAct`: Thought, Action, Observation을 모두 사용

따라서 논문에서 CoT는 **Reasoning-only**, Act는 **Acting-only**, ReAct는 **Reasoning + Acting**에 해당합니다.

다만 시스템적인 관점에서는 CoT와 ReAct 사이에 상당한 차이가 있습니다.

CoT는 LLM에 Prompt를 입력하고 하나의 Generation으로 문제를 해결할 수 있지만, ReAct는 Action을 실행할 Environment와 Observation을 다시 LLM에 전달하는 Interaction Loop가 필요하기 때문입니다.

이 부분은 이후 Critical Review에서 다시 살펴보고자 합니다.

## 3.2 Internal Knowledge vs. External Knowledge

ReAct와 CoT를 비교하면서 또 하나 헷갈렸던 부분은 **Internal Knowledge와 External Knowledge의 관계**였습니다.

처음에는 다음과 같이 이해했습니다.

`CoT = LLM 내부 지식 사용`

`ReAct = 외부 지식 사용`

하지만 이는 정확한 구분이 아닙니다.

ReAct 역시 LLM의 **Internal Knowledge(내부 지식)**를 사용합니다.

ReAct의 Thought는 단순히 외부에서 가져온 Observation을 정리하는 역할만 수행하지 않습니다.

논문에서는 Thought가 다음과 같은 역할도 수행한다고 설명합니다.

- Task의 목표를 여러 Subgoal로 분해
- Action Plan 생성
- Commonsense Reasoning 수행
- Arithmetic Reasoning 수행
- Observation에서 중요한 정보 추출
- 현재 진행 상황 추적
- 기존 Plan 수정

특히 Commonsense Reasoning은 LLM이 Pre-training 과정에서 학습한 내부 지식을 사용하는 대표적인 예입니다.

예를 들어 ALFWorld에서 LLM이,

```
Thought: lettuce는 fridge나 dining table에 있을 가능성이 높다.
```

라고 판단한다면 이 정보는 Environment에서 직접 전달받은 것이 아니라, LLM이 Pre-training 과정에서 습득한 Commonsense Knowledge를 이용한 것입니다.

따라서 ReAct를 다음과 같이 보는 것이 더 정확합니다.

`Internal Knowledge + External Observation → Reasoning → Action`

반면 CoT는 외부 Environment와 상호작용하는 Action이 없기 때문에, **이 논문의 Question-only 실험 설정에서는** 기본적으로 Question과 현재 Prompt에 포함된 정보, 그리고 LLM이 기존에 가지고 있던 Internal Knowledge를 이용하여 답을 도출합니다.

이를 비교하면 다음과 같습니다.

| Method | Internal Knowledge | External Knowledge 획득 | External Action |
| --- | --- | --- | --- |
| CoT | O | X | X |
| ReAct | O | O | O |

즉 ReAct의 특징은 Internal Knowledge를 External Knowledge로 대체했다는 것이 아닙니다.

**LLM이 기존에 가지고 있던 Internal Knowledge를 이용해 Reasoning하면서, 필요한 경우 Action을 통해 External Knowledge를 추가로 가져올 수 있다는 것**이 핵심입니다.

예를 들어 다음과 같은 과정이 가능합니다.

```
Thought:
질문의 두 인물의 직업을 비교해야 한다.
먼저 첫 번째 인물을 검색해야 한다.

Action:
Search[Person A]

Observation:
Person A was a mathematician ...

Thought:
Person A는 수학자이다.
이제 Person B의 직업을 검색해야 한다.

Action:
Search[Person B]

Observation:
Person B was a mathematician and computer scientist ...

Thought:
두 사람 모두 mathematician이므로 공통된 직업은 mathematician이다.
```

첫 번째 Thought에서 어떤 정보를 찾아야 할지 판단하는 과정과 마지막 두 정보를 비교하는 과정은 LLM의 Reasoning입니다.

반면 각 인물에 대한 실제 정보는 External Environment를 통해 가져옵니다.

따라서 ReAct의 Reasoning은 **내부 지식과 외부에서 새롭게 얻은 정보를 함께 이용할 수 있는 Reasoning**이라고 볼 수 있습니다.

## 3.3 ReAct와 CoT-SC를 결합한 이유

논문에서는 ReAct와 CoT를 비교한 결과 두 방법이 서로 다른 장단점을 가지고 있다고 설명합니다.

ReAct는 Wikipedia와 같은 External Knowledge Base에 접근할 수 있기 때문에 Reasoning 과정이 비교적 **Factual하고 Grounded**하다는 장점이 있습니다.

반면 CoT는 External Environment와 상호작용하지 않기 때문에 잘못된 사실을 생성하는 Hallucination이 발생할 수 있습니다.

하지만 ReAct가 항상 CoT보다 Reasoning을 잘하는 것은 아니었습니다.

ReAct에서는 Thought와 Action, Observation을 계속 연결해야 하기 때문에 Reasoning 과정 자체가 일정한 구조적 제약을 받습니다.

논문의 Error Analysis에서는 ReAct가 동일한 Thought와 Action을 반복하거나, 잘못된 Search 결과를 받은 뒤 기존 Reasoning을 수정하지 못하는 문제가 나타났습니다.

반대로 CoT는 외부 Action을 수행할 필요가 없기 때문에 Reasoning Structure를 비교적 자유롭게 구성할 수 있었습니다.

즉 두 방법에는 다음과 같은 Trade-off가 존재합니다.

```
CoT
장점: Reasoning Structure를 자유롭게 구성
단점: External Knowledge를 확인할 수 없어 Hallucination 발생 가능

ReAct
장점: External Knowledge를 이용하여 보다 Grounded한 Reasoning 가능
단점: Action과 Observation에 Reasoning이 묶이면서 Reasoning Flexibility 감소
```

논문의 저자들은 이 두 방법의 장점을 함께 이용하기 위해 **ReAct와 CoT-SC(Self-Consistency)를 결합**합니다.

CoT-SC는 하나의 CoT Reasoning만 생성하는 것이 아니라 여러 개의 CoT Trajectory를 Sampling한 뒤, 가장 많이 등장한 Answer를 최종 답으로 선택하는 방법입니다.

논문의 실험에서는 Temperature `0.7`로 총 21개의 CoT Trajectory를 Sampling한 뒤 Majority Answer를 사용합니다.

그리고 두 가지 전환 전략을 사용합니다.

### ReAct → CoT-SC

먼저 ReAct로 문제를 해결합니다.

하지만 일정한 Step 안에 ReAct가 Answer를 반환하지 못하면 CoT-SC로 전환합니다.

논문에서는 Maximum Step을 다음과 같이 설정했습니다.

- HotpotQA: 7 Steps
- FEVER: 5 Steps

즉 다음과 같은 구조입니다.

```
ReAct 수행
    ↓
외부 검색을 반복했지만 일정 Step 안에 답을 찾지 못함
    ↓
CoT-SC로 전환
    ↓
Internal Knowledge를 이용하여 Reasoning
```

### CoT-SC → ReAct

반대 전략도 사용합니다.

먼저 여러 개의 CoT Trajectory를 생성합니다.

그런데 Majority Answer가 충분히 강하게 나타나지 않는다면 모델의 Internal Knowledge만으로 문제에 자신 있게 답하기 어렵다고 판단합니다.

논문에서는 $n$개의 CoT-SC Sample 가운데 Majority Answer의 등장 횟수가 $n/2$보다 작은 경우 ReAct로 전환합니다.

즉,

```
CoT-SC 수행
    ↓
여러 Reasoning 결과의 Answer가 서로 일치하지 않음
    ↓
Internal Knowledge만으로 확신하기 어려움
    ↓
ReAct로 전환
    ↓
External Knowledge 검색
```

과 같은 구조입니다.

따라서 ReAct와 CoT-SC를 결합한 이유는 **ReAct의 Reasoning이 CoT와 완전히 다르기 때문이라기보다, 두 문제 해결 방식이 사용할 수 있는 정보와 Reasoning의 특성에서 서로 다른 장단점을 가지고 있기 때문**이라고 이해할 수 있습니다.

저자들은 이러한 결과를 통해 Internal Knowledge와 External Knowledge를 적절하게 결합하는 것이 Knowledge-intensive Reasoning Task에서 중요하다고 설명합니다.

## 3.4 Reasoning과 Acting은 어떻게 서로를 돕는가?

ReAct 논문의 제목은 **"Synergizing Reasoning and Acting in Language Models"**입니다.

따라서 ReAct의 핵심은 단순히 Reasoning과 Acting이라는 두 기능을 하나의 시스템에 넣었다는 것보다, **두 기능이 서로의 문제 해결을 돕도록 만들었다는 것**에 있습니다.

논문에서는 이를 크게 두 가지 방향으로 설명합니다.

### Reason to Act

첫 번째는 **Reasoning을 이용하여 더 좋은 Action을 선택하는 것**입니다.

Agent는 Thought를 통해,

- 현재 문제에서 무엇을 알아야 하는지
- 어떤 정보를 먼저 찾아야 하는지
- 어떤 Subgoal을 완료했는지
- 다음에는 어떤 Action을 수행해야 하는지
- 기존 Action이 실패했을 때 무엇을 해야 하는지

등을 판단할 수 있습니다.

예를 들어 다음과 같습니다.

```
Question:
Colorado orogeny의 eastern sector가 이어지는 지역의 elevation range는?

Thought:
먼저 Colorado orogeny를 검색해서 eastern sector가 어디까지 이어지는지 찾아야 한다.

Action:
Search[Colorado orogeny]
```

여기서는 Reasoning을 통해 `Search[Colorado orogeny]`라는 Action이 결정됩니다.

Observation을 확인한 이후에는,

```
Observation:
The Colorado orogeny was an episode of mountain building ...

Thought:
현재 결과에는 eastern sector에 대한 정보가 없다.
"eastern sector"를 Lookup해야 한다.

Action:
Lookup[eastern sector]
```

처럼 이전 Observation을 바탕으로 다음 Action을 수정합니다.

이것이 **Reason to Act**입니다.

### Act to Reason

두 번째는 반대로 **Action을 통해 Reasoning에 필요한 새로운 정보를 얻는 것**입니다.

LLM의 Internal Knowledge만으로 현재 문제를 해결하기 어렵다면 External Environment에 Action을 수행합니다.

```
Action:
Lookup[eastern sector]
```

Environment에서 다음 Observation을 얻었다고 가정합니다.

```
Observation:
The eastern sector extends into the High Plains.
```

그러면 이 정보가 다시 Reasoning에 사용됩니다.

```
Thought:
eastern sector는 High Plains까지 이어진다.
그렇다면 이제 High Plains의 elevation range를 찾아야 한다.
```

즉 Action이 새로운 Observation을 만들고, 이 Observation이 다음 Thought의 근거가 됩니다.

이것이 **Act to Reason**입니다.

따라서 ReAct의 전체 구조는 다음과 같은 Feedback Loop로 볼 수 있습니다.

```text
Reasoning
   ↓
어떤 Action을 해야 하는지 결정
   ↓
Action
   ↓
External Environment와 Interaction
   ↓
Observation
   ↓
새로운 정보를 Reasoning에 반영
   ↓
Reasoning
   ↓
...
```

즉, `Reason to Act` 와 `Act to Reason` 이 반복적으로 연결됩니다.

이러한 구조 때문에 ReAct에서 Thought와 Action은 서로 독립적인 두 기능이 아니라, **하나의 Task-solving Trajectory 안에서 서로의 입력을 만들어주는 관계**라고 볼 수 있습니다.

## 3.5 ReAct에서 말하는 Reasoning은 정말 CoT인가?

ReAct 논문을 처음 읽었을 때 가장 혼란스러웠던 부분이 바로 이것이었습니다.

ReAct에서 Thought를 생성한다면 그것은 결국 CoT를 사용하는 것이 아닌가 하는 의문이 들었습니다.

실제로 두 방법은 상당히 유사한 점이 있습니다.

CoT와 ReAct 모두 LLM이 최종 Answer를 바로 출력하지 않고 **자연어 형태의 Intermediate Reasoning Trace(중간 추론 과정)**를 생성하도록 합니다.

따라서 ReAct의 Thought를 **CoT-like Reasoning**이라고 이해하는 것은 어느 정도 타당합니다.

하지만 논문에서 정의하는 CoT Prompting과 ReAct의 Thought를 완전히 동일한 것으로 보기는 어렵습니다.

### CoT의 Reasoning

CoT에서는 Reasoning Trace 자체가 문제 해결 과정의 중심입니다.

```text
Question
→ Reasoning
→ Reasoning
→ Reasoning
→ Answer
```

Reasoning 중간에 External Environment에서 새로운 Observation이 들어오지 않습니다.

### ReAct의 Reasoning

ReAct의 Thought는 전체 Agent Trajectory의 일부입니다.

```text
Question
→ Thought
→ Action
→ Observation
→ Thought
→ Action
→ Observation
→ ...
```

따라서 ReAct의 Thought는 단순히 정답을 향해 Reasoning을 이어가는 역할뿐만 아니라,

- 어떤 Action을 수행할지 결정
- Observation의 의미를 해석
- 검색 결과가 부족한지 판단
- 다음 검색어를 결정
- 기존 계획을 수정

하는 역할까지 수행합니다.

즉 두 방법 모두 자연어 형태의 Reasoning Trace를 사용하지만 **Reasoning이 놓여 있는 Context와 역할이 다릅니다.**

이를 간단하게 표현하면,

```
CoT
Reasoning을 통해 Answer를 도출

ReAct
Reasoning을 통해 Action을 결정하고,
Action의 결과를 다시 Reasoning하면서 Answer를 도출
```

라고 볼 수 있습니다.

따라서 ReAct를 단순히

`ReAct = CoT + Tool`

이라고 정의하는 것은 ReAct의 핵심 구조를 지나치게 단순화할 수 있습니다.

보다 정확하게 표현하면, **ReAct는 CoT에서 보여준 자연어 Reasoning Trace의 아이디어를 Agent의 Action-Observation Trajectory에 결합하여, Reasoning과 Acting을 Interleave한 Prompt-based Paradigm** 이라고 보는 것이 적절하다고 생각합니다.

그리고 이것이 논문에서 ReAct와 CoT를 별도의 Method로 구분하여 비교할 수 있는 이유이기도 합니다.

# 4. Engineering & Implementation Details

ReAct 논문을 읽으면서 가장 많은 의문이 들었던 부분은 실제 구현 방식이었습니다.

논문에서는 ReAct를 **Prompt-based Paradigm**이라고 설명하고 있으며, 사람이 작성한 Thought, Action, Observation의 Trajectory를 Few-shot Example로 제공하면 LLM이 새로운 문제에서도 유사한 형태의 Trajectory를 생성한다고 설명합니다.

하지만 실제 동작을 생각해 보면 단순히 Prompt 하나를 LLM에 입력하는 것만으로는 ReAct를 구현할 수 없습니다.

예를 들어 LLM이,

```
Action: Search[Colorado orogeny]
```

를 생성했다고 하더라도 Language Model 자체가 Wikipedia API를 자동으로 호출하는 것은 아닙니다.

또한 Wikipedia에서 얻은 검색 결과인 Observation 역시 LLM이 생성하는 것이 아닙니다.

따라서 ReAct의 실제 동작을 이해하기 위해서는 크게 다음과 같은 구성 요소를 함께 생각할 필요가 있습니다.

```text
LLM
+
ReAct Prompt
+
Action을 실행할 Environment
+
Action 실행 결과를 다시 LLM에 전달하는 반복 구조
```

즉 ReAct에서 Prompt는 LLM에게 **어떻게 Reasoning하고 어떤 Action을 생성할 것인지**를 알려주는 역할을 하며, 실제 Action의 실행과 Observation의 전달은 LLM 외부의 Environment와 실행 구조를 통해 이루어집니다.

## 4.1 LLM은 어떻게 Wikipedia를 검색하는가?

HotpotQA와 FEVER 실험에서 ReAct는 외부 지식을 얻기 위한 Environment로 Wikipedia를 사용합니다.

논문에서는 이를 위해 매우 단순한 Wikipedia API를 구성하고 다음 세 가지 Action을 정의합니다.

### Search

```text
Search[entity]
```

특정 Entity의 Wikipedia Page를 검색합니다.

해당 Page가 존재하는 경우 Wikipedia 문서의 처음 5개 문장을 Observation으로 반환합니다.

예를 들어,

```
Action: Search[Colorado orogeny]
```

라는 Action을 실행하면,

```text
Observation:
The Colorado orogeny was an episode of mountain building ...
```

와 같은 결과를 얻을 수 있습니다.

만약 정확히 일치하는 Page가 존재하지 않는 경우에는 Wikipedia Search Engine에서 유사한 Entity 5개를 반환합니다.

### Lookup

```
Lookup[string]
```

현재 검색된 Wikipedia Page 안에서 특정 String이 포함된 다음 문장을 찾습니다. 논문에서는 이를 Browser의 `Ctrl + F` 기능과 유사한 기능이라고 설명합니다.

예를 들어,

```
Action: Lookup[eastern sector]
```

를 실행하면,

```
Observation:
The eastern sector extends into the High Plains ...
```

와 같은 결과를 얻을 수 있습니다.

### Finish

```
Finish[answer]
```

현재 Task를 종료하고 최종 Answer를 반환합니다.

예를 들어,

```text
Action: Finish[1,800 to 7,000 ft]
```

가 생성되면 Agent는 더 이상 Wikipedia를 검색하지 않고 해당 값을 최종 Answer로 사용합니다.

여기서 중요한 점은 **LLM이 Wikipedia API 자체를 실행하는 것은 아니라는 것**입니다.

LLM의 역할은 현재 Context를 바탕으로,

```
Search[Colorado orogeny]
```

와 같은 **정해진 형식의 Action을 텍스트로 생성하는 것**입니다.

실제 시스템을 구현한다면 LLM이 생성한 문자열을 프로그램에서 확인한 뒤 어떤 Action인지 판단하고 해당 함수를 실행해야 합니다.

예를 들어 개념적으로 다음과 같이 구현할 수 있습니다.

```python
if action.startswith("Search["):
    entity = parse_entity(action)
    observation = wikipedia.search(entity)

elif action.startswith("Lookup["):
    keyword = parse_keyword(action)
    observation = wikipedia.lookup(keyword)

elif action.startswith("Finish["):
    answer = parse_answer(action)
    return answer
```

위 코드는 ReAct 논문에 직접 제시된 구현 코드가 아니라, **논문에서 정의한 Action Space를 실제 프로그램 관점에서 풀어쓴 Pseudo Code**입니다.

즉 ReAct에서 LLM은, **"무엇을 검색할지 결정하는 역할"** 을 담당하고, 외부 프로그램은, **"LLM이 선택한 검색 Action을 실제로 실행하는 역할"** 을 담당한다고 볼 수 있습니다.

## 4.2 Observation은 누가 생성하는가?

ReAct를 이해할 때 Thought, Action, Observation의 생성 주체를 구분하는 것이 중요합니다.

HotpotQA를 기준으로 보면 다음과 같습니다.

```
Thought       → LLM이 생성
Action        → LLM이 생성
Observation   → Environment가 생성
```

예를 들어 LLM이 다음과 같이 생성했다고 가정해 보겠습니다.

```
Thought 1:
Colorado orogeny를 검색하여 eastern sector가 어디까지 이어지는지 찾아야 한다.

Action 1:
Search[Colorado orogeny]
```

여기까지는 LLM의 출력입니다.

하지만 그 다음에 등장하는,

```text
Observation 1:
The Colorado orogeny was an episode of mountain building ...
```

은 LLM이 만들어낸 문장이 아닙니다.

`Search[Colorado orogeny]`라는 Action을 Wikipedia Environment에서 실제로 실행한 결과입니다.

이 차이는 ReAct와 일반적인 CoT를 구분하는 데 중요합니다.

만약 Observation까지 LLM이 생성한다면,

```
Thought
→ Action
→ LLM이 상상한 Observation
→ Thought
```

이 되어 실제 External Environment와 상호작용한다고 보기 어렵습니다.

반면 ReAct에서는,

```
LLM
→ Action
→ External Environment
→ Observation
→ LLM
```

이라는 과정이 만들어집니다.

따라서 LLM이 잘못된 사실을 기억하고 있더라도 Search Action을 수행하여 External Environment로부터 새로운 정보를 얻을 수 있습니다.

물론 이는 반대로 **Environment가 잘못되거나 유용하지 않은 Observation을 반환하면 이후 Reasoning도 영향을 받을 수 있다는 것**을 의미합니다.

이 문제는 이후 Critical Review에서 다시 다루고자 합니다.

## 4.3 Thought와 Observation은 어떻게 다시 LLM에 전달되는가?

ReAct가 계속해서 Reasoning을 이어가기 위해서는 이전에 생성한 Thought와 Action, 그리고 Environment에서 얻은 Observation을 다음 LLM Inference에서도 사용할 수 있어야 합니다.

예를 들어 최초의 Context가 다음과 같다고 가정해 보겠습니다.

```
[ReAct Few-shot Examples]

Question:
Colorado orogeny의 eastern sector가 이어지는 지역의 elevation range는?
```

첫 번째 LLM Inference에서 다음 결과가 생성됩니다.

```
Thought 1:
먼저 Colorado orogeny를 검색해야 한다.

Action 1:
Search[Colorado orogeny]
```

Action을 실행한 뒤 다음 Observation을 얻습니다.

```
Observation 1:
The Colorado orogeny was an episode of mountain building ...
```

그러면 다음 LLM Inference에서는 새로운 Question만 입력하는 것이 아니라 지금까지의 Trajectory를 Context에 포함합니다.

```
[ReAct Few-shot Examples]

Question:
Colorado orogeny의 eastern sector가 이어지는 지역의 elevation range는?

Thought 1:
먼저 Colorado orogeny를 검색해야 한다.

Action 1:
Search[Colorado orogeny]

Observation 1:
The Colorado orogeny was an episode of mountain building ...
```

이 Context를 바탕으로 LLM이 다시 다음 Thought와 Action을 생성합니다.

```
Thought 2:
현재 검색 결과에는 eastern sector에 대한 정보가 없다.
eastern sector를 Lookup해야 한다.

Action 2:
Lookup[eastern sector]
```

다시 Environment에서 Observation을 얻습니다.

```
Observation 2:
The eastern sector extends into the High Plains ...
```

그러면 다음 Context는 더욱 길어집니다.

```
Question
+ Thought 1
+ Action 1
+ Observation 1
+ Thought 2
+ Action 2
+ Observation 2
```

즉 ReAct에서는 **지금까지 만들어진 Task-solving Trajectory가 다음 LLM Inference의 Context로 계속 누적**됩니다.

Section 2에서 살펴본 다음 수식이 바로 이러한 구조를 나타냅니다.

$$
c_t = (o_1, a_1, \dots, o_{t-1}, a_{t-1}, o_t)
$$

그리고 Thought 역시 이후 Reasoning이나 Acting에 사용할 수 있도록 Context에 추가됩니다.

$$
c_{t+1} = (c_t, \hat{a}_t)
$$

여기서 $\hat{a}_t \in L$은 Language Space에 속하는 Thought를 의미합니다.

따라서 ReAct의 다음 Action은 단순히 최초의 Question만 보고 결정되는 것이 아닙니다.

지금까지의,

- Question
- Thought
- Action
- Observation

전체 History가 다음 Action을 결정하기 위한 Context가 됩니다.

## 4.4 ReAct의 반복은 누가 제어하는가?

처음 ReAct 논문을 읽었을 때에는 LLM이 스스로,

```
Thought
→ Action
→ Observation
→ Thought
→ Action
→ Observation
```

을 반복하는 것처럼 이해했습니다.

하지만 엄밀하게 보면 LLM이 자기 자신을 다시 호출하는 것은 아닙니다.

LLM은 호출될 때마다 현재 Context를 입력받아 다음 Thought 또는 Action을 생성합니다.

그 이후,

1. LLM의 Action을 확인하고
2. 해당 Action을 Environment에서 실행하고
3. Observation을 얻고
4. Observation을 기존 Context에 추가한 뒤
5. 다시 LLM을 호출하는

반복 구조가 필요합니다.

이를 프로그램 관점에서 단순화하면 다음과 같이 표현할 수 있습니다.

```python
context = few_shot_examples + question

while True:

    # 현재 Context를 바탕으로 LLM 호출
    output = LLM(context)

    # LLM이 생성한 Thought와 Action 추출
    thought, action = parse(output)

    context += thought
    context += action

    # 종료 Action
    if is_finish(action):
        return get_answer(action)

    # Action을 외부 Environment에서 실행
    observation = environment.execute(action)

    # 실행 결과를 다음 Context에 추가
    context += observation
```

다시 말하지만 위의 코드는 논문에서 제공한 실제 Source Code가 아니라 **논문에서 설명한 ReAct의 동작을 프로그램의 Control Flow 관점에서 표현한 Pseudo Code**입니다.

이 관점으로 보면 ReAct는 다음과 같은 Closed-loop 구조를 가집니다.

```
┌───────────────────────┐
│          LLM          │
│ Thought / Action 생성  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│    Action Executor    │
│     Action 해석/실행   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│      Environment      │
│    Observation 반환   │
└───────────┬───────────┘
            │
            ▼
      Context에 추가
            │
            └───────────────→ LLM 재호출
```

따라서 ReAct의 반복을 실제로 제어하는 것은 LLM 자체라기보다 **LLM과 Environment 사이를 연결하는 외부 실행 구조**라고 이해하는 것이 적절합니다.

다만 ReAct 논문에서 이 실행 구조 자체를 새로운 Agent Framework로 제안한 것은 아닙니다.

논문의 주된 관심은 이러한 Agent-Environment Interaction이 존재한다는 전제 아래, LLM이 단순히 Action만 생성하는 것이 아니라 **Thought와 Action을 함께 생성하도록 Prompting하는 것**에 있습니다.

## 4.5 LLM은 언제 Thought를 생성할지 어떻게 결정하는가?

ReAct에서는 모든 Action 앞에서 반드시 Thought가 생성되는 것은 아닙니다.

이 부분은 Task의 성격에 따라 다릅니다.

### Reasoning 중심 Task

HotpotQA와 FEVER와 같이 Reasoning 자체가 중요한 Task에서는 저자들이 **Dense Thought** 방식을 사용합니다.

대체로 다음 구조를 반복합니다.

```
Thought 1
→ Action 1
→ Observation 1

Thought 2
→ Action 2
→ Observation 2

Thought 3
→ Action 3
→ ...
```

즉 Reasoning Task에서는 Thought와 Action이 거의 번갈아 나타납니다.

HotpotQA의 실제 ReAct Example 역시 다음과 같은 구조입니다.

```
Thought 1:
Colorado orogeny를 검색해야 한다.

Action 1:
Search[Colorado orogeny]

Observation 1:
...

Thought 2:
현재 결과에는 eastern sector가 없다.
Lookup해야 한다.

Action 2:
Lookup[eastern sector]

Observation 2:
...
```

따라서 Knowledge-intensive Reasoning Task에서는 Thought 발생 시점의 자유도가 상대적으로 크지 않습니다.

### Decision Making 중심 Task

반면 ALFWorld와 같이 많은 수의 Action을 수행해야 하는 Task에서는 매 Action마다 Thought를 생성할 필요가 없습니다.

예를 들어 다음과 같은 Trajectory가 가능합니다.

```
Thought:
먼저 lettuce를 찾아야 한다.
fridge부터 확인해보자.

Action:
go to fridge 1

Observation:
The fridge 1 is closed.

Action:
open fridge 1

Observation:
The fridge is open ...

Action:
go to diningtable 1

Observation:
On the diningtable ... lettuce 1 ...

Thought:
lettuce를 찾았다.
이제 lettuce를 집어야 한다.

Action:
take lettuce 1 from diningtable 1
```

여기서는,

```
Action
→ Observation
→ Action
→ Observation
→ Action
→ Observation
```

이 여러 번 이어진 뒤 필요한 시점에서만 Thought가 등장합니다.

논문에서는 이를 **Sparse Thought**라고 볼 수 있으며, 이런 Decision Making Task에서는 Thought와 Action이 발생하는 시점을 미리 엄격하게 고정하지 않고 Language Model이 현재 Context를 기반으로 결정하도록 합니다.

그렇다면 LLM은 이것을 어떻게 배우는가라는 의문이 생깁니다.

ReAct에서는 사람이 작성한 Few-shot Trajectory 자체가 이러한 Pattern을 보여줍니다.

예를 들어 Demonstration에,

```
think: 먼저 lettuce를 찾아야 한다.
go to fridge 1
open fridge 1
go to diningtable 1
think: lettuce를 찾았다. 이제 집어야 한다.
take lettuce 1 ...
```

와 같은 Pattern을 보여주면 LLM은 새로운 Task에서도 현재 Context에 따라 다음에,

```
think:
```

를 생성할지,

```
go to ...
```

와 같은 Environment Action을 생성할지를 결정합니다.

즉 별도의 Classifier가,

```
지금은 Thought를 생성해야 함
```

이라고 판단해 주는 구조가 논문에서 제안된 것은 아닙니다.

Few-shot Example을 통해 **Thought 역시 선택 가능한 Language Action 중 하나라는 Pattern을 학습하게 하는 방식**에 가깝습니다.

## 4.6 ReAct의 종료 조건

ReAct는 Thought와 Action을 무한정 반복하는 것이 아니라 Task별 종료 조건을 가집니다.

### HotpotQA / FEVER

Knowledge-intensive Reasoning Task에서는 다음 Action이 생성되면 Task가 종료됩니다.

```
Finish[answer]
```

예를 들어,

```
Thought:
High Plains의 elevation range는 1,800 ~ 7,000 ft이다.
따라서 정답은 1,800 ~ 7,000 ft이다.

Action:
Finish[1,800 to 7,000 ft]
```

가 생성되면 해당 Answer를 최종 결과로 사용합니다.

또한 논문에서 ReAct와 CoT-SC를 결합한 실험에서는 ReAct가 일정 Step 안에 Answer를 반환하지 못할 경우 CoT-SC로 전환하도록 합니다.

Maximum Step은 다음과 같습니다.

- HotpotQA: 7 Steps
- FEVER: 5 Steps

즉 해당 실험에서는 다음과 같은 추가적인 종료 또는 전환 조건이 존재합니다.

```
Finish[answer] 생성
→ ReAct 정상 종료

또는

Maximum Step 도달
→ ReAct 중단
→ CoT-SC로 전환
```

### ALFWorld

ALFWorld에서는 특정 Answer 문자열을 생성하는 것이 목적이 아니라 Environment에서 실제 Goal State를 달성하는 것이 목적입니다.

예를 들어,

```
put a clean lettuce in diningtable
```

이라는 Task라면,

1. lettuce를 찾고
2. lettuce를 집고
3. sinkbasin에서 씻고
4. diningtable에 놓으면

Environment의 Goal이 달성되며 Episode가 종료됩니다.

### WebShop

WebShop에서는 사용자의 요구조건을 만족하는 상품을 선택한 뒤,

```
Buy Now
```

Action을 수행하여 구매가 완료되면 Episode가 종료됩니다.

따라서 ReAct에는 모든 Task에 공통으로 적용되는 하나의 종료 문자열이 존재하는 것이 아니라, **Agent가 연결된 Environment의 Task 정의에 따라 종료 조건도 달라집니다.**

결국 ReAct의 실행 과정을 가장 단순하게 정리하면 다음과 같습니다.

```
1. 최초의 prompt(Context)를 LLM에 입력
2. LLM이 Thought 또는 Action 생성
3. Environment Action이면 실제로 실행
4. Observation을 Context에 추가
5. 새로운 Context로 LLM 재호출
6. Task별 종료 조건을 만족할 때까지 반복
```

이 구조를 이해하고 나면 ReAct를 단순히, **"Thought → Action → Observation이라는 문장을 Prompt에 넣는 방법"** 으로 보는 것보다, **"LLM이 Reasoning을 통해 Action을 선택하고, Environment에서 얻은 Observation을 다시 Reasoning에 반영할 수 있도록 Prompt와 Agent-Environment Interaction을 연결한 구조"**로 이해하는 것이 더 정확하다고 생각합니다.

# 5. Experiments & Results

ReAct 논문에서는 Reasoning과 Acting을 결합하는 것이 실제로 효과가 있는지를 확인하기 위해 크게 두 종류의 Task에서 실험을 진행합니다.

첫 번째는 **Knowledge-Intensive Reasoning Task**입니다. 질문에 답하거나 주어진 Claim이 사실인지 판단하기 위해 외부 지식을 검색해야 하는 Task로, HotpotQA와 FEVER를 사용합니다.

두 번째는 **Interactive Decision Making Task**입니다. Agent가 긴 시간 동안 여러 Action을 수행하면서 목표를 달성해야 하는 Task로, ALFWorld와 WebShop을 사용합니다.

이 두 종류의 실험은 각각 ReAct의 서로 다른 방향을 검증한다고 볼 수 있습니다.

- Knowledge-Intensive Reasoning: **Acting이 Reasoning을 얼마나 도와주는가?**
- Interactive Decision Making: **Reasoning이 Acting을 얼마나 도와주는가?**

즉 논문 제목인 `Synergizing Reasoning and Acting`에서 말하는 두 방향의 효과를 각각 실험으로 확인합니다.

## 5.1 Knowledge-Intensive Reasoning: HotpotQA & FEVER

Knowledge-Intensive Reasoning 실험에서는 **HotpotQA**와 **FEVER** 두 Dataset을 사용합니다.

### HotpotQA

HotpotQA는 하나의 Wikipedia 문서만 확인해서 정답을 찾는 것이 아니라, **두 개 이상의 Wikipedia Passage에 있는 정보를 연결하여 Reasoning해야 하는 Multi-hop Question Answering Benchmark**입니다.

예를 들어 다음과 같은 질문이 있다고 가정할 수 있습니다.

```text
Colorado orogeny의 eastern sector가 이어지는 지역의 elevation range는 얼마인가?
```

이 문제를 풀기 위해서는 단순히 `Colorado orogeny`에 대한 정보만 알아서는 안 됩니다.

먼저,

```
Colorado orogeny
→ eastern sector
→ High Plains
```

라는 관계를 찾아야 하고, 이후 다시 High Plains에 대한 정보를 검색하여 elevation range를 찾아야 합니다.

즉,

```
Question
→ 첫 번째 정보 검색
→ 검색 결과를 이용한 Reasoning
→ 두 번째 정보 검색
→ 최종 Answer
```

과 같은 Multi-hop Reasoning이 필요합니다.

평가 지표로는 **Exact Match(EM)**를 사용합니다.

이는 모델이 생성한 최종 Answer가 정답과 정확하게 일치하는지를 기준으로 평가합니다.

### FEVER

FEVER는 **Fact Verification(사실 검증)** Task입니다.

하나의 Claim이 주어지면 Wikipedia의 정보를 기준으로 다음 세 가지 Label 중 하나를 선택합니다.

- `SUPPORTS`: Claim을 뒷받침할 수 있음
- `REFUTES`: Claim이 사실이 아님
- `NOT ENOUGH INFO`: 판단할 충분한 정보가 없음

예를 들어,

```
Claim:
Stranger Things is set in Bloomington, Indiana.
```

라는 Claim이 주어진다면 Wikipedia에서 Stranger Things를 검색하여 실제 배경이 어디인지 확인해야 합니다.

검색 결과가,

```
Set primarily in the fictional town of Hawkins, Indiana ...
```

라면 Bloomington이 아니라 Hawkins이므로,

```
REFUTES
```

를 선택할 수 있습니다.

FEVER의 평가 지표는 **Accuracy(Acc)**입니다.

### Question-only Setting

중요한 점은 두 Task 모두 **Question-only Setting**으로 실험한다는 것입니다.

즉 모델에게 Supporting Paragraph를 미리 제공하지 않습니다.

모델은 오직 Question 또는 Claim만 입력받으며,

1. LLM 내부의 Internal Knowledge를 사용하거나
2. Wikipedia Environment와 상호작용하여 필요한 정보를 직접 검색해야 합니다.

이를 통해 External Knowledge를 가져올 수 있는 ReAct가 Reasoning Task에서 어떤 영향을 주는지를 확인합니다.

## 5.2 Standard / CoT / CoT-SC / Act / ReAct 비교

ReAct 논문에서는 ReAct의 어떤 요소가 성능에 영향을 주는지 확인하기 위해 동일한 ReAct Trajectory에서 특정 요소를 제거하는 방식으로 Baseline을 구성합니다.

### Standard Prompting

Standard Prompting에서는 ReAct Trajectory의,

- Thought
- Action
- Observation

을 모두 제거합니다.

즉,

```
Question
→ Answer
```

형태의 가장 기본적인 Prompting입니다.

### CoT

Chain-of-Thought(CoT)는 ReAct Trajectory에서 **Action과 Observation을 제거하고 Thought만 남긴 방식**입니다.

```text
Question
→ Thought
→ Answer
```

따라서 논문에서는 CoT를 **Reasoning-only Baseline**으로 사용합니다.

### CoT-SC

CoT-SC는 **Chain-of-Thought with Self-Consistency**입니다.

하나의 CoT 결과만 사용하는 것이 아니라, Temperature `0.7`을 이용하여 **21개의 CoT Trajectory를 Sampling**하고 가장 많이 등장한 Answer를 최종 Answer로 선택합니다.

개념적으로는 다음과 같습니다.

```text
Question
 ├─ CoT 1 → Answer A
 ├─ CoT 2 → Answer B
 ├─ CoT 3 → Answer A
 ├─ ...
 └─ CoT 21 → Answer A

Majority Answer → A
```

### Act

Act는 ReAct Trajectory에서 **Thought만 제거한 방식**입니다.

```text
Question
→ Action
→ Observation
→ Action
→ Observation
→ ...
→ Answer
```

즉 External Environment와 상호작용할 수는 있지만 명시적인 Reasoning Trace는 사용하지 않습니다.

### ReAct

ReAct는 모든 요소를 사용합니다.

```text
Question
→ Thought
→ Action
→ Observation
→ Thought
→ Action
→ Observation
→ ...
```

따라서 논문의 Baseline 구성을 간단하게 정리하면 다음과 같습니다.

| Method | Thought | Action | Observation |
| --- | :---: | :---: | :---: |
| Standard | X | X | X |
| CoT | O | X | X |
| Act | X | O | O |
| ReAct | O | O | O |

이러한 Ablation 방식은 **Reasoning만 사용할 때, Acting만 사용할 때, 그리고 두 능력을 결합할 때 어떤 차이가 발생하는지**를 비교하기 위한 것입니다.

## 5.3 ReAct vs. CoT 실험 결과

PaLM-540B를 사용한 HotpotQA와 FEVER의 주요 실험 결과는 다음과 같습니다.

| Prompt Method | HotpotQA EM | FEVER Acc |
| --- | ---: | ---: |
| Standard | 28.7 | 57.1 |
| CoT | 29.4 | 56.3 |
| CoT-SC | 33.4 | 60.4 |
| Act | 25.7 | 58.9 |
| ReAct | 27.4 | 60.9 |
| CoT-SC → ReAct | 34.2 | **64.6** |
| ReAct → CoT-SC | **35.1** | 62.0 |
| Supervised SoTA | 67.5 | 89.5 |

가장 먼저 확인할 수 있는 것은 **ReAct가 Act보다 두 Task 모두에서 높은 성능을 기록했다는 것**입니다.

HotpotQA에서는,

```
Act   : 25.7
ReAct : 27.4
```

FEVER에서는,

```
Act   : 58.9
ReAct : 60.9
```

였습니다.

Act와 ReAct의 가장 큰 차이는 Thought의 존재이기 때문에 저자들은 이를 **Reasoning이 Acting을 Guide하는 것이 효과가 있다는 결과**로 해석합니다.

하지만 ReAct와 CoT의 결과를 비교하면 조금 다른 이야기가 나옵니다.

### HotpotQA

```text
CoT   : 29.4
ReAct : 27.4
```

오히려 **CoT가 ReAct보다 높은 성능**을 보입니다.

### FEVER

```text
CoT   : 56.3
ReAct : 60.9
```

반대로 FEVER에서는 **ReAct가 CoT보다 높은 성능**을 보입니다.

따라서 이 실험 결과를 단순히, **"Reasoning과 Acting을 결합한 ReAct가 CoT보다 항상 뛰어나다."** 라고 해석하는 것은 잘못된 해석입니다.

저자들은 FEVER의 경우 SUPPORTS와 REFUTES가 작은 사실 차이에 의해 결정되는 경우가 많기 때문에 **정확하고 최신의 외부 정보를 직접 검색하는 능력이 특히 중요하다**고 설명합니다.

반면 HotpotQA에서는 외부 검색을 추가했다고 해서 반드시 더 높은 성능을 보이지는 않았습니다.

이 차이를 분석하기 위해 논문에서는 ReAct와 CoT의 실제 Trajectory에 대한 Error Analysis를 진행합니다.

## 5.4 Error Analysis: Hallucination vs. Reasoning Flexibility

저자들은 HotpotQA에서 ReAct와 CoT의 행동 차이를 분석하기 위해 각 방법에서,

- 정답 Trajectory 50개
- 오답 Trajectory 50개

를 무작위로 추출합니다.

따라서 ReAct 100개와 CoT 100개, 총 **200개의 Trajectory를 사람이 직접 분석**합니다.

분석 결과는 다음과 같습니다.

| Type | Definition | ReAct | CoT |
| --- | --- | ---: | ---: |
| Success - True Positive | Reasoning과 Fact가 모두 정확함 | 94% | 86% |
| Success - False Positive | Hallucinated Reasoning 또는 Fact 포함 | 6% | 14% |
| Failure - Reasoning Error | 잘못된 Reasoning 또는 반복 Step에서 회복하지 못함 | 47% | 16% |
| Failure - Search Result Error | 검색 결과가 없거나 유용하지 않음 | 23% | - |
| Failure - Hallucination | Hallucinated Reasoning 또는 Fact | 0% | 56% |
| Failure - Label Ambiguity | Prediction은 맞지만 Label과 정확히 일치하지 않음 | 29% | 28% |

여기서 주의해야 할 점은 위 비율이 전체 Dataset에 대한 Error Rate가 아니라, **사람이 따로 Sampling하여 분석한 성공/실패 Trajectory 내부의 구성 비율**이라는 점입니다.

이 Error Analysis에서는 ReAct와 CoT 사이에 매우 흥미로운 Trade-off가 나타납니다.

### CoT: Hallucination 문제

CoT가 실패한 Trajectory에서는 Hallucination이 가장 큰 문제로 나타났습니다.

분석된 CoT Failure 가운데 **56%가 Hallucinated Reasoning Trace 또는 Fact와 관련되어 있었습니다.**

CoT는 외부 Environment를 통해 자신이 사용하는 사실을 검증하지 않기 때문에 Reasoning 초기에 잘못된 Fact가 생성되면 이후 Reasoning 역시 그 Fact에 영향을 받을 수 있습니다.

```
잘못된 Fact
→ 그 Fact를 이용한 Reasoning
→ 추가 Reasoning
→ 잘못된 Answer
```

즉 Section 1에서 설명한 **Error Propagation**이 실제 Error Analysis에서도 확인됩니다.

### ReAct: Reasoning Error 문제

반면 ReAct에서는 분석된 Failure 가운데 **Reasoning Error가 47%**를 차지합니다.

ReAct는,

```
Thought
→ Action
→ Observation
→ Thought
→ Action
```

이라는 구조를 계속 유지해야 하기 때문에 CoT보다 Reasoning을 구성하는 방식에 제약이 생깁니다.

특히 논문에서는 ReAct가 이전 Thought와 Action을 반복하는 문제가 자주 나타났다고 설명합니다.

예를 들어,

```
Thought:
A를 검색해야 한다.

Action:
Search[A]

Observation:
유용한 정보 없음

Thought:
A를 검색해야 한다.

Action:
Search[A]

...
```

와 같이 Loop에서 빠져나오지 못할 수 있습니다.

저자들은 이러한 현상을 **ReAct의 Structural Constraint가 Reasoning Flexibility를 감소시키는 문제**로 해석합니다.

### Search Result Error

또 다른 ReAct의 특징적인 Failure는 **Search Result Error**입니다.

분석된 ReAct Failure 중 23%에서 Search가 비어 있거나 문제 해결에 유용하지 않은 정보를 반환했습니다.

중요한 것은 단순히 검색 한 번이 실패하는 것으로 끝나는 것이 아니라는 점입니다.

ReAct에서는 Observation이 다음 Thought의 Context에 포함되므로,

```
유용하지 않은 Search Result
→ 잘못된 방향의 Thought
→ 잘못된 Action
→ 추가적인 잘못된 Observation
```

처럼 이후 Reasoning까지 잘못된 방향으로 흘러갈 수 있습니다.

따라서 ReAct가 External Knowledge를 사용한다는 것은 분명 Hallucination을 줄이는 데 도움을 주지만, 반대로 **External Retrieval의 품질에 영향을 받는 새로운 Failure Mode**를 만들어냅니다.

논문의 결과를 정리하면 다음과 같은 Trade-off를 확인할 수 있습니다.

```
CoT
→ 높은 Reasoning Flexibility
→ 그러나 Hallucination에 취약

ReAct
→ 높은 Groundedness / Factuality
→ 그러나 Reasoning의 구조적 제약과 Search Failure에 취약
```

이러한 결과는 저자들이 ReAct와 CoT를 다시 결합하게 되는 중요한 근거가 됩니다.

## 5.5 ReAct + CoT-SC

ReAct와 CoT가 서로 다른 장단점을 보였기 때문에 논문의 저자들은 두 방법을 결합합니다.

실험 결과 가장 높은 Prompting 성능은 다음과 같습니다.

### HotpotQA

```text
ReAct → CoT-SC : 35.1
```

### FEVER

```text
CoT-SC → ReAct : 64.6
```

즉 어느 하나의 방법이 모든 Task에서 가장 좋은 것이 아니라, **두 방법을 조건에 따라 전환했을 때 가장 높은 성능**을 보였습니다.

ReAct → CoT-SC에서는 ReAct가 정해진 Step 안에 Answer를 찾지 못하면 CoT-SC로 전환합니다.

반대로 CoT-SC → ReAct에서는 여러 CoT Sample의 Answer가 충분히 일치하지 않으면 Internal Knowledge만으로 문제를 확신하기 어렵다고 판단하고 ReAct로 전환하여 External Knowledge를 검색합니다.

특히 논문에서는 두 Hybrid Method가 다양한 CoT-SC Sample 수에서도 CoT-SC 단독보다 지속적으로 높은 결과를 보였으며, **CoT-SC가 21개의 Sample을 사용할 때 얻는 성능 수준을 3~5개의 Sample만으로도 달성할 수 있었다**고 설명합니다.

저자들은 이를 **Internal Knowledge와 External Knowledge를 적절하게 결합하는 것이 Reasoning Task에 도움이 된다는 결과**로 해석합니다.

다만 ReAct와 CoT를 이렇게 별도의 Method로 구분한 뒤 다시 결합하는 방식에는 개인적으로 몇 가지 의문이 들었으며, 이 부분은 이후 Critical Review에서 자세히 다루고자 합니다.

## 5.6 Decision Making: ALFWorld

Knowledge-intensive Reasoning 실험이 **Acting이 Reasoning에 도움을 주는지**를 주로 살펴봤다면, ALFWorld와 WebShop에서는 반대로 **Reasoning이 Acting에 어떤 도움을 주는지**를 확인합니다.

ALFWorld는 가상의 Household Environment에서 Agent가 Text Action을 이용하여 특정 목표를 달성해야 하는 Text-based Game입니다.

예를 들어 다음과 같은 Task가 주어질 수 있습니다.

```
examine paper under desklamp
```

Agent는 Environment를 탐색하면서,

```
go to coffeetable 1
take paper 2
use desklamp 1
```

같은 Action을 수행해야 합니다.

ALFWorld의 문제는 하나의 Task에 50개가 넘는 Location이 존재할 수 있고, Expert Policy조차 50 Step 이상을 수행해야 하는 경우가 있다는 것입니다.

따라서 단순히 현재 Observation을 보고 다음 Action 하나를 선택하는 것만으로는 문제를 해결하기 어렵습니다.

Agent는,

- 전체 Goal을 Subgoal로 분해하고
- 어떤 Subgoal이 완료되었는지 기억하고
- 다음 Subgoal을 결정하고
- 물체가 어디에 있을 가능성이 높은지 Commonsense Reasoning을 수행해야 합니다.

ReAct에서는 이를 위해 **Sparse Thought**를 사용합니다.

ALFWorld 실험 결과는 다음과 같습니다.

| Method | Overall Success Rate |
| --- | ---: |
| Act (best of 6) | 45% |
| ReAct (average) | 57% |
| **ReAct (best of 6)** | **71%** |
| ReAct-IM (average) | 48% |
| ReAct-IM (best of 6) | 53% |
| BUTLERg (best of 8) | 22% |
| BUTLER (best of 8) | 37% |

가장 좋은 ReAct Trial은 **71%의 Success Rate**를 기록했습니다.

이는,

```
Act     : 45%
BUTLER  : 37%
ReAct   : 71%
```

로 상당한 차이를 보입니다.

논문에서는 심지어 가장 낮은 ReAct Trial의 Success Rate인 48%도 Act와 BUTLER의 Best Trial보다 높았다고 설명합니다.

또한 동일한 조건으로 비교한 6개의 ReAct와 Act Trial 모두에서 ReAct가 더 높은 성능을 보였습니다.

저자들이 실제 Trajectory를 분석한 결과 Thought가 없는 Act에서는 주로,

- Goal을 올바르게 Subgoal로 분해하지 못하거나
- 현재 Environment State를 놓치는

문제가 나타났습니다.

즉 ALFWorld 실험은 단순히 Action을 생성하는 것보다 **Action 사이에서 현재 상태와 앞으로 해야 할 일을 자연어로 Reasoning하는 것이 Long-horizon Decision Making에 도움이 될 수 있음**을 보여줍니다.

### ReAct-IM

논문에서는 추가적으로 ReAct와 Inner Monologue 스타일의 ReAct-IM을 비교합니다.

ReAct-IM은 현재 Environment State나 현재 해야 할 Subgoal과 같은 External Feedback 중심의 Thought를 보다 촘촘하게 사용합니다.

결과는,

```
ReAct best    : 71%
ReAct-IM best : 53%
```

였습니다.

저자들은 ReAct-IM이,

- Subgoal이 언제 끝났는지 판단
- 다음 Subgoal 선택
- 물체가 어디에 있을지 Commonsense Reasoning

하는 부분에서 어려움을 보였다고 설명합니다.

즉 단순히 Observation을 자연어로 반복하거나 현재 해야 할 행동을 계속 기록하는 것보다, **고수준의 Goal Decomposition과 Commonsense Reasoning을 포함하는 유연한 Thought가 중요하다는 것**을 보여주기 위한 Ablation Experiment입니다.

## 5.7 WebShop

WebShop은 ALFWorld보다 실제 환경에 가까운 Web Interaction Task입니다.

WebShop에는 약 **118만 개의 실제 상품**과 약 **12,000개의 사람이 작성한 Instruction**이 존재합니다.

예를 들어 사용자가 다음과 같이 요청할 수 있습니다.

```
서랍이 달린 nightstand를 원한다.
nickel finish여야 하고 가격은 140달러 미만이어야 한다.
```

Agent는 이 요구사항을 만족하는 상품을 찾기 위해,

- 검색어 입력
- 상품 선택
- 상품 Option 선택
- 이전 페이지 이동
- 구매

등의 Action을 수행해야 합니다.

WebShop의 평가 지표는 두 가지입니다.

### Score

선택한 상품이 사용자가 요구한 Attribute를 얼마나 만족하는지를 나타냅니다.

### Success Rate

선택한 상품이 사용자의 **모든 요구사항을 만족한 Episode의 비율**입니다.

실험 결과는 다음과 같습니다.

| Method | Score | Success Rate |
| --- | ---: | ---: |
| Act | 62.3 | 30.1% |
| **ReAct** | **66.6** | **40.0%** |
| IL | 59.9 | 29.1% |
| IL + RL | 62.4 | 28.7% |
| Human Expert | 82.1 | 59.6% |

Thought가 없는 Act만 사용해도 기존 IL과 IL+RL 방식과 유사한 수준의 성능을 보였습니다.

하지만 여기에 Sparse Reasoning을 추가한 ReAct는,

```
Act   : 30.1%
ReAct : 40.0%
```

의 Success Rate를 기록합니다.

저자들은 ReAct가 Noisy한 Observation과 실제 Action 사이에서 Reasoning을 수행하면서 **Instruction과 관련 있는 Product와 Option을 더 잘 선택할 수 있었기 때문**이라고 분석합니다.

예를 들어 상품을 확인한 후,

```
이 상품은 bright citrus 옵션과
3 ounce 옵션을 모두 가지고 있으므로 구매하기 적절하다.
```

와 같이 사용자의 요구조건과 현재 상품의 Attribute를 직접 비교한 뒤 구매 여부를 결정할 수 있습니다.

그러나 ReAct 역시 Human Expert의,

```
Score        : 82.1
Success Rate : 59.6%
```

에는 상당히 미치지 못했습니다.

논문에서는 Human Expert가 Agent보다 더 많은 상품을 탐색하고 Search Query를 여러 번 수정하는 경향이 있었으며, 이러한 행동은 Prompting 기반 방식에는 여전히 어려운 문제라고 설명합니다.

따라서 WebShop 결과는 ReAct가 Interactive Decision Making에서 효과를 보이기는 하지만 **실제 인간 수준의 탐색 능력에 도달한 것은 아니라는 점**도 함께 보여줍니다.

## 5.8 Fine-tuning ReAct

ReAct 논문의 주요 실험은 Few-shot Prompting을 중심으로 진행되지만, 저자들은 ReAct를 Fine-tuning했을 때의 가능성도 실험합니다.

문제는 사람이 Thought와 Action을 포함한 ReAct Trajectory를 대규모로 직접 작성하는 것이 매우 어렵다는 점입니다.

이를 해결하기 위해 저자들은 LLM에 사람이 직접 작성한 Few-shot ReAct Prompting을 적용하여 LLM 모델이 생성한 Trajectory 가운데 **최종 Answer가 정답인 3,000개의 Trajectory**를 이용하여 더 작은 모델인,

- PaLM-8B
- PaLM-62B

를 Fine-tuning합니다.

Fine-tuning 시 모델은 Question 또는 Claim을 입력받아 전체 Trajectory를 생성하도록 학습합니다.

즉,

```text
Question
→ Thought
→ Action
→ Observation
→ ...
```

의 형태를 학습하게 됩니다.

흥미로운 점은 Prompting과 Fine-tuning에서 결과가 크게 달랐다는 것입니다.

### Prompting

PaLM-8B와 PaLM-62B와 같이 상대적으로 작은 모델에서는 ReAct Prompting이 Standard, CoT, Act, ReAct 가운데 **가장 낮은 성능**을 보였습니다.

저자들은 작은 모델이 몇 개의 In-context Example만 보고,

```
Reasoning
+
Acting
```

두 가지 능력을 동시에 학습하는 것이 어렵기 때문이라고 설명합니다.

### Fine-tuning

하지만 3,000개의 Trajectory로 Fine-tuning한 이후에는 ReAct가 네 가지 방법 가운데 가장 좋은 결과를 보였습니다.

논문에서는,

- **PaLM-8B Fine-tuned ReAct**가 모든 PaLM-62B Prompting Method보다 높은 성능
- **PaLM-62B Fine-tuned ReAct**가 모든 PaLM-540B Prompting Method보다 높은 성능

을 보였다고 보고합니다.

저자들은 Standard나 CoT를 Fine-tuning하는 경우 모델에게 특정 Knowledge Fact를 직접 기억하도록 가르치는 성격이 강한 반면, Act와 ReAct는,

```
Wikipedia에서 필요한 정보를 어떻게 얻을 것인가
```

를 학습하게 된다고 해석합니다. 특히 ReAct는 여기에,

```
어떻게 Reasoning하면서 정보를 찾을 것인가
```

까지 함께 학습합니다.

저자들은 이러한 능력이 특정 Fact를 암기하는 것보다 **Knowledge Reasoning에서 더 Generalizable한 Skill**일 가능성이 있다고 설명합니다.

다만 논문의 모든 Prompting Method는 여전히 Domain-specific Supervised State-of-the-art와 큰 성능 차이가 존재했습니다.

예를 들어 Table 1에서 Supervised SoTA는,

```
HotpotQA : 67.5
FEVER    : 89.5
```

인 반면 ReAct를 포함한 Prompting Method들은 이에 크게 미치지 못합니다.

따라서 저자들은 ReAct가 Few-shot Prompting만으로 모든 기존 방법을 대체했다고 주장하기보다는, **더 많은 고품질의 Human-written Trajectory를 이용한 Fine-tuning이 ReAct의 잠재력을 더 끌어낼 수 있을 것**이라고 결론 내립니다.

# 6. Critical Review & Follow-up

ReAct 논문을 읽기 전에는 ReAct를 단순히 **CoT에 외부 검색 기능을 추가한 Prompting Technique** 정도로 생각했습니다.

하지만 논문을 읽고 실제 동작 방식을 하나씩 확인하면서 ReAct는 단순히 Prompt의 형식만 변경하는 방법이라고 보기에는 조금 더 복잡한 구조를 가지고 있다는 생각이 들었습니다.

특히 논문에서는 ReAct를 지속적으로 **Prompt-based Paradigm**으로 설명하지만, 실제 ReAct가 동작하기 위해서는 LLM 외부에서 Action을 실행하고, Environment가 반환한 Observation을 다시 Context에 추가한 뒤, LLM을 다시 호출하는 Closed-loop Interaction이 필요합니다.

또한 논문의 중요한 비교 대상인 CoT와 ReAct가 실제 시스템 구조에서는 상당히 다른 방식이라는 점도 논문을 읽으면서 계속 의문으로 남았습니다.

이번 섹션에서는 이러한 의문을 중심으로 ReAct 논문을 비판적으로 살펴보고자 합니다.

## 6.1 ReAct는 정말 단순한 Prompting Technique인가?

ReAct 논문에서는 ReAct를 다음과 같이 정의합니다.

**"A novel prompt-based paradigm to synergize reasoning and acting in language models"**

즉 저자들은 ReAct를 새로운 Model Architecture나 별도의 학습 알고리즘이라기보다 **Prompting을 기반으로 Reasoning과 Acting을 결합하는 Paradigm**으로 바라봅니다.

실제로 ReAct의 핵심적인 학습 방식도 Few-shot In-context Learning입니다.

사람이 작성한,

```
Question
→ Thought
→ Action
→ Observation
→ Thought
→ Action
→ Observation
→ ...
```

형태의 Trajectory를 Prompt에 넣고, Frozen LLM이 새로운 문제에서도 같은 방식으로 문제를 해결하도록 유도합니다.

이 관점에서 보면 ReAct를 Prompting Technique이라고 부르는 것은 충분히 이해할 수 있습니다.

하지만 실제 실행 과정까지 생각하면 일반적인 CoT Prompting과는 상당한 차이가 있습니다.

CoT는 기본적으로 다음과 같이 실행할 수 있습니다.

```text
Prompt
→ LLM
→ Reasoning Trace
→ Answer
```

반면 ReAct는 다음과 같은 구조가 필요합니다.

```text
Prompt
→ LLM
→ Thought / Action
→ Action 실행
→ Environment
→ Observation
→ Context 갱신
→ LLM 재호출
→ ...
```

즉 Prompt 이외에도,

- Action을 해석하고 실행하는 부분
- External Environment
- Observation을 다시 Context에 추가하는 부분
- LLM을 반복적으로 호출하는 Control Loop

가 필요합니다.

따라서 ReAct를 단순히,

**"Prompt의 작성 방법을 변경하는 기법"**

이라고만 이해하면 실제 동작 과정의 중요한 부분이 빠지게 됩니다.

다만 이것이 논문의 설명이 잘못되었다는 의미는 아닙니다.

논문의 관점에서는 Agent가 Environment와 상호작용하는 구조 자체가 ReAct의 새로운 Contribution이 아니며, **그 구조 안에서 LLM이 Reasoning과 Acting을 어떤 방식으로 수행하도록 유도할 것인가**가 핵심 Contribution이기 때문입니다.

따라서 개인적으로 ReAct는,

**"Agent-Environment Interaction 위에서 LLM이 Reasoning과 Acting을 함께 수행하도록 유도하는 Prompt-based Paradigm"**

이라고 이해하는 것이 가장 정확하다고 생각합니다.

## 6.2 ReAct와 CoT는 같은 관점에서 비교할 수 있는가?

ReAct 논문에서는 CoT를 매우 중요한 비교 대상으로 사용합니다.

논문의 실험에서는 ReAct Trajectory에서 Action과 Observation을 제거하여 CoT Baseline을 만들고,

```text
CoT   = Reasoning-only
Act   = Acting-only
ReAct = Reasoning + Acting
```

이라는 구조로 비교합니다.

논문의 연구 목적을 생각하면 이러한 비교 방식은 자연스럽습니다.

저자들이 확인하고 싶은 것은, **"Reasoning만 사용할 때, Acting만 사용할 때, 그리고 둘을 함께 사용할 때 어떤 차이가 발생하는가?"** 이기 때문입니다.

하지만 시스템의 실행 관점에서 보면 ReAct와 CoT는 완전히 동일한 조건의 비교라고 보기는 어렵습니다.

CoT는,

```
LLM
+
Prompt
```

만으로 문제를 해결합니다.

반면 ReAct는,

```
LLM
+
Prompt
+
External Environment
+
Action Execution
+
Observation Feedback
```

를 사용할 수 있습니다.

특히 HotpotQA와 FEVER에서 ReAct는 Wikipedia를 통해 LLM의 Parameter 내부에 존재하지 않거나 정확하지 않을 수 있는 정보를 직접 검색할 수 있습니다.

따라서 ReAct와 CoT의 차이는 단순히,

```
Reasoning 방식의 차이
```

만 존재하는 것이 아니라,

```
Reasoning 방식
+
External Knowledge 접근 여부
+
Environment Interaction 구조
```

까지 함께 달라집니다.

이 때문에 예를 들어 ReAct가 CoT보다 Hallucination이 적었다는 결과를, **"ReAct Prompt 자체가 CoT Prompt보다 Hallucination을 줄였다."** 라고만 해석하기에는 무리가 있다고 생각합니다.

보다 정확하게는, **"External Knowledge를 검색하여 Reasoning 과정에 반영할 수 있는 ReAct의 문제 해결 구조가 CoT보다 Grounded한 Trajectory를 생성하는 데 도움을 주었다."** 라고 보는 것이 적절합니다.

반대로 ReAct와 Act의 비교는 상대적으로 더 직접적인 Ablation이라고 생각합니다.

두 방법 모두 동일하게 Environment에 접근할 수 있지만,

```
Act   = Action + Observation
ReAct = Thought + Action + Observation
```

이라는 차이가 있기 때문입니다.

따라서 ReAct가 Act보다 높은 성능을 보인 결과는 **Acting 과정에 명시적인 Reasoning을 추가한 효과**를 비교적 직접적으로 확인할 수 있는 실험이라고 볼 수 있습니다.

결국 ReAct와 CoT의 비교가 잘못되었다기보다, **두 방법의 차이를 단순한 Prompt A와 Prompt B의 비교로만 해석해서는 안 된다고 생각합니다.**

## 6.3 ReAct의 성능은 External Environment의 품질에 얼마나 의존하는가?

ReAct의 가장 큰 장점 중 하나는 External Environment에서 직접 정보를 가져와 Reasoning에 사용할 수 있다는 것입니다.

하지만 이것은 동시에 ReAct가 **External Environment가 반환하는 Observation의 품질에 의존하게 된다는 것**을 의미합니다.

HotpotQA Error Analysis에서 ReAct Failure의 **23%가 Search Result Error**로 분석되었습니다.

Search Result Error는 검색 결과가 비어 있거나 문제 해결에 유용한 정보가 포함되어 있지 않은 경우를 의미합니다.

예를 들어 Agent가 올바른 정보를 찾기 위해 다음 Action을 수행했다고 가정할 수 있습니다.

```
Thought:
A에 대한 정보를 찾아야 한다.

Action:
Search[A]
```

하지만 검색 결과가,

```
Observation:
Could not find [A].
```

이거나 전혀 관련 없는 정보라면 ReAct의 다음 Thought 역시 잘못된 방향으로 영향을 받을 수 있습니다.

```
잘못되거나 불충분한 Observation
→ 잘못된 Thought
→ 잘못된 Action
→ 또 다른 잘못된 Observation
```

이라는 Error Propagation이 발생할 수 있습니다.

즉 ReAct에서 Grounding이 이루어진다고 해서 Reasoning이 자동으로 정확해지는 것은 아닙니다.

**Groundedness는 "외부 정보에 근거한다"는 의미이지, 그 외부 정보가 반드시 문제 해결에 적절하고 정확하게 검색되었다는 의미는 아닙니다.**

오히려 CoT에는 존재하지 않았던 새로운 Failure Point가 생기는 것으로 볼 수도 있습니다.

```
CoT
→ Internal Knowledge의 오류와 Hallucination에 취약

ReAct
→ Internal Knowledge뿐 아니라
   Retrieval / Environment의 품질에도 영향을 받음
```

따라서 ReAct의 성능을 실제 System에서 활용하려면 LLM 자체의 Reasoning 능력뿐 아니라 **Action Space와 Retrieval System의 품질 역시 중요한 구성요소**라고 생각합니다.

## 6.4 Groundedness와 Reasoning Flexibility의 Trade-off

ReAct 논문의 Error Analysis에서 가장 흥미로운 부분은 ReAct가 단순히 CoT의 문제를 모두 해결한 것이 아니라, **Hallucination을 줄이는 대신 새로운 종류의 Reasoning Error가 증가했다는 점**입니다.

CoT의 Failure 가운데 **56%가 Hallucination**과 관련된 반면, 분석된 ReAct Failure에서는 Hallucination이 나타나지 않았습니다.

External Knowledge Base를 통해 실제 정보를 검색할 수 있기 때문에 ReAct의 Trajectory가 더 Fact-driven하고 Grounded해진 것입니다.

하지만 반대 문제가 발생했습니다. 분석된 ReAct Failure 가운데 **47%가 Reasoning Error**였습니다. CoT의 동일한 Error 비율은 16%였습니다.

논문에서는 ReAct가,

```
Thought
→ Action
→ Observation
→ Thought
→ Action
→ Observation
```

이라는 구조를 계속 유지해야 하기 때문에 이러한 **Structural Constraint가 Reasoning의 Flexibility를 감소시킬 수 있다**고 설명합니다.

대표적인 문제는 같은 Thought와 Action을 반복하면서 Loop에서 빠져나오지 못하는 것입니다.

```
Thought:
A를 검색해야 한다.

Action:
Search[A]

Observation:
유용한 정보가 없음.

Thought:
A를 검색해야 한다.

Action:
Search[A]

Observation:
유용한 정보가 없음.

...
```

CoT에서는 Reasoning 과정 중 자유롭게 다른 방향의 추론을 수행할 수 있지만, ReAct에서는 다음 Reasoning이 직전 Observation과 Action 구조에 강하게 연결되기 때문에 잘못된 방향으로 들어갔을 때 회복하기 어려울 수 있습니다.

따라서 ReAct와 CoT의 관계를 다음과 같은 Trade-off로 볼 수 있습니다.

```
CoT
높은 Reasoning Flexibility
        ↕
Hallucination 위험

ReAct
높은 Groundedness / Factuality
        ↕
Reasoning Flexibility 감소
External Observation에 대한 의존
```

흥미로운 점은 논문의 저자들 역시 이러한 결과를 바탕으로 어느 한 방법을 완전히 선택하기보다 ReAct와 CoT-SC를 결합했다는 것입니다.

즉 실험 결과 자체가 **External Grounding과 Internal Reasoning의 어느 한쪽만으로는 충분하지 않을 수 있음**을 보여준다고 생각합니다.

## 6.5 ReAct가 검색에 실패했을 때 왜 내부 Reasoning으로 자연스럽게 전환하지 않은가?

논문을 읽으면서 개인적으로 가장 큰 의문이 들었던 부분 중 하나입니다.

ReAct의 Thought는 External Observation만 사용하는 것이 아닙니다.

앞서 살펴본 것처럼 ReAct는 LLM의 Pre-training Knowledge를 이용하여,

- Commonsense Reasoning
- Arithmetic Reasoning
- Goal Decomposition
- Action Planning

등을 수행할 수 있습니다.

즉 ReAct 역시 Internal Knowledge를 사용할 수 있습니다.

그렇다면 다음과 같은 상황을 생각해 볼 수 있습니다.

```
Thought:
A를 검색해야 한다.

Action:
Search[A]

Observation:
유용한 정보를 찾을 수 없다.
```

여기서 ReAct가,

```
Thought:
검색으로는 정보를 찾을 수 없다.
내가 알고 있는 정보와 지금까지의 Observation을 이용해서
문제를 직접 Reasoning해보자.
```

와 같이 External Search를 중단하고 Reasoning-only 방식으로 자연스럽게 전환할 수도 있지 않을까 하는 의문이 들었습니다.

하지만 논문의 Knowledge-intensive Reasoning 실험에서는 이러한 전환을 ReAct 내부의 하나의 Policy로 구성하지 않습니다.

대신 별도의 Heuristic을 사용합니다.

```
ReAct
↓
정해진 Step 안에 Answer를 생성하지 못함
↓
ReAct 중단
↓
CoT-SC 실행
```

즉 `ReAct → CoT-SC`라는 두 개의 Method를 연결하는 방식으로 처리합니다.

개인적으로는 이 부분에서 다음과 같은 의문이 남습니다.

**"ReAct의 Thought 자체가 Internal Knowledge를 사용할 수 있다면, 외부 검색의 실패를 감지하고 Reasoning-only Mode로 전환하는 것까지 ReAct의 Thought-Action Policy 안에서 해결할 수 있지 않았을까?"**

만약 가능하다면,

```
Search가 유용함
→ External Action 계속 수행

Search가 반복적으로 실패함
→ External Action 중단
→ Internal Reasoning 수행

Internal Knowledge로도 불확실함
→ 다시 External Search
```

처럼 보다 유연한 하나의 Trajectory를 구성할 수도 있을 것입니다.

다만 이것은 **ReAct 논문이 실험적으로 검증한 내용이 아니라 논문을 읽으며 개인적으로 생긴 설계상의 의문**입니다.

따라서 논문의 한계라고 단정하기보다는, ReAct의 `Reasoning + Acting`이라는 개념을 조금 더 유연하게 확장한다면 생각해 볼 수 있는 방향이라고 보는 것이 적절하다고 생각합니다.

## 6.6 Human-written Trajectory에 대한 의존성

ReAct Prompting을 사용하기 위해서는 사람이 먼저 문제를 해결하면서 Thought와 Action을 포함한 Demonstration을 작성해야 합니다.

HotpotQA에서는 6개의 ReAct Trajectory를, FEVER에서는 3개의 ReAct Trajectory를 Few-shot Example로 사용합니다.

이 정도의 작은 수에서는 Human Annotation Cost가 크게 문제되지 않을 수 있습니다.

하지만 Task가 복잡해지고 Action Space가 커질수록 문제가 발생합니다.

Action의 종류가 많아질수록 LLM에게 다양한 상황에서 어떻게 Reasoning하고 Acting해야 하는지를 보여주기 위해 더 많은 Demonstration이 필요할 수 있기 때문입니다.

논문의 Conclusion에서도 저자들은 **Large Action Space를 가진 복잡한 Task에서는 더 많은 Demonstration이 필요하며, 이것이 In-context Learning의 Input Length Limit을 쉽게 초과할 수 있다**고 직접 언급합니다.

즉 Prompting 기반 ReAct에는 다음과 같은 Scaling 문제가 존재합니다.

```
Task가 복잡해짐
↓
Action Space 증가
↓
더 다양한 ReAct Demonstration 필요
↓
Prompt 길이 증가
↓
Context Length Limit
```

또한 높은 품질의 Thought를 사람이 직접 작성하는 것 자체도 비용이 발생합니다.

저자들이 Fine-tuning 실험에서 Bootstrapping 방식을 사용한 이유 역시 이 문제와 관련이 있습니다.

사람이 수천 개의 ReAct Trajectory를 직접 작성하는 대신,

```
소수의 Human-written ReAct Example
↓
ReAct Prompting으로 많은 Trajectory 자동 생성
↓
최종 Answer가 정답인 Trajectory 선별
↓
3,000개의 Training Data 생성
↓
PaLM-8B / PaLM-62B Fine-tuning
```

이라는 방법을 사용합니다.

하지만 논문의 저자들은 동시에 더 높은 성능을 얻기 위해서는 **더 많은 High-quality Human Annotation이 필요할 수 있다**고 설명합니다.

따라서 ReAct의 Few-shot Prompting은 적은 Example로 상당한 성능을 낼 수 있다는 장점이 있지만, 더 복잡한 Agent Task로 확장할 때는 Demonstration의 수와 품질, Context Length가 중요한 Scaling Bottleneck이 될 수 있습니다.

논문에서는 이에 대한 Follow-up 방향으로 Fine-tuning뿐 아니라 Multi-task Training과 Reinforcement Learning과의 결합 가능성도 제시합니다.

## 6.7 ReAct를 'Agent'로 봐야하는가, 'Prompting Paradigm'으로 봐야하는가?

ReAct 논문을 읽으면서 끝까지 가장 많이 고민했던 부분입니다.

처음에는 ReAct를 다음과 같이 이해했습니다.

```text
ReAct = CoT와 유사한 Prompting Technique
```

하지만 실제 동작 과정을 확인하고 나면,

```text
LLM
→ Thought
→ Action
→ Environment
→ Observation
→ Context Update
→ LLM 재호출
```

이라는 명확한 Agent-Environment Interaction이 존재합니다.

실제로 논문에서도 Section 2를 시작할 때부터 **Agent가 Environment와 상호작용하는 일반적인 구조**를 먼저 정의합니다.

또한 Decision Making 실험에서는 ReAct를 LLM을 Interactive Environment에 적용한 **Closed-loop System**으로 설명합니다.

그렇다면 ReAct를 Agent라고 불러야 하는 것인지, Prompting Technique이라고 불러야 하는 것인지 의문이 생깁니다.

개인적으로는 둘 중 하나만 선택해서 정의하는 것은 적절하지 않다고 생각합니다.

먼저 ReAct 자체가 완전히 새로운 Agent Framework를 제안한 것은 아닙니다.

Agent가,

```text
Observation을 받고
→ Policy를 이용하여 Action을 결정하고
→ Environment에 Action을 수행하고
→ 다시 Observation을 받는
```

구조 자체는 기존에 존재하던 개념입니다.

ReAct의 새로운 아이디어는 이러한 Agent의 Action Space에 Language Space를 추가하여,

```text
Environment Action
+
Language Action(Thought)
```

을 함께 사용할 수 있도록 한 것입니다.

그리고 이러한 Policy를 새로운 Reinforcement Learning Algorithm 등을 통해 학습시키는 대신, **Human-written Trajectory를 Few-shot Prompt로 제공하여 Frozen LLM에서 유도**합니다.

이 관점에서 보면 논문의 저자들이 ReAct를 `Prompt-based Paradigm`이라고 부르는 것은 자연스럽습니다.

하지만 반대로 ReAct를 단순한 Prompt Template으로만 보는 것도 정확하지 않습니다.

ReAct가 실제로 동작하려면 External Environment와 상호작용하고 Observation을 다시 LLM에 전달하는 Closed-loop 구조가 필요하기 때문입니다.

따라서 제가 ReAct 논문을 읽고 최종적으로 이해한 정의는 다음과 같습니다.

**ReAct는 새로운 Agent Framework 그 자체라기보다, Agent-Environment Interaction을 전제로 하여 LLM이 Thought와 Action을 하나의 Trajectory 안에서 함께 선택하도록 만드는 Prompt-based Agent Policy Paradigm이다.**

즉,

```text
단순한 Prompt Template        X
완전히 새로운 Agent Framework X

Agent-Environment Loop
        +
ReAct Few-shot Prompting
        +
LLM을 이용한 Reasoning / Action Policy
        O
```

라고 보는 것이 가장 적절하다고 생각합니다.

ReAct 논문의 가장 중요한 의미 역시 단순히 LLM에 Wikipedia 검색 기능을 연결했다는 데 있지 않습니다.

기존에는,

```text
Reasoning
```

과,

```text
Acting
```

을 서로 분리된 능력으로 연구하는 경우가 많았다면 ReAct는,

```text
Thought
→ Action
→ Observation
→ Thought
→ Action
→ Observation
```

이라는 하나의 Task-solving Trajectory 안에서 Reasoning과 Acting을 서로 연결했습니다.

Reasoning은 다음 Action을 결정하는 데 도움을 주고,

```text
Reason to Act
```

Action을 통해 얻은 Observation은 다시 Reasoning의 근거가 됩니다.

```text
Act to Reason
```

결국 ReAct의 핵심적인 Contribution은 **LLM의 자연어 Reasoning을 Agent의 Action Selection 과정 안에 명시적으로 포함시키고, External Environment에서 얻은 정보를 다시 Reasoning 과정에 연결했다는 것**이라고 생각합니다.

논문 자체에서도 ReAct가 모든 문제를 해결했다고 주장하지는 않습니다.

Large Action Space에서 필요한 Demonstration의 증가, In-context Learning의 Input Length Limit, Human Annotation의 필요성 등 여러 문제가 남아 있으며, 저자들은 Fine-tuning, Multi-task Training, Reinforcement Learning과의 결합을 향후 발전 방향으로 제시합니다.

그럼에도 ReAct는 LLM을 단순한 Text Generator나 Reasoning Model로만 사용하는 것이 아니라, **Reasoning하면서 외부 세계와 상호작용하는 Agent의 Policy로 사용할 수 있다는 가능성을 보여주었다는 점**에서 의미가 있는 논문이라고 생각합니다.

# 마무리

최근 몇 년 사이 생성형 AI는 굉장히 빠른 속도로 발전하고 있습니다. 불과 몇 년 전까지만 해도 인공지능, 특히 Deep Learning은 관련 연구자나 개발자가 아니면 쉽게 접하기 어려운 분야라는 인식이 강했다고 생각합니다. 하지만 ChatGPT를 비롯한 생성형 AI가 대중화되면서 상황은 크게 달라졌습니다.

이제는 인공지능을 전공하지 않은 사람들도 일상이나 업무에서 자연스럽게 생성형 AI를 사용하고 있으며, 저 역시 업무를 하면서 생성형 AI에 상당히 의존하고 있다고 느낄 정도가 되었습니다.

이러한 변화와 함께 최근에는 생성형 AI를 조금이라도 더 잘 사용하기 위해 Prompt를 반복적으로 수정하고 최적화하는 Prompt Engineering에도 많은 관심이 생겼습니다.

저 역시 생성형 AI를 사용하면서 다른 사람보다 뒤처지지 않아야 한다는 생각도 있었고, 비용을 지불하면서 사용하는 생성형 AI를 조금이라도 더 효과적으로 활용하고 싶다는 생각이 있었습니다. 또한 최근 기업들이 단순히 하나의 Deep Learning Model을 개발할 수 있는 능력뿐만 아니라 LLM, RAG, Agent 등 생성형 AI 전반에 대한 이해를 요구하고 있다는 것을 취업 준비 과정에서 직접 느끼게 되었습니다.

특히 최근 관심을 가지고 공부하고 있는 Agentic RAG 역시 단순히 Retrieval Model과 LLM을 연결하는 것에서 끝나는 것이 아니라, LLM이 상황에 따라 어떤 정보를 검색하고 어떤 Tool을 사용할지를 결정해야 하기 때문에 Prompting과 Agent에 대한 이해가 필요하다고 생각했습니다.

그 과정에서 Agentic RAG와 LLM Agent를 이해하기 위한 중요한 선행 연구 중 하나로 ReAct 논문을 접하게 되었고, Prompt Engineering을 공부한다는 생각으로 이 논문을 읽기 시작했습니다.

하지만 논문을 모두 읽고 난 뒤에는 조금 다른 생각이 들었습니다.

ReAct 논문에서는 사람이 직접 Thought, Action, Observation으로 구성된 Task-solving Trajectory를 작성하고 이를 Few-shot Example로 제공하여 LLM이 유사한 문제 해결 방식을 따르도록 합니다.

이 과정을 보면서 가장 먼저 들었던 생각은, **"실제로 일반 사용자가 이런 Prompt를 직접 설계할 수 있을까?"** 였습니다.

단순히 문장의 표현을 조금 다듬는 수준이 아니라 문제를 어떻게 분해하고, 어느 시점에 어떤 Action을 수행하며, Observation이 주어졌을 때 다음 Reasoning을 어떻게 이어갈 것인지까지 Demonstration으로 설계해야 하기 때문입니다.

따라서 처음에는, **"이렇게 연구자가 실험을 위해 정교하게 설계한 Prompting 방법을 연구하는 것이 실제 일반 사용자에게 얼마나 가치가 있을까?"** 라는 의문도 들었습니다.

하지만 논문을 끝까지 읽으면서 ReAct의 의미를 단순히 **"좋은 Prompt를 만드는 방법"**에서 찾는 것은 적절하지 않다는 생각이 들었습니다.

ReAct의 핵심은 Prompt 문장을 잘 작성하는 것 자체보다, `Reasoning → Acting → Observation → Reasoning` 이라는 구조를 통해 **LLM이 자신의 내부 지식만으로 문제를 해결하는 것을 넘어 외부 Environment와 상호작용하면서 문제를 해결할 수 있도록 만들었다는 것**에 있다고 생각합니다.

과거의 LLM이 사용자의 Question을 입력받고 자신의 Parameter 안에 존재하는 Knowledge를 기반으로 Answer를 생성하는 것에 가까웠다면, ReAct는 LLM이 문제를 해결하기 위해 어떤 정보가 필요한지를 Reasoning하고, 필요한 경우 External Environment에 Action을 수행하고, 그 결과로 얻은 새로운 Observation을 다시 Reasoning에 사용할 수 있다는 가능성을 보여주었습니다.

물론 ReAct가 이러한 모든 아이디어의 최초라고 할 수는 없습니다. ReAct 이전에도 외부 정보를 검색하거나 Environment와 상호작용하는 여러 연구가 존재했습니다.

그럼에도 개인적으로 ReAct는 현재의 LLM Agent와 Agentic RAG로 이어지는 흐름에서 매우 중요한 선행 연구이자 하나의 전환점이라고 생각합니다.

ChatGPT가 처음 등장했을 당시와 비교하면 현재의 생성형 AI는 단순히 Model Parameter 내부의 Knowledge만으로 답변하지 않습니다. 필요하다면 Web Search와 다양한 Tool을 사용하고, 여러 단계의 작업을 수행하며, 그 결과를 다시 다음 판단에 활용합니다.

ReAct 논문을 읽기 전에는 이러한 기능들을 단순히 "LLM에 검색 기능을 붙인 것" 정도로 생각했습니다.

하지만 이번 논문을 읽으면서 중요한 것은 Tool 자체가 아니라, **"LLM이 언제 어떤 Tool을 사용해야 하는지를 Reasoning하고, Tool을 사용하여 얻은 결과를 다시 다음 Reasoning에 연결하는 구조"** 라는 것을 이해하게 되었습니다.

결국 제가 ReAct 논문을 읽으면서 가장 크게 얻은 것은 특정 Prompt를 작성하는 방법이 아니었습니다.

처음에는 Prompt Engineering을 배우기 위해 읽기 시작했지만, 오히려 논문을 다 읽고 난 뒤에는 **LLM을 단순한 Text Generation Model이 아니라 외부 Environment와 상호작용하며 문제를 해결하는 Agent의 Policy로 바라보는 관점**을 배우게 되었습니다.

이러한 관점에서 ReAct는 지금의 생성형 AI가 단순히 "질문에 답하는 모델"을 넘어 "Reasoning하면서 행동하는 시스템"으로 발전하는 흐름을 이해하는 데 의미가 큰 논문이었다고 생각합니다.