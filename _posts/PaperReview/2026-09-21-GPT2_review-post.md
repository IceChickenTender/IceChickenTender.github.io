---
title: "[PaperReview] GPT-2 논문 Language Models are Unsupervised Multitask Learners 리뷰"
categories:
  - PaperReview
tags:
  - PaperReview

use_math: true
toc: true
toc_sticky: true
toc_label: "GPT-2 논문 Language Models are Unsupervised Multitask Learners 리뷰"
---

# 1. Introduction: Context & Motivation

GPT-2를 이해하기 위해서는 먼저 이 논문이 새로운 Transformer Architecture를 제안하는 연구라기보다, Language Model을 학습하고 활용하는 방식 자체에 새로운 질문을 던진 연구라는 점을 이해할 필요가 있습니다.

GPT-2 이전에도 NLP에서는 Pre-training을 활용한 연구가 빠르게 발전하고 있었습니다. 하지만 대부분의 방법은 결국 특정 Downstream task를 수행하기 위해 별도의 Labeled Dataset과 Supervised Training을 필요로 했습니다.

GPT-2는 여기서 한 단계 더 나아가 "충분히 크고 다양한 Text로 학습한 Language Model이라면, Task별 Supervised Fine-tuning 없이도 여러 NLP Task를 직접 수행할 수 있지 않을까?"를 고민한 연구입니다.

## 1.1 GPT-2 이전 NLP와 Narrow Expert 문제

GPT-2 이전의 전형적인 NLP 시스템은 특정 Task를 잘 수행하도록 설계되었습니다. 예를 들어 Question Answering 모델을 만들고 싶다면 Question Answering용 Dataset을 구축하고, 각 Question에 대한 정답을 Label로 제공하여 모델을 Supervised Learning 방식으로 학습했습니다. Translation, Setiment Analysis, Reading Comprehension 등 다른 Task도 기본적으로 비슷한 방식을 사용했습니다. 

이러한 방식은 해당 Task에서 높은 성능을 얻는 데에는 효과적이었습니다. 그러나 GPT-2 저자들은 이러한 시스템을 범용적인 `Genearlist`라기보다 **`Narrow Expert`**에 가깝다고 보았습니다.

특정 Dataset과 Task에 맞추어 학습된 `Narrow Expert`한 모델은 학습 당시와 비슷한 환경에서는 잘 동작하지만, Data Distribution이나 Task Specification이 조금만 달라져도 성능이 크게 떨어질 수 있는 문제가 존재했습니다. 또한 새로운 Task가 등장할 때마다 다시 Labeled Dataset을 구축하고 학습해야 한다는 점입니다.

GPT-2 저자들은 이러한 문제점들에 주목해 "Task가 바뀔 때마다 새로운 Dataset과 Supervised Learning을 요구하는 방식으로 과연 범용적인 Language System을 만들 수 있는가?"라는 의문을 품었습니다.

## 1.2 GPT-1에서 GPT-2로

GPT-2의 아이디어롤 이해하기 위해서는 GPT-1과의 차이를 함께 보는 것이 중요합니다.

GPT-1은 Transformer의 Decoder 구조를 이용한 Autoregressive Language Model을 대규모 Unlabeled Text로 먼저 Pre-training한 뒤, 특정 Downstream Task의 Labeled Dataset을 사용하여 Supervised Fine-tuning 하는 방식을 사용했습니다.

GPT-1의 중요한 의미는 대규모 Text를 이용한 Language Model Pre-training이 다양한 NLP Task에 사용할 수 있는 Representation을 학습할 수 있음을 보여주었다는 점입니다. 하지만 최종적으로 특정 Task를 수행하기 위해서는 여전히 Supervised Fine-tuning이 필요했습니다.

GPT-2는 바로 이 마지막 단계를 문제 삼고 있습니다. GPT-2는 "Fine-tuning 없이 Pre-trained Language Model자체가 Task를 수행할 수 있는가?"를 확인하고자 한 것입니다. 또한 GPT-2는 Architecture의 기본 구조가 GPT-1과 마찬가지로 Decoder-only Transformer입니다. Layer Normalization 의 위치, Initialization, Context Length 등의 변경은 있었지만, 논문의 핵심적인 변화는 Architecture 바체 보다는 Model Capacity와 Training Data의 규모 및 다양성을 크게 확장했다는 것에 있습니다.

GPT-1과 GPT-2의 가장 중요한 차이를 단순화하면 다음과 같습니다.

|                 | GPT-1                            | GPT-2                      |
| --------------- | -------------------------------- | -------------------------- |
| 기본 Architecture | Decoder-only Transformer         | Decoder-only Transformer   |
| Pre-training    | Language Modeling                | Language Modeling          |
| Downstream Task | Supervised Fine-tuning 필요        | Fine-tuning 없이 수행하는 것을 목표  |
| 핵심 질문           | Pre-training이 Transfer에 도움이 되는가? | LM 자체가 여러 Task를 수행할 수 있는가? |

GPT-1에서는 Langauge Model이 좋은 Representaion을 얻기 위한 Pre-training 단계였다면, GPT-2에서는 Language Model 그 자체가 여러 Task를 수행하는 범용적인 Task Learner가 될 가능성을 탐구하기 시작했다고 볼 수 있습니다.

## 1.3 GPT-2의 핵심 가설

GPT-2 저자들은 "어떻게 하면 Pre-training 만으로 여러 Task를 수행할 수 있을까?"라는 질문에 대한 답을 자연어 데이터 자체에서 찾았습니다. 인터넷에 존재하는 Text는 단순한 평문만으로 구성되어 있지 않습니다. 저자들은 자연어에는 이미 다양한 형태의 Input과 Output 관계가 포함되어 있다고 보았습니다. 예를 들면 다음과 같습니다.

Web에는 다음과 같이 번역된 내용이 들어 있을 수 있습니다.

```
English:
I love you.

French:
Je t'aime.
```

혹은 QA 형태가 존재할 수도 있습니다.

```
Question:
Who wrote The Origin of Species?

Answer:
Charles Darwin
```

사람의 관점에서는 각각 다른 Task이지만 Language Model 관점에서는 모두 하나의 연속된 Token Sequence일 뿐입니다. 저자들은 이 점에 주목했습니다. Langauge Model의 Training Objective는 기본적으로 이전 Token들을 보고 다음 Token을 예측하는 것입니다. 그런데 모델이 위와 같은 Text를 정확하게 예측하려면 단순히 개별 단어의 등장 빈도만 학습하는 것보다 

```
English Sentence ↔ French Sentence

Question ↔ Answer

Article ↔ Summary
```

와 같은 관계를 어느 정도 모델링하는 것이 유리합니다. 따라서 저자들은 충분한 Capacity를 가진 Language Model이 매우 다양하고 방대한 자연어를 학습한다면, 다음 Token을 더 잘 예측하는 과정에서 자연오 속에 존재하는 Task Demonstration까지 암묵적으로 학습하기 시작할 수 있다고 가정했습니다.

여기서 중요한 것은 GPT-2가 학습할 때 Translation Loss, QA Loss, Summarization Loss를 별도로 사용하지 않는다는 점입니다. 모델에게 직접 주어진 Objective는 오직 Language Modeling뿐입니다. 그럼에도 자연어 안에 존재하는 다양한 Task Demonstration을 학습한 결과 여러 Task를 수행하는 Behavior가 나타난다면, 하나의 Language Modeling Objective만으로 사실상 여러 Task를 동시에 학습한 것으로 볼 수도 있습니다.

GPT-2 저자들은 이러한 가능성을 `Unsupervised Multitask Learning`이라고 표현하였습니다. 이를 한 문장으로 정리를 한다면 "충분히 크고 다양한 자연어 데이터로 충분한 Capacity의 Language Model을 학습하면, 별도의 Task-Specific Supervision 없이도 자연어 속에 존재하는 Task Demonstration을 통해 여러 Task를 수행하는 방법을 학습하기 시작할 수 있다."

GPT-2 저자들은 이 가설을 검증하기 위해 더 다양한 WebText Dataset을 구축하고 Model Size를 크게 확장한 뒤, 별도의 Task-specific Fine-tuning 없이 Language Modeling, Reading Comprehension, Summarization, Translation, Questino Answering 등 여러 Task에서 모델의 성능을 평가합니다.

# 2. Architecture & Objective

## 2.1 GPT-2 Architecture

GPT-2는 Transformer를 기반으로 한 Language Model이며, 전체적인 Architecture는 GPT-1에서 사용한 OpenAI GPT의 구조를 대부분 그대로 따릅니다. GPT 계열에서 사용하는 구조는 Transformer의 Decoder부분만을 사용하는 Decoder-only Transformer입니다.

GPT-2 논문에서는 GPT-1의 Architecture를 완전히 새롭게 설계하기 보다 몇 가지 수정과 함께 Model Capacity를 크게 증가시큰 것에 초점을 두었습니다. 주요 변경 사항은 다음과 같습니다.

- Layer Normalization을 각 Sub-block의 입력 위치로 이동
- 마지막 Self-Attention Block 이후에 추가 Layer Normalization 적용
- Model Depth에 따른 Residual Path의 누적 효과를 고려한 Initialization 적용
- Vocabuliary Size를 `50,257`로 확장
- Context Size를 `512->1024 Token`으로 증가
- Batch Size를 `512`로 증가

Residual Layer의 Weight는 Initialization 단계에서 $\frac{1}{\sqrt{N}}$ 비율로 Scaling 하였습니다. 여기서 $N$은 Residual Layer의 개수를 의미합니다.

GPT-2 연구에서는 Model Capacity의 영향을 확인하기 위해 총 네 가지 크기의 모델을 학습하였습니다.

| Parameters | Layers | \(d_{model}\) |
| ---------: | -----: | ------------: |
|       117M |     12 |           768 |
|       345M |     24 |          1024 |
|       762M |     36 |          1280 |
|     1.542B |     48 |          1600 |

가장 작은 117M 모델은 기존 GPT-1과 동일한 규모이며, 가장 큰 1.542B Parameter 모델을 논문에서는 GPT-2라고 부릅니다. 

GPT-2 논문에서 Model Size는 단순히 성능을 높이기 위한 Engineering 요소가 아니라, Language Model이 여러 Task를 수행하는 능력이 Capacity 증가에 따라 나타나는지를 확인하기 위한 주용한 실험 변수이기도 합니다.

## 2.2 Autoregressive Language Modeling

GPT-2의 Training Objective는 `Autoregressive Language Modeling`입니다. Langauge Model의 목적은 주어진 이전 Token들을 이용하여 다음 Token의 Probability Distributino을 예측하는 것입니다.

하나의 Text Sequence를 다음과 같다고 하면,

$$
x=(s_1, s_2, \ldots, s_n)
$$

전체 Sequence의 Probability는 다음과 같이 표현할 수 있습니다.

$$
p(x)=\prod_{i=1}^{n} p(s_i \mid s_1, \ldots, s_{i-1})
$$

GPT-2는 이전 모델인 GPT-1과 달리 Fine-tuning 없이 이전 Token들의 정보만으로 다음 Token을 예측하도록 학습하는 Language Modeling 학습 방식의 Pre-Training만 진행합니다.

## 2.3 Task를 Natural Language로 표현한다는 아이디어

GPT-2의 핵심 가설을 이해하기 위해 가장 중요한 부분 중 하나가 바로 `Task Conditioning`입니다. 일반적으로 하나의 Task를 Probability 관점에서 표현하면 다음과 같이 나타낼 수 있습니다.

$$
p(\text{output} \mid \text{input})
$$

하지만 하나의 General System이 여러 Task를 수행하려면 같은 Input에 대해서도 어떤 Task를 수행해야 하는지 알아야 합니다. 따라서 GPT-2 논문에서는 이를 다음과 같이 확장합니다.

$$
p(\text{output} \mid \text{input}, \text{task})
$$

Output에는 Input뿐 아니라 Task에도 Condition 되어야 합니다.

기존 Multitask Learning에서는 이를 해결하기 위해 Task마다 다른 Encoder나 Decoder를 사용하거나, Task를 구분할 수 있는 별도의 Architecture 또는 Training Procedure를 사용하는 경우가 있었습니다. GPT-2 저자들이 주목한 것은 Task 자체도 Natural Language로 표현할 수 있을 것이라는 점입니다.

예를 들어 Training Corpus에 다음과 같은 Text가 자연스럽게 존재한다고 해보겠습니다.

```
English: Hello.
French: Bonjour.

English: Thank you.
French: Merci.
```

GPT-2는 이것을 명시적인 Translation Dataset으로 인식하여 학습하지 않습니다. Language Model에게는 단지 하나의 Text Sequence일 뿐입니다. 하지만 다음 Token을 정확하게 예측하기 위해서는

```
English Sentence
        ↓
Corresponding French Sentence
```

라는 관계를 어느 정도 학습하는 것이 유리합니다. 다른 Task들도 마찬가지입니다.

논문에서는 명시적인 Supervised Objective는 Sequence 일부의 Output만을 평가하는 반면, Language Modeling Objective는 Sequence 전체를 예측한다는 점도 주목합니다.

단순화된 환경에서는 Supervised Objective가 요구하는 Output까지 정확하게 모델링하는 것이 결국 전체 Sequence를 정확하게 모델링하는 Language Modeling Objective에도 포함될 수 있다는 것입니다. 저자들은 실제 Preliminary Experiment에서도 충분히 큰 Language Model이 이러한 인위적인 Multitask 환경에서는 여러 Task를 학습할 수 있었지만, 명시적인 Supervised Learning  보다 학습 소도는 훨씬 느렸다고 설명합니다.

물론 실제 Web Text는 이렇게 깔끔하게 정리된 Training Dataset과는 다릅니다. 글머에도 저자들은 충분히 큰 Language Model이 다양한 자연어를 더 잘 예측하려다 보면, 그 안에 자연스럽게 포함되어 있는 Task Demonstration을 추론하고 수행하는 방법까지 배우기 싲가할 수 있다고 가정하였습니다.

# 3. Deep Dive: Unsupervised Multitask Learning

GPT-2 논문의 핵심은 단순히 더 큰 Language Model을 만든 것이 아닙니다. 저자들이 정말 확인하고 싶었던 것은 하나의 Langauge Modeling Objective만으로 여러 NLP Task를 학습할 수 있는가였습니다.

GPT-2는 Translation, Question Answering, Summarization을 위한 별도의 Training Objective를 사용하지 않습니다. 그럼에도 여러 Task에서 일정 수준의 성능이 나타났고, 저자들은 이를 `Unsupervised Multitask Learning`의 가능성으로 해석했습니다.

## 3.1 Unsupervised Multitask Learning

GPT-2 이전에는 일반적인 Multitask Learning에서는 여러 Task를 위해 각각의 Dataset과 Objective를 준비합니다. 

```
Translation Dataset → Translation Objective
QA Dataset          → QA Objective
Summarization Data  → Summarization Objective
```

반면 GPT-2에는 이러한 Task-specific Objective가 존재하지 않습니다. 다만 Training Corpus 안에는 여러 종류의 Task Demonstration이 포함될 수 있습니다. 따라서 GPT-2 저자들의 가설은 다음과 같이 정리할 수 있습니다.

```
다양한 Natural Language
        ↓
Next-token Prediction
        ↓
Text 내부의 여러 관계를 학습
        ↓
여러 Task를 수행하는 Behavior가 나타남
```

저자들은 Language Model이 이러한 방식으로 여러 Task를 추론하고 수행한다면 사실상 `Unsupervised Multitask Learning`을 수행하는 것이라고 보았습니다. 다만 이것은 주의해서 해석할 필요가 있습니다.

GPT-2 실험에서 확인한 것은 Translation, QA, Summarization과 같은 Task-like Behavior가 나타났다는 것입니다. 모델 내부에 Translation Module이나 QA Module처럼 독립적인 Task Representation이 형성되었다는 것을 입증한 것은 아닙니다.

## 3.2 Zero-shot과 Task Conditioning

GPT-2 논문에서는 Downstream Task를 수행할 때 Parameter나 Architecture를 수정하지 않는 것을 `Zero-shot`의 핵심 조건으로 봅니다. 하지만 GPT-2는 아무런 정보 없이 Task를 수행한 것은 아닙니다. Summarization에서는 Article 뒤에 `TL;DR:`을 추가했고, Translation에서는 Exmaple Context를 제공했습니다. Question Answering에서도 Question-Answer Example을 앞부분에 제공하여 모델이 원하는 Answer Format을 추론하도록 했습니다.

즉 GPT-2의 `Zero-shot`은 Task-specific Fine-tuning이나 Parameter Update를 수행하지 않는다는 의미가 강합니다. 그리고 GPT-2 논문에서 더 중요한 발견은 용어 자체보다 "모델의 Parameter를 수정하지 않고 Natural Language Context만 변경해도 모델이 수행하는 Task Behavior를 바꿀 수 있었다" 입니다.

# 4. Engineering & Implementation Details

## 4.1 WebText

기존 Language Model 연구에서는 News, Wikipedia, Book처럼 비교적 하나의 Domain에 집중된 Dataset을 사용하는 경우가 많았습니다.

하지만 GPT-2 논문의 가설은 다양한 자연어 안에 존재하는 여러 Task Demonstration을 Language Model이 학습할 수 있다는 것이었기 때문에, 저자들은 가능한 크고 다양한 Corpus가 필요하다고 보았습니다.

웹 전체를 수집할 수 있는 Common Crawl은 좋은 후보였지만, 품질이 낮거나 이해하기 어려운 Document가 많이 포함된다는 문제가 있었습니다. 저자들은 이를 해결하기 위해 Reddit을 일종의 `Human Curation Filter`로 사용했습니다.

저자들은 Reddit 게시글 자체를 Training Data로 사용한 것이 아니라, Reddit에 공유된 외부 Web Page를 수집했다고 합니다. 또한 `3Karma`라는 기준은 해당 Document에 Translation이나 QA같은 Task가 포함되어 있기 때문이 아니라, 사람들이 어느 정도 가치 있는 Content라고 판단했는지를 나타내는 단순한 Heuristic으로 사용되었습니다. 이 방식으로 구축한 Dataset이 `WebText`입니다.

논문에서 사용한 초기 WebText는 Cleaning과 deduplication 이후 다음 정도의 규모를 가집니다.

- 약 8 백 만개의 문서
- 약 40GB의 Text
- 2017년 12월 이후 생성된 Link는 제외
- 다른 NLP Dataset과의 Overlap 문제를 줄이기 위해 Wikipedia Document 제거

## 4.2 Byte-level BPE

GPT-2 논문은 Byte-Level BPE를 선택한 이유와 일부 설계 원칙은 설명하지만, 실제 Tokenization 과정이 코드 수준에서 어떻게 동작하는지까지 상세하게 설명하지는 않습니다. NLP 모델에서는 이러한 Tokenizer가 굉장히 중요한데 논문에서는 자세한 설명이 없어 실제 구현을 하려고 할 때나 구현 레벨에서 이러한 Tokenizer에 대해서 물어볼 때 어려움을 겪는 경우가 많습니다. 특히나 저는 논문 리뷰 포스트를 작성하는 이유로 시간이 지난 후의 재학습을 용이하게 하고자 하는 목적이 크기 때문에 저는 GPT-2 논문에 더해 BPE를 NLP에 적용한 Sennrich et al(2015)의 연구와 OpenAI가 공개한 GPT-2의 encoder.py 구현을 참고하여 동작 원리를 정리하고자 합니다.

### 4.2.1 Byte-level BPE가 필요한 이유

Word-level Tokenization은 Vocabulary에 존재하지 않는 단어를 처리하기 어렵다는 문제가 있습니다. 이와는 반대로 Character 단위로 모든 Text를 처리하면 새로운 단어도 표현할 수 있지만 Sequence가 지나치게 길어진다는 문제가 있습니다.

`BPE` 방식은 이 두 방식의 중간에 위치합니다. 처음에는 작은 Symbol에서 시작한 뒤 Training Corpus에 자주 함께 등장하는 인접 Symbol Pair를 반복적으로 합쳐, 자주 등장하는 문자열은 큰 Token으로, 드문 문자열은 작은 Token들의 조합으로 표현합니다.

Sennrich et al.(2015)은 이러한 BPE를 Subword Tokenization에 적용하여 Rare Word와 Unknown Word를 더 작은 단위의 조합으로 표현하는 방법을 제안했습니다. 하지만 일반적인 BPE를 모든 Unicode Character에 직접 적용하려면 매우 큰 Base Vocabulary가 필요합니다.

GPT-2는 이 문제를 해결하기 위해 Character보다 더 작은 **Byte**를 기본 단위로 사용합니다. 모든 Text는 UTF-8 Encoding을 통해 Byte Sequence로 변환할 수 있으며, 하나의 Byte가 가질 수 있는 값은 `0~255`이므로 256개의 Base Symbol만으로 어떤 Unicode 문자열도 표현할 수 있습니다.

### 4.2.2 GPT-2 Byte-level BPE의 동작 과정

OpenAI가 공개한 GPT-2의 `encoder.py`를 기준으로 Tokenization 과정은 크게 다음과 같습니다.

```
Raw Text
   ↓
1. 정규식을 이용한 Pre-tokenization
   ↓
2. UTF-8 Byte 변환
   ↓
3. Byte → Unicode Symbol Mapping
   ↓
4. 학습된 BPE Merge 적용
   ↓
5. Token ID 변환
```

1. Pre-tokenization

    GPT-2는 전체 Text에 바로 BPE를 적용하지 않습니다. 먼저 정규식을 이용해 Letter, Number, Punctuation, Whitespace, 영어 축약형 등을 구분합니다. 실제 구현에서는 다음과 같은 Pattern을 사용합니다.

    ```python
    self.pat = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d|
            ?\p{L}+|
            ?\p{N}+|
            ?[^\s\p{L}\p{N}]+|
            \s+(?!\S)|
            \s+"""
    )
    ```

    예시로 설명을 하자면 "Hello World!"는 다음과 같이 나누어 집니다.

    ```
    "Hello"
    " world"
    "!"
    ```
    
    여기서 `" world"`처럼 앞쪽 공백이 단어와 함께 처리될 수 있다는 점이 GPT-2 Tokenizer의 특징 중 하나입니다.

2. UTF-8 Byte로 변환

    각 문자열은 UTF-8 Byte Sequence로 변환됩니다. 예를 들어 영어 알파벳 `A`는 하나의 Byte로 표현할 수 있지만, 한글과 같은 Unicode Character는 여러 Byte로 표현됩니다.

    ```
    "한"
    ↓ UTF-8
    ED 95 9C
    ↓
    237, 149, 156
    ```

    따라서 GPT-2는 Vocabulary에 `"한"`이라는 Character가 존재하지 않더라도 해당 문자열을 Byte의 조합으로 항상 표현할 수 있습니다.

3. Byte를 Unicode Symbol로 Mapping

    GPT-2 구현에서는 Byte 값을 그대로 BPE에 넣지 않고, 각 Byte를 1:1 대응되는 Unicode Symbol로 변환합니다. 이를 담당하는 함수가 `bytes_to_unicode()`입니다. 이 때문에 GPT-2 Tokenizer의 결과를 보면 `Ġ`라는 특수문자가 자주 등장합니다. 이 문자는 원래 Text에 존재하는 문자가 아니라 Space Byte를 내부적으로 표현할 Symbol입니다.

    따라서 `" world"`가 Tokenizer 내부에서는 `"Ġworld"`처럼 표현될 수 있습니다. 

4. BPE Merge 적용

    이제 Byte를 표현하는 Symbol들에 BPE를 적용합니다. BPE 학습 과정에서는 Training Corpus에서 자주 등장하는 인접 Pair를 반복적으로 합칩니다. 단순한 예를 들어 설명하자면, `lower`에서 `(l, o)`가 자주 등장한다면 `l + o → lo`라는 Merge Rule을 만들 수 있습니다. 이후 `(lo, w)`도 자주 등장한다면 `lo + w → low`처럼 더 큰 Token이 만들어질 수 있습니다.

    실제 GPT-2에서는 이러한 Merge Rule이 `vocab.bpe`에 저장되어 있으며, Tokenization 시 새로운 Merge Rule을 다시 학습하는 것이 아니라 이미 학습된 Merge Priority를 순서대로 적용합니다.

    OpenAI의 `bpe()` 함수는 현재 Symbol의 인접 Pair를 확인한 뒤 `bpe_ranks`에서 가장 높은 Priority를 가진 Pair를 반복적으로 Merge합니다.

5. Token ID로 변환

    BPE가 끝나면 최종 Token을 Vocabulary에서 Integer ID로 변환합니다. 전체 과정을 단순화하면 다음과 같습니다.

    ```
    "Hello world!"
            ↓
    Pre-tokenization
            ↓
    "Hello" / " world" / "!"
            ↓
    UTF-8 Byte
            ↓
    Byte → Unicode Mapping
            ↓
    BPE Merge
            ↓
    ["Hello", "Ġworld", "!"]
            ↓
    Token ID
            ↓
    Transformer Input
    ```

    모델은 실제 문자열을 직접 입력받는 것이 아니라 최종적으로 만들어진 Token ID Sequence를 Embedding으로 변환하여 처리합니다.

    반대로 생성된 Token ID를 Text로 복원할 때는 이 과정을 역순으로 수행합니다. OpenAI의 `decode()` 구현에서도 Token ID를 BPE 문자열로 변환한 뒤 다시 Byte Sequence와 UTF-8 Text로 복원합니다.

    GPT-2의 Vocabulary Size는 `50,257`이며, OpenAI의 현재 tokenizer 정의에서도 GPT-2 Encoding의 Vocabulary Size와 `<|endoftext|>` Special Token을 확인할 수 있습니다.
    

## 4.3 Training Set과 Test Set을 비교한 방법

GPT-2 연구진은 Training Data와 Evaluation Data 사이의 중복 정도를 확인하기 위해 8-gram Overlap을 측정했습니다. 먼저 비교 대상 문자열을 다음과 같이 정규화 했다고 합니다.

- 모든 문자를 소문자로 변환
- Alphanumeric Word만 유지
- Word 사이를 하나의 Space로 통일

이후 WebText Training Set에서 연속된 8개의 Token으로 이루어진 8-gram을 생성하고, 이를 `Bloom Filter`에 저장했다고 합니다. Bloom Filter는 매우 많은 8-gram에 대해 특정 8-gram이 WebText에 존재하는지를 빠르게 검사하기 위해 사용되었습니다.

Bloom Filter는 어떤 값이 거대한 집합 안에 존재하는지를 빠르고 Memory-efficient하게 검사하는 확률적 자료구조입니다. 원리는 다음과 같습니다.

먼저 거대한 Bit Array를 준비합니다.

```
Index

0 1 2 3 4 5 6 7 8 9 ...
───────────────────────
0 0 0 0 0 0 0 0 0 0 ...
```

WebText에서 어떤 8-gram을 하나 가져옵니다. 이를 여러 Hash Function에 넣습니다.

```
hash1 → 2
hash2 → 7
hash3 → 9
```

그리고 해당 위치의 Bit를 `1`로 만듭니다.

```
0 1 2 3 4 5 6 7 8 9
────────────────────
0 0 1 0 0 0 0 1 0 1
```

WebText의 수 많은 8-gram에 대해 이 작업을 반복합니다. 결과적으로 Bloom Filter는 "WebText에 존재했던 8-gram들의 압축된 Membership 정보"를 가지게 됩니다.

# 5. Experiment & Conclusion

GPT-2의 실험은 Task-specifc Fine-tuning 없이 Language Model 자체가 얼마나 다양한 NLP Task를 수행할 수 있는지를 확인하는데 초점을 두고 있습니다. 이를 위해 117M, 345M, 762M, 1.542B Parameter의 네 모델을 비교했으며, 전반적으로 Model Capacity가 증가할수록 여러 Task의 성능도 향상되는 경향을 보였습니다.

## 5.1 Zero-shot Task Experiments

GPT-2는 별도의 Task-specific Training 없이 Language Modeling, Reading Comprehension, Summarization, Translation, Question Answering 등 다양한 Task에서 평가되었습니다. 대표적인 결과를 정리하면 다음과 같습니다.

| Task | 주요 결과 |
|---|---|
| Language Modeling | 8개 Benchmark 중 7개에서 기존 SOTA 갱신 |
| LAMBADA | 8.63 PPL, 63.24% Accuracy |
| Winograd Schema | 70.70% Accuracy |
| CoQA Reading Comprehension | 55 F1 |
| Summarization | Summary 형태의 Text 생성에는 성공했지만 낮은 ROUGE |
| Translation | En→Fr 5 BLEU, Fr→En 11.5 BLEU |
| Natural Questions | 4.1% Exact Match |

특히 CoQA에서는 127,000개 이상의 Question-Answer Pair로 학습한 일부 Supervised Baseline과 비슷하거나 더 높은 55F1을 기록했습니다. 반면 Translation이나 Open-domain Question Answering에서는 Task Behavior가 나타나기 시작했지만 당시의 전문적인 시스템과 비교하면 성능 차이가 컸습니다.

따라서 실험 결과를 단순히 GPT-2가 모든 NLP Task를 잘 수행했다고 해석하기보다는, Model Capacity가 증가하면서 하나의 Language Modeling Objective만으로도 다양한 Task-specific Behavior가 나타나기 시작했다는 점에 의미가 있습니다.

## 5.2 `TL;DR:`과 Task Behavior

Summarization 실험을 GPT-2 논문의 핵심 가설을 비교적 직관적으로 보여주고 있습니다. 연구진은 CNN/Daily Mail의 Articl 뒤에 단순히 `TL;DR:`을 추가한 뒤 GPT-2가 이어서 Text를 생성하도록 했습니다.

실험에서는 100개의 Token을 Top-k Smapling(k=2)으로 생성하고, 그중 처음 3개의 Sentence를 Summary로 사용했습니다. 생성된 Text는 정성적으로 Summary와 유사했지만, ROUGE 기준으로는 기존 Summarization Model보다 낮은 성능을 보였습니다. 흥미로운 점은 `TL;DR:` Hint를 제거했을 때입니다.

| 설정 | R-AVG |
|---|---:|
| GPT-2 + `TL;DR:` | 21.40 |
| GPT-2 without Hint | 15.03 |

`TL;DR:`을 제거하면 Aggregate ROUGE가 약 6.4 Point 감소했습니다. 저자들은 이를 Natural Language Context가 Language Model에서 특정 Task Behaivor를 유도할 수 있다는 증거로 해석했습니다.

## 5.3 Generalization vs Memorization

GPT-2는 WebText라는 대규모 Web Corpus로 학습되었기 때문에, 높은 성능이 실제 Generalization 때문인지 Evaluation Data와의 중복 때문인지도 분석했습니다.

앞서 살펴본 8-gram 기반 Overlap 분석 결과, 일반적은 Langauge Modeling Benchmark의 Test Set은 WebText Trainig Set과 평균 약 3.2%의 Overlap을 보였습니다. 반면 동일 Benchmark의 자체 Training Set과 Test Set 사이의 평균 Overlap은 약 5.9% 였습니다.

GPT-2는 LAMBADA에서는 Overlap이 존재하는 Example을 모두 제외하고 다시 평가를 진행하였습니다. 평가 결과 Perplexity는 `8.6`에서 `8.7`로, Accuracy는 `63.2%`에서 `62.9%`로 성능이 소폭 감소했기 때문에 연구진은 Data Overlap이 성능에 작지만 일관된 이점을 제공하긴 하지만, 주요 결과 전체를 Overlap만으로 설명하기는 어렵다고 해석했습니다.

또한 WebText 자체의 Training Set과 Held-out Set에서도 Model Size가 증가할수록 성능이 함께 개선되었습니다. 저자들은 이를 근거로 가장 큰 GPT-2조차 WebText를 완전히 Fit한 상태가 아니라 여전히 Underfitting하고 있다고 판단했습니다.

## 5.4 GPT-2의 결론

GPT-2 논문의 결론은 Architecture의 혁신보다는 Language Modeling을 바라보는 관점의 변화에 있습니다. 저자들이 실험을 통해 제시한 방향은 다음과 같이 정리할 수 있습니다.

```
Large & Diverse Dataset
        +
High-capacity Language Model
        +
Single Language Modeling Objective
        ↓
Task-specific Fine-tuning 없이
여러 Task Behavior가 나타남
```

GPT-2는 Langauge Modeling Benchmark 8개 중 7개에서 Zero-shot SOTA를 기록했고, Reading Comprehension, Summarization, Translation, Question Answering에서도 정도의 차이는 있지만 Task에 해당하는 Behavior를 보였습닏아. 저자들은 이를 바탕으로 충분히 다양한 Text Corpus의 Likelihood를 학습하는 High-capacity Language Model이 명시적인 Supervision 없이도 여러 Task를 수행하는 방법을 학습하기 시작할 수 있다고 결론을 내렸습니다.

다만 논문 자체에서도 GPT-2의 Zero-shot Performance가 실제 Application에 사용하기에는 아직 부족하며, Summarization과 Translation, Questino Anmswring 등에서는 성능이 제한적이라는 점을 인정하고 있습니다.

# 6. Critical Review & Follow-up

GPT-2는 Langauge Model을 단순한 Pre-training Model이 아니라 여러 Task를 직접 수행할 가능성이 있는 범용적인 Model로 바라보게 했다는 점에서 중요한 연구입니다. 하지만 논문을 읽으며 실험 방법과 주장에 대해 몇 가지 의문도 들었습니다. 이번 섹션에서는 논문의 결과 자체를 부정하기 보다는, 어떤 부분까지 실험으로 확인되었고 어떤 부분은 추가 검증이 필요한지를 중심으로 정리해 보았습니다.

## 6.1 학습 데이터와 평가 데이터 Set Overlap 방식에 대한 의문

논문에서 이 방식을 읽으며 가장 먼저 생긴 의문은 8-gram을 생성할 때 원본 Text의 단위를 어떻게 설정했는가였습니다.

예를 들어 하나의 Sequence를 Sentence 단위로 나누었다면 8 Token보다 짧은 Sentence에서는 8-gram 자체를 생성할 수 없으며, Sentence Boundary를 넘는 중복 역시 탐지하지 못합니다. 반대로 Document 전체를 대상으로 했다면 훨씬 많은 8-gram을 생성할 수 있습니다.

하지만 논문에서는 8-gram을 생성할 때 Sentence, Paragraph, Document 중 어떤 단위를 사용했는지 명확하게 설명하지 않습니다. 따라서 해당 실험을 동일하게 재현하려면 필요한 세부 구현 정보가 일부 부족하다고 느꼈습니다.

## 6.2 Translation과 Summarization Task에서 GPT-2가 생성한 Text를 이용한 평가 방식에 대한 의문

GPT-2의 Translation과 Summarization 실험을 읽으며 가장 의문이 들었던 부분 중 하나는 모델이 실제로 생성한 전체 Text가 아니라 연구진이 선택한 일부 Text만을 최종 평가에 사용했다는 점입니다.

Summarization에서는 Article 뒤에 `TL;DR:`을 추가하여 100Token을 생성한 뒤 처음 3개의 Sentence만 Summary로 사용하여 ROUGE를 계산했습니다. Translation에서는 Example Pair를 Context로 제공하여 Greedy Decoding을 수행한 뒤 첫 번째 생성 Sentence만 Translation 결과로 사용하여 BLEU를 계산했습니다.

여기서 드는 의문은 이러한 방식으로 진행되어 보고된 성능이 Output Truncatino Rule에 얼마나 의존하는지 확인할 수 없다는 것입니다. 특히 Summarization의 경우 왜 정확히 3개의 Sentence를 선택했는지, 생성된 100Token 전체를 평가하거나 첫 1, 2, 4 Sentence를 사용했을 경우 ROUGE가 어떻게 달라지는지에 대한 비교 실험이 없습니다.

만약 다음과 같은 Ablation이 함께 제공되었다면 평가 방식을 더 명확하게 이해할 수 있었을 것입니다.

```
Generated 100 Tokens 전체
vs.
First 1 Sentence
vs.
First 2 Sentences
vs.
First 3 Sentences
vs.
First 4 Sentences
```

GPT-2가 생성한 뒷부분에 Task와 관계없는 Text가 포함되어 있었다면 첫 3 Sentence만 사용했을 때 ROUGE가 전체 Generatino보다 높아질 가능성도 있습니다. 반대로 뒷부분에 올바른 Summary Content가 있었다면 Truncation이 Score를 낮출 수도 있습니다. 논문에는 Full Generation과의 비교가 없기 때문에 실제로 어느 방향으로 영향을 주었는지는 알 수 없습니다.

Translation 역시 첫 Sentence를 Translation 결과로 사용하는 것은 Source와 Target이 Sentence 단위라는 점에서 이해할 수 있습니다. 하지만 Task-specific Translation Model이 자신의 Decoding Procdure를 통해 Translation의 종료 시점을 결정하는 것과 달리, GPT-2에서는 연구자가 외부에서 첫 Sentence라는 종료 Rule을 적용하고 있습니다.

GPT-2 저자들이 전체 Generation을 사용했을 때 성능이 낮았기 때문에 의도적으로 일부만 선택했다고 단정할 근거는 없습니다. 다만 Full Generation 또는 다른 Truncation Strategy와의 비교 결과가 제공되지 않았기 때문에 그러한 가능성을 실험 결과만으로 배제할 수도 없다고 생각합니다.

## 6.3 좋은 성능의 사전 학습을 위한 고품질 사전 학습 데이터셋을 구축하는 것에 대한 어려움

GPT-2는 Task-specific Dataset과 Supervised Fine-tuning에 의존하지 않고, 크고 다양한 Pre-training Corpus에서 학습한 하나의 Language Model이 여러 Task를 수행할 수 있는지를 탐구했습니다. 그러나 논문을 읽으며 한 가지 의문이 들었습니다. Task-specific Dataset을 만들지 않는다고 해서 Data 구축 비용 자체가 사라지는 것은 아니기 때문입니다.

GPT-2에서도 좋은 Langauge Model을 만들기 위해서는 대규모이면서 동시에 품질이 높은 Training Corpus가 필요했습니다. 연구진은 Common Crawl의 낮은 품질 문제를 피하기 위해 Reddit에서 3 Karma 이상을 얻은 Outbound Link를 Human Curation의 Heuristic으로 사용하여 WebText를 구축했습니다. 하지만 이러한 방식 역시 Data Collection, Filtering, Cleaning, Deduplication과 같은 상당한 Data Curation 과정을 요구합니다.

또한 GPT-2는 Summarization, Translation, Questino Answering 등의 여러 Task에서 Task-specific 또는 전문 시스템보다 낮은 성능을 보였습니다. 따라서 GPT-2만 놓고 본다면 대규모 Pre-trainig Dataset과 Model을 추구하는 비용을 지불하면서도, 개별 Task에서는 기존의 전문 시스템보다 부족한 경우가 많았습니다.

이러한 점에서 GPT-2가 **DataSet 구축 문제를 해결했다**고 보기는 어렵다고 생각합니다. 오히려 기존의 Task별 Labeling 문제를 범용적인 Pre-trainig Dataset을 어떻게 대규모로 수집하고 품질 관리할 것인가라는 새로운 문제로 전환했다고 생각합니다.

다만 두 방식에는 중요한 차이가 있습니다. Task-specific Dataset은 새로운 Task마다 다시 구축해야 하지만, Pre-training Dataset은 하나의 Model을 학습한 뒤 여러 Task에 재사용할 수 있습니다. 따라서 GPT-2가 제안한 방향의 의미는 단순한 Data 구축 비용 절감보다는 Task가 증가하더라도 매번 새로운 Dataset과 Objective를 설계하지 않는 방향으로 Training Paradigm을 변경했다는 것에 있다고 볼 수 있습니다.

## 6.4 GPT-2는 GPT-1과 같은 구조라고 볼 수 있을까?

GPT-2를 설명할 때 흔히 "GPT-1과 Architecture는 같고 Model Size만 키웠다"라고 단순화할 수 있습니다. 큰 관점에서는 맞는 설명입니다. GPT-1과 GPT-2 모두 Decoder-only Transformer 기반의 Autoregressive Language Model이라는 동일한 기본 Architecutre를 사용합니다. 실제로 GPT-2 논문에서도 Model이 OpenAI GPT의 세부사항을 **Largely follows**한다고 표현합니다. 하지만 정확히 동일한 Architecture라고 말하기에는 몇 가지 변경 사항이 존재합니다. 대표적으로 다음과 같습니다.

- Layer Normalization을 각 Sub-block의 Input으로 이동
- 마지막 Self-Attention Block 이후 추가 Layer Normalization 적용
- Residual Path를 고려한 Initialization 변경
- Context Length 512 → 1024
- Vocabulary Size 50,257로 증가
- Batch Size 512 사용

따라서 개인적으로는 GPT-1과 GPT-2의 관계를 다음과 같이 정리하는 것이 가장 정확하다고 생각합니다.

> GPT-2는 GPT-1과 동일한 Decoder-only Transformer 계열의 기본 Architecture를 사용하지만, 세부 Architecture와 Training Configuration에는 여러 수정이 존재하므로 완전히 동일한 구조라고 표현하는 것은 정확하지 않다.

## 6.5 GPT-2가 남긴 의미

GPT-2의 가장 중요한 의미는 단순히 1.5B Parameter의 큰 Tranformer를 만들었다는 데 있지 않습니다. GPT-1에서 Pre-trained Language Model은 기본적으로 다음과 같은 역할을 했습니다.

```
Language Model Pre-training
        ↓
좋은 Representation 획득
        ↓
Supervised Fine-tuning
        ↓
Downstream Task 수행
```

반면 GPT-2에서는 다음 가능성을 실험하기 시작했습니다.

```
Large & Diverse Text
        ↓
Language Modeling
        ↓
Context를 통해 Task 지정
        ↓
Parameter Update 없이 Task 수행
```

즉 Pre-training이 단순히 Downstream Model의 초기 Parameter를 제공하는 단계가 아니라, Langauge Modeling 그 자체가 여러 Task를 수행하는 학습 방법이 될 수 있는가라는 질문으로 관점이 확장되었습니다.

GPT-2는 이 가능성을 완전히 증명한 Model은 아닙니다. 논문에서도 Summarizatino은 아직 rudimentary한 수준이고, Translation과 Question Answering 역시 충분한 Model Capacity가 있어야 단순 Baseline을 넘기 시작했으며 실제 Application에 사용하기에는 Zero-shot 성능이 부족하다고 인정합니다.

또 하나 주의할 점은 논문에서 말하는 `Zero-shot`입니다. GPT-2에서는 Task-specific Fine-tuning이나 Parameter Update를 하지 않았다는 의미에서 Zero-shot이라고 표현하지만, Translation과 Question Answering에서는 Context에 Exmaple을 제공합니다.

따라서 오늘날의 Prompting 용어로 보면 모든 실험을 엄격한 의미의 `No-exmaple Zero-shot`이라고 보기는 어렶브니다. 마찬가지로 GPT-2가 여러 Task에서 Beahvior를 보였다는 사실이 모델 내부에 명확하게 구분된 `Translation Task`, `QA Task`, `Summarizatino Task`가 Representation되어 있다는 것을 직접 증명하는 것도 아닙니다. 실험에서 확실하게 확인된 것은 보다 제한적입니다.

실험을 통해 얻을 수 있는 결론은 "하나의 Language Modeling Objective만으로 학습한 High-capacity Model이 적절한 Natural Language Context가 주어졌을 때 여러 종류의 Task-like Behavior를 보일 수 있었다." 라고 생각합니다. 그럼에도 이 결과는 중요합니다. GPT-2 이전의 주요 관점이 "Pre-trained Model을 각 Task에 어떻게 Fine-tuning할 것인가?" 였다면, GPT-2는 "충분히 큰 Langauge Model 자체가 Context를 통해 Task를 수행할 수 있지 않을까?"라는 질문을 본격적으로 제시했기 때문입니다.

논문의 결론 역시 충분히 크고 다양한 Dataset에서 Likelihood를 학습한 High-capacity Language Model이 명시적인 Supervision 없이 여러 Task를 수행하는 방법을 학습하기 시작할 가능성을 제시합니다. 따라서 GPT-2 논문의 의미를 한 문장으로 정리를 해보자면 다음과 같습니다.

>GPT-2는 Language Model을 단순한 Pre-training 도구에서, Natural Language Context를 통해 여러 Task를 직접 수행할 가능성을 가진 General-purpose Model로 바라보는 관점의 전환을 보여준 연구이다.

# 마치며

LLM의 기초를 공부하고자 GPT-3 논문을 읽으려고 했지만 GPT-2 논문을 먼저 읽고 GPT-3 논문을 읽는 것이 좋다는 내용을 보고 GPT-2 논문을 읽었고, GPT-2 논문의 관련 내용과 논문을 읽으면서 들었던 생각과 의견들을 정리해 GPT-2 논문 리뷰 포스트를 작성했습니다.

일단 GPT-2 저자들의 Langauge Model의 Parameter에 주목한 것과, 이를 바탕으로 현재의 Prompt라고 볼 수 있는 Task Behavior를 생각해 이를 실험으로 입증했다는 부분이 정말 감탄스러웠습니다.

다음은 이번에 정리한 GPT-2의 내용을 기반으로 GPT-3 논문에서는 어떤 부분에 주목하여 실험을 진행했는지 알아보고, 논문 리뷰를 작성해 보도록 하겠습니다.

# 참조

- GPT-2 원 논문 [Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.](https://cdn.openai.com/better-language-models/language-models.pdf?utm_source=chatgpt.com)

- BPE 원 논문 [Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. Proceedings of ACL 2016, 1715–1725. DOI: 10.18653/v1/P16-1162.](https://aclanthology.org/P16-1162/?utm_source=chatgpt.com)

- [OpenAI GPT-2 Repository](https://github.com/openai/gpt-2?utm_source=chatgpt.com)

- [OpenAI tiktoken의 GPT-2 Encoding 정의](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py?utm_source=chatgpt.com)

- [Bottom-Up Abstractive Summarization](https://aclanthology.org/D18-1443/?utm_source=chatgpt.com)