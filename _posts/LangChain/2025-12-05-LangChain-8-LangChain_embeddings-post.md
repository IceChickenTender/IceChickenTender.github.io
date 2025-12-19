---
title: "[LLM/RAG] LangChain - 8. LangChain에서의 임베딩(Embedding) 알아보기"
categories:
  - LangChain
tags:
  - LangChain
  
use_math: true  
toc: true
toc_sticky: true
toc_label: "LangChain에서의 임베딩(Embedding) 알아보기"
---

# 1. Embedding 개요

임베딩(Embedding)은 텍스트 데이터를 숫자로 이루어진 벡터로 변환시킨 것을 말합니다. 이러한 벡터 표현을 사용하면, 텍스트 데이터를 벡터 공간 내에서 수학적으로 다룰 수 있게 되며, 이를 통해 텍스트 간의 유사성을 계산하거나, 텍스트 데이터를 기반으로 하는 다양한 머신러닝 및 자연어 처리 작업을 수행할 수 있습니다. 임베딩 과정은 텍스트의 의미적인 정보를 보존하도록 설계 되어 있어, 벡터 공간에서 가까이 위치한 텍스트 조각들은 의미적으로도 유사한 것으로 간주됩니다.

임베딩의 주요 활용 사례

- 의미 검색(Semantic Search) : 벡터 표현을 활용하여 의미적으로 유사한 텍스트를 검색하는 과정으로, 사용자가 입력한 쿼리에 대해 가장 관련성 높은 문서나 정보를 찾아내는 데 사용됩니다.

- 문서 분류(Document Classification) : 임베딩된 텍스트 벡터를 사용하여 문서를 특정 카테고리나 주제에 할당하는 분류 작업에 사용됩니다.

- 텍스트 유사도 계산(Text Similarity Calculation) : 두 텍스트 벡터 사이의 거리를 계산하여, 텍스트 간의 유사성 정도를 정량적으로 평가합니다.

임베딩 모델 제공자
- OpenAI : GPT와 같은 언어 모델을 통해 텍스트의 임베딩 벡터를 생성할 수 있는 API 를 제공합니다.
- Hugging Face : Transformers 라이브러리를 통해 다양한 오픈소스 임베딩 모델을 제공합니다.
- Google : Gemini, Gemma 등 언어 모델에 적용되는 임베딩 모델을 제공합니다.

임베딩 메서드
- embed_documents : 이 메소드는 문서 객체의 집합을 입력으로 받아, 각 문서를 벡터 공간에 임베딩합니다. 주로 대량의 텍스트 데이터를 배치 단위로 처리할 때 사용됩니다.
- embed_query : 이 메소드는 단일 텍스트 쿼리를 입력으로 받아, 쿼리를 벡터 공간에 임베딩합니다. 주로 사용자의 검색 쿼리를 임베딩하여, 문서 집합 내에서 해당 쿼리와 유사한 내용을 찾아내는 데 사용됩니다.

임베딩은 텍스트 데이터를 머신러닝 모델이 이해할 수 있는 형태로 변환하는 핵심 과정입니다. 다양한 자연어 처리 작업의 기반이 되는 중요한 작업입니다.

# 2. OpenAIEmbeddings

`OpenAIEmbeddings` 클래스는 OpenAI 의 API 를 활용하여, 각 문서를 대응하는 임베딩 벡터로 변환합니다. `langchain_openai` 라이브러리에서 `OpenAIEmbeddings` 클래스를 직접 임포트합니다.

아래 코드에서 `embed_documents` 메소드는 입력 받은 5개의 문서 객체를 각각 별도의 벡터로 임베딩합니다. `embeddings` 변수에는 각 텍스트에 대한 벡터 표현을 담고 있는 리스트가 할당됩니다. `len(embeddings)` 는 입력된 텍스트 리스트의 개수와 동일하며, 이는 임베딩 과정을 거친 문서의 총 수를 나타냅니다.

`len(embeddings[0])` 는 첫 번째 문서의 벡터 표현의 차원을 나타냅니다. 일반적으로 이 차원 수는 선택된 모델에 따라 정해지며, 모든 임베딩 벡터는 동일한 차원을 가집니다. OpenAI 의 임베딩 모델을 사용할 경우 임베딩 벡터의 차원은 1536 이라는 것을 확인할 수 있습니다.

```python
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()

embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)

len(embeddings), len(embeddings[0])
```

```
실행 결과

(5, 1536)
```

첫 번째 문서의 변환된 임베딩 벡터를 출력해 봅니다. 1536 차원 중에서 앞에서 20차원에 해당하는 원소만을 출력합니다. 이처럼 각 문서를 임베딩으로 변환하면 숫자를 원소로 갖는 긴 벡터 형태를 갖게 됩니다.

```python
print(embeddings[0][:20])
```

```
실행 결과

[-0.010458921082317829, -0.013548481278121471, -0.006539991125464439, -0.01863865926861763, -0.018246132880449295, 0.016625380143523216, -0.009211701340973377, 0.0039442540146410465, -0.007413678336888552, 0.01007272582501173, 0.011775783263146877, -0.006723592057824135, -0.02538757584989071, -0.022538594901561737, -0.004830603487789631, -0.021804191172122955, 0.025286277756094933, -0.017651012167334557, 0.007939157076179981, -0.017840944230556488]
```

`embed_query` 메소드는 단일 쿼리 문자열을 받아 이를 벡터 공간에 임베딩합니다. 주로 검색 쿼리나 질문 같은 단일 텍스트를 임베딩할 때 유용하며, 생성된 임베딩을 사용해 유사한 문서나 답변을 찾을 수 있습니다.

```python
from langchain_openai import OpenAIEmbeddings

embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')
embedded_query[:5]
```

```
실행 결과

[0.003640108974650502,
 -0.024275783449411392,
 0.010910888202488422,
 -0.04110145568847656,
 -0.004543057177215815]
```

코사인 유사도는 두 벡터 간의 코사인 각을 이용하여 유사성을 측정하는 방법입니다. 두 벡터의 바향이 완전히 동일하면 코사인 유사도는 1이 됩니다. 90도로 수직이면 0, 반대 방향이면 -1이 됩니다. 이는 텍스트 임베딩과 같이 고차원에서도 벡터 간 유사도를 측정하는 데 유용하게 사용됩니다.

주어진 cos_sim 함수는 두 벡터 A와 B 사이의 코사인 유사도를 계산합니다. `dot(A, B)` 는 두 벡터의 내적을, `norm(A)`와 `norm(B)`는 각각 벡터 A와 B의 노름(크기)을 계산합니다. 이 함수는 내적 값과 두 벡터 크기의 곱으로 나눈 값으로 코사인 유사도를 계산합니다.

다음 예시는 앞에서 임베딩 변환한 문서들(embeddings)과 하나의 임베딩된 쿼리(embedded_query) 사이의 코사인 유사도를 계산하여 출력합니다. 각 문서 임베딩에 대해 cos_sim 함수를 호출하여, 해당 문서가 쿼리와 얼마나 유사한지를 숫자로 나타냅니다. 유사도가 높은 문서일수록 쿼리와 더 관련이 깊다고 볼 수 있습니다.

```python

# 코사인 유사도

import numpy as np
from numpy import dot
from numpy.linalg import norm
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()

embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)

embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')

def cos_sim(A, B):
  return dot(A, B) / (norm(A)*norm(B))

for embedding in embeddings:
  print(cos_sim(embedding, embedded_query))

```

다음은 임베딩된 모든 문서와 쿼리 사이의 코사인 유사도를 출력한 결과입니다. "인사"와 "이름"이라는 두 가지 토픽에 대해서 상대적으로 관련성이 높은 문서들이 유사도가 높은 것을 확인할 수 있습니다. 이러한 방식으로 문서 검색, 추천 시스템 등 다양한 자연어 처리 작업에서 유사도 기반 필터링이나 정렬을 수행할 수 있습니다.

```
실행 결과

0.8347781524360807
0.8153837322339593
0.8843960106566056
0.7899011862340304
0.7468198077293927
```

# 3. HuggingFaceEmbeddings

`sentence-transformers` 라이브러리를 사용하면 HuggingFace 모델에서 사용된 사전 훈련된 임베딩 모델을 다운로드 받아서 적용할 수 있습니다. OpenAI 임베딩 모델을 사용할 때는 API 사용료가 부과되지만, HuggingFace 의 오픈소스 기반의 임베딩 모델을 사용하면 요금이 부과되지 않습니다.

먼저 `sentence-transformers` 라이브러리를 설치합니다.

```
!pip install -U sentence-transformers
```

HuggingFaceEmbeddings 클래스는 Hugging Face 의 트랜스포머 모델을 사용하여 문서 또는 문장을 임베딩하는 데 사용됩니다. 다음은 주요 매개변수의 설정 값을 설명합니다.

- `model_name` : 사용할 모델을 지정합니다. 여기서는 한국어 자연어 추론(Natural Language Inference, NLI)에 최적화된 ko-sroberta 모델을 사용합니다.

- `model_kwargs` : 모델이 cpu 에서 실행되도록 설정합니다. gpu 를 사용할 수 있는 환경이라면 cuda 로 설정할 수도 있습니다.

- `encode_kwargs` : 임베딩을 정규화하여 모든 벡터가 같은 범위의 값을 갖도록 합니다. 이는 유사도 계산 시 일관성을 높여줍니다.

`embeddings_model` 을 출력해보면 `Pooling` 레이어의 `word_embedding_dimension` 값에서 임베딩 벡터의 크기를 확인할 수 있습니다. 768 차원의 벡터라는 것을 알 수 있습니다.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name = "jhgan/ko-sroberta-nli",
    model_kwargs = {'device' : 'cuda'},
    encode_kwargs = {'normalize_embeddings':True}
)

embeddings_model
```

```python
실행 결과

HuggingFaceEmbeddings(client=SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False, 'architecture': 'RobertaModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
), model_name='jhgan/ko-sroberta-nli', cache_folder=None, model_kwargs={'device': 'cuda'}, encode_kwargs={'normalize_embeddings': True}, multi_process=False, show_progress=False)
```

`embed_documents` 메소드는 주어진 5개의 문장으로 구성된 텍스트 리스트를 임베딩합니다. 임베딩 벡터는 768차원으로 확인됩니다.

```python
embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)
len(embeddings), len(embeddings[0])

```

```
실행 결과

(5, 768)
```

`embed_query` 메소드는 단일 쿼리 문장을 임베딩합니다. 이렇게 생성된 임베딩은 cos_sim 함수를 사용하여 쿼리와 각 문서 간의 코사인 유사도를 계산합니다. 이 유사도 점수를 통해 쿼리와 가장 관련이 깊은 문서를 파악할 수 있습니다.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
  return dot(A, B) / (norm(A)*norm(B))

embeddings_model = HuggingFaceEmbeddings(
    model_name = "jhgan/ko-sroberta-nli",
    model_kwargs = {'device' : 'cuda'},
    encode_kwargs = {'normalize_embeddings':True}
)

#embeddings_model

embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)
len(embeddings), len(embeddings[0])

embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')

for embedding in embeddings:
    print(cos_sim(embedding, embedded_query))

```

```
실행 결과

0.5899016046274541
0.41826309410089174
0.7240604881408343
0.05702663569167117
0.4316417573777297
```

---

# 4. GoogleGenerativeAIEmbeddings

`langchain_google_genai` 라이브러리와 `GoogleGenerativeAIEmbeddings` 클래스를 사용하면 Google 의 생성형 AI 모델을 활용하여 문서나 문장을 임베딩할 수 있습니다.

먼저 `langchain_google_genai` 라이브러리를 설치합니다. `-q` 플래그는 로그 출력을 최소화합니다.

```
!pip install -q langchain_google_genai
```

Google API 사용을 위해선 환경 변수에 API 키를 설정합니다. `GOOGLE_API_KEY`는 실제 사용자의 API 키로 대체합니다.

```python
import os

os.environ['GOOGLE_API_KEY'] = 'GOOGLE_API_KEY'
```

`GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')`을 사용하여 임베딩 모델의 인스턴스를 생성합니다. 이 때, 사용할 Google 의 생성형 AI 모델을 `model` 인자를 통해 지정합니다. 여기서는 `models/gemini-embedding-001` 을 사용하도록 지정하고 있습니다.

`embed_documents` 메소드를 호출하여, 주어진 텍스트 리스트를 임베딩합니다. 이 메소드는 각 문서 또는 문장을 벡터 공간에 매핑된 임베딩으로 변환합니다. 

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)
len(embeddings), len(embeddings[0])
```

```
실행 결과

(5, 768)
```

이번엔 쿼리에 대한 임베딩을 생성하고, 코사인 유사도를 구해보도록 하겠습니다.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings_model = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)
len(embeddings), len(embeddings[0])

embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')

for embedding in embeddings:
    print(cos_sim(embedding, embedded_query))

```

```
실행 결과

0.9483334221872353
0.9563804554562886
0.9988387578259003
0.9707770904712603
0.8775081156240261
```

# 5. OllamaEmbeddings

이전 LangChain에서의 LLM 포스트에서도 설명했지만 다시 한 번 설명을 하자면 Ollama는 로컬 환경에서 대규모 언어 모델을 쉽게 실행할 수 있게 해주는 오픈 소스 프로젝트입니다. 이 도구는 다양한 LLM을 간단한 명령어로 다운로드하고 실행할 수 있게 해주며, 개발자들이 AI 모델을 자신의 컴퓨터에서 직접 실험하고 사용할 수 있도록 지원합니다. Ollama는 사용자 친화적인 인터페이스와 빠른 성능으로 AI개발 및 실험을 더욱 접근하기 쉽고 효율적으로 만들어주는 도구입니다.

Ollama를 이용한 임베딩을 진행하기 전에 우선 Ollama 설치부터 진행해 주도록 하겠습니다. 터미널에 다음 두 명령어를 실행해 Ollama 설치와 실행을 진행해 줍니다.

```bash
# 올라마 설치 명령어
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
# 올라마 실행 명령어
ollama serve &
```

그리고 이번에 사용할 임베딩 모델도 Ollama에 pull 해줍니다. 이번 예제에서 사용할 모델은 Ollama 설치 시 설치가 되어 있지 않기 때문에 직접 설치를 해주어야 합니다.

```bash
ollama pull nomic-embed-text
```

그럼 이제 예제를 실행하기 위한 라이브러리 설치를 진행해 줍니다.

```bash
pip install langchain-ollama
```

아래는 예제에서 사용할 문서 리스트입니다. 

```python
texts = [
    "안녕, 만나서 반가워.",
    "LangChain simplifies the process of building applications with large language models",
    "랭체인 한국어 튜토리얼은 LangChain의 공식 문서, cookbook 및 다양한 실용 예제를 바탕으로 하여 사용자가 LangChain을 더 쉽고 효과적으로 활용할 수 있도록 구성되어 있습니다. ",
    "LangChain은 초거대 언어모델로 애플리케이션을 구축하는 과정을 단순화합니다.",
    "Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.",
]
```

좀 전에 Ollama에 설치했던 `nomic-embed-text` 모델을 불러와 줍니다.

```python
from langchain_ollama import OllamaEmbeddings

ollama_embeddings = OllamaEmbeddings(
    model = "nomic-embed-text",
)
```

미리 선언해 두었던 임베딩 모델을 이용해 문서와 쿼리를 임베딩 해 줍니다.

```python
# 문서 임베딩
embedded_documents = ollama_embeddings.embed_documents(texts)

# 쿼리 임베딩
embedded_query = ollama_embeddings.embed_query("LangChain에 대해서 상세히 알려주세요.")
print(len(embedded_query))
```

문서 임베딩과 쿼리 임베딩을 이용해 유사도를 계산해 그 결과를 출력해 봅니다.

```python
import numpy as np

# 유사도 계산 결과 출력
similarity = np.array(embedded_query) @ np.array(embedded_documents).T

# 유사도 기준 내림차순 정렬
sorted_idx = (np.array(embedded_query) @ np.array(embedded_documents).T).argsort()[::-1]

# 결과 출력
print("[Query] LangChain에 대해서 알려주세요\n==========================================")
for i, idx in enumerate(sorted_idx):
    print(f"[{i}] 유사도: {similarity[idx]:.3f} | {texts[idx]}")
    print()
```

```
Output:
[Query] LangChain에 대해서 알려주세요
==========================================
[0] 유사도: 0.978 | LangChain은 초거대 언어모델로 애플리케이션을 구축하는 과정을 단순화합니다.

[1] 유사도: 0.935 | 안녕, 만나서 반가워.

[2] 유사도: 0.733 | Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.

[3] 유사도: 0.686 | 랭체인 한국어 튜토리얼은 LangChain의 공식 문서, cookbook 및 다양한 실용 예제를 바탕으로 하여 사용자가 LangChain을 더 쉽고 효과적으로 활용할 수 있도록 구성되어 있습니다. 

[4] 유사도: 0.555 | LangChain simplifies the process of building applications with large language models
```

Ollama에서 지원하는 임베딩 모델은 <https://ollama.com/library>에서 확인할 수 있습니다.

# 6. GPT4ALL 임베딩

GPT4ALL은 무료로 사용할 수 있는 로컬 실행 기반의 개인정보를 고려한 챗봇입니다. GPU나 인터넷 연결이 필요하지 않으며, GPT4ALL Falcon, Wizard 등 인기 있는 모델과 자체 모델을 제공합니다. LangChain과 함께 GPT4ALL embeddings를 사용하는 방법에 대해서 알아보도록 하겠습니다.

우선 GPT4ALL을 실행하기 위해서 GPT4ALL Python 바인딩 설치를 먼저 진행해 주어야 합니다. 아래 명령어를 실행해 주세요.

```python
%pip install --upgrade --quiet gpt4all > /dev/null
```

그리고 GPT4ALLEmbeddings를 사용하기 위해 langchain-community 라이브러리 설치도 진행합니다.

```python
pip install langchain-community
```

langchain_community.embeddings 모듈에서 GPT4AllEmbeddings 클래스를 임포트합니다. GPT4AllEmbeddings는 GPT4ALL을 사용하여 텍스트 데이터를 벡터로 임베딩하는 기능을 제공하는 클래스입니다. 이 클래스는 LangChain 프레임워크의 임베딩 인터페이스를 구현하여, LangChain의 다양한 기능과 함께 사용할 수 있습니다. GPT4All은 CPU에 최적화된 대조 학습 문장 변환기를 사용하여 임의 길이의 텍스트 문서에 대한 고품질 임베딩 생성을 지원합니다. 이러한 임베딩은 OpenAI를 사용하는 많은 작업에서 품질이 비슷합니다.

그렇다면 이전 Ollama 임베딩에서와 같이 예제 문서와 쿼리를 임베딩해서 유사도 계산을 진행해 보도록 하겠습니다.

```python
texts = [
    "안녕, 만나서 반가워.",
    "LangChain simplifies the process of building applications with large language models",
    "랭체인 한국어 튜토리얼은 LangChain의 공식 문서, cookbook 및 다양한 실용 예제를 바탕으로 하여 사용자가 LangChain을 더 쉽고 효과적으로 활용할 수 있도록 구성되어 있습니다. ",
    "LangChain은 초거대 언어모델로 애플리케이션을 구축하는 과정을 단순화합니다.",
    "Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.",
]
```

```python
from langchain_community.embeddings import GPT4AllEmbeddings

gpt4all_embed = GPT4AllEmbeddings()

gpt4all_documents_embeddings = gpt4all_embed.embed_documents(
    texts
)

gpt4all_query_embeddings = gpt4all_embed.embed_query(
    "LangChain에 대해서 상세히 알려주세요"
)

print(len(gpt4all_query_embeddings))
```

```
Output:
384
```

```python
import numpy as np

# 유사도 계산 결과 출력
similarity = np.array(gpt4all_documents_embeddings) @ np.array(gpt4all_query_embeddings).T

# 유사도 기준 내림차순 정렬
sorted_idx = (np.array(gpt4all_query_embeddings) @ np.array(gpt4all_documents_embeddings).T).argsort()[::-1]

# 결과 출력
print("[Query] LangChain에 대해서 알려주세요\n==========================================")
for i, idx in enumerate(sorted_idx):
    print(f"[{i}] 유사도: {similarity[idx]:.3f} | {texts[idx]}")
    print()
```

유사도 계산 결과를 보면 Ollama에서 사용했던 nomic-embed-text 모델보다는 한국어에서는 좋은 유사도 결과를 보여주고 있는 것을 확인할 수 있습니다.

```
Output:
[Query] LangChain에 대해서 알려주세요
==========================================
[0] 유사도: 0.761 | 랭체인 한국어 튜토리얼은 LangChain의 공식 문서, cookbook 및 다양한 실용 예제를 바탕으로 하여 사용자가 LangChain을 더 쉽고 효과적으로 활용할 수 있도록 구성되어 있습니다. 

[1] 유사도: 0.698 | LangChain은 초거대 언어모델로 애플리케이션을 구축하는 과정을 단순화합니다.

[2] 유사도: 0.555 | 안녕, 만나서 반가워.

[3] 유사도: 0.421 | LangChain simplifies the process of building applications with large language models

[4] 유사도: 0.075 | Retrieval-Augmented Generation (RAG) is an effective technique for improving AI responses.
```

# 마치며

LangChain에서 사용하는 임베딩을 제공하는 여러 플랫폼들에 대해서 알아보았습니다. 이 포스트를 준비하면서 여러가지 임베딩을 제공하는 플랫폼들이 있는 것을 알게 되었고, 사용하고자 하는 목적이나 환경에 따라서 플랫폼을 유연하게 사용할 수 있다는 것을 알게 되었습니다. 예를 들면, GPU가 없는 환경이거나, 극적인 성능을 기대하는 작업을 한다면 비용이 들더라도 OpenAI나 Google 등과 같이 성능이 보장되어 있는 임베딩을 사용하고, GPU가 있거나 GPU가 없어도 Ollama나 GPT4ALL을 이용해 비용을 최대한 아끼면서 임베딩을 생성할 수 있습니다.

긴 글 읽어주셔서 감사드리며, 본문 내용 중에 잘못된 내용이나 오타, 궁금하신 사항이 있으시다면 댓글 달아주시기 바랍니다.

# 참조

- 판다스 스튜디오 - 랭체인(LangChain) 입문부터 응용까지(https://wikidocs.net/book/14473)
- 테디노트 - LangChain 한국어 튜토리얼KR(https://wikidocs.net/book/14314)