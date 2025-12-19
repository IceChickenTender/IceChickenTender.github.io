---
title: "[LLM/RAG] LangChain - 10. LangChain 의 검색기(Retriever)"
categories:
  - LangChain

tags:
  - LangChain
  
use_math: true  
toc: true
toc_sticky: true
toc_label: "LangChain의 검색기(Retriever)"
---

# 1. 검색기(Retriever) 개요

검색기(Retriever)는 RAG(Retriever-Augmented Generation) 시스템의 다섯 번째 단계로, 저장된 벡터 데이터베이스에서 사용자의 질문과 관련된 문서를 검색하는 과정입니다. 이 단계는 사용자 질문에 가장 적합한 정보를 신속하게 찾아내는 것이 목표이며, RAG 시스템의 전반적인 성능과 직결되는 매우 중요한 과정입니다.

## 1.1 검색기의 필요성

1. 정확한 정보 제공: 검색기는 사용자의 질문과 가장 관련성 높은 정보를 검색하여, LLM이 정확하고 유용한 답변을 생성할 수 있도록 합니다. 이 과정이 효과적으로 이루어지지 않으면, 결과적으로 제공되는 답변의 품질이 떨어질 수 있습니다.

2. 응답 시간 단축: 효율적인 검색 알고리즘을 사용하여 데이터베이스에서 적절한 정보를 빠르게 검색함으로써, 전체적인 시스템 응답 시간을 단축시킵니다. 사용자 경험 향상에 직접적인 영향을 미칩니다.

3. 최적화: 효과적인 검색 과정을 통해 필요한 정보만을 추출함으로써 시스템 자원의 사용을 최적화하고, 불필요한 데이터 처리를 줄일 수 있습니다.

## 1.2 동작 방식

1. 질문의 벡터화: 사용자의 질문을 벡터 형태로 변환합니다. 이 과정은 임베딩 단계와 유사한 기술을 사용하여 진행됩니다. 변환된 질문 벡터는 후속 검색 작업의 기준점으로 사용됩니다.

2. 벡터 유사성 비교: 저장된 문서 벡터들과 질문 벡터 사이의 유사성을 계산합니다. 이는 주로 코사인 유사성(cosine similarity), Max Marginal Relevance(MMR)등의 수학적 방법을 사용하여 수행됩니다.

3. 상위 문서 선정: 계산된 유사성 점수를 기준으로 상위 N개의 가장 관련성 높은 문서를 선정합니다. 이 문서들은 다음 단계에서 사용자의 질문에 대한 답변을 생성하는데 사용됩니다.

4. 문서 정보 반환: 선정된 문서들의 정보를 다음 단계(프롬프트 생성)로 전달합니다. 이 정보에는 문서의 내용, 위치, 메타데이터 등이 포함될 수 있습니다.

## 1.3 검색기의 중요성

검색기는 RAG 시스템에서 정보 검색의 질을 결정하는 핵심적인 역할을 합니다. 효율적인 검색기 없이는 대규모 데이터베이스에서 관련 정보를 신속하고 정확하게 찾아내는 것이 매우 어렵습니다. 또한 검색기는 사용자 질문에 대한 적절한 컨텍스트(문맥)를 제공하여, LLM이 보다 정확한 답변을 생성할 수 있도록 돕습니다. 따라서 검색기의 성능은 RAG 시스템의 전반적인 효율성과 사용자 만족도에 직접적인 영향을 미칩니다.

## 1.4 Sparse Retriever & Dense Retriever

Sparse Retriever와 Dense Retriever는 정보 검색 시스템에서 사용되는 두 가지 주요 방법입니다. 이들은 자연어 처리 분야, 특히 대규모 문서 집합에서 관련 문서를 검색할 때 사용됩니다.

### 1.4.1 Sparse Retriever

Sparse Retriever는 문서와 쿼리를 이산적인 키워드 벡터인 Sparse Vector로 변환하여 문서를 검색하는 방법입니다. 이 방법은 주로 TF-IDF나 BM25와 같은 전통적인 정보 검색 기법을 사용합니다. Sparse Retriever의 특징은 각 단어의 존재 여부만을 고려하기 때문에 계산 비용이 낮고, 구현이 간단하다는 점입니다. 그러나 이 방법은 단어의 의미적 연관성을 고려하지 않으며, 검색 결과의 품질이 키워드의 선택에 크게 의존합니다.

### 1.4.2 Dense Retriever

Dense Retriever는 최신 딥러닝 기법을 사용하여 문서와 쿼리를 연속적인 고차원 벡터인 Dense Vector로 인코딩하고, 인코딩된 Dense Vector를 이용해 문서 검색을 하는 방법입니다. Dense Retriever는 문서의 의미적 내용을 보다 풍부하게 표현할 수 있으며, 키워드가 완벽히 일치하지 않더라도 의미적으로 관련된 문서를 검색할 수 있습니다. Dense Retriever는 벡터 공간에서의 거리(예시: 코사인 유사도)를 사용하여 쿼리와 가장 관련성 높은 문서를 찾습니다. 이 방식은 특히 언어의 뉘앙스와 문맥을 이해하는 데 유리하며, 복잡한 쿼리에 대해 더 정확한 검색 결과를 제공할 수 있습니다.

### 1.4.3 Sparse와 Dense의 차이점

1. 표현 방식: Sparse Retriever는 이산적인 키워드 기반의 표현을 사용하는 반면, Dense Retriever는 연속적인 벡터 공간에서 의미적 표현을 사용합니다.

2. 의미적 처리 능력: Dense Retriever는 문맥과 의미를 더 깊이 파악할 수 있어, 키워드가 정확히 일치하지 않아도 관련 문서를 검색할 수 있습니다. Sparse Retriever는 이러한 의미적 뉘앙스를 덜 반영합니다.

3. 적용 범위: 복잡한 질문이나 자연어 쿼리에 대해서는 Dense Retrieve가 더 적합할 수 있으며, 간단하고 명확한 키워드 검색에는 Sparse Retriever가 더 유용할 수 있습니다.



# 2. RAG Retriever

RAG (Retrieval Augmented Generation)에서 검색도구(Retrievers)는 벡터 저장소에서 문서를 검색하는 도구입니다. LangChain 은 간단한 의미 검색도구부터 성능 향상을 위해 고려된 다양한 검색 알고리즘을 지원합니다. 이번 챕터에서는 LangChain 에서 제공하는 다양한 검색도구에 대해서 알아보겠습니다.

## 2.1 Vector Store Retriever

벡터스토어 검색도구(Vector Store Retriever)를 사용하면 대량의 텍스트 데이터에서 관련 정보를 효율적으로 검색할 수 있습니다. 다음 코드에서는 LangChain 의 벡터 스토어와 임베딩 모델을 사용하여 문서들의 임베딩을 생성하고, 그 후 저장된 임베딩들을 기반으로 검색 쿼리에 가장 관련 있는 문서들을 검색하는 방법을 설명합니다.

### 2.1.1 사전 준비

우선 실행에 필요한 라이브러리 설치부터 진행해 줍니다.

```bash
pip install langchain langchain-openai tiktoken langchain-chroma langchain-community pypdf sentence-transformers pymupdf faiss-cpu langchain-huggingface
```

이전 Chroma 와 FAISS 를 공부하면서 진행했던 pdf 문서를 로드하고, 텍스트 분리를 진행해 chunking 을 진행하고, 임베딩 모델을 이용해 문서 임베딩을 벡터스토어에 저장합니다. 이전에 진행했던 것과 같이 데이터는 "카카오뱅크 2022 지속가능경영보고서.pdf" 데이터를 사용하며, 임베딩 모델은 `HuggingFaceEmbeddings` 의 `jhgan/ko-sbert-nli` 를 사용하며, 벡터DB 는 FAISS 를 사용합니다. 전체 소스코드는 다음과 같습니다.

```python
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load data -> Text split

loader = PyMuPDFLoader("/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

documents = text_splitter.split_documents(data)

# 벡터스토어 db 인스턴스를 생성
embeddings_model = HuggingFaceEmbeddings(
    model_name = "jhgan/ko-sroberta-multitask",
    model_kwargs = {'device' : 'cuda'},
    encode_kwargs = {'normalize_embeddings':True}
)

vectorstore = FAISS.from_documents(documents,
                                    embedding = embeddings_model,
                                    distance_strategy = DistanceStrategy.COSINE
                                    )

```

### 2.1.2 단일 문서 검색

검색 쿼리를 정의한 후, `as_retriever` 메소드를 사용하여 벡터스토어에서 Retriever 객체를 생성합니다. `search_kwargs`에서 `k:1`을 설정하여 가장 유사도가 높은 하나의 문서를 검색합니다.

```python
# 검색 쿼리
query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘'

# 가장 유사도가 높은 문장을 하나만 추출
retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

docs = retriever.get_relevant_documents(query)
docs[0]

```

```
실행 결과

/tmp/ipython-input-2101203039.py:8: LangChainDeprecationWarning: The method `BaseRetriever.get_relevant_documents` was deprecated in langchain-core 0.1.46 and will be removed in 1.0. Use :meth:`~invoke` instead.
  docs = retriever.get_relevant_documents(query)
Document(id='46c68410-503b-436f-9285-736d729386d4', metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.1 (Macintosh)', 'creationdate': '2023-06-21T17:01:54+09:00', 'source': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'file_path': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'total_pages': 99, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'moddate': '2023-06-21T17:02:34+09:00', 'trapped': '', 'modDate': "D:20230621170234+09'00'", 'creationDate': "D:20230621170154+09'00'", 'page': 21}, page_content='카카오뱅크는 중대성 평가를 통해 도출된 6가지 중대 주제에 대한 활동을 공개하고 성과를 평가하며 지속가능한 경영 보고 체계를 수립하고 있습니다.\nManagement Approach\n구분\nESG 경영 이행\n환경경영체계\n구축 및\n운영 내재화\n인권경영 확대\n이사회 건전성 강화\n포용적 금융을 통한\n경제 및 사회적\n가치 창출\n정보보안 및\n고객정보 관리\n주요 성과\n• 2022 MSCI 평가등급 BBB\n• 2022 ESG 펀드 신규 투자\n• 2022 ESG위원회 5회 개최\n• 2023 ISO 14001 획득\n• 2022 UN Global Compact 가입\n• 2023 인권영향평가 시행\n• 2022 ESG위원회 신설\n• 2022 여성 사외이사 신규 선임\n• 2022 중신용대출 취급 확대,\n\t 개인사업자 신용대출 출시\n• 2023 사회적 가치 측정 첫 시도\n• 2022 정보보호 관련 인증 유지\n\t (ISMS, ISO 27001) \n• 개인정보 관련 인증 최초 획득\n\t (ISO 27701)\n중대 주제 선정 이유 및 영향\n카카오뱅크는 설립 단계부터 비즈니스의 환경적, 사회적 가치 창출을 고려하였습니다.\n카카오뱅크는 기술과 혁신을 통해 기존 금융권에서 소외되어 있던 계층을 포함한 모든 사회\n구성원의 금융 접근성을 향상했습니다.\n카카오뱅크는 ‘더 나은 세상을 위해 환경을 지키는 것’을 환경경영의 목표로 삼고, 기업활동\n전반에 걸쳐 발생하는 환경영향을 최소화하여 지속가능한 미래에 기여하고자 합니다.\n특히, 카카오뱅크의 비즈니스가 지속적으로 성장함에 따라 온실가스 배출량을 비롯한 환경에의\n영향 확대가 불가피한 현실에서 환경 리스크를 전사 차원의 주요 리스크로 인식하고 관리해야\n함을 인지하고 있습니다.\n카카오뱅크는 임직원 뿐만 아니라 협력사, 고객, 지역사회와 같은 모든 이해관계자의 인권이\n마땅히\xa0존중받는 ‘사람 중심 경영’을 실천하고자 노력합니다. 인권헌장이 바탕이 된 인권경영과\n이해관계자의 참여 및 적극적인 리스크 관리로 진정성 있는 인권존중을 실현하고자 하며,')
```

### 2.1.3 MMR 검색

다양성을 고려한 MMR 검색을 사용하여 상위 5개 문서를 검색합니다. 여기서 `fetch_k : 50`는 후보 집합으로 선택되는 문서의 수를 의미하고, `k:5` 는 최종적으로 반환되는 문서의 수입니다. `lambda_mult : 0.5` 설정은 유사도와 다양성 사이에서 적용될 수준을 의미합니다. 0.5를 사용하면 중립적으로 적용하게 됩니다.

```python
# MMR - 다양성 고려 (lambda_mult = 0.5)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50}
)

docs = retriever.get_relevant_documents(query)
print(len(docs))
docs[0]

```

```
실행 결과

5
Document(id='46c68410-503b-436f-9285-736d729386d4', metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.1 (Macintosh)', 'creationdate': '2023-06-21T17:01:54+09:00', 'source': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'file_path': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'total_pages': 99, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'moddate': '2023-06-21T17:02:34+09:00', 'trapped': '', 'modDate': "D:20230621170234+09'00'", 'creationDate': "D:20230621170154+09'00'", 'page': 21}, page_content='카카오뱅크는 중대성 평가를 통해 도출된 6가지 중대 주제에 대한 활동을 공개하고 성과를 평가하며 지속가능한 경영 보고 체계를 수립하고 있습니다.\nManagement Approach\n구분\nESG 경영 이행\n환경경영체계\n구축 및\n운영 내재화\n인권경영 확대\n이사회 건전성 강화\n포용적 금융을 통한\n경제 및 사회적\n가치 창출\n정보보안 및\n고객정보 관리\n주요 성과\n• 2022 MSCI 평가등급 BBB\n• 2022 ESG 펀드 신규 투자\n• 2022 ESG위원회 5회 개최\n• 2023 ISO 14001 획득\n• 2022 UN Global Compact 가입\n• 2023 인권영향평가 시행\n• 2022 ESG위원회 신설\n• 2022 여성 사외이사 신규 선임\n• 2022 중신용대출 취급 확대,\n\t 개인사업자 신용대출 출시\n• 2023 사회적 가치 측정 첫 시도\n• 2022 정보보호 관련 인증 유지\n\t (ISMS, ISO 27001) \n• 개인정보 관련 인증 최초 획득\n\t (ISO 27701)\n중대 주제 선정 이유 및 영향\n카카오뱅크는 설립 단계부터 비즈니스의 환경적, 사회적 가치 창출을 고려하였습니다.\n카카오뱅크는 기술과 혁신을 통해 기존 금융권에서 소외되어 있던 계층을 포함한 모든 사회\n구성원의 금융 접근성을 향상했습니다.\n카카오뱅크는 ‘더 나은 세상을 위해 환경을 지키는 것’을 환경경영의 목표로 삼고, 기업활동\n전반에 걸쳐 발생하는 환경영향을 최소화하여 지속가능한 미래에 기여하고자 합니다.\n특히, 카카오뱅크의 비즈니스가 지속적으로 성장함에 따라 온실가스 배출량을 비롯한 환경에의\n영향 확대가 불가피한 현실에서 환경 리스크를 전사 차원의 주요 리스크로 인식하고 관리해야\n함을 인지하고 있습니다.\n카카오뱅크는 임직원 뿐만 아니라 협력사, 고객, 지역사회와 같은 모든 이해관계자의 인권이\n마땅히\xa0존중받는 ‘사람 중심 경영’을 실천하고자 노력합니다. 인권헌장이 바탕이 된 인권경영과\n이해관계자의 참여 및 적극적인 리스크 관리로 진정성 있는 인권존중을 실현하고자 하며,')
```

### 2.1.4 유사도 점수 임계값 기반 검색

이 방식은 설정한 `score_threshold` 유사도 점수 이상인 문서만을 대상으로 추출합니다. 여기서 임계값은 0.3으로 설정되어 있습니다. 이는 쿼리 문장과 최소한 0.3 이상의 유사도를 가진 문서만을 검색 결과로 반환하게 됩니다. 따라서 유사도가 높은 문서만 필터링하고 싶을 때 유용합니다.

```python
# Similarity score threshold (기준 스코어 이상인 문서를 대상으로 추출)
retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.3}
)

docs = retriever.get_relevant_documents(query)
print(len(docs))
docs[0]
```

실행 결과로 하나의 문서만 검색된 것을 확인할 수 있습니다.

```
실행 결과

1
Document(id='46c68410-503b-436f-9285-736d729386d4', metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.1 (Macintosh)', 'creationdate': '2023-06-21T17:01:54+09:00', 'source': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'file_path': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'total_pages': 99, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'moddate': '2023-06-21T17:02:34+09:00', 'trapped': '', 'modDate': "D:20230621170234+09'00'", 'creationDate': "D:20230621170154+09'00'", 'page': 21}, page_content='카카오뱅크는 중대성 평가를 통해 도출된 6가지 중대 주제에 대한 활동을 공개하고 성과를 평가하며 지속가능한 경영 보고 체계를 수립하고 있습니다.\nManagement Approach\n구분\nESG 경영 이행\n환경경영체계\n구축 및\n운영 내재화\n인권경영 확대\n이사회 건전성 강화\n포용적 금융을 통한\n경제 및 사회적\n가치 창출\n정보보안 및\n고객정보 관리\n주요 성과\n• 2022 MSCI 평가등급 BBB\n• 2022 ESG 펀드 신규 투자\n• 2022 ESG위원회 5회 개최\n• 2023 ISO 14001 획득\n• 2022 UN Global Compact 가입\n• 2023 인권영향평가 시행\n• 2022 ESG위원회 신설\n• 2022 여성 사외이사 신규 선임\n• 2022 중신용대출 취급 확대,\n\t 개인사업자 신용대출 출시\n• 2023 사회적 가치 측정 첫 시도\n• 2022 정보보호 관련 인증 유지\n\t (ISMS, ISO 27001) \n• 개인정보 관련 인증 최초 획득\n\t (ISO 27701)\n중대 주제 선정 이유 및 영향\n카카오뱅크는 설립 단계부터 비즈니스의 환경적, 사회적 가치 창출을 고려하였습니다.\n카카오뱅크는 기술과 혁신을 통해 기존 금융권에서 소외되어 있던 계층을 포함한 모든 사회\n구성원의 금융 접근성을 향상했습니다.\n카카오뱅크는 ‘더 나은 세상을 위해 환경을 지키는 것’을 환경경영의 목표로 삼고, 기업활동\n전반에 걸쳐 발생하는 환경영향을 최소화하여 지속가능한 미래에 기여하고자 합니다.\n특히, 카카오뱅크의 비즈니스가 지속적으로 성장함에 따라 온실가스 배출량을 비롯한 환경에의\n영향 확대가 불가피한 현실에서 환경 리스크를 전사 차원의 주요 리스크로 인식하고 관리해야\n함을 인지하고 있습니다.\n카카오뱅크는 임직원 뿐만 아니라 협력사, 고객, 지역사회와 같은 모든 이해관계자의 인권이\n마땅히\xa0존중받는 ‘사람 중심 경영’을 실천하고자 노력합니다. 인권헌장이 바탕이 된 인권경영과\n이해관계자의 참여 및 적극적인 리스크 관리로 진정성 있는 인권존중을 실현하고자 하며,')
```

### 2.1.5 메타데이터 필터링을 사용한 검색

메타데이터의 특정 필드에 대해서 기준(예:`'format', 'PDF 1.4'`)을 설정하고 조건을 충족하는 문서만을 필터링하여 검색합니다. 특정 형식이나 조건을 만족하는 문서를 검색할 때 유용합니다.

```python
# 문서 객체의 metadata 를 이용한 필터링

retriever = vectorstore.as_retriever(
    search_kwargs={'filter' : {'format':'PDF 1.4'}}
)

docs = retriever.get_relevant_documents(query)
print(len(docs))
docs[0]
```

```
실행 결과

4
Document(id='46c68410-503b-436f-9285-736d729386d4', metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.1 (Macintosh)', 'creationdate': '2023-06-21T17:01:54+09:00', 'source': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'file_path': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'total_pages': 99, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'moddate': '2023-06-21T17:02:34+09:00', 'trapped': '', 'modDate': "D:20230621170234+09'00'", 'creationDate': "D:20230621170154+09'00'", 'page': 21}, page_content='카카오뱅크는 중대성 평가를 통해 도출된 6가지 중대 주제에 대한 활동을 공개하고 성과를 평가하며 지속가능한 경영 보고 체계를 수립하고 있습니다.\nManagement Approach\n구분\nESG 경영 이행\n환경경영체계\n구축 및\n운영 내재화\n인권경영 확대\n이사회 건전성 강화\n포용적 금융을 통한\n경제 및 사회적\n가치 창출\n정보보안 및\n고객정보 관리\n주요 성과\n• 2022 MSCI 평가등급 BBB\n• 2022 ESG 펀드 신규 투자\n• 2022 ESG위원회 5회 개최\n• 2023 ISO 14001 획득\n• 2022 UN Global Compact 가입\n• 2023 인권영향평가 시행\n• 2022 ESG위원회 신설\n• 2022 여성 사외이사 신규 선임\n• 2022 중신용대출 취급 확대,\n\t 개인사업자 신용대출 출시\n• 2023 사회적 가치 측정 첫 시도\n• 2022 정보보호 관련 인증 유지\n\t (ISMS, ISO 27001) \n• 개인정보 관련 인증 최초 획득\n\t (ISO 27701)\n중대 주제 선정 이유 및 영향\n카카오뱅크는 설립 단계부터 비즈니스의 환경적, 사회적 가치 창출을 고려하였습니다.\n카카오뱅크는 기술과 혁신을 통해 기존 금융권에서 소외되어 있던 계층을 포함한 모든 사회\n구성원의 금융 접근성을 향상했습니다.\n카카오뱅크는 ‘더 나은 세상을 위해 환경을 지키는 것’을 환경경영의 목표로 삼고, 기업활동\n전반에 걸쳐 발생하는 환경영향을 최소화하여 지속가능한 미래에 기여하고자 합니다.\n특히, 카카오뱅크의 비즈니스가 지속적으로 성장함에 따라 온실가스 배출량을 비롯한 환경에의\n영향 확대가 불가피한 현실에서 환경 리스크를 전사 차원의 주요 리스크로 인식하고 관리해야\n함을 인지하고 있습니다.\n카카오뱅크는 임직원 뿐만 아니라 협력사, 고객, 지역사회와 같은 모든 이해관계자의 인권이\n마땅히\xa0존중받는 ‘사람 중심 경영’을 실천하고자 노력합니다. 인권헌장이 바탕이 된 인권경영과\n이해관계자의 참여 및 적극적인 리스크 관리로 진정성 있는 인권존중을 실현하고자 하며,')
```

### 2.1.6 답변 생성

이번에는 실제로 사용자 쿼리(`카카오뱅크의 환경목표와 세부추진내용을 알려줘`)에 대한 답변을 생성해보겠습니다. 벡터 저장소에서 문서를 검색한 다음, 이를 기반으로 ChatGPT 모델에 쿼리를 수행하는 end-to-end 프로세스를 구현합니다. 이 과정을 통해 사용자의 질문에 대한 의미적으로 관련이 있는 답변을 생성할 수 있습니다.

1. 검색(Retrieval) : `vectorstore.as_retriever` 를 사용하여 MMR 검색 방식으로 문서를 검색합니다. `search_kwargs`에 `k:5` 와 `lambda_mult:0.15` 를 설정하여 상위 5개의 관련성이 높으면서도 다양한 문서를 선택합니다.

2. 프롬프트 생성(Prompt) : `ChatPromptTemplate` 를 사용하여 쿼리에 대한 답변을 생성하기 위한 템플릿을 정의합니다. 여기서 `{context}`는 검색된 문서의 내용이고, `{question}`은 사용자의 쿼리입니다.

3. 모델(Model) : `ChatOpenAI`를 사용하여 OpenAI의 GPT 모델을 초기화합니다. 사용하는 모델은 `gpt-4o-mini`를 사용하며, `temperature` 를 0으로 설정하여 결정론적인 응답을 생성하고, `max_tokens`를 500으로 설정하여 응답의 길이를 제한합니다.

4. 문서 포맷팅(Formatting Docs) : 검색된 문서(`docs`)를 포맷팅하는 `format_docs` 함수를 정의합니다. 이 함수는 각 문서의 `page_content` 를 가져와 두 개의 문단 사이에 두 개의 줄바꿈을 삽입하여 문자열로 결합합니다.

5. 체인 실행(Chain Execution) : `prompt | llm | StrOutputParser()` 를 사용하여 LLM 체인을 구성하고, 실행합니다. 프롬프트를 통해 정의된 쿼리를 모델에 전달하고, 모델의 응답을 문자열로 파싱합니다.

6. 실행(Run) : `chain.invoke` 메소드를 사용하여 체인을 실행합니다. `context` 로는 포맷팅된 문서 내용이고, `question`은 사용자의 쿼리입니다. 최종 응답은 `response` 변수에 저장됩니다.

위 과정을 진행하기 전에 우선 단순히 LLM 에 "카카오뱅크의 환경목표와 세부추진내용을 알려줘" 라는 질문을 해보고 답변을 받아보도록 하겠습니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=500,
    )

chain = llm | StrOutputParser()

response = chain.invoke("카카오뱅크의 환경목표와 세부추진내용을 알려줘")
print(response)
```

단순히 gpt-4o-mini 에게 물어보면 아래와 같은 답변을 얻을 수 있습니다.

```
실행 결과

카카오뱅크는 환경 지속 가능성을 중요시하며, 다양한 환경목표와 세부 추진내용을 수립하고 있습니다. 카카오뱅크의 환경목표와 관련된 구체적인 내용은 다음과 같습니다.

### 환경목표
1. **탄소중립 달성**: 카카오뱅크는 운영 과정에서 발생하는 탄소 배출량을 줄이고, 최종적으로 탄소중립을 목표로 하고 있습니다.
2. **친환경 금융 상품 확대**: 환경 친화적인 금융 상품 및 서비스 제공을 통해 지속 가능한 투자와 대출을 촉진하고 있습니다.
3. **자원 효율성 증대**: 운영 과정에서 자원의 효율성을 높이고, 폐기물을 줄이는 방향으로 운영 방침을 설정하고 있습니다.

### 세부 추진 내용
1. **온실가스 감축 활동**: 에너지 효율성을 개선하고 재생 가능 에너지를 활용하여 온실가스 배출량을 지속적으로 모니터링하고 관리합니다.
2. **ESG 투자 및 대출**: 기업의 환경, 사회, 거버넌스(ESG) 지표를 반영한 금융 상품 개발을 통해 지속 가능한 프로젝트에 대한 투자를 장려합니다.
3. **디지털 소통 강화**: 비대면 서비스와 디지털 플랫폼을 통해 종이 사용을 줄이고, 고객과의 소통에서 환경친화적인 방법을 도입합니다.
4. **사회 공헌 프로그램**: 환경 보호 활동과 연계한 다양한 사회 공헌 프로그램을 운영하여 지역사회와 함께 지속 가능한 환경을 만들어가는 데 기여합니다.

### 결론
카카오뱅크는 지속 가능한 금융을 실현하기 위해 다양한 노력을 기울이고 있으며, 이를 통해 환경보호와 경제적 가치 창출을 동시에 추구하고 있습니다. 더 구체적인 내용은 카카오뱅크의 공식 홈페이지나 지속 가능성 보고서를 통해 확인할 수 있습니다.
```

```python

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Retrieval
retriever = vectorstore.as_retriever(
    search_type = 'mmr',
    search_kwargs={'k':5, 'lambda_mult':0.15}
)

query = "카카오뱅크의 환경목표와 세부추진내용을 알려줘"

docs = retriever.get_relevant_documents(query)

# Prompt
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

# Model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1000,
)

def format_docs(docs):
  return '\n\n'.join([d.page_content for d in docs])

# Chain
chain = prompt | llm | StrOutputParser()

# Run

response = chain.invoke({'context':(format_docs(docs)), 'question' : query})
print(response)
```

RAG 를 적용할 경우 이전에 문서 검색을 할 때 출력되던 문서의 내용들이 포함되어 좀 더 구체적인 정보들이 많이 포함된 것을 확인할 수 있습니다.

```
실행 결과

카카오뱅크의 환경목표는 '더 나은 세상을 위해 환경을 지키는 것'으로, 기업활동 전반에 걸쳐 발생하는 환경영향을 최소화하여 지속가능한 미래를 위해 노력하는 것입니다. 

세부 추진 내용은 다음과 같습니다:

1. **환경경영 관리체계 구축**: 환경경영 전략과 정책을 수립하고 이를 이행하기 위한 기반을 마련합니다.
2. **환경 리스크 및 성과 관리**: 환경 리스크를 관리하고 성과를 모니터링합니다.
3. **단기, 중기, 장기 과제 도출 및 이행**: 환경영향 저감을 위한 다양한 과제를 단계적으로 이행합니다.
4. **ESG팀 및 환경 TF 구성**: 환경경영 이행을 전담하는 조직을 구성하여 환경경영 업무를 고도화하고 혁신 과제를 발굴합니다.
5. **환경 방침 수립**: 전사 환경경영 정책, 녹색 구매 지침, 환경 지표 설정 및 성과 관리, 자원 사용량 관리, 기후변화 포함 환경 리스크 관리체계를 마련합니다.
6. **온실가스 배출 관리**: Scope 1, 2, 3 온실가스 배출량을 관리하고 저감하기 위한 노력을 지속합니다.

이러한 목표와 추진 내용을 통해 카카오뱅크는 환경문제에 적극적으로 대응하고 지속 가능한 경영을 실현하고자 합니다.
```

## 2.2 Multi Query Retriever

멀티 쿼리 검색도구(MultiQueryRetriever)는 벡터스토어 검색도구(Vector Store Retriever)의 한계를 극복하기 위해 고안된 방법입니다. 사용자가 입력한 쿼리의 의미를 다각도로 포착하여 검색 효율성을 높이고, LLM을 활용하여 사용자에게 보다 관련성 높고 정확한 정보를 제공하는 것을 목표로 합니다.

단일 쿼리의 의미를 다양한 관점으로 확장하여 멀티 쿼리를 자동 생성하고, 이러한 모든 쿼리에 대한 검색 결과를 결합하여 처리합니다. 다양한 문장을 생성하기 위하여 LLM 을 사용하여 사용자의 입력 문장을 다양한 관점으로 패러프레이징(Paraphrasing)하는 방식으로 구현됩니다.

다음 코드는 `MultiQueryRetriever` 클래스를 사용하여 여러 쿼리에 기반한 문서 검색 과정을 설정하고 실행하는 방법을 보여줍니다.

1. MultiQueryRetriever 설정 : `from_llm` 메서드를 통해, 기존 벡터 저장소 검색도구(vectorstore.as_retriever())와 LLM 모델을 결합하여 `MultiQueryRetriever` 인스턴스를 생성합니다. 이때 LLM은 다양한 관점의 쿼리를 생성하는 데 사용됩니다.

2. 로깅 설정 : 로깅을 설정하여 `MultiQueryRetriever`에 의해 생성되고 실행되는 쿼리들에 대한 정보를 로그로 기록하고 확인할 수 있습니다. 검색 과정에서 어떤 쿼리들이 생성되고 사용되었는지 이해하는 데 도움이 됩니다.

3. 문서 검색 실행 : `get_relevant_documents` 메서드를 사용하여 주어진 사용자 쿼리(`question`)에 대해 멀티 쿼리 기반의 문서 검색을 실행합니다. 생성된 모든 쿼리에 대해 문서를 검색하고, 중복을 제거하여 고유한 문서들만을 결과로 반환합니다.

4. 결과 확인 : 검색을 통해 반환된 고유 문서들의 수를 확인합니다. 멀티 쿼리 접근 방식을 통해 얼마나 많은 관련 문서가 검색되었는지를 나타냅니다.

```python
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# Load data

loader = PyMuPDFLoader("/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name = 'cl100k_base'
)

documents = text_splitter.split_documents(data)

embeddings_model = HuggingFaceEmbeddings(
    model_name = "jhgan/ko-sroberta-multitask",
    model_kwargs = {'device' : 'cpu'},
    encode_kwargs = {'normalize_embeddings' : True}
)

vectorstore = FAISS.from_documents(documents,
                                   embedding = embeddings_model,
                                   distance_strategy = DistanceStrategy.COSINE
                                   )

# 멀티 쿼리 생성
question = "카카오뱅크의 최근 영업실적을 알려줘."

llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature=0,
    max_tokens=500,
)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

unique_docs = retriever_from_llm.get_relevant_documents(query=question)
len(unique_docs)

```

```
실행 결과

INFO:langchain.retrievers.multi_query:Generated queries: ['카카오뱅크의 최신 재무 성과에 대한 정보를 제공해줄 수 있나요?  ', '최근 카카오뱅크의 영업 실적에 대한 업데이트가 있나요?  ', '카카오뱅크의 최근 분기 영업 실적은 어떻게 되나요?']
6
```

앞에서 정의한 MultiQueryRetriever(`retriever_from_llm`)를 활용하여 여러 쿼리를 생성하고 검색된 문서를 기반으로 사용자 질문에 답변하는 과정을 살펴봅니다. 우선 LLM 에 RAG 를 적용하지 않고 단순히 `query` 를 던져 답을 받아 보도록 하겠습니다.

```python
from langchain_core.output_parsers import StrOutputParser

chain = llm | StrOutputParser

response = chain.invoke(query)

print(response)

```

RAG 를 적용하지 않고 LLM 에게 질문을 던지면 정보가 없어 답변을 해주지 못하는 것을 확인할 수 있습니다.

```
실행 결과

죄송하지만, 2023년 10월 이후의 카카오뱅크의 영업 실적에 대한 구체적인 정보를 제공할 수 없습니다. 하지만 카카오뱅크는 일반적으로 디지털 뱅킹 서비스와 관련된 다양한 금융 상품을 제공하며, 최근 몇 년간 빠른 성장세를 보였습니다. 최신 영업 실적이나 재무 정보는 카카오뱅크의 공식 웹사이트나 금융 관련 뉴스 매체를 통해 확인하실 수 있습니다.
```

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Prompt
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
  return '\n\n'.join([d.page_content for d in docs])

# Chain
chain = (
    {'context' : retriever_from_llm | format_docs, 'question' : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run
response = chain.invoke("카카오뱅크의 최근 영업실적을 요약해서 알려주세요")
print(response)
```

놀랍게도 LLM 에 질문만 했을 때에는 정보가 없어 전혀 답변을 하지 못했지만 구체적인 정보를 포함해주니 문서의 내용을 토대로 답변 해주는 것을 확인할 수 있습니다.

```
실행 결과

카카오뱅크는 2022년 말 기준으로 총 고객 수가 2,042만 명에 달하며, 수신 규모는 33.1조 원, 여신 규모는 27.9조 원을 기록했습니다. 특히, 차별적인 수신 상품을 통해 자금조달의 경쟁우위를 확보하였고, 주택담보대출 상품의 성공적인 안착으로 리테일 뱅킹 포트폴리오를 완성했습니다. 또한, 개인사업자 뱅킹 서비스 출시로 기업 금융의 발판을 마련하였으며, 제휴를 통한 연계 서비스도 성장하여 증권계좌개설 실적이 누적 614만 좌, 연계 대출이 누적 5.7조 원에 달했습니다. 청소년을 위한 카카오뱅크 mini는 2022년 말 기준으로 누적 가입 고객 수가 160만 명 이상에 이르렀습니다.
```

---

## 2.3 Contextual compression

컨텍스트 압축 기법은 검색된 문서 중에서 쿼리와 관련된 정보만을 추출하여 반환하는 것을 목표로 합니다. 쿼리와 무관한 정보를 제거하는 방식으로 답변의 품질을 높이고 비용을 줄일 수 있습니다. 먼저 기본 검색기를 정의합니다.

### 2.3.1 기본 검색기(Base Retriever) 정의

1. 기본 검색기 설정

  `vectorstore.as_retriever` 함수를 사용하여 기본 검색기를 설정합니다. 여기서 `search_type='mmr'` 와 `search_kwargs={'k':7, 'fetch_k':20}`는 검색 방식을 설정합니다. `mmr` 검색 방식은 다양성을 고려한 검색 결과를 제공하여, 단순히 가장 관련성 높은 문서만 반환하는 대신 다양한 관점에서 관련된 문서들을 선택합니다.

2. 쿼리 처리 및 문서 검색

  `base_retriever.get_relevant_documents(question)` 함수를 사용하여 주어진 쿼리에 대한 관련 문서를 검색합니다. 이 함수는 쿼리와 관련성 높은 문서들을 반환합니다.

3. 결과 출력

  `print(len(docs))`를 통해 검색된 문서의 수를 출력합니다.

```python
# 기본 검색기

question = "카카오뱅크의 최근 영업실적을 알려줘"

llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature=0,
    max_tokens=500,
)

base_retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k':7, 'fetch_k':20}
)

docs = base_retriever.get_relevant_documents(question)
print(len(docs))
```

```
실행 결과

7
```

### 2.3.2 문서 압축기의 구성과 작동 방식

문서 압축기는 기본 검색기로부터 얻은 문서들을 더욱 효율적으로 압축하여, 쿼리와 가장 관련이 깊은 내용만을 추려내는 것을 목표로 합니다. `LLMChainExtractor` 와 `ContextualCompressionRetriever` 클래스를 사용합니다.

1. LLMChainExtractor 설정

  `LLMChainExtractor.from_llm(llm)`를 사용하여 문서 압축기를 설정합니다. LLM 을 사용하여 문서 내용을 압축합니다.

2. ContextualCompressionRetriever 설정

  `ContextualCompressionRetriever` 인스턴스를 생성할 때, `base_compressor` 와 `base_retriever` 를 인자로 제공합니다. `base_compressor` 는 앞서 설정한 `LLMChainExtractor` 인스턴스이며, `base_retriever` 는 기본 검색기 인스턴스입니다 . 이 두 구성 요소를 결합하여 검색된 문서들을 압축하는 과정을 처리합니다.

3. 압축된 문서 검색

  `compression_retriever.get_relevant_documents(question)` 함수를 사용하여 주어진 쿼리에 대한 압축된 문서들을 검색합니다. 기본 검색기를 통해 얻은 문서들을 문서 압축기를 사용하여 내용을 압축하고, 쿼리와 가장 관련된 내용만을 추려냅니다.

4. 결과 출력

  `print(len(compressed_docs))`를 통해 압축된 문서의 수를 출력합니다.

```python
# 문서 압축기를 연결하여 구성

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

compressed_docs = compression_retriever.get_relevant_documents(question)
print(len(compressed_docs))

```

저는 wiki docs 와는 다르게 `카카오뱅크의 최근 영업실적을 알려줘` 라는 쿼리를 사용했을 때 문서 압축기로 추출되는 문서가 없어 쿼리를 `카카오뱅크가 생각하는 환경문제에 대해서 알려줘`로 바꾸어서 진행해 보았습니다.

```
실행 결과

6
```

```python
compressed_docs
```

최종적으로 압축된 문서의 내용을 출력하여 확인합니다. 이 방식은 효율적인 정보 검색과 내용 압축을 통해 RAG 답변의 품질을 높일 수 있는 유용한 접근법입니다. 기본 검색기로부터 유사도 기반으로 추출된 문서들 중에서 실제로 사용자의 쿼리와 관련된 정보만을 압축하여, 정보를 더욱 집약적으로 제공하는 것이 목적입니다.

```
실행 결과

[Document(metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.1 (Macintosh)', 'creationdate': '2023-06-21T17:01:54+09:00', 'source': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'file_path': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'total_pages': 99, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'moddate': '2023-06-21T17:02:34+09:00', 'trapped': '', 'modDate': "D:20230621170234+09'00'", 'creationDate': "D:20230621170154+09'00'", 'page': 25}, page_content='카카오뱅크는 아래와 같은 환경방침을 수립하여 운영합니다.\n• 전사 환경경영 정책 수립\n• 녹색 구매 지침 수립\n• 환경 지표 설정 및 성과 관리\n• 용수, 폐기물, 에너지 등 자원 사용량 관리\n• 기후변화를 포함한 환경 리스크 관리체계 마련\n• Scope 1&2&3 온실가스 배출량 모니터링\n• 탄소 가격 도입을 통한 환경 비용 관리\n• 신재생 에너지 사용 확대\n• 녹색채권 발행 기반 마련\n• 환경경영 조직 및 관리 체계 구축\n• 환경영향평가 체계 구축'),
 Document(metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.1 (Macintosh)', 'creationdate': '2023-06-21T17:01:54+09:00', 'source': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'file_path': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'total_pages': 99, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'moddate': '2023-06-21T17:02:34+09:00', 'trapped': '', 'modDate': "D:20230621170234+09'00'", 'creationDate': "D:20230621170154+09'00'", 'page': 14}, page_content='카카오뱅크는 무점포 영업구조를 가진 모바일 전문 은행으로 혁신적인 금융과 기술을 통해 출범 시점부터 저탄소 경영과 환경보호 등 친환경 가치를 실현하고 있습니다.  \n탄소발자국  \nZERO  \n• 영업점 非 구축 및 非 운영  \n• 모바일 only 정책으로 고객의 은행 방문 無  \n• 현금 보관 및 현수송에 소요되는 에너지 無  \n종이 사용  \nZERO  \n• 100% 모바일 통장 운영 및 디지털명세서 발급  \n• 대출 신청, 심사, 결과 안내까지 전과정 디지털화  \n• 수표와 어음에 사용되는 종이가 필요 없는 전자송금 방식  \n그린 IT  \n• 에너지 효율성과 기후변화를 고려한 가상 서버  \n(Virtual Machine Server) 등의 혁신 기술 활용  \n환경경영  \n• ISO 14001(환경경영시스템) 인증 취득  \n• 금융배출량 포함 탄소배출량 측정 및 공개  \n• 친환경 투자 확대'),
 Document(metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.1 (Macintosh)', 'creationdate': '2023-06-21T17:01:54+09:00', 'source': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'file_path': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'total_pages': 99, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'moddate': '2023-06-21T17:02:34+09:00', 'trapped': '', 'modDate': "D:20230621170234+09'00'", 'creationDate': "D:20230621170154+09'00'", 'page': 94}, page_content='카카오뱅크는 기후리스크가 주요 리스크임을 인지하고 포트폴리오의 배출량을 산정 및 공개하고 있고 내·외부의 ESG 리스크 및 기회를 식별하기 위해 이중 중대성 평가를 시행하고 있습니다. 카카오뱅크는 투자의사결정 시 환경, 사회, 지배구조 요소가 충분히 고려되도록 투자 가이드라인에 ESG 요소를 포함합니다. 카카오뱅크는 2022년 친환경 건축물 인증제도(LEED)를 받은 판교테크원에 입주하여 용수 사용량을 저감하고 고효율 LED 및 자동점등 센서를 사용하여 에너지를 저감했습니다. 고객 대상으로는 교통카드 온라인 충전 서비스를 제공하여 종이 영수증 발행을 감소시켰으며, 추후에는 고객의 친환경 활동에 대해 우대금리를 제공하는 상품을 개발할 예정입니다. 카카오뱅크는 중요 리스크에 기후리스크를 포함한 바 있으며, 이를 근거로 향후 기업 대출 강화 등 포트폴리오 다변화에 따른 기후리스크 평가 시스템을 구축해 나갈 예정입니다. 연 1회 환경영향평가를 실시하여 이해관계자 요구사항을 반영한 주요 리스크와 기회를 식별 및 관리하고 있습니다. 도출된 이슈를 해결하기 위한 목표와 세부 추진계획을 세워 환경영향을 최소화하고 있으며, 관련 지표들을 모니터링하고 있습니다.'),
 Document(metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.1 (Macintosh)', 'creationdate': '2023-06-21T17:01:54+09:00', 'source': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'file_path': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'total_pages': 99, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'moddate': '2023-06-21T17:02:34+09:00', 'trapped': '', 'modDate': "D:20230621170234+09'00'", 'creationDate': "D:20230621170154+09'00'", 'page': 61}, page_content='•\t탄소 배출 절감 도시 숲 조성  \n•\t코트디부아르 재활용 플라스틱 벽돌 학교 지원 등'),
 Document(metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.1 (Macintosh)', 'creationdate': '2023-06-21T17:01:54+09:00', 'source': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'file_path': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'total_pages': 99, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'moddate': '2023-06-21T17:02:34+09:00', 'trapped': '', 'modDate': "D:20230621170234+09'00'", 'creationDate': "D:20230621170154+09'00'", 'page': 29}, page_content='카카오뱅크는 기후위기에 대응하고 임직원의 환경의식을 제고하기 위해 다양한 환경 교육과 캠페인을 실시하고 있습니다. 전자문서 보고체계를 구축하여 페이퍼리스 문화를 조성하였습니다. 또한 제안서를 메일로만 접수받고 제안 발표 자료 출력을 최소화하는 등 협력사의 환경영향까지 고려하고 있습니다. 카카오뱅크는 폐기물 발생량 최소화를 위해 분리배출 시스템을 운영하며 임직원들이 자원순환에 적극적으로 동참할 수 있도록 다양한 캠페인을 실시하고 있습니다. 또한 미사용 사무용 가구와 노트북을 사회복지시설에 기부하여 자원을 재활용하였습니다. 사내 일회용품 사용을 줄이기 위해 생분해성 친환경 컵과 빨대를 사용하고 있으며 텀블러 사용을 활성화하기 위해 세척기를 도입하였습니다. 또한, 사내 카페에서 텀블러 사용 시 할인 혜택을 제공하는 등 일회용품 사용 저감을 위한 다양한 활동을 운영하고 있습니다.'),
 Document(metadata={'producer': 'Adobe PDF Library 17.0', 'creator': 'Adobe InDesign 18.1 (Macintosh)', 'creationdate': '2023-06-21T17:01:54+09:00', 'source': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'file_path': '/content/drive/MyDrive/LangChain/카카오뱅크 2022 지속가능경영보고서.pdf', 'total_pages': 99, 'format': 'PDF 1.4', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'moddate': '2023-06-21T17:02:34+09:00', 'trapped': '', 'modDate': "D:20230621170234+09'00'", 'creationDate': "D:20230621170154+09'00'", 'page': 94}, page_content='연 1회 환경영향평가를 실시하여 이해관계자 요구사항을 반영한 주요 리스크와 기회를 식별 및 관리하고 있습니다. 도출된 이슈를 해결하기 위한 목표와 세부 추진계획을 세워 환경영향을 최소화하고 있으며, 관련 지표들을 모니터링하고 있습니다. 카카오뱅크는 의무적인 기후 또는 환경 규제(배출권 거래제 등)에 적용을 받지 않고, Scope 1&2로 배출되는 탄소배출량이 배출권 거래제에서 요구하는 법적 수치보다 현저히 적습니다. 그러나 환경영향을 정확하게 파악하고 개선하기 위해 탄소배출량을 모니터링하고 금융배출량을 포함한 업스트림, 다운스트림 Scope 3 탄소배출량을 산정하여 공개하고 있습니다. 카카오뱅크는 사업 확장으로 미래 배출량을 예측하기는 어려우나, 에너지 및 탄소배출량 집약도를 지속적으로 모니터링하고 있습니다. 기후변화의 영향도를 보다 면밀하게 분석하고 관리하기 위해 과거 Scope2 탄소배출량 공개 이외에 Scope 1과 3도 추가로 공개하는 등 관리 체계를 고도화 하고 있습니다. 카카오뱅크는 사업 확장으로 인해 예상되는 환경리스크 및 기회도 식별하며 사업을 추진할 계획입니다.')]
```

## 2.4 한글 형태소 분석기와 BM25Retriever의 결합

검색기에서 Dense Vector를 이용한 의미 검색은 때때로 의미에만 초점을 맞추기 때문에 검색 쿼리에 있는 키워드와의 의미적 유사성만을 고려해 검색 쿼리에 있는 키워드들이 하나도 없는 문서들이 top k 문서에 포함될 수 있습니다. 의미적으로는 맞다고 볼 수 있지만 우리 사람들이 보았을 때는 검색 쿼리에 있는 키워드들이 없어 결과가 엉터리인데? 라고 생각하기도 합니다. 그럴 때 사용되는 것이 Sparse Vector를 이용한 검색으로 주로 BM25Retriever를 이용합니다. 하지만 우리 한국어는 명사뒤에 조사가 붙는다는 특징이 있어 한국어에 맞는 tokenizer를 사용하지 않으면 명사 키워드들이 정확하게 분리가 되지 않아 Sparse Vector를 이용한 키워드 검색의 성능이 낮아지는 문제가 발생합니다. 이러한 문제를 해결하기 위해 BM25Retriever 검색기에 한글 형태소 분석기를 결합하기도 합니다. 따라서 이번에는 BM25Retriever에 한글 형태소 분석기를 결합해 보고, 한글 형태소 분석기를 사용하지 않았을 때와 결합 했을 때의 검색 결과가 어떻게 다른지 한 번 살펴보도록 하겠습니다.

### 2.4.1 Kiwi 토크나이저를 이용한 BM25Retriever

누구나 사용 가능한 한국어 형태소 Kiwi와 BM25Retriever를 결합해 보도록 하겠습니다. 실행 전에 우선 실행에 필요한 라이브러리 설치를 먼저 진행해 주시기 바랍니다. 

```bin
!pip install -U langchain langchain-community langchain-core kiwipiepy konlpy
```

그리고 langchain-community의 retrievers의 BM25Retriever과 langchain-teddynote에서 제공해 주는 KiwiBM25Retriever를 이용해 결과를 비교해 보려고 했으나 langchain-community의 BM25Retriever는 각 결과의 score값을 얻을 수가 없었고, langchain-teddynote의 KiwiBM25Retriever의 경우 사용되는 tokenizer가 kiwi tokenizer로 고정되어 있어 KiwiBM25Retriever 클래스를 가져와서 필요한 tokenizer를 쓸 수 있게 아주 조금 개량하였습니다. from_texts() 메서드의 파라미터인 proprocess_func의 값을 무조건 kiwi_preprocessing_func을 사용하도록 하는 것이 아니라 입력으로 받는 메서드를 사용하도록 하기 위해 초기값을 None으로 주었습니다.

```python
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional
from operator import itemgetter
import numpy as np

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field
from langchain_core.retrievers import BaseRetriever

try:
    from kiwipiepy import Kiwi
except ImportError:
    raise ImportError(
        "Could not import kiwipiepy, please install with `pip install " "kiwipiepy`."
    )

kiwi_tokenizer = Kiwi()


def kiwi_preprocessing_func(text: str) -> List[str]:
    return [token.form for token in kiwi_tokenizer.tokenize(text)]


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


class KiwiBM25Retriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any
    """ BM25 vectorizer."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], List[str]] = kiwi_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = None,
        **kwargs: Any,
    ) -> KiwiBM25Retriever:
        """
        Create a KiwiBM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A KiwiBM25Retriever instance.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )

        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = kiwi_preprocessing_func,
        **kwargs: Any,
    ) -> KiwiBM25Retriever:
        """
        Create a KiwiBM25Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A KiwiBM25Retriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        return return_docs

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def argsort(seq, reverse):
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

    def search_with_score(self, query: str, top_k=None):
        normalized_score = KiwiBM25Retriever.softmax(
            self.vectorizer.get_scores(self.preprocess_func(query))
        )

        if top_k is None:
            top_k = self.k

        score_indexes = KiwiBM25Retriever.argsort(normalized_score, True)

        docs_with_scores = []
        for i, doc in enumerate(self.docs):
            metadata = doc.metadata.copy()
            metadata["score"] = normalized_score[i]
            document = Document(page_content=doc.page_content, metadata=metadata)
            docs_with_scores.append(document)

        score_indexes = score_indexes[:top_k]

        # Creating an itemgetter object
        getter = itemgetter(*score_indexes)

        # Using itemgetter to get items
        selected_elements = getter(docs_with_scores)
        return selected_elements
```

잘 정돈된 이쁜 결과를 보기위한 pretty_print 메서드도 정의해 주었습니다.

```python
def pretty_print(docs):
    for i, doc in enumerate(docs):
        if "score" in doc.metadata:
            print(f"[{i+1}] {doc.page_content} ({doc.metadata['score']:.4f})")
        else:
            print(f"[{i+1}] {doc.page_content}")

```

그럼 예제에 사용할 문서를 정의해 주도록 하겠습니다.

```python
sample_texts = [
    "금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다.",
    "금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다.",
    "금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요.",
    "금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다.",
]
```

검색기를 정의해 주도록 하겠습니다. normal_bm25_retriever는 일반적은 BM25Retriever에서 사용되는 것처럼 공백을 기준으로 tokenizing 하며, kiwi_bm25_retriever는 kiwi의 tokenizer를 이용해 tokenizing을 진행합니다.

```python
# 검색기를 생성합니다.

normal_bm25_retriever = KiwiBM25Retriever.from_texts(sample_texts, preprocess_func=default_preprocessing_func)
kiwi_bm25_retriever = KiwiBM25Retriever.from_texts(sample_texts, preprocess_func=kiwi_preprocessing_func)
```

"금융보험"이라는 키워드에 대한 검색 결과를 출력하도록 해보았습니다.

```python
pretty_print(normal_bm25_retriever.search_with_score("금융보험"))
print()

pretty_print(kiwi_bm25_retriever.search_with_score("금융보험"))
```

그냥 공백을 기준으로 tokenizing을 하도록 한 normal_bm25_retriever는 검색된 4개의 문서가 모두 같은 score를 가지고 있었고, kiwi tokenizer를 사용한 kiwi_bm25_retriever의 경우 "금융보험"이 들어간 문장이 명사와 조사로 잘 나누어져 "금융보험"이 있는 문장이 가장 높은 점수를 가지는 것을 확인할 수 있었습니다.

```
Output:
[1] 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다. (0.2500)
[2] 금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다. (0.2500)
[3] 금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요. (0.2500)
[4] 금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다. (0.2500)

[1] 금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다. (0.2750)
[2] 금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다. (0.2688)
[3] 금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요. (0.2288)
[4] 금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다. (0.2273)
```

## 2.5 Ensemble Retriever

EnsembleRetriever는 여러 검색기를 결합하여 더 강력한 검색 결과를 제공하는 LangChain의 기능입니다. 이 검색기는 다양한 검색 알고리즘의 장점을 활용하여 단일 알고리즘보다 더 나은 성능을 달성할 수 있습니다.

주요 특징은 다음과 같습니다. 

1. 여러 검색기 통합: 다양한 유형의 검색기를 입력으로 받아 결과를 결합합니다.
2. 결과 재순위화: Reciprocal Rank Fusion 알고리즘을 사용하여 결과의 순위를 조정합니다.
3. 하이브리드 검색: 주로 sparse retriever와 dense retriever를 결합하여 사용합니다.

이러한 상호 보완적인 특성으로 인해 Ensemble Retriever는 다양한 검색 시나리오에서 향상된 성능을 제공할 수 있습니다.

## 2.5.1 기본적인 Ensemble Retriever

그러면 예제 코드로 Ensemble Retriever에 대해서 알아보도록 하겠습니다. 그 전에 우선 필요한 라이브러리들 설치부터 진행해 주도록 하겠습니다.

```bin
pip install faiss-cpu langchain langchain-openai langchain-core langchain-community langchain-experimental langchain-classic rank-bm25
```

BM25Retriever와 FAISS 검색기를 결합하고, 각 검색기의 가중치를 설정합니다.

```python
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 샘플 문서 리스트
doc_list = [
    "I like apples",
    "I like apple company",
    "I like apple's iphone",
    "Apple is my favorite company",
    "I like apple's ipad",
    "I like apple's macbook",
]

# bm25와 retriever와 faiss retriever를 초기화합니다.
bm25_retriever = BM25Retriever.from_texts(
    doc_list,
)

bm25_retriever.k = 1 # BM25Retriever의 검색 결과 개수를 1로 설정합니다.

embedding = OpenAIEmbeddings() # OpenAI 임베딩을 사용합니다.
faiss_vectorstore = FAISS.from_texts(
    doc_list,
    embedding,
)

faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k":1})

# 앙상블 retriever를 초기화합니다.
ensemble_retriever = EnsembleRetriever(
    retrievers = [bm25_retriever, faiss_retriever],
    weights=[0.7, 0.3],
)
```

ensemble_retriever 객체의 get_relevant_documents() 메서드를 호출하여 관련성 높은 문서를 검색합니다.

```python
# 검색 결과 문서를 가져옵니다.
query = "My favorite fruit is apple"
ensemble_result = ensemble_retriever.invoke(query)
bm25_result = bm25_retriever.invoke(query)
faiss_result = faiss_retriever.invoke(query)

# 가져온 문서를 출력합니다.
print("[Ensemble Retriever]")
for doc in ensemble_result:
    print(f"Content: {doc.page_content}")
    print()

print("[BM25 Retriever]")
for doc in bm25_result:
    print(f"Content: {doc.page_content}")
    print()

print("[FAISS Retriever]")
for doc in faiss_result:
    print(f"Content: {doc.page_content}")
    print()
```

검색 결과를 보면 Ensemble Retriever의 결과에 bm25와 faiss 검색 결과가 모두 포함된 것을 확인할 수 있습니다.

```
Output:
[Ensemble Retriever]
Content: Apple is my favorite company

Content: I like apple's iphone

[BM25 Retriever]
Content: Apple is my favorite company

[FAISS Retriever]
Content: I like apple's iphone
```

## 2.5.2 Convex Combination이 적용된 Ensemble Retriever

우리는 이전 항목에서 Sparse Vector와 Dense Vector를 이용해 검색된 문서들에 RRF(Reciprocal Rank Fusion) 기법을 적용해 두 가지 방식의 검색 방법을 최종 검색 결과에 반영하는 방법에 대해서 알아보았습니다. 이번엔 일반적으로 사용되는 RRF 방식보다 조금 번거롭지만 튜닝만 잘 한다면 RRF보다 더 좋은 성능을 뽑아낼 수 있는 Convex Combination이 적용된 Ensemble Retriever에 대해서 알아보도록 하겠습니다.

Convex Combination은 Ensmeble Retriever에 사용된 두 가지 Retriever의 검색 결과에서 RRF와 같이 순위를 사용하는 것이 아니라 각 Retriever의 결과에 있는 score값을 그대로 사용하는 방식입니다. 이 방식은 간단히 말하자면 가중 평균 혹은 가중 합 방식으로 두 검색기에 가중치를 두어 두 검색기의 score 값에 설정한 비중을 적용해 최종적으로 산출되는 score 값을 이용해 두 검색기가 검색한 문서들의 순위를 다시 매기는 방식입니다. 수식은 아래와 같습니다.

$$Score_{final} = \alpha \cdot Score_{vector} + (1 - \alpha) \cdot Score_{bm25}$$

Convex Combination 방식을 사용할 경우 핵심적인 전제 조건이 따라야 합니다. 그 전제 조건은 Score Normalization(점수 정규화)로 두 검색기의 점수 체계(Scale)이 다르기 때문에, 그냥 더하면 안되고 반드시 0과 1 사이로 범위를 맞춰줘야 합니다.

예를 들어 보도록 하겠습니다. VectorStore Search의 경우 보통 Cosine Similarity를 사용하므로 0~1 사이의 값이 나오게 되어 정규화를 해주지 않아도 됩니다. 하지만 BM25를 사용하게 될 경우 문서 길이나 빈도에 따라 점수가 15.5, 30.2 등과 같은 점수가 나오게 됩니다. 이럴 경우 단순히 두 스코어를 사용하게 되면 BM25의 결과가 최종 결과를 독식하게 되는 문제가 발생합니다. 따라서 Min-Max Scaling 등을 통해 0~1사이로 값을 압축한 뒤 연산을 진행해야 정확한 결과를 얻을 수 있습니다.

일반적으로 Ensemble Retriever에서는 RRF 방식을 주로 사용하지만 RRF 방식은 순위를 사용하기 때문에 확실한 문서와 애매한 문서의 차이를 구분할 수 없는 문제가 존재합니다. 예를 들면 두 검색기 A와 B에서 각각 3개의 결과가 나왔다고 한다면 A 검색기에서는 (0.005, 0.005, 0.99), B 검색기에서는 (0.335, 0.333, 0.332) 가 나왔다고 가정해 보겠습니다. 이때 A검색기의 0.99 값을 가지는 문서의 경우 가장 확실한 1위 문서라고 볼 수 있습니다. 하지만 B 검색기의 0.335 값을 가지는 문서의 경우 다른 두 문서와 큰 차이가 없으며, RRF를 적용하게 되면 A 검색기에서 확실하다고 볼 수 있는 문서의 순위가 뒤로 밀려나게 될 수도 있습니다. Convex Combination 방식은 이러한 부작용을 없애고자 좀 더 객관적이라고 볼 수 있는 score 값을 이용해 재순위화를 진행하는 방식입니다.

그럼 예제를 통해 직접 Convex Combination Ensemble Retriever를 구축해 보도록 하겠습니다.

우선 실행에 필요한 라이브러리 설치 먼저 진행해 줍니다.

```bin
pip install langchain langchain-community langchain-openai langchain-teddynote langchain-text-splitters pdfplumber faiss-cpu
```

데이터 로드, 청킹, 그리고 벡터 저장소, 검색기를 정의합니다.

```python
from langchain_classic.retrievers import EnsembleRetriever as OriginalEnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_teddynote.retrievers import KiwiBM25Retriever

# 문서 로드(Load Documents)
loader = PDFPlumberLoader("/content/drive/MyDrive/LangChain/19년 디지털 정부혁신 추진계획.pdf")

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
split_documents = loader.load_and_split(text_splitter)

# 임베딩 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# FaissRetriever 생성

faiss = FAISS.from_documents(split_documents, embeddings).as_retriever(search_kwargs={"k":5})

# KiwiBM25Retriever 생성
bm25 = KiwiBM25Retriever.from_documents(documents=split_documents, embedding=embeddings)
bm25.k = 5

# LangChain 버전의 EnsembleRetriever
original_ensemble_retriever = OriginalEnsembleRetriever(retrievers=[faiss, bm25])
```

CC(Convex Combination)과 RRF 방식의 EnsembleRetriever를 생성합니다.

```python
from langchain_teddynote.retrievers import(
    EnsembleRetriever,
    EnsembleMethod,
)

# RRF 방식의 EnsembleRetriever
rrf_ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss, bm25], method=EnsembleMethod.RRF
)

# CC 방식의 EnsembleRetriever
cc_ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss, bm25], method=EnsembleMethod.CC
)
```

그리고 정돈된 출력 값을 보도록 하기 위해 pretty_print 메서드도 정의해 줍니다.

```python
def pretty_print(query):
    for i, (original_doc, cc_doc, rrf_doc) in enumerate(
        zip(
            original_ensemble_retriever.invoke(query),
            cc_ensemble_retriever.invoke(query),
            rrf_ensemble_retriever.invoke(query),
        )
    ):
        print(f"[{i}] [Original] Q: {query}", end="\n\n")
        print(original_doc.page_content)
        print("-" * 100)
        print(f"[{i}] [RRF] Q: {query}", end="\n\n")
        print(rrf_doc.page_content)
        print("-" * 100)
        print(f"[{i}] [CC] Q: {query}", end="\n\n")
        print(cc_doc.page_content)
        print("=" * 100, end="\n\n")
```

검색 결과를 한 번 비교해 보도록 하겠습니다.

```python
# 검색 결과 비교
pretty_print("디지털 트랜스포메이션이란 무엇인가요?")
```

예제의 실행 결과로 검색된 결과 자체를 봤을 때 CC 방식이 RRF 방식보다 좋은가?에 대해서 잘 모르겠다라는 느낌입니다만 그래도 CC 방식과 RRF 방식에서 결과가 다르게 출력되는 것을 확인할 수 있습니다.

```
Output:
[0] [Original] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ (시스템) 디지털 신기술의 적기 도입과 활용 곤란
- 기존 복잡한 용역개발 방식은 혁신주기가 짧은 디지털 전환에 부적합
----------------------------------------------------------------------------------------------------
[0] [RRF] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ (시스템) 디지털 신기술의 적기 도입과 활용 곤란
- 기존 복잡한 용역개발 방식은 혁신주기가 짧은 디지털 전환에 부적합
----------------------------------------------------------------------------------------------------
[0] [CC] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ (시스템) 디지털 신기술의 적기 도입과 활용 곤란
- 기존 복잡한 용역개발 방식은 혁신주기가 짧은 디지털 전환에 부적합
====================================================================================================

[1] [Original] Q: 디지털 트랜스포메이션이란 무엇인가요?

Ⅰ. 개 요
□ 추진 배경
○ 우리나라는 국가적 초고속 정보통신망 투자와 적극적인 공공정보화
사업 추진에 힘입어 세계 최고수준의 전자정부를 구축‧운영
----------------------------------------------------------------------------------------------------
[1] [RRF] Q: 디지털 트랜스포메이션이란 무엇인가요?

Ⅰ. 개 요
□ 추진 배경
○ 우리나라는 국가적 초고속 정보통신망 투자와 적극적인 공공정보화
사업 추진에 힘입어 세계 최고수준의 전자정부를 구축‧운영
----------------------------------------------------------------------------------------------------
[1] [CC] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ (디지털 고지‧수납) 각종 고지서·안내문* 등을 온라인(공공‧민간)
으로 받고, 간편하게 납부할 수 있도록 디지털 고지‧수납 활성화
====================================================================================================

[2] [Original] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ (디지털 고지‧수납) 각종 고지서·안내문* 등을 온라인(공공‧민간)
으로 받고, 간편하게 납부할 수 있도록 디지털 고지‧수납 활성화
----------------------------------------------------------------------------------------------------
[2] [RRF] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ (디지털 고지‧수납) 각종 고지서·안내문* 등을 온라인(공공‧민간)
으로 받고, 간편하게 납부할 수 있도록 디지털 고지‧수납 활성화
----------------------------------------------------------------------------------------------------
[2] [CC] Q: 디지털 트랜스포메이션이란 무엇인가요?

참고 1 디지털 정부혁신 추진전략
디지털로 여는 좋은 세상
□ 비전
※ 부제 : 대한민국이 먼저 갑니다.
□ 추진원칙 △ 최종 이용자의 관점에서
△ 공공서비스 수준 향상을 목표로
====================================================================================================

[3] [Original] Q: 디지털 트랜스포메이션이란 무엇인가요?

<종이 증명서·고지서 감축에 따른 비용절감 효과>
☞ 민원처리시 자기정보 활용이나 전자증명서 대체로 종이증명서가 연간 10% 감축되는
----------------------------------------------------------------------------------------------------
[3] [RRF] Q: 디지털 트랜스포메이션이란 무엇인가요?

<종이 증명서·고지서 감축에 따른 비용절감 효과>
☞ 민원처리시 자기정보 활용이나 전자증명서 대체로 종이증명서가 연간 10% 감축되는
----------------------------------------------------------------------------------------------------
[3] [CC] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ 오픈소스 중심의 디지털정부 생태계와 공공시장 수요를 바탕으로
첨단 디지털 산업의 혁신 가속화와 글로벌 도약을 위한 전기 마련
====================================================================================================

[4] [Original] Q: 디지털 트랜스포메이션이란 무엇인가요?

참고 1 디지털 정부혁신 추진전략
디지털로 여는 좋은 세상
□ 비전
※ 부제 : 대한민국이 먼저 갑니다.
□ 추진원칙 △ 최종 이용자의 관점에서
△ 공공서비스 수준 향상을 목표로
----------------------------------------------------------------------------------------------------
[4] [RRF] Q: 디지털 트랜스포메이션이란 무엇인가요?

참고 1 디지털 정부혁신 추진전략
디지털로 여는 좋은 세상
□ 비전
※ 부제 : 대한민국이 먼저 갑니다.
□ 추진원칙 △ 최종 이용자의 관점에서
△ 공공서비스 수준 향상을 목표로
----------------------------------------------------------------------------------------------------
[4] [CC] Q: 디지털 트랜스포메이션이란 무엇인가요?

디지털 정부혁신 추진계획
2019. 10. 29.
관계부처 합동
====================================================================================================

[5] [Original] Q: 디지털 트랜스포메이션이란 무엇인가요?

* DaaS 이용 시 기존 방식보다 약 70% 예산절감효과 추정
** (필요성) 특정 OS 종속 개선, 플러그인 사용관행 원천 제거 계기
----------------------------------------------------------------------------------------------------
[5] [RRF] Q: 디지털 트랜스포메이션이란 무엇인가요?

* DaaS 이용 시 기존 방식보다 약 70% 예산절감효과 추정
** (필요성) 특정 OS 종속 개선, 플러그인 사용관행 원천 제거 계기
----------------------------------------------------------------------------------------------------
[5] [CC] Q: 디지털 트랜스포메이션이란 무엇인가요?

Ⅰ. 개 요
□ 추진 배경
○ 우리나라는 국가적 초고속 정보통신망 투자와 적극적인 공공정보화
사업 추진에 힘입어 세계 최고수준의 전자정부를 구축‧운영
====================================================================================================

[6] [Original] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ 오픈소스 중심의 디지털정부 생태계와 공공시장 수요를 바탕으로
첨단 디지털 산업의 혁신 가속화와 글로벌 도약을 위한 전기 마련
----------------------------------------------------------------------------------------------------
[6] [RRF] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ 오픈소스 중심의 디지털정부 생태계와 공공시장 수요를 바탕으로
첨단 디지털 산업의 혁신 가속화와 글로벌 도약을 위한 전기 마련
----------------------------------------------------------------------------------------------------
[6] [CC] Q: 디지털 트랜스포메이션이란 무엇인가요?

<종이 증명서·고지서 감축에 따른 비용절감 효과>
☞ 민원처리시 자기정보 활용이나 전자증명서 대체로 종이증명서가 연간 10% 감축되는
====================================================================================================

[7] [Original] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ 장기적으로 정부의 디지털 전환이 적극 추진되어 스마트시티 등
도시행정 전반의 혁신으로 파급되는 경우 막대한 경제효과 예상
----------------------------------------------------------------------------------------------------
[7] [RRF] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ 장기적으로 정부의 디지털 전환이 적극 추진되어 스마트시티 등
도시행정 전반의 혁신으로 파급되는 경우 막대한 경제효과 예상
----------------------------------------------------------------------------------------------------
[7] [CC] Q: 디지털 트랜스포메이션이란 무엇인가요?

* DaaS 이용 시 기존 방식보다 약 70% 예산절감효과 추정
** (필요성) 특정 OS 종속 개선, 플러그인 사용관행 원천 제거 계기
====================================================================================================

[8] [Original] Q: 디지털 트랜스포메이션이란 무엇인가요?

디지털 정부혁신 추진계획
2019. 10. 29.
관계부처 합동
----------------------------------------------------------------------------------------------------
[8] [RRF] Q: 디지털 트랜스포메이션이란 무엇인가요?

디지털 정부혁신 추진계획
2019. 10. 29.
관계부처 합동
----------------------------------------------------------------------------------------------------
[8] [CC] Q: 디지털 트랜스포메이션이란 무엇인가요?

○ 장기적으로 정부의 디지털 전환이 적극 추진되어 스마트시티 등
도시행정 전반의 혁신으로 파급되는 경우 막대한 경제효과 예상
====================================================================================================

[9] [Original] Q: 디지털 트랜스포메이션이란 무엇인가요?

ㅇ 양육수당, 아동수당 등 각종 출산지원 서비스를 출생신고 시 한 번에 신청
(부처별 전국공통 서비스 7종 + 지자체별 서비스 3∼10종)
----------------------------------------------------------------------------------------------------
[9] [RRF] Q: 디지털 트랜스포메이션이란 무엇인가요?

ㅇ 양육수당, 아동수당 등 각종 출산지원 서비스를 출생신고 시 한 번에 신청
(부처별 전국공통 서비스 7종 + 지자체별 서비스 3∼10종)
----------------------------------------------------------------------------------------------------
[9] [CC] Q: 디지털 트랜스포메이션이란 무엇인가요?

ㅇ 양육수당, 아동수당 등 각종 출산지원 서비스를 출생신고 시 한 번에 신청
(부처별 전국공통 서비스 7종 + 지자체별 서비스 3∼10종)
====================================================================================================

```

## 2.6 Parent Document Retriever

문서 검색 과정에서 문서를 적절한 크기의 청크로 나누는 것은 두 가지 중요한 요소를 고려해야 합니다. 첫 번째는 크기가 작은 청크로 나누는 경우입니다. 이 경우 문서 임베딩의 의미를 가장 정확하게 반영할 수 있습니다. 하지만 청크의 크기가 작아 정보가 충분히 포함되지 않을 수도 있습니다. 두 번째는 청크의 맥락이 유지되도록 충분히 큰 청크로 나누는 방법입니다. 하지만 이 방법은 임베딩으로 바꿀 청크가 너무 길면 임베딩의 의미를 잃어버릴 수 있다는 문제가 존재합니다.

이 두 요구 사항 사이의 균형을 맞추기 위해 ParentDocumentRetriever라는 도구가 사용됩니다. 이 도구는 문서를 작은 청크로 나누고, 이 청크들을 관리합니다. ParentDocumentRetriever를 사용하기 위해선 벡터 DB 말고 또 다른 저장소가 필요합니다. 바로 부모 문서의 청크 id 값을 key 값으로 가지고 value 값으로는 큰 크기의 청크 값을 가지는 일종의 dictionary 저장소가 필요합니다. ParentDocumentetriever의 작동 원리는 우선 child_splitter와 parent_splitter로 작은 크기, 큰 크기 청크로 분할합니다(만약 parent_splitter 가 설정 되지 않으면 문서 전체를 부모 청크로 사용합니다.). 자식 청크를 저장할 때 저장되는 메타 데이터에는 부모 청크의 doc_id 값이 저장됩니다. 그리고 벡터 DB에는 작은 크기의 청크를 저장하고 또 다른 dictionary 저장소에는 부모 청크를 저장해 둡니다. 그리고 invoke 메서드를 이용해 검색을 진행하게 되면 우선 자식 청크에서 검색을 진행합니다. 검색된 자식 청크에 있는 부모 청크의 doc_id를 dictionary 저장소에서 찾아 최종적으로 LLM에 사용할 문서로 제공해 주는 방식입니다.

즉, ParentDocumentRetriever를 사용하게 되면 문서 간의 계층 구조를 활용하게 되므로 검색의 효율과 LLM 답변 성능이 향상되게 됩니다. 벡터 DB에서 쿼리와 관련된 문서를 찾을 때는 작은 크기의 청크로 이루어진 문서를 찾고, LLM에게 전달한 문서는 작은 크기로 분할된 청크의 원본 문서 혹은 좀 더 크게 분할된 청크를 전달하여 LLM에게는 좀 더 많은 정보를 주도록 하여 검색의 효율과 LLM 답변 성능이 향상되게 됩니다. 그럼 예제 코드로 한 번 알아보도록 하겠습니다. 예제 코드 실행을 위해 실행에 필요한 라이브러리 설치부터 진행해 줍니다.

```bin
pip install langchain langchain-core langchain-community langchain-chroma chromadb langchain-text-splitters langchain-openai
```

우선 예제에 사용할 텍스트 파일을 로드합니다.

```python
from langchain_classic.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever

loaders = [
    # 파일을 로드
    TextLoader("/content/drive/MyDrive/LangChain/appendix-keywords.txt"),
]

docs = []
for loader in loaders:
    # 로더를 사용하여 문서를 로드하고 docs 리스트에 추가
    docs.extend(loader.load())
```

### 2.6.1 작은 청크와 관련된 전체 문서 검색

큰 크기 청크의 분할을 따로 진행하지 않고 작은 작은 크기 청크의 분할만 진행하고, LLM에 사용할 문서는 전체 문서에서 가져오도록 하겠습니다. 우선 작은 크기로 청크 문서들을 만들어 주고, 벡터 DB, 검색기를 정의합니다.

```python
# 자식 분할기를 생성
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200
)

# DB를 생성합니다.
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)

store = InMemoryStore()

# Retriever를 생성합니다.
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)
```

ParentDocumentRetriever의 add_documents 메서드로 문서 목록을 추가합니다.

```python
# 문서를 검색기에 추가합니다. docs는 문서 목록이고, ids는 문서의 고유 식별자 목록입니다.
retriever.add_documents(docs, ids=None, add_to_docstore=True)
```

그럼 이제 작은 청크들을 저장하고 있으니, 유사도 검색으로 작은 청크가 출력되는지 확인해 보도록 하겠습니다.

```python
# 유사도 검색을 수행
sub_docs = vectorstore.similarity_search("Word2Vec")

print(sub_docs[0].page_content)
```

```
Output:
정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: Word2Vec 모델에서 "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
```

그럼 이제 큰 청크인 전체 문서에서 검색해서 쿼리와 관련된 전체 문서를 가져오도록 해보겠습니다.

```python
# 문서를 검색하여 가져옵니다.
retrieved_docs = retriever.invoke("Word2Vec")

# 검색된 문서의 문서의 페이지 내용의 길이를 출력합니다.
print(
    f"문서의 길이: {len(retrieved_docs[0].page_content)}",
    end="\n\n=================\n\n"
)

# 문서의 일부를 출력합니다.
print(retrieved_docs[0].page_content[2000:2500])
```

```
Output:
문서의 길이: 5733

=================

 컴퓨팅을 도입하여 데이터 저장과 처리를 혁신하는 것은 디지털 변환의 예입니다.
연관키워드: 혁신, 기술, 비즈니스 모델

Crawling

정의: 크롤링은 자동화된 방식으로 웹 페이지를 방문하여 데이터를 수집하는 과정입니다. 이는 검색 엔진 최적화나 데이터 분석에 자주 사용됩니다.
예시: 구글 검색 엔진이 인터넷 상의 웹사이트를 방문하여 콘텐츠를 수집하고 인덱싱하는 것이 크롤링입니다.
연관키워드: 데이터 수집, 웹 스크래핑, 검색 엔진

Word2Vec

정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: Word2Vec 모델에서 "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
LLM (Large Language Model)

정의: LLM은 대규모의 텍스트 데이터로 훈련된 큰 규모의 언어 모델을
```

### 2.6.2 더 큰 Chunk의 크기를 조절

이전의 결과처럼 전체 문서가 너무 커서 있는 그대로 검색하기에는 부적합할 수 있습니다. 이런 경우, 실제로 우리가 하고 싶은 것은 먼저 원시 문서를 더 큰 청크로 분할한 다음, 더 작은 청크로 분할하는 것입니다. 그런 다음 작은 청크들을 인덱싱하지만, 검색 시에는 더 큰 청크를 검색합니다.

RecursiveCharacterTextSplitter를 이용해 child_splitter, parent_splitter를 정의하고, 사용할 벡터 DB와 dictionary 저장소도 정의합니다.

```python
# 부모 문서를 생성하는데 사용되는 텍스트 분할기
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# 자식 문서를 생성하는데 사용되는 텍스트 분할기입니다.
# 부모보다 작은 문서를 생성해야 합니다.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# 자식 청크를 인덱싱하는데 사용할 벡터 저장소
vectorestore = Chroma(
    collection_name = "split_parents", embedding_function=OpenAIEmbeddings()
)

# 부모 문서의 저장 계층
store = InMemoryStore()
```

그리고 ParentDocumentRetriever를 정의합니다.

```python
retriever = ParentDocumentRetriever(
    # 벡터 저장소를 지정
    vectorstore=vectorstore,
    
    # 문서 저장소를 지정
    docstore=store,

    #하위 문서 분할기를 지정
    child_splitter=child_splitter,

    # 상위 문서 분할기를 지정
    parent_splitter=parent_splitter,
)
```

retriever 객체에 미리 작업해 두었던 docs를 추가합니다.

```python
retriever.add_documents(docs)
```

부모 청크가 저장된 store의 길이를 출력해 보면 7로 좀 전에 전체 문서의 길이 5733을 chunk_size 1000으로 나누었을 때 대략 7개 정도로 분할되어 7로 나오는 것을 확인할 수 있습니다.

```python
# 저장소에서 키를 생성하고 리스트로 변환한 후 길이를 반환
len(list(store.yield_keys()))
```

```
Output:
7
```

그럼 벡터 저장소의 similarity_search를 이용해 검색하면 여전히 작은 크기의 청크 즉, 자식 청크를 검색하는지 확인해 보도록 하겠습니다.

```python
# 유사도 검색을 수행
sub_docs = vectorstore.similarity_search("Word2Vec")

# sub_docs 리스트의 첫 번째 요소의 page_content 속성을 출력합니다.
print(sub_docs[0].page_content)
```

이전과 동일하게 작은 크기의 청크를 검색해 줍니다.

```
Output:
정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: Word2Vec 모델에서 "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
```

그럼 이번엔 retriever의 invoke 메서드를 사용하면 문서를 검색해 보도록 하겠습니다.

```python
# 문서를 검색하여 가져옵니다.
retrieved_docs = retriever.invoke("Word2Vec")

# 검색된 문서의 첫 번째 페이지 내용 길이를 반환합니다.
print(retrieved_docs[0].page_content)
```

page_content 전체를 출력하도록 했음에도 이전과 달리 출력 내용이 적은 것을 확인할 수 있습니다.

```
Output:
정의: 트랜스포머는 자연어 처리에서 사용되는 딥러닝 모델의 한 유형으로, 주로 번역, 요약, 텍스트 생성 등에 사용됩니다. 이는 Attention 메커니즘을 기반으로 합니다.
예시: 구글 번역기는 트랜스포머 모델을 사용하여 다양한 언어 간의 번역을 수행합니다.
연관키워드: 딥러닝, 자연어 처리, Attention

HuggingFace

정의: HuggingFace는 자연어 처리를 위한 다양한 사전 훈련된 모델과 도구를 제공하는 라이브러리입니다. 이는 연구자와 개발자들이 쉽게 NLP 작업을 수행할 수 있도록 돕습니다.
예시: HuggingFace의 Transformers 라이브러리를 사용하여 감정 분석, 텍스트 생성 등의 작업을 수행할 수 있습니다.
연관키워드: 자연어 처리, 딥러닝, 라이브러리

Digital Transformation

정의: 디지털 변환은 기술을 활용하여 기업의 서비스, 문화, 운영을 혁신하는 과정입니다. 이는 비즈니스 모델을 개선하고 디지털 기술을 통해 경쟁력을 높이는 데 중점을 둡니다.
예시: 기업이 클라우드 컴퓨팅을 도입하여 데이터 저장과 처리를 혁신하는 것은 디지털 변환의 예입니다.
연관키워드: 혁신, 기술, 비즈니스 모델

Crawling

정의: 크롤링은 자동화된 방식으로 웹 페이지를 방문하여 데이터를 수집하는 과정입니다. 이는 검색 엔진 최적화나 데이터 분석에 자주 사용됩니다.
예시: 구글 검색 엔진이 인터넷 상의 웹사이트를 방문하여 콘텐츠를 수집하고 인덱싱하는 것이 크롤링입니다.
연관키워드: 데이터 수집, 웹 스크래핑, 검색 엔진

Word2Vec

정의: Word2Vec은 단어를 벡터 공간에 매핑하여 단어 간의 의미적 관계를 나타내는 자연어 처리 기술입니다. 이는 단어의 문맥적 유사성을 기반으로 벡터를 생성합니다.
예시: Word2Vec 모델에서 "왕"과 "여왕"은 서로 가까운 위치에 벡터로 표현됩니다.
연관키워드: 자연어 처리, 임베딩, 의미론적 유사성
LLM (Large Language Model)
```

# 마치며

RAG에서의 핵심 중 하나인 Retriever(검색기)에 대해서 다뤄보았습니다. 특히나 여러가지 검색기들이 있었지만 실무에서 가장 많이 사용되는 검색기들 위주로 다루어 보았고, BM25Retriever를 사용할 때는 공백 단위로 tokenizing을 하는 것보다는 한국어의 특징에 맞게 좀 더 잘 동작하도록 하는 키워드 검색을 위한 형태소 분석기에서 제공하는 tokenizer를 이용한 BM25Retriever에 대해서도 다루어 보았습니다. 이제 다음 포스트에서는 여태까지 배웠던 것을 통합하여 진짜 RAG 시스템에 대해서 알아보도록 하겠습니다.

긴 글 읽어주셔서 감사드리며 잘못된 내용이나 오타, 궁금하신 사항이 있으실 경우 댓글 달아주시기 바랍니다.

# 참조

- 판다스 스튜디오 - 랭체인(LangChain) 입문부터 응용까지(https://wikidocs.net/book/14473)
- 테디노트 - LangChain 한국어 튜토리얼KR(https://wikidocs.net/book/14314)