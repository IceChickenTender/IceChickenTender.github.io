---
title: "[LLM/RAG] LangChain - 7. LangChain에서의 Text Splitter"
categories:
  - LangChain
tags:
  - LangChain
  
use_math: true  
toc: true
toc_sticky: true
toc_label: "LangChain에서의 Text Splitter"
---

# 1. Text Splitter 개요

LangChain 은 긴 문서를 작은 단위인 청크(chunk)로 나누는 텍스트 분리 도구를 다양하게 지원합니다. 텍스트를 분리하는 작업을 청킹(chunking)이라고 부르기도 합니다. 이렇게 문서를 작은 조각으로 나누는 이유는 LLM 모델의 입력 토큰의 개수가 정해져 있기 때문입니다. 허용 한도를 넘는 텍스트는 모델에서 입력으로 처리할 수 없기 때문이죠 한편, 텍스트가 너무 긴 경우에는 핵심 정보 이외에 불필요한 정보들이 많이 포함될 수 있어서 RAG 품질이 낮아지는 요인이 될 수도 있습니다. 핵심 정보가 유지될 수 있는 적절한 크기로 나누는 것이 매우 중요합니다.

LangChain 이 지원하는 다양한 텍스트 분리기(Text Splitter)는 분할하려는 텍스트 유형과 사용 사례에 맞춰 선택할 수 있는 다양한 옵션이 제공됩니다. 크게 두 가지 차원에서 검토가 필요합니다.

1. 텍스트가 어떻게 분리되는지

	텍스트를 나눌 때 각 청크가 독립적으로 의미를 갖도록 나눠야 합니다. 이를 위해 문장, 구절, 단락 등 문서 구조를 기준으로 나눌 수 있습니다.

2. 청크 크기가 어떻게 측정 되는지

	각 청크의 크기를 직접 조정할 수 있습니다. LLM 모델의 입력 크기와 비용 등을 종합적으로 고려하여 애플리케이션에 적합한 최적 크기를 결정하는 기준입니다. 예를 들면 단어 수, 문자 수 등을 기준으로 나눌 수 있습니다.

# 2. CharacterTextSplitter

## 2.1 문서를 개별 문자 단위로 나누기(`separator=""`)

`CharacterTextSplitter` 클래스는 주어진 텍스트를 문자 단위로 분할하는데 사용됩니다. Python 의 split 함수라고 생각하시면 됩니다. 다음 코드에서 적용된 주요 매개변수는 다음과 같습니다.

- `separator` : 분할된 각 청크를 구분할 때 기준이 되는 문자열입니다. 여기서는 빈 문자열('')을 사용하므로, 각 글자를 기준으로 분할합니다.

- `chunk_size` : 각 청크의 최대 길이입니다. 여기서는 '500'으로 설정되어 있으므로, 최대 500자까지의 텍스트가 하나의 청크에 포함됩니다.

- `chunk_overlap` : 인접한 청크 사이에 중복으로 포함될 문자의 수입니다. 여기서는 '100'으로 설정되어 있으므로, 각 청크들은 연결 부분에서 100자가 중복됩니다.

- `length_function` : 청크의 길이를 계산하는 함수입니다. 여기서는 `len` 함수가 사용되었으므로, 문자열의 길이를 기반으로 청크의 길이를 계산합니다.

`split_text` 메소드는 주어진 텍스트를 위에서 설정한 매개변수에 따라 분할하고, 분할된 청크의 리스트를 반환합니다. `len(texts)` 는 분할된 청크의 총 수를 나타냅니다. 여기서는 3개의 청크로 분할됩니다.

여기서 중요한 것은 각 청크의 크기가 `chunk_size` 를 초과하지 않으며, 인접한 청크 사이에는 `chunk_overlap` 만큼의 문자가 중복되어 있음을 이해하는 것입니다. 이렇게 함으로써, 텍스트의 의미적 연속성을 유지하면서도 큰 데이터를 더 작은 단위로 분할할 수 있습니다.

사용한 데이터는 이전에 TextLoader 클래스를 설명할 때 사용한 'history.txt' 파일에 있는 데이터를 사용했습니다.

```python
# 각 문자를 구분하여 분할

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# 구글 드라이브에서 history.txt 파일을 로드
file_path = "/content/drive/MyDrive/LangChain/history.txt"

loader = TextLoader(file_path)
data = loader.load()

# 텍스트 분리 진행
text_splitter = CharacterTextSplitter(
    separator = '',
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len,
)

texts = text_splitter.split_text(data[0].page_content)

print(len(texts))
```

```
실행 결과

3
```

분할된 텍스트 조각 중에서 첫 번째 청크의 길이를 확인해보면 정확하게 500자임을 알 수 있습니다.

```python
# 각 문자를 구분하여 분할

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# 구글 드라이브에서 history.txt 파일을 로드
file_path = "/content/drive/MyDrive/LangChain/history.txt"

loader = TextLoader(file_path)
data = loader.load()

# 텍스트 분리 진행
text_splitter = CharacterTextSplitter(
    separator = '',
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len,
)

texts = text_splitter.split_text(data[0].page_content)

print(len(texts[0]))
```

```
실행 결과

500
```

---

## 2.2 문서를 특정 문자열 기준으로 나누기 (`separator="문자열"`)

`CharacterTextSplitter` 클래스의 `separator` 매개변수를 줄바꿈 문자로 설정하는 예제입니다. 이렇게 하면 각 청크를 나누는 기준을 줄바꿈 문자로 설정하는 것입니다.

우선 이전에 `separator` 를 지정하지 않았을 경우 각 청크들의 길이가 어떻게 되는지 출력을 해보면 다음과 같습니다.

```
실행 결과

500, 499 434
```

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

file_path = "/content/drive/MyDrive/LangChain/history.txt"

loader = TextLoader(file_path)
data = loader.load()

text_splitter = CharacterTextSplitter(
    separator = '\n',
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len,
)

texts = text_splitter.split_text(data[0].page_content)

print(len(texts))
print(len(texts[0]), len(texts[1]), len(texts[2]))
```

분할된 각 청크의 길이를 확인해보면 정확하게 500자 단위로 나누어지지 않았습니다. 이처럼 줄바꿈 문자를 기준으로 최대 500자를 맞출 수 있는 위치를 찾아서 분할하게 됩니다.

```
실행 결과

3
411 386 427
```

---

# 3. RecursiveCharacterTextSplitter

`RecursiveCharacterTextSplitter` 클래스는 텍스트를 재귀적으로 분할하여 의미적으로 관련 있는 텍스트 조각들이 같이 있도록 하는 목적으로 설계되었습니다. 이 과정에서 문자 리스트(`['\n\n', '\n', ' ', '']`)의 문자를 순서대로 사용하여 텍스트를 분할하며, 분할된 청크들이 설정된 `chunk_size` 보다 작아질 때까지 이 과정을 반복합니다. 여기서 `chunk_overlap` 은 분할된 텍스트 조각들 사이에서 중복으로 포함될 문자 수를 정의합니다. `length_function = len` 코드는 분할의 기준이 되는 길이를 측정하는 함수로 문자열의 길이를 반환하는 `len` 함수를 사용한다는 의미입니다.

`texts = text_splitter.split_text(data[0].page_content)` 코드는 `data[0].page_content` 에서 첫 번째 문서의 내용을 `RecursiveCharacterTextSplitter` 를 사용하여 분할하고, 결과를 `texts` 변수에 할당합니다. `data` 리스트에서 첫 번째 문서의 내용을 기반으로 분할 작업을 수행하게 됩니다. `len(texts)` 는 분할된 텍스트 조각들의 총 수를 반환합니다.

사용한 데이터는 `history.txt` 파일을 사용하였습니다.

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "/content/drive/MyDrive/LangChain/history.txt"

loader = TextLoader(file_path)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len,
)

texts = text_splitter.split_text(data[0].page_content)

print(len(texts))
len(texts[0]), len(texts[1]), len(texts[2])
```

```
실행 결과

3
413, 388, 429
```

# 4. 토큰 수를 기준으로 텍스트 분할(Tokenizer 활용)

대규모 언어 모델(LLM)을 사용할 때 모델이 처리할 수 있는 토큰 수에는 한계가 있습니다. 입력 데이터를 모델의 제한을 초과하지 않도록 적절히 분할하는 것이 중요합니다. 이 때 LLM 모델에 적용되는 토크나이저를 기준으로 텍스트를 토큰으로 분할하고, 이 토큰들의 수를 기준으로 텍스트를 청크로 나누면 모델 입력 토큰 수를 조절할 수 있습니다.

## 4.1 tiktoken을 이용한 Text 분할

OpenAI API의 경우 `tiktoken` 라이브러리를 통해 해당 모델에서 사용하는 토크나이저를 기준으로 분할할 수 있습니다. `CharacterTextSplitter.from_tiktoken_encoder` 메서드는 글자 수 기준으로 분할할 때 `tiktoken` 토크나이저를 기준으로 글자수를 계산하여 분할합니다. 여기서 `encoding_name='cl100k_base`는 텍스트를 토큰으로 변환하는 인코딩 방식을 나타냅니다.

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

file_path = "/content/drive/MyDrive/LangChain/history.txt"

loader = TextLoader(file_path)
data = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 600,
    chunk_overlap = 200,
    encoding_name = 'cl100k_base'
)

docs = text_splitter.split_documents(data)

print(len(docs))

print(len(docs[0].page_content))
print(docs[0].page_content)
```

```
실행 결과

3
525
한국의 역사는 수천 년에 걸쳐 이어져 온 긴 여정 속에서 다양한 문화와 전통이 형성되고 발전해 왔습니다. 고조선에서 시작해 삼국 시대의 경쟁, 그리고 통일 신라와 고려를 거쳐 조선까지, 한반도는 많은 변화를 겪었습니다.

고조선은 기원전 2333년 단군왕검에 의해 세워졌다고 전해집니다. 이는 한국 역사상 최초의 국가로, 한민족의 시원이라 할 수 있습니다. 이후 기원전 1세기경에는 한반도와 만주 일대에서 여러 소국이 성장하며 삼한 시대로 접어듭니다.

4세기경, 고구려, 백제, 신라의 삼국이 한반도의 주요 세력으로 부상했습니다. 이 시기는 삼국이 각각 문화와 기술, 무력을 발전시키며 경쟁적으로 성장한 시기로, 한국 역사에서 중요한 전환점을 마련했습니다. 특히 고구려는 북방의 강대국으로 성장하여 중국과도 여러 차례 전쟁을 벌였습니다.

7세기 말, 신라는 당나라와 연합하여 백제와 고구려를 차례로 정복하고, 한반도 최초의 통일 국가인 통일 신라를 건립합니다. 이 시기에 신라는 불교를 국교로 채택하며 문화와 예술이 크게 발전했습니다.
```

## 4.2 SentenceTransformer의 Tokenizer를 이용한 방법

허깅페이스의 transformers의 sentence-transformer에 있는 tokenizer를 사용하는 방법도 있습니다. LangChain에서는 langchain-text-splitters의 SentenceTransformersTokenTextSplitter를 이용해 사용할 수 있습니다. 따로 모델을 지정하지 않으면 huggingface의 sentence-transformers/all-MiniLM-L6-v2 모델에 있는 tokenizer를 기본적으로 사용한다고 합니다. 그러면 SentenceTransformersTokenTextSplitter를 이용한 텍스트 분할을 한 번 진행해 보도록 하겠습니다.

우선 진행에 필요한 라이브러리를 먼저 설치해 줍니다.

```bash
pip install langchain-text-splitters
```

SentenceTransformersTokenTextSplitter로 chunk_size는 200, chunk_overlap은 0으로 설정해서 splitter 변수에 할당해 줍니다.

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

splitter = SentenceTransformersTokenTextSplitter(chunk_size=200, chunk_overlap=0)
```

테스트로 사용할 텍스트 파일을 읽어와 줍니다.

```python
with open("/content/drive/MyDrive/LangChain/history.txt") as f:
    file = f.read()

print(file[:350])
```

```
Output:
한국의 역사는 수천 년에 걸쳐 이어져 온 긴 여정 속에서 다양한 문화와 전통이 형성되고 발전해 왔습니다. 고조선에서 시작해 삼국 시대의 경쟁, 그리고 통일 신라와 고려를 거쳐 조선까지, 한반도는 많은 변화를 겪었습니다.

고조선은 기원전 2333년 단군왕검에 의해 세워졌다고 전해집니다. 이는 한국 역사상 최초의 국가로, 한민족의 시원이라 할 수 있습니다. 이후 기원전 1세기경에는 한반도와 만주 일대에서 여러 소국이 성장하며 삼한 시대로 접어듭니다.

4세기경, 고구려, 백제, 신라의 삼국이 한반도의 주요 세력으로 부상했습니다. 이 시기는 삼국이 각각 문화와 기술, 무력을 발전시키며 경쟁적으로 성장한 시기로, 한국 역사에
```

splitter를 이용해 텍스트 분할을 진행해 주고 첫 번째 청크를 출력해 봅니다.

```python
text_chunks = splitter.split_text(text=file)

print(text_chunks[0])
```

출력 결과를 보면 중간 중간에 [UNK]가 있는 것을 확인할 수 있습니다. SentenceTransformersTokenTextSplitter는 따로 모델을 지정해 두지 않으면 기본 모델인 sentence-transformers/all-MiniLM-L6-v2 모델을 사용하는데 해당 모델은 주로 영어 코퍼스로 학습을 했기 때문에 한국어에는 취약합니다. 그래서 조금만 어려운 한글이나 한자가 나오면 바로 [UNK]를 처리해버립니다.

```
Output:
한국의 역사는 수천 년에 걸쳐 이어져 온 긴 여정 속에서 [UNK] 문화와 전통이 [UNK] 발전해 [UNK]. 고조선에서 시작해 삼국 시대의 경쟁, 그리고 통일 신라와 고려를 거쳐 [UNK], 한반도는 [UNK] 변화를 [UNK]. 고조선은 기원전 2333년 단군왕검에 의해 [UNK] 전해집니다. 이는 한국 역사상 [UNK] 국가로, 한민족의 시원이라 할 수 [UNK]. 이후 기원전 1세기경에는 한반도와 만주 일대에서 여러 소국이 성장하며 삼한 시대로 접어듭니다. 4세기ᄀ
```

그렇다면 한국어 데이터로 학습한 모델로 텍스트 분할을 한 번 진행해 보도록 하겠습니다. 사용한 모델은 jhgan/ko-sroberta-multitask 모델을 사용했습니다.

```python
splitter = SentenceTransformersTokenTextSplitter(
    model_name="jhgan/ko-sroberta-multitask",
    chunk_size=200,
    chunk_overlap=0
)

text_chunks = splitter.split_text(text=file)

print(text_chunks[0])
```

출력 결과를 보면 기본 모델인 sentence-transformers/all-MiniLM-L6-v2의 결과와 비교했을 때 [UNK]가 하나도 없고 청킹이 잘 된 것을 확인할 수 있습니다.

```
Output:
한국의 역사는 수천 년에 걸쳐 이어져 온 긴 여정 속에서 다양한 문화와 전통이 형성되고 발전해 왔습니다. 고조선에서 시작해 삼국 시대의 경쟁, 그리고 통일 신라와 고려를 거쳐 조선까지, 한반도는 많은 변화를 겪었습니다. 고조선은 기원전 2333년 단군왕검에 의해 세워졌다고 전해집니다. 이는 한국 역사상 최초의 국가로, 한민족의 시원이라 할 수 있습니다. 이후 기원전 1세기경에는 한반도와 만주 일대에서 여러 소국이 성장하며 삼한 시대로 접어듭니다. 4세기경, 고구려
```

# 5. MarkdownHeaderTextSplitter

문서의 전체적인 맥락과 구조를 고려하여 의미 있는 방식으로 텍스트를 임베딩하는 과정은, 광범위한 의미와 주제를 더 잘 포착할 수 있는 포괄적인 벡터 표현을 생성하는데 큰 도움이 됩니다. 이러한 맥락에서, 기술 문서나 블로그 글은 대부분 #(H1), ##(H2) 같은 헤더 구조를 가지며, 일반적으로 마크다운 포맷을 이용해 작성되는 경우가 많습니다. 여기서 MarkdownHeaderTextSplitter는 Header와 같은 메타데이터를 청크에 같이 붙여줘서 검색 정확도를 높여줍니다. 그럼 예제 코드로 한 번 알아보도록 하겠습니다.

예제 실행 전에 우선 라이브러리 설치를 먼저 진행해 줍니다.

```bash
pip install langchain-text-splitters
```

MarkdownHeaderTextSplitter를 사용하여 마크다운 형식의 텍스트를 헤더 단위로 분할합니다.

- 마크다운 문서의 헤더(`#`, `##`, `###`)를 기준으로 텍스트를 분할하는 역할을 합니다.
- markdown_document 변수에 마크다운 형식의 문서가 할당됩니다.
- headers_to_split_on 리스트에는 마크다운 헤더 레벨과 해당 레벨의 이름이 튜플 형태로 정의됩니다
- MarkdownHeaderTextSplitter 클래스를 사용하여 markdown_splitter 객체를 생성하며, headers_to_split_on 매개변수로 분할 기준이 되는 헤더 레벨을 전달합니다.
- split_text 메서드를 호출하여 markdown_document를 헤더 레벨에 따라 분할합니다.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 마크다운 형식의 문서를 문자열로 정의합니다.
markdown_document = "# Title\n\n## 1. SubTitle\n\nHi this is Jim\n\nHi this is Joe\n\n### 1-1. Sub-SubTitle \n\nHi this is Lance \n\n## 2. Baz\n\nHi this is Molly"
print(markdown_document)
```

```
Output:
# Title

## 1. SubTitle

Hi this is Jim

Hi this is Joe

### 1-1. Sub-SubTitle 

Hi this is Lance 

## 2. Baz

Hi this is Molly
```

```python
headers_to_split_on = [  # 문서를 분할할 헤더 레벨과 해당 레벨의 이름을 정의합니다.
    (
        "#",
        "Header 1",
    ),  # 헤더 레벨 1은 '#'로 표시되며, 'Header 1'이라는 이름을 가집니다.
    (
        "##",
        "Header 2",
    ),  # 헤더 레벨 2는 '##'로 표시되며, 'Header 2'라는 이름을 가집니다.
    (
        "###",
        "Header 3",
    ),  # 헤더 레벨 3은 '###'로 표시되며, 'Header 3'이라는 이름을 가집니다.
]

# 마크다운 헤더를 기준으로 텍스트를 분할하는 MarkdownHeaderTextSplitter 객체를 생성합니다.
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# markdown_document를 헤더를 기준으로 분할하여 md_header_splits에 저장합니다.
md_header_splits = markdown_splitter.split_text(markdown_document)
# 분할된 결과를 출력합니다.
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

```
Output:
Hi this is Jim  
Hi this is Joe
{'Header 1': 'Title', 'Header 2': '1. SubTitle'}
=====================
Hi this is Lance
{'Header 1': 'Title', 'Header 2': '1. SubTitle', 'Header 3': '1-1. Sub-SubTitle'}
=====================
Hi this is Molly
{'Header 1': 'Title', 'Header 2': '2. Baz'}
=====================
```

기본적으로 MarkdownHeaderTextSplitter는 분할되는 헤더를 출력 청크의 내용에서 제거합니다. 하지만 strip_headers의 값을 False로 설정하여 비활성화 할 수 있습니다.

```python
markdown_splitter = MarkdownHeaderTextSplitter(
    # 분할할 헤더를 지정합니다.
    headers_to_split_on=headers_to_split_on,
    # 헤더를 제거하지 않도록 설정합니다.
    strip_headers = False,
)

# 마크다운 문서를 헤더를 기준으로 분할합니다.
md_header_splits = markdown_splitter.split_text(markdown_document)

# 분할된 결과를 출력합니다.
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")
```

strip_headers의 값을 False로 바꾸면 위의 결과와 달리 header 내용까지 포함된 것을 확인할 수 있습니다.

```
Output:
# Title  
## 1. SubTitle  
Hi this is Jim  
Hi this is Joe
{'Header 1': 'Title', 'Header 2': '1. SubTitle'}
=====================
### 1-1. Sub-SubTitle  
Hi this is Lance
{'Header 1': 'Title', 'Header 2': '1. SubTitle', 'Header 3': '1-1. Sub-SubTitle'}
=====================
## 2. Baz  
Hi this is Molly
{'Header 1': 'Title', 'Header 2': '2. Baz'}
=====================
```

# 6. SemanticChunker

텍스트를 의미론적 유사성에 기반하여 분류합니다. 단순히 줄바꿈이나 글자 수가 아니라, 임베딩 유사도(Embedding Similarity)를 계산해서 주제가 바뀌는 지점에서 자릅니다. 이는 사람이 글을 읽는 호흡과 가장 비슷하게 자르는 효과가 있는 방법입니다.

## 6.1 breakpoints 인자를 이용한 방법

예제에서 사용한 임베딩 모델은 Google gemini-embedding-001을 사용했습니다.

### 6.1.1 percentile 방법

백분위수(Percentile)를 이용한 방법입니다. 이 방법에서는 문장 간의 모든 차이를 계산한 다음, 지정한 백분위수를 기준으로 분리합니다.

우선 예제 실행에 필요한 라이브러리 설치를 진행해 주도록 하겠습니다.

```bash
pip install langchain-huggingface langchain-experimental langchain-google-genai
```

먼저 예제에 사용할 텍스트 파일을 읽어옵니다.

```python
# data/appendix-keywords.txt 파일을 열어서 f라는 파일 객체를 생성합니다.
with open("/content/drive/MyDrive/LangChain/appendix-keywords.txt") as f:
    file = f.read()  # 파일의 내용을 읽어서 file 변수에 저장합니다.
```

SemanticChunker 클래스에서 breakpoint_threshold_type을 `percentile`로 맞춰주고 breakpoint_threshold_amount 값을 70으로 설정합니다. 

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
)
```

```python
docs = text_splitter.create_documents([file])
for i, doc in enumerate(docs[:5]):
    print(f"[Chunk {i}]", end="\n\n")
    print(doc.page_content)
    print("==="*20)
```

각 청크별로 살펴보면 어느정도 비슷한 문장끼리 묶어서 분할된 것을 확인할 수 있습니다.

```
Output:
[Chunk 0]

Semantic Search

정의: 의미론적 검색은 사용자의 질의를 단순한 키워드 매칭을 넘어서 그 의미를 파악하여 관련된 결과를 반환하는 검색 방식입니다. 예시: 사용자가 "태양계 행성"이라고 검색하면, "목성", "화성" 등과 같이 관련된 행성에 대한 정보를 반환합니다. 연관키워드: 자연어 처리, 검색 알고리즘, 데이터 마이닝

Embedding

정의: 임베딩은 단어나 문장 같은 텍스트 데이터를 저차원의 연속적인 벡터로 변환하는 과정입니다. 이를 통해 컴퓨터가 텍스트를 이해하고 처리할 수 있게 합니다.
============================================================
[Chunk 1]

예시: "사과"라는 단어를 [0.65, -0.23, 0.17]과 같은 벡터로 표현합니다. 연관키워드: 자연어 처리, 벡터화, 딥러닝

Token

정의: 토큰은 텍스트를 더 작은 단위로 분할하는 것을 의미합니다. 이는 일반적으로 단어, 문장, 또는 구절일 수 있습니다.
============================================================
[Chunk 2]

예시: 문장 "나는 학교에 간다"를 "나는", "학교에", "간다"로 분할합니다. 연관키워드: 토큰화, 자연어 처리, 구문 분석

Tokenizer

정의: 토크나이저는 텍스트 데이터를 토큰으로 분할하는 도구입니다. 이는 자연어 처리에서 데이터를 전처리하는 데 사용됩니다.
============================================================
[Chunk 3]

예시: "I love programming."이라는 문장을 ["I", "love", "programming", "."]으로 분할합니다. 연관키워드: 토큰화, 자연어 처리, 구문 분석

VectorStore

정의: 벡터스토어는 벡터 형식으로 변환된 데이터를 저장하는 시스템입니다. 이는 검색, 분류 및 기타 데이터 분석 작업에 사용됩니다.
============================================================
[Chunk 4]

예시: 단어 임베딩 벡터들을 데이터베이스에 저장하여 빠르게 접근할 수 있습니다. 연관키워드: 임베딩, 데이터베이스, 벡터화

SQL

정의: SQL(Structured Query Language)은 데이터베이스에서 데이터를 관리하기 위한 프로그래밍 언어입니다. 데이터 조회, 수정, 삽입, 삭제 등 다양한 작업을 수행할 수 있습니다.
============================================================
```

### 6.1.2 Standard Deviation 방법

이 방법은 지정한 breakpoint_threshold_amount 표준편차보다 큰 차이가 있는 경우 분할됩니다.

```python
text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.25,
)
```

```python
docs = text_splitter.create_documents([file])
for i, doc in enumerate(docs[:5]):
    print(f"[Chunk {i}]", end="\n\n")
    print(doc.page_content)
    print("==="*20)
```

결과를 확인해 보면 각 청크의 길이가 조금 더 길어졌지만 몇몇 문장은 청크에 있는 문장들과 큰 유사성은 없어 보이는 문장들도 보입니다.

```
Output:
[Chunk 0]

Semantic Search

정의: 의미론적 검색은 사용자의 질의를 단순한 키워드 매칭을 넘어서 그 의미를 파악하여 관련된 결과를 반환하는 검색 방식입니다. 예시: 사용자가 "태양계 행성"이라고 검색하면, "목성", "화성" 등과 같이 관련된 행성에 대한 정보를 반환합니다. 연관키워드: 자연어 처리, 검색 알고리즘, 데이터 마이닝

Embedding

정의: 임베딩은 단어나 문장 같은 텍스트 데이터를 저차원의 연속적인 벡터로 변환하는 과정입니다. 이를 통해 컴퓨터가 텍스트를 이해하고 처리할 수 있게 합니다.
============================================================
[Chunk 1]

예시: "사과"라는 단어를 [0.65, -0.23, 0.17]과 같은 벡터로 표현합니다. 연관키워드: 자연어 처리, 벡터화, 딥러닝

Token

정의: 토큰은 텍스트를 더 작은 단위로 분할하는 것을 의미합니다. 이는 일반적으로 단어, 문장, 또는 구절일 수 있습니다. 예시: 문장 "나는 학교에 간다"를 "나는", "학교에", "간다"로 분할합니다. 연관키워드: 토큰화, 자연어 처리, 구문 분석

Tokenizer

정의: 토크나이저는 텍스트 데이터를 토큰으로 분할하는 도구입니다. 이는 자연어 처리에서 데이터를 전처리하는 데 사용됩니다. 예시: "I love programming."이라는 문장을 ["I", "love", "programming", "."]으로 분할합니다. 연관키워드: 토큰화, 자연어 처리, 구문 분석

VectorStore

정의: 벡터스토어는 벡터 형식으로 변환된 데이터를 저장하는 시스템입니다. 이는 검색, 분류 및 기타 데이터 분석 작업에 사용됩니다.
============================================================
[Chunk 2]

예시: 단어 임베딩 벡터들을 데이터베이스에 저장하여 빠르게 접근할 수 있습니다. 연관키워드: 임베딩, 데이터베이스, 벡터화

SQL

정의: SQL(Structured Query Language)은 데이터베이스에서 데이터를 관리하기 위한 프로그래밍 언어입니다. 데이터 조회, 수정, 삽입, 삭제 등 다양한 작업을 수행할 수 있습니다.
============================================================
[Chunk 3]

예시: SELECT * FROM users WHERE age > 18;은 18세 이상의 사용자 정보를 조회합니다. 연관키워드: 데이터베이스, 쿼리, 데이터 관리

CSV

정의: CSV(Comma-Separated Values)는 데이터를 저장하는 파일 형식으로, 각 데이터 값은 쉼표로 구분됩니다. 표 형태의 데이터를 간단하게 저장하고 교환할 때 사용됩니다. 예시: 이름, 나이, 직업이라는 헤더를 가진 CSV 파일에는 홍길동, 30, 개발자와 같은 데이터가 포함될 수 있습니다. 연관키워드: 데이터 형식, 파일 처리, 데이터 교환

JSON

정의: JSON(JavaScript Object Notation)은 경량의 데이터 교환 형식으로, 사람과 기계 모두에게 읽기 쉬운 텍스트를 사용하여 데이터 객체를 표현합니다. 예시: {"이름": "홍길동", "나이": 30, "직업": "개발자"}는 JSON 형식의 데이터입니다. 연관키워드: 데이터 교환, 웹 개발, API

Transformer

정의: 트랜스포머는 자연어 처리에서 사용되는 딥러닝 모델의 한 유형으로, 주로 번역, 요약, 텍스트 생성 등에 사용됩니다. 이는 Attention 메커니즘을 기반으로 합니다.
============================================================
[Chunk 4]

예시: 구글 번역기는 트랜스포머 모델을 사용하여 다양한 언어 간의 번역을 수행합니다. 연관키워드: 딥러닝, 자연어 처리, Attention

HuggingFace

정의: HuggingFace는 자연어 처리를 위한 다양한 사전 훈련된 모델과 도구를 제공하는 라이브러리입니다. 이는 연구자와 개발자들이 쉽게 NLP 작업을 수행할 수 있도록 돕습니다. 예시: HuggingFace의 Transformers 라이브러리를 사용하여 감정 분석, 텍스트 생성 등의 작업을 수행할 수 있습니다. 연관키워드: 자연어 처리, 딥러닝, 라이브러리

Digital Transformation

정의: 디지털 변환은 기술을 활용하여 기업의 서비스, 문화, 운영을 혁신하는 과정입니다. 이는 비즈니스 모델을 개선하고 디지털 기술을 통해 경쟁력을 높이는 데 중점을 둡니다.
============================================================
```

### 6.1.3 Interquartile

이 방법은 사분위수 범위(interquartile range)를 사용하여 청크를 분할하는 방법으로 사분위수 범위란 제3사분위수에서 제1사분위수를 뺀 값으로, 전체 데이터의 중간이 얼마나 퍼져있는지를 알 수 있는 척도입니다. 사분위수를 구한 데이터들의 값과 비교해 사분위수 범위값이 클수록 자료의 흐트러진 정도가 큽니다.

```python
text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    breakpoint_threshold_type="interquartile",
    breakpoint_threshold_amount=0.5,
)
```

```python
docs = text_splitter.create_documents([file])

for i, doc in enumerate(docs[:5]):
    print(f"[Chunk {i}]", end="\n\n")
    print(doc.page_content)
    print("==="*20)
```

```
Output:
[Chunk 0]

Semantic Search

정의: 의미론적 검색은 사용자의 질의를 단순한 키워드 매칭을 넘어서 그 의미를 파악하여 관련된 결과를 반환하는 검색 방식입니다. 예시: 사용자가 "태양계 행성"이라고 검색하면, "목성", "화성" 등과 같이 관련된 행성에 대한 정보를 반환합니다. 연관키워드: 자연어 처리, 검색 알고리즘, 데이터 마이닝

Embedding

정의: 임베딩은 단어나 문장 같은 텍스트 데이터를 저차원의 연속적인 벡터로 변환하는 과정입니다. 이를 통해 컴퓨터가 텍스트를 이해하고 처리할 수 있게 합니다.
============================================================
[Chunk 1]

예시: "사과"라는 단어를 [0.65, -0.23, 0.17]과 같은 벡터로 표현합니다. 연관키워드: 자연어 처리, 벡터화, 딥러닝

Token

정의: 토큰은 텍스트를 더 작은 단위로 분할하는 것을 의미합니다. 이는 일반적으로 단어, 문장, 또는 구절일 수 있습니다.
============================================================
[Chunk 2]

예시: 문장 "나는 학교에 간다"를 "나는", "학교에", "간다"로 분할합니다. 연관키워드: 토큰화, 자연어 처리, 구문 분석

Tokenizer

정의: 토크나이저는 텍스트 데이터를 토큰으로 분할하는 도구입니다. 이는 자연어 처리에서 데이터를 전처리하는 데 사용됩니다. 예시: "I love programming."이라는 문장을 ["I", "love", "programming", "."]으로 분할합니다. 연관키워드: 토큰화, 자연어 처리, 구문 분석

VectorStore

정의: 벡터스토어는 벡터 형식으로 변환된 데이터를 저장하는 시스템입니다. 이는 검색, 분류 및 기타 데이터 분석 작업에 사용됩니다.
============================================================
[Chunk 3]

예시: 단어 임베딩 벡터들을 데이터베이스에 저장하여 빠르게 접근할 수 있습니다. 연관키워드: 임베딩, 데이터베이스, 벡터화

SQL

정의: SQL(Structured Query Language)은 데이터베이스에서 데이터를 관리하기 위한 프로그래밍 언어입니다. 데이터 조회, 수정, 삽입, 삭제 등 다양한 작업을 수행할 수 있습니다.
============================================================
[Chunk 4]

예시: SELECT * FROM users WHERE age > 18;은 18세 이상의 사용자 정보를 조회합니다. 연관키워드: 데이터베이스, 쿼리, 데이터 관리

CSV

정의: CSV(Comma-Separated Values)는 데이터를 저장하는 파일 형식으로, 각 데이터 값은 쉼표로 구분됩니다. 표 형태의 데이터를 간단하게 저장하고 교환할 때 사용됩니다.
============================================================
```

# 마치며

LangChain에서의 Text Splitter에 대해서 알아보았습니다. 매우 다양한 Text Splitter들이 있었지만 그 중에서도 자주 사용되는 것들만 알아보았습니다.

긴 글 읽어주셔서 감사드리며 본문 내용 중 잘못된 내용이나 오타가 있을 경우 댓글 달아주시기 바랍니다.

# 참조

- 판다스 스튜디오 - 랭체인(LangChain) 입문부터 응용까지(https://wikidocs.net/book/14473)
- 테디노트 - LangChain 한국어 튜토리얼KR(https://wikidocs.net/book/14314)
