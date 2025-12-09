---
title: "[ToyProject] 모델별 개체명 인식 성능 비교 - 1. BERT 모델을 이용한 개체명 인식기 구축과 성능 평가"
categories:
  - ToyProject
tags:
  - ToyProject
  - Deeplearning
  - LLM/RAG

use_math: true
toc: true
toc_sticky: true
toc_label: "BERT 모델을 이용한 개체명 인식기 구축과 성능 평가"
---

저는 LLM 모델과 RAG라는 기술의 전반적인 내용에 대해 독학을 진행하고 있었습니다. 그리고 공부 초기에 어느 정도 공부가 되었다고 판단이 되면 토이 프로젝트도 한 번 진행해 봐야겠다는 생각도 하고 있었습니다. 제대로 공부한지는 올해 9월 부터 공부를 했고, 3개월 정도 공부를 한 결과 이제 약간이나마 개념이 잡힌 듯 하다고 생각해 이전부터 해보고 싶던 토이 프로젝트를 진행해 보고 그 과정들을 블로그에 기록해 보고자 합니다. 다음은 제가 이전부터 해보고 싶었던 토이 프로젝트의 내용입니다.

제가 해보고 싶은 토이 프로젝트는 머신러닝부터 시작해서 현재의 LLM 모델까지 각 시대를 대표한 모델을 이용해 개체명 인식 모델을 직접 구현해 보고, 각 모델별로 성능 평가와 실행 속도 등을 구해 각 모델별 차이를 한 번 살펴보고자 합니다. 제가 생각한 토이 프로젝트에서 사용할 모델들은 다음과 같습니다.

- 머신러닝 : CRF 모델
- 딥러닝 : Bi-LSTM-CRF 모델
- 딥러닝 : BERT 모델
- LLM : sLLM 모델

제가 위 모델들을 고른 이유는 개체명 인식이라는 태스크에서 머신러닝 기반의 모델은 아주 오랫동안 사용되었으며, 여러 딥러닝 모델의 기반이 되는 모델이기 때문에 BaseLine으로써의 역할로 CRF 모델을 선택했습니다. Bi-LSTM-CRF 모델은 제가 대학원 때 주로 다루었던 모델이며, 그 당시 개체명 인식과 같은 시퀀스 레이블링 문제에서는 가장 높은 성능을 기록하던 모델이기 때문입니다. 세 번째 BERT 모델은 제가 대학원을 졸업할 때 쯤에 나온 모델로 그 당시 제가 졸업과 여러 프로젝트를 진행하고 있었기 때문에 다루고 싶어도 다룰 수 없었던 모델이고, 최근에는 huggingface에서 BERT와 BERT의 개선된 버전이 RoBERTa 모델이 많이 사용되고 있어 선택하였습니다. 마지막으로 sLLM 모델은 현재 전 세계적으로 LLM 모델의 뛰어난 성능이 주목 받고 있습니다. 하지만 대용량 모델과 대용량 학습 데이터를 개인이 구축하기에는 큰 어려움이 따릅니다. 그래도 최근 여러 연구에서도 그렇고 기업도 어느 정도 개인이 돌려 볼 수 있는 공개되어 있고, 사전 학습된 sLLM 모델을 이용해 프롬프트로 만든 학습 데이터로 미세 조정을 진행하면 특정 도메인과 태스크에서는 어느 정도 높은 성능을 보여준다는 점에서 작지만 직접 LLM 모델을 학습 시켜는 경험도 쌓을 수 있고, 다른 사람과 좀 더 차별점이 생길 수도 있겠단 생각에 선택을 하게 되었습니다.

우선 모델별 개체명 인식 성능 비교 토이 프로젝트 포스트의 첫 번째 내용은 huggingface를 이용해 쉽게 학습 시켜볼 수 있는 BERT 모델을 이용한 개체명 인식을 먼저 구축해 보려고 합니다. 

# 1. 데이터

이번 모델별 개체명 인식 성능 비교 토이 프로젝트에서 사용할 데이터는 KLUE의 NER 데이터입니다. KLUE 데이터셋은 한국어 언어 모델의 공정한 평가를 위한 목적으로 8개 종류(뉴스 헤드라인 분류, 문장 유사도 비교, 자연어 추론, 개체명 인식, 관계 추출, 형태소 및 의존 구문 분석, 기계 독해 이해, 대화 상태 추적)의 한국어 자연어 이해 문제가 포함된 데이터 집합체입니다. 특히 다양한 한국어 언어 모델이 동일한 평가 선상에서 정확하게 비교될 수 있는 평가기준과 토대가 된다는 점에서 KLUE NER 데이터를 사용하기로 하였습니다. 또한 huggingface에 공개되어 있어 언제든 사용할 수 있다는 장점도 있기 때문입니다.

# 2. BERT 모델

BERT 모델은 2019년에 발표된 모델로 2017년 Transformer 모델을 이용한 기계 번역 태스크 모델에서 Transformer를 이용한 Encoder 부분만 가지고 와서 비정형 텍스트 데이터를 이용해 사전 학습을 진행해 현재의 LLM과 같은 모델을 미리 구성한 후 각 태스크를 위한 학습 데이터를 이용해 미세 조정을 진행해 하나의 모델로 여러 태스크를 수행할 수 있는 모델로써 그 당시 획기적인 모델로 주목을 받았고, 현재까지도 이 모델을 이용해 허깅페이스에서 Sentence-Transformer로 단어 및 문장 임베딩을 제공하거나, 사전 학습된 모델을 이용해 태스크별 학습 데이터로 미세 조정을 통해 여러 기업에서도 많이 쓰이고 있는 모델입니다. 저에게 있어선 대학원 졸업년도에 발표된 모델인데 제가 너무 바쁜 나머지 논문만 대충 읽고 직접 다뤄보지 못했다는 것이 아직 마음에 남아 있어 직접 다뤄보고 싶다는 마음에 이 모델을 이용한 개체명 인식 모델을 구현해 보고자 생각했습니다. 그리고 무엇보다도 huggingface에 이 모델의 사전 학습 모델이 많고 huggingface에서 제공하는 여러 기능들로 다루기 쉽다는 장점이 있어 가장 먼저 다루게 되었습니다.

BERT 모델을 이용한 개체명 인식 모델 구현에서 사용할 사전 학습한 BERT 모델은 KLUE 데이터를 이용해 사전 학습을 진행한 klue/bert-base 모델을 사용하고자 합니다. 

# 3. 개체명 인식 모델 구현

모델 구현에 앞서 실행에 필요한 라이브러리 설치를 먼저 진행해 주도록 하겠습니다.

```bash
pip install transformers huggingface_hub hf_transfer datasets evaluate seqeval accelerate scikit-learn
```

## 3.1 데이터 전처리

자연어처리 분야에선 모델 구현도 중요하지만 그보다 더 중요한 것이 데이터입니다. 데이터 전처리를 통해 학습에 불필요한 정보를 없애거나, 기존 데이터셋을 모델에 맞게 변형시키는 등의 전처리 작업이 수행되어야 모델이 잘 학습해서 모델이 좋은 성능을 뽐낼 수 있습니다. 특히 BERT 모델부터는 각 모델이 가지는 tokenizer가 있기 때문에 이 tokenizer를 이용해 모델이 사용하는 tokenizer에 맞게 데이터를 변형시키는 작업이 필수 입니다. 그 이유는 사전 학습 시에도 모델이 가지고 있는 tokenizer를 이용해 분리된 token들을 이용해 사전 학습이 진행되기 때문입니다. 그리고 미세 조정 학습을 진행할 때에 사용되는 입력도 모델의 tokenizer를 이용해 분할된 token을 이용해야 사전 학습된 모델로부터 token의 임베딩 값을 얻어올 수 있기 때문에 BERT 모델에서는 필수로 진행되어야 합니다. 우선 데이터 전처리를 진행하기 전에 KLUE NER 데이터셋의 구성요소와 데이터 형태를 한 번 살펴보도록 하겠습니다.

datasets 라이브러리를 이용해 huggingface에 있는 KLUE NER 데이터를 가져와서 한 번 출력해 보도록 하겠습니다.

```python
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("klue", "ner")

print(dataset)
```

딕셔너리 형태로 train 데이터셋과 validation 데이터셋이 있는 것을 확인할 수 있으며, train 데이터셋은 21,008 문장, validation 데이터셋은 5,000 문장으로 구성되어 있는 것을 확인할 수 있습니다. 또한 train과 validation 데이터셋 내부에는 ['sentence', 'tokens', 'ner_tags'] 로 구성되어 있습니다.

```
Output:
DatasetDict({
    train: Dataset({
        features: ['sentence', 'tokens', 'ner_tags'],
        num_rows: 21008
    })
    validation: Dataset({
        features: ['sentence', 'tokens', 'ner_tags'],
        num_rows: 5000
    })
})
```

그럼 train 데이터셋에서 첫 번째 데이터를 가져와 첫 번째 데이터에 있는 sentence, tokens, ner_tags를 한 번 보도록 하겠습니다.

```python
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("klue", "ner")

train_dataset = dataset['train']

print(f"sentence : {train_dataset['sentence'][0]}")
print(f"tokens : {train_dataset['tokens'][0]}")
print(f"tokens size : {len(train_dataset['tokens'][0])}")
print(f"ner_tags : {train_dataset['ner_tags'][0]}")
print(f"ner_tags size : {len(train_dataset['ner_tags'][0])}")
```

출력 결과를 보면 sentence는 문장이며 각 개체에 `<entity:tag>` 형식이 적용되어 있으며, tokens는 공백을 포함해 한 음절씩 나뉘어져 있습니다. 그리고 ner_tags는 숫자로 구성되어 있는 huggingface의 KLUE 데이터 사이트를 참조해 보면 

"ner_tags: a list of classification labels, with possible values including B-DT(0), I-DT(1), B-LC(2), I-LC(3), B-OG(4), I-OG(5), B-PS(6), I-PS(7), B-QT(8), I-QT(9), B-TI(10), I-TI(11), O(12)"

라는 설명이 되어 있습니다. 즉, 숫자를 이용해 각 ner 태그를 나타내고 있으며, 사용하는 개체명 태그를 보면 BIO 레이블 방식을 사용하고 있는 것으로 확인이 됩니다. 또한 tokens와 ner_tags 포함된 각 리스트의 크기도 동일한 것으로 보아 공백에도 ner tag를 부착하고 있는 것으로 확인됩니다.

```
Output:
sentence : 특히 <영동고속도로:LC> <강릉:LC> 방향 <문막휴게소:LC>에서 <만종분기점:LC>까지 <5㎞:QT> 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.
tokens : ['특', '히', ' ', '영', '동', '고', '속', '도', '로', ' ', '강', '릉', ' ', '방', '향', ' ', '문', '막', '휴', '게', '소', '에', '서', ' ', '만', '종', '분', '기', '점', '까', '지', ' ', '5', '㎞', ' ', '구', '간', '에', '는', ' ', '승', '용', '차', ' ', '전', '용', ' ', '임', '시', ' ', '갓', '길', '차', '로', '제', '를', ' ', '운', '영', '하', '기', '로', ' ', '했', '다', '.']
tokens size : 66
ner_tags : [12, 12, 12, 2, 3, 3, 3, 3, 3, 12, 2, 3, 12, 12, 12, 12, 2, 3, 3, 3, 3, 12, 12, 12, 2, 3, 3, 3, 3, 12, 12, 12, 8, 9, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
ner_tags size : 66
```

그렇다면 실제로 모든 데이터의 tokens는 음절 단위로 나누어져 있는지 확인을 해보도록 하겠습니다. train 데이터셋의 모든 tokens를 순회하면서 요소 중에 길이가 1보다 크면 카운트를 세도록 하고, 카운트 값이 1보다 클 경우에는 `klue ner datasets tokens is not a syllable` 를 출력하도록 했습니다. 그게 아니라면 `klue ner datasets tokens is a syllable` 를 출력하도록 해보았습니다.

```python
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("klue", "ner")

train_dataset = dataset['train']

all_tokens = train_dataset['tokens']

cnt = 0

for tokens in all_tokens:
    for token in tokens:
        if len(token) > 1:
            cnt+=1

if cnt > 0:
    print(f"klue ner datasets tokens is not a syllable")
else:
    print(f"klue ner datasets tokens is a syllable")
```

출력 결과로 `klue ner datasets tokens is a syllable`가 출력된 것을 보아 데이터셋의 모든 tokens 데이터는 음절 단위로 쪼개져있고, 개체명 태그 부착도 음절 단위로 되어 있습니다.

```
Output:
klue ner datasets tokens is a syllable
```

그렇다면 여기서 우리는 KLUE NER 데이터셋은 왜 음절 단위로 자른 것인지에 대해서 생각해볼 필요가 있습니다. 우선 저는 KLUE NER 데이터셋은 BERT 모델에 사용할 데이터이니 KLUE와 관련된 모델의 tokenizer 때문에 음절 단위로 잘린 것인가 하는 생각이 들어 제가 사용할 klue/bert-base 모델의 tokenizer에 우리가 사용할 데이터의 문장을 넣어 어떻게 token 단위로 분할해 주는지 한 번 확인해 보았습니다.

train 데이터셋의 첫 번째 tokens에 있는 token 값들을 이어 붙여 문장으로 만들고, 그 문장을 모델의 tokenizer를 이용해 token 단위로 분할해 출력하도록 해보았습니다.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

sentence = "".join(train_dataset['tokens'][0])
print(f"sentence : {sentence}\n")

tokenized_sentence = tokenizer(sentence)
print(f"tokenized sentence : {tokenizer.convert_ids_to_tokens(tokenized_sentence['input_ids'])}")
```

출력 결과 klue/bert-base 모델의 tokenizer는 음절 단위가 아니었으며, KLUE NER 데이터셋도 모델의 tokenizer에 맞춰서 음절 단위로 데이터를 구성한 것이 아니라는 것을 알 수 있습니다.

```
Output:
sentence : 특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5㎞ 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.

tokenized sentence : ['[CLS]', '특히', '영동고속도로', '강릉', '방향', '문', '##막', '##휴', '##게', '##소', '##에서', '만', '##종', '##분', '##기', '##점', '##까', '##지', '5', '##㎞', '구간', '##에', '##는', '승용차', '전용', '임시', '갓', '##길', '##차', '##로', '##제', '##를', '운영', '##하기', '##로', '했', '##다', '.', '[SEP]']
```

그렇다면 우리는 KLUE NER 데이터셋을 모델의 tokenizer에 맞게 수정하는 전처리 작업을 진행해야 합니다. 즉, 음절 단위로 부착되어 있는 개체명 태그를 tokenizer가 잘라주는 token들에 맞게 부착을 해주어야 합니다. 이를 보통 토큰 정렬(Token Alignment) 작업이라고 합니다. 이 작업을 위해 저는 tokenize_and_align_labels() 라는 함수를 만들어 주었습니다. 그럼 이 함수가 어떠한 작업을 하는지 알아보도록 하겠습니다.

tokenize_and_align_labels() 함수는 모델의 tokenizer에 있는 offset 정보를 사용합니다. offset 정보란 이런 스테밍(stemming)에 원본 위치 정보를 추가해 주는 것입니다. 한글의 경우 최소 의미를 가지는 단위가 형태소이며, 형태소 중에는 불규칙 형태소가 있어 불규칙 형태소는 원본 문장의 형태를 유지하지 못하고 바뀌거나 파괴되는 문제가 있는데, 이러한 형태소 분석 결과와 원본 문장과의 일치 정보가 필요해 자주 쓰이는 기법입니다. BERT 모델의 tokenizer는 파라미터로 `return_offsets_mapping` 옵션에 True로 주면 각 토큰이 가지는 offset 정보가 결과로 출력되는데 우리는 이를 이용해 모델의 tokenizer로 분리된 token에 개체명 태그를 부착해 줄겁니다.

우선 한 음절도 구성되어 있는 데이터들을 모두 하나로 합쳐 원본 문장을 만들어주고, 모델의 tokenizer를 이용해 token으로 나누어 줍니다. 이 때 `return_offsets_mapping` 파라미터의 값을 True로 해줍니다.

그리고 tokenzied_inputs에 저장된 offset_mapping 정보와 개체명 태그 정보가 저장되어 있는 ner_tags에 있는 리스트들을 불러옵니다. 그리고 offset_mapping 정보에 있는 원본 문장에서의 시작과 끝 위치값을 가져오고, 시작값과 끝값이 같을 경우 -100값을 저장합니다. 이렇게 진행하는 이유는 BERT 모델의 tokenizer는 token으로 분리를 할 때 각 문장의 처음에 ['CLS'] token을 넣어줍니다. 그리고 BERT 모델은 태스크 중에 두 문장이 이어지는 문장인지 아닌지를 판단하는 태스크가 있으며, 이 태스크를 잘 수행하기 위해 사전 학습 시 한 문장 안에 두 개의 문장이 있다고 판단이 될 경우 문장 사이에 분리의 의미를 지니는 ['SEP'] 토큰을 넣어줍니다. 그리고 max_length 혹은 가장 긴 문장을 기준으로 모든 문장의 크기를 동일하게 맞추고, 만약 이보다 짧은 문장일 경우 남은 부분에 Padding의 의미를 지니는 ['PAD']라는 토큰을 넣어주는데 이러한 3개의 토큰을 구별하고자 -100값을 넣어줍니다.

그리고 offset_mapping의 시작값이 ner_tags가 저장된 리스트의 길이보다 작을 경우 ner_tags가 저장된 리스트의 index 값에 offset_mapping의 시작값을 넣어주고 필요한 개체명 태그를 가져옵니다. 그리고 offset_mapping의 시작값이 ner_tags가 저장된 리스트의 길이보다 클 경우에는 ['PAD']토큰이 들어가 있으므로 -100을 채워넣어 줍니다.

이렇게 만들어진 labels 리스트를 token 결과값이 있는 tokenized_inputs의 'labels'를 키값으로 하는 곳에 value로 넣어줍니다. 이렇게 하면 토큰 정렬이 완성되고 token 별로 개체명 태그가 잘 부착되는 것을 확인할 수 있습니다.

```python
def tokenize_and_align_labels(examples):
    # 1) 음절 단위 리스트를 하나의 문장으로 병합
    raw_inputs = ["".join(x) for x in examples["tokens"]]

    # 2) 재토큰화 및 Offset Mapping 반환
    tokenized_inputs = tokenizer(
        raw_inputs,
        truncation=True,
        return_offsets_mapping=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    for i, (doc_tags, offset_mapping) in enumerate(zip(examples["ner_tags"], tokenized_inputs["offset_mapping"])):
        encoded_labels = []
        
        for offset in offset_mapping:
            start, end = offset
            
            # 특수 토큰([CLS], [SEP], [PAD]) 처리
            if start == end:
                encoded_labels.append(-100)
                continue
            
            # 해당 Subword가 원본의 어느 음절에서 시작했는지 확인
            origin_char_idx = start
            
            # 원본 라벨 매핑
            if origin_char_idx < len(doc_tags):
                encoded_labels.append(doc_tags[origin_char_idx])
            else:
                encoded_labels.append(-100)
        
        labels.append(encoded_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```

```python
tokenized_datasets = dataset.map(
    tokenize_and_align_labels, 
    batched=True,
    )

print(f"tokens size : {len(tokenized_datasets['train']['input_ids'][0])}")
print(f"tokens labels size : {len(tokenized_datasets['train']['labels'][0])}")
print(f"tokens offset_mapping size : {len(tokenized_datasets['train']['offset_mapping'][0])}")

convert_tokenized = tokenizer.convert_ids_to_tokens(tokenized_datasets['train']['input_ids'][0])
labels = tokenized_datasets['train']['labels'][0]

for token, label in zip(convert_tokenized, labels):
    print(f"{token}\t{label}")
```

다음은 첫 번째 문장인 "특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5㎞ 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다." 의 token별 개체명 태그 부착 결과입니다. 그리고 모델의 token 별로 나눠진 뒤 token의 id 값을 가지고 있는 input_ids, 방금 작업을 해준 labels, 그리고 offset_mapping 리스트의 크기를 출력해보면 모두 tokenizer 객체를 선언할 때 설정해준 max_length 값인 128인 것을 확인할 수 있습니다.

```
Output:
tokens size : 128
tokens labels size : 128
tokens offset_mapping size : 128

특히	12
영동고속도로	2
강릉	2
방향	12
문	2
##막	3
##휴	3
##게	3
##소	3
##에서	12
만	2
##종	3
##분	3
##기	3
##점	3
##까	12
##지	12
5	8
##㎞	9
구간	12
##에	12
##는	12
승용차	12
전용	12
임시	12
갓	12
##길	12
##차	12
##로	12
##제	12
##를	12
운영	12
##하기	12
##로	12
했	12
##다	12
.	12
[SEP]	-100
[PAD]	-100
... 생략 ...
[PAD]	-100
```

추가적으로 우리는 이제 tokenizer의 결과값을 이용해 모델을 학습시킬 것이기 때문에 사용하지 않는 컬럼들을 제거해 주도록 하겠습니다. 우리는 기존 dataset에 있던 컬럼들이 필요가 없기 때문에 dataset["train"]에서 컬럼 이름을 가져와 제거할 컬럼값으로 넣어줍니다.

```python
tokenized_datasets = dataset.map(
    tokenize_and_align_labels, 
    batched=True,
    remove_columns=dataset["train"].column_names #제거할 컬럼들
    )
```

이렇게 모델 학습에 사용하기 위한 데이터 전처리 과정이 끝났습니다. 자연어처리를 공부하다 보면 이런 데이터 전처리 작업이 굉장히 많고 중요하다는 것을 알게됩니다. 특히나 저는 회사를 다닐 때 데이터만 다루는 동료 직원 혹은 고객사 직원으로부터 이러한 전처리 요청을 굉장히 많이 받았고, 처리해 주느라 제 업무를 보지 못한 경우도 많았을 정도로 그 만큼 자연어처리에서는 데이터 전처리가 굉장히 중요하다고 볼 수 있습니다.

## 3.2 모델 학습

그럼 이제 모델 학습에 필요한 데이터는 준비가 끝났습니다. 그러면 이제 모델 학습을 진행해 보도록 하겠습니다. 참고로 저는 RunPod라는 GPU 공유 사이트를 이용해 RTX3090 Pod를 빌려 학습을 진행했습니다. RunPod 사용 방법이 궁금하시다면 [RunPod 시작하기](https://icechickentender.github.io/mlops/engineering/runpod/runpod-1-start_runpod-post/)와 [Github와 연동하기](https://icechickentender.github.io/mlops/engineering/runpod/runpod-1-start_runpod-post/) 포스트를 참고하시기 바랍니다.

우선 모델 학습 후 우리 눈으로 확인하기 쉽게 숫자로 매핑되어 있는 개체명 태그를 텍스트와 함께 매핑 시켜주도록 하겠습니다.

```python
# 라벨 리스트 추출 (예: ['B-DT', 'I-DT', ... 'O'])
label_list = dataset["train"].features["ner_tags"].feature.names

# ID <-> Label 변환 딕셔너리 생성
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

print(id2label)
# 출력 예: {0: 'B-DT', 1: 'I-DT', ..., 12: 'O'}
```


저는 huggingface에서 지원하는 Trainer를 이용해 굉장히 간단하게 모델 학습을 진행해 보고자 합니다. 우선 가장 중요한 성능 평가 함수를 먼저 정의합니다. 개체명 인식 태스크의 성능 평가 지표로는 F1-score를 사용하며 최근에는 예전과 달리 성능 지표 라이브러리도 지원을 하기 때문에 직접 구현하지 않고 seqeval이라는 라이브러리를 사용하도록 하겠습니다. 다음과 같이 성능 평가에 사용될 compute_metrics라는 함수를 정의해 줍니다.

```python
import evaluate
import numpy as np
from transformers import DataCollatorForTokenClassification

# 1. 평가 지표 로드 (seqeval)
metric = evaluate.load("seqeval")

# 앞서 만든 id2label이 있다고 가정합니다. (0: 'B-DT', 1: 'I-DT' ...)
# label_list는 ['B-DT', 'I-DT', ..., 'O'] 순서여야 합니다.
label_list = dataset["train"].features["ner_tags"].feature.names

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # -100으로 설정된 특수 토큰 및 패딩을 제외하고 실제 예측값만 추출
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 2. 데이터 콜레이터 (배치 구성을 도와줌)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

이제 학습에 사용할 모델과 모델의 하이퍼 파라미터를 설정해 줍니다.

```python
from transformers import AutoModelForTokenClassification

# 라벨 맵핑 생성 (앞선 대화의 코드를 참고)
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

model = AutoModelForTokenClassification.from_pretrained(
    "klue/bert-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./klue-ner-result", # 결과 저장 경로
    eval_strategy="epoch",    # 에포크마다 평가 수행
    save_strategy="epoch",          # 에포크마다 모델 저장
    learning_rate=2e-5,             # 학습률 (BERT 계열은 보통 2e-5 ~ 5e-5 사용)
    per_device_train_batch_size=32, # GPU 메모리에 따라 조절 (16 or 32 추천)
    per_device_eval_batch_size=32,
    num_train_epochs=3,             # 데이터가 많으므로 3~5 에포크면 충분
    weight_decay=0.01,              # 오버피팅 방지
    load_best_model_at_end=True,    # 학습 완료 후 가장 성능 좋은 모델 로드
    metric_for_best_model="f1",     # 최적 모델 선정 기준은 F1 score
    logging_dir='./logs',           # 텐서보드 로그 저장
    logging_steps=100,
)
```

이제 huggingface에서 제공하는 학습 API인 Trainer를 이용해 학습을 진행해줍니다. Trainer 객체를 가지고 와서 Trainer 객체에 필요한 파라미터들을 설정해 주고 학습을 진행합니다.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 학습 시작!
trainer.train()

# 학습된 모델 저장 (이 경로에 저장된 모델을 나중에 불러와 씁니다)
trainer.save_model("./final_ner_model")
```

학습을 모두 진행하고 나니 F1-score 88.29라는 성능을 확인할 수 있었습니다. 그래서 혹시 모르니 KLUE 데이터에 대한 논문인 [KLUE](https://arxiv.org/pdf/2105.09680)를 한 번 살펴보았습니다. 논문에서는 제가 진행한 것과 똑같이 validation 데이터셋 기준 KLUE-BERT-BASE 모델의 경우 F1-score가 83.71으로 제가 진행한 것 보다 낮은 점수를 보이고 있었습니다. 조금 이상함을 느껴 논문의 환경과 최대한 똑같이 맞추기 위해 모델의 하이퍼 파라미터를 논문과 동일하게 맞추고, 또 논문에서는 각 에폭별 F1-score의 평균을 최종 F1-score로 사용했을 수도 있기 때문에 최대한 논문과 비슷하게 구현을 해보고자 했습니다. 또한 여태까지 진행한 코드는 상세 정보들이 출력되지 않는 것 같아 에폭별 걸린 시간등과 같은 상세 정보도 출력되도록 코드 수정을 진행하였습니다.

우선 에폭별 실행 시간을 보기 위해 다음과 같은 콜백 함수를 정의해주고, Trainer 객체에 콜백 함수를 전달해 주었습니다.

```python
from transformers import TrainerCallback

class TimeHistoryCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_duration)

... 생략 ...

# 콜백 인스턴스 생성
    time_callback = TimeHistoryCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[time_callback] # 생성한 콜백 인스턴스를 Trainer 객체에 넣어줌
    )

... 생략 ...

trainer.train()

print(f"\n[1] Epoch별 소요 시간")
    if time_callback.epoch_times:
        for i, duration in enumerate(time_callback.epoch_times):
            print(f" - Epoch {i+1}: {duration:.2f} 초")
        avg_time = sum(time_callback.epoch_times) / len(time_callback.epoch_times)
        print(f" - 평균 소요 시간: {avg_time:.2f} 초")
    else:
        print(" - 시간 데이터가 기록되지 않았습니다.")
```

논문에서 설정한 모델의 하이퍼 파라미터를 맞추기 위해 논문에서 사용한 하이퍼 파라미터를 가져와 사용했습니다. 사실 논문에서는 여러 Seed를 두고 다양한 하이퍼 파라미터로 실험을 진행했지만, 저는 하나의 Seed만 가지고 진행했습니다.

```python
SEED = 42  # 재현성을 위한 시드 고정

# 논문 설정값 적용
HYPERPARAMETERS = {
    # 1. 저장 및 로깅 설정
    "output_dir": "./klue-ner-paper-repro", # 학습된 모델 체크포인트와 로그가 저장될 경로
    "logging_steps": 100,                   # 100 step마다 Loss와 Learning Rate를 출력 (진행상황 모니터링)
    "save_strategy": "epoch",               # 매 Epoch이 끝날 때마다 모델을 저장 (중간 저장)
    "eval_strategy": "epoch",         # 매 Epoch이 끝날 때마다 Validation 셋으로 성능(F1) 평가

    # 2. 핵심 학습 파라미터 (논문 재현의 핵심)
    "num_train_epochs": 5,             
    # [논문 설정] 고정값 5. 
    # BERT Fine-tuning은 보통 3~5 epoch 내에 빠르게 수렴합니다. 
    # 너무 길게 잡으면 과적합(Overfitting) 위험이 있습니다.

    "learning_rate": 5e-5,             
    # [논문 설정] 탐색 범위 (1e-5, 3e-5, 5e-5) 중 상한값 선택.
    # NER 같은 Token Classification은 일반 분류보다 조금 높은 학습률(5e-5)에서 
    # Local Minima를 잘 탈출하는 경향이 있습니다.

    "per_device_train_batch_size": 32, 
    # [논문 설정] 탐색 범위 (16, 32).
    # 배치 사이즈 32는 Gradient 추정의 안정성과 학습 속도 간의 균형이 가장 좋습니다.
    # (GPU 메모리가 부족하면 16으로 줄여야 합니다.)

    "per_device_eval_batch_size": 32,  
    # 평가 시 배치 사이즈 (학습 성능에 영향 없음, 속도에만 영향)

    # 3. 최적화 및 규제 (Regularization) - 모델의 일반화 성능 향상
    "warmup_ratio": 0.1,               
    # [논문 설정] 전체 학습 스텝의 10%.
    # "준비 운동"과 같습니다. 학습 초반 10% 동안은 학습률을 0에서 5e-5까지 서서히 올립니다.
    # 이유: 초반부터 높은 학습률을 때려버리면, 잘 학습된 사전학습(Pre-trained) 가중치가 
    # 급격하게 망가지는 것(Catastrophic Forgetting)을 방지하기 위함입니다.

    "weight_decay": 0.01,              
    # [논문 설정] 고정값 0.01 (L2 Regularization).
    # 가중치(Weight) 값이 너무 커지는 것을 억제하는 페널티를 줍니다.
    # 모델이 특정 데이터에만 과도하게 의존하는 것을 막아 과적합을 방지합니다.

    # 4. 모델 선택 및 하드웨어 가속
    "load_best_model_at_end": True,    
    # 학습이 다 끝나면, 저장된 체크포인트 중 가장 성능이 좋았던 모델을 다시 불러옵니다.
    # (마지막 Epoch의 모델이 항상 Best라는 보장이 없기 때문입니다.)

    "metric_for_best_model": "f1",     
    # Best 모델을 선정하는 기준은 'Loss'가 아닌 'F1-score'로 설정합니다.

    "fp16": True,                      
    # [가속 옵션] 16-bit Mixed Precision 사용.
    # 최신 NVIDIA GPU(T4, 3090, A100 등)에서 학습 속도를 2배 가까이 높이고 메모리를 절약합니다.
    # 성능 저하는 거의 없으므로 무조건 켜는 것이 이득입니다.
}
```

마지막으로 Trainer 객체의 state.log_history에 남아 있는 정보를 이용해 출력하고자 했던 전체 에폭 기준 평균 F1-score와 학습 중 가장 높은 F1-score를 모두 출력하는 코드도 추가했습니다.

```python
eval_logs = [log for log in trainer.state.log_history if 'eval_f1' in log]

    if eval_logs:
        f1_scores = [log['eval_f1'] for log in eval_logs]
        epochs = [log['epoch'] for log in eval_logs]
        
        best_f1 = max(f1_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)
        best_epoch = epochs[f1_scores.index(best_f1)]

        print(f"\n[2] F1-score 분석 (총 {len(f1_scores)}회 평가)")
        print(f" - 전체 평균 F1-score : {avg_f1:.4f}")
        print(f" - 최고(Best) F1-score: {best_f1:.4f} (at Epoch {int(best_epoch)})")
        
        print("\n[상세 기록]")
        for ep, f1 in zip(epochs, f1_scores):
            print(f" - Epoch {int(ep)}: F1 {f1:.4f}")
    else:
        print("\n[!] 평가 로그를 찾을 수 없습니다.")
```

아래는 위 모든 과정을 하나의 py 파일에 합친 전체 코드입니다.

```python
import numpy as np
from datasets import load_dataset
import evaluate
import time
from transformers import(
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
    TrainerCallback
)

# 에포크별 걸린 시간을 측정하기 위한 콜백 함수 정의
class TimeHistoryCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_duration)

# =============================================================================
# 1. 환경 설정 및 하이퍼파라미터 (KLUE 논문 Appendix C 기반)
# =============================================================================
MODEL_ID = "klue/bert-base"
DATASET_ID = "klue"
TASK_NAME = "ner"
SEED = 42  # 재현성을 위한 시드 고정

# 논문 설정값 적용
HYPERPARAMETERS = {
    # 1. 저장 및 로깅 설정
    "output_dir": "./klue-ner-paper-repro", # 학습된 모델 체크포인트와 로그가 저장될 경로
    "logging_steps": 100,                   # 100 step마다 Loss와 Learning Rate를 출력 (진행상황 모니터링)
    "save_strategy": "epoch",               # 매 Epoch이 끝날 때마다 모델을 저장 (중간 저장)
    "eval_strategy": "epoch",         # 매 Epoch이 끝날 때마다 Validation 셋으로 성능(F1) 평가

    # 2. 핵심 학습 파라미터 (논문 재현의 핵심)
    "num_train_epochs": 5,             
    # [논문 설정] 고정값 5. 
    # BERT Fine-tuning은 보통 3~5 epoch 내에 빠르게 수렴합니다. 
    # 너무 길게 잡으면 과적합(Overfitting) 위험이 있습니다.

    "learning_rate": 5e-5,             
    # [논문 설정] 탐색 범위 (1e-5, 3e-5, 5e-5) 중 상한값 선택.
    # NER 같은 Token Classification은 일반 분류보다 조금 높은 학습률(5e-5)에서 
    # Local Minima를 잘 탈출하는 경향이 있습니다.

    "per_device_train_batch_size": 32, 
    # [논문 설정] 탐색 범위 (16, 32).
    # 배치 사이즈 32는 Gradient 추정의 안정성과 학습 속도 간의 균형이 가장 좋습니다.
    # (GPU 메모리가 부족하면 16으로 줄여야 합니다.)

    "per_device_eval_batch_size": 32,  
    # 평가 시 배치 사이즈 (학습 성능에 영향 없음, 속도에만 영향)

    # 3. 최적화 및 규제 (Regularization) - 모델의 일반화 성능 향상
    "warmup_ratio": 0.1,               
    # [논문 설정] 전체 학습 스텝의 10%.
    # "준비 운동"과 같습니다. 학습 초반 10% 동안은 학습률을 0에서 5e-5까지 서서히 올립니다.
    # 이유: 초반부터 높은 학습률을 때려버리면, 잘 학습된 사전학습(Pre-trained) 가중치가 
    # 급격하게 망가지는 것(Catastrophic Forgetting)을 방지하기 위함입니다.

    "weight_decay": 0.01,              
    # [논문 설정] 고정값 0.01 (L2 Regularization).
    # 가중치(Weight) 값이 너무 커지는 것을 억제하는 페널티를 줍니다.
    # 모델이 특정 데이터에만 과도하게 의존하는 것을 막아 과적합을 방지합니다.

    # 4. 모델 선택 및 하드웨어 가속
    "load_best_model_at_end": True,    
    # 학습이 다 끝나면, 저장된 체크포인트 중 가장 성능이 좋았던 모델을 다시 불러옵니다.
    # (마지막 Epoch의 모델이 항상 Best라는 보장이 없기 때문입니다.)

    "metric_for_best_model": "f1",     
    # Best 모델을 선정하는 기준은 'Loss'가 아닌 'F1-score'로 설정합니다.

    "fp16": True,                      
    # [가속 옵션] 16-bit Mixed Precision 사용.
    # 최신 NVIDIA GPU(T4, 3090, A100 등)에서 학습 속도를 2배 가까이 높이고 메모리를 절약합니다.
    # 성능 저하는 거의 없으므로 무조건 켜는 것이 이득입니다.
}

# 시드 고정
set_seed(SEED)

def main():
    print(f">>> 데이터셋 로드 및 전처리 시작 (Model: {MODEL_ID})")

    # =========================================================================
    # 2. 데이터셋 및 토크나이저 로드
    # =========================================================================
    dataset = load_dataset(DATASET_ID, TASK_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 라벨 리스트 및 ID 맵핑 생성
    label_list = dataset["train"].features["ner_tags"].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    # =========================================================================
    # 3. 데이터 전처리 (음절 -> 문장 -> Subword 재정렬 & Offset Mapping)
    # =========================================================================
    def tokenize_and_align_labels(examples):
        # 1) 음절 단위 리스트를 하나의 문장으로 병합
        raw_inputs = ["".join(x) for x in examples["tokens"]]

        # 2) 재토큰화 및 Offset Mapping 반환
        tokenized_inputs = tokenizer(
            raw_inputs,
            truncation=True,
            return_offsets_mapping=True,
            padding="max_length",
            max_length=128
        )

        labels = []
        for i, (doc_tags, offset_mapping) in enumerate(zip(examples["ner_tags"], tokenized_inputs["offset_mapping"])):
            encoded_labels = []
            
            for offset in offset_mapping:
                start, end = offset
                
                # 특수 토큰([CLS], [SEP], [PAD]) 처리
                if start == end:
                    encoded_labels.append(-100)
                    continue
                
                # 해당 Subword가 원본의 어느 음절에서 시작했는지 확인
                origin_char_idx = start
                
                # 원본 라벨 매핑
                if origin_char_idx < len(doc_tags):
                    encoded_labels.append(doc_tags[origin_char_idx])
                else:
                    encoded_labels.append(-100)
            
            labels.append(encoded_labels)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # 전처리 적용 (불필요한 컬럼 제거)
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )

    # =========================================================================
    # 4. 평가 함수 정의 (seqeval)
    # =========================================================================
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # -100 (특수 토큰) 제외하고 실제 라벨만 복원
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # =========================================================================
    # 5. 모델 초기화 및 학습 설정
    # =========================================================================
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(**HYPERPARAMETERS)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 콜백 인스턴스 생성
    time_callback = TimeHistoryCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[time_callback]
    )

    # =========================================================================
    # 6. 학습 실행 및 저장
    # =========================================================================
    print(">>> 학습 시작...")
    trainer.train()

    # 7. 결과 분석 및 보고서 출력
    print("\n" + "="*50)
    print(" >>> [상세 분석 보고서] <<<")
    print("="*50)

    # (1) 에포크별 시간 출력
    print(f"\n[1] Epoch별 소요 시간")
    if time_callback.epoch_times:
        for i, duration in enumerate(time_callback.epoch_times):
            print(f" - Epoch {i+1}: {duration:.2f} 초")
        avg_time = sum(time_callback.epoch_times) / len(time_callback.epoch_times)
        print(f" - 평균 소요 시간: {avg_time:.2f} 초")
    else:
        print(" - 시간 데이터가 기록되지 않았습니다.")

    
    # (2) F1-score 분석 (Log History 파싱)
    eval_logs = [log for log in trainer.state.log_history if 'eval_f1' in log]

    if eval_logs:
        f1_scores = [log['eval_f1'] for log in eval_logs]
        epochs = [log['epoch'] for log in eval_logs]
        
        best_f1 = max(f1_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)
        best_epoch = epochs[f1_scores.index(best_f1)]

        print(f"\n[2] F1-score 분석 (총 {len(f1_scores)}회 평가)")
        print(f" - 전체 평균 F1-score : {avg_f1:.4f}")
        print(f" - 최고(Best) F1-score: {best_f1:.4f} (at Epoch {int(best_epoch)})")
        
        print("\n[상세 기록]")
        for ep, f1 in zip(epochs, f1_scores):
            print(f" - Epoch {int(ep)}: F1 {f1:.4f}")
    else:
        print("\n[!] 평가 로그를 찾을 수 없습니다.")

    # 모델 저장
    print("\n>>> Best 모델 저장 중...")
    trainer.save_model("./final_ner_model_paper_ver")

if __name__ == "__main__":
    main()
```

이렇게 수정한 후 학습을 진행한 결과 다음과 같은 결과를 얻을 수 있었습니다. Best F1-score는 88.97으로 이전 88.29보다 대략 0.7 정도 높은 성능을 보이는 것을 확인할 수 있었습니다. 평균 F1-score 또한 88.22로 논문에 나와있는 83.71보다 높은 성능을 보이는 것을 확인할 수 있었습니다. 아마도 논문보다 높은 성능을 보이는 이유는 논문에서 실험한 환경과 완전히 동일하지 않기 때문입니다. 일단 논문에서 사용한 평가 지표는 똑같은 F1-score긴 하지만 논문에서 좀 더 정확한 성능 평가를 위해 F1-score 계산 로직을 직접 구현하였거나 몇가지 추가된 사항이 있을 수 있습니다. 또한 논문에서는 다양한 하이퍼 파라미터 값으로 학습 시키기 위해 여러 Seed를 사용해 굉장히 많이 학습을 진행하고 평가를 진행하였지만 저는 단지 하나의 Seed만 정해서 학습하고 평가를 진행하는 등 모든 실험 환경을 똑같이 재현하지 않았기 때문에 이렇게 성능 차이가 나는 듯 합니다. 아니면 시간이 지난 후 klue/bert-base 모델의 사전 학습 데이터가 추가되어 논문에 사용된 사전 학습 모델과 다른 모델일 수도 있는 등 여러가지 원인이 있을 수 있습니다.

```
Output:
==================================================
 >>> [상세 분석 보고서] <<<
==================================================

[1] Epoch별 소요 시간
 - Epoch 1: 55.03 초
 - Epoch 2: 54.47 초
 - Epoch 3: 54.49 초
 - Epoch 4: 54.45 초
 - Epoch 5: 54.55 초
 - 평균 소요 시간: 54.60 초

[2] F1-score 분석 (총 5회 평가)
 - 전체 평균 F1-score : 0.8822
 - 최고(Best) F1-score: 0.8897 (at Epoch 3)

[상세 기록]
 - Epoch 1: F1 0.8653
 - Epoch 2: F1 0.8853
 - Epoch 3: F1 0.8897
 - Epoch 4: F1 0.8852
 - Epoch 5: F1 0.8855
```

## 3.3 모델 평가

이제 모델 학습을 진행했으니 실제로 학습한 모델과 학습하지 않은 모델에서 얼마 정도 성능 차이가 발생하는지 보도록 하겠습니다. 평가에는 아래 코드를 이용했습니다. 학습시 사용했던 토큰 정렬 함수와 평가 함수 그리고 Trainer 객체를 이용해 평가를 진행했습니다. 우선은 미세 조정 학습을 하지 않은 klue/bert-base 모델을 이용해 평가를 진행해 보도록 하겠습니다. `model_id` 값에 학습한 모델의 경로가 아닌 huggingface에 학습하지 않은 모델이 있는 model_id 값을 넣어주고 실행해 봅니다.

```python
import torch
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset

# 1. 저장된 모델과 토크나이저 불러오기 (경로 주의)
model_id = "klue/bert-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

# 2. 데이터셋 및 평가 지표 준비 (학습 때와 동일해야 함)
dataset = load_dataset("klue", "ner")
metric = evaluate.load("seqeval")
label_list = dataset["train"].features["ner_tags"].feature.names

# 3. 전처리 함수 (학습 때 사용한 함수와 100% 동일해야 함)
# (앞서 작성했던 offset_mapping을 사용하는 함수를 그대로 가져옵니다)
def tokenize_and_align_labels(examples):
    raw_inputs = ["".join(x) for x in examples["tokens"]]
    tokenized_inputs = tokenizer(
        raw_inputs,
        truncation=True,
        return_offsets_mapping=True,
        padding="max_length",
        max_length=128
    )
    
    labels = []
    for i, (doc_tags, offset_mapping) in enumerate(zip(examples["ner_tags"], tokenized_inputs["offset_mapping"])):
        encoded_labels = []
        for offset in offset_mapping:
            start, end = offset
            if start == end: 
                encoded_labels.append(-100)
                continue
            origin_char_idx = start
            if origin_char_idx < len(doc_tags):
                encoded_labels.append(doc_tags[origin_char_idx])
            else:
                encoded_labels.append(-100)
        labels.append(encoded_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Validation 데이터셋 전처리
eval_dataset = dataset["validation"].map(tokenize_and_align_labels, batched=True)

# 4. 평가 함수 정의
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 5. Trainer를 이용해 평가 수행
trainer = Trainer(
    model=model,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

print(">>> 검증 시작...")
metrics = trainer.evaluate()

print(f"\n>>> [검증 결과] F1-score: {metrics['eval_f1']:.4f}")
print(f">>> (학습 로그의 Best F1과 비교해 보세요)")
```

다음과 같은 경고문과 에러가 뜨면서 평가 자체가 되질 않는 것을 볼 수 있습니다. 아래 출력과 같이 에러가 발생하는 이유는 학습하지 않은 klue/bert-base 모델의 num_labels 인자를 따로 설정하지 않으면, 기본적으로 2개로 분류기 층이 만들어집니다. 하지만 KLUE NER 데이터셋의 라벨은 총 13개입니다. 모델은 출력으로 2개짜리 벡터를 내보내는데, 정답이 2개를 벗어난 값이 들어오면 Loss 계산 시 인덱스 범위를 벗어나게 되어 `CUDA device-side assert`에러가 발생합니다.

```
Output:
Some weights of BertForTokenClassification were not initialized from the model checkpoint at klue/bert-base and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
^[[A^[[B/workspace/bert_eval.py:74: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
>>> 검증 시작...
Traceback (most recent call last):
  File "/workspace/bert_eval.py", line 83, in <module>
    metrics = trainer.evaluate()
              ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 4489, in evaluate
    output = eval_loop(
             ^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 4685, in evaluation_loop
    losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 4905, in prediction_step
    loss = loss.detach().mean()
           ^^^^^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

```

그렇다면 model_id 값에 우리가 학습했던 모델이 저장되어 있는 경로 `./final_ner_model_paper_ver`로 해준 뒤 실행해 보도록 하겠습니다. 실행 결과 아래와 같이 F1-score가 우리가 학습했을 때 얻었던 성능인 88.97을 볼 수 있었습니다.

```
Output:
>>> [검증 결과] F1-score: 0.8897
```

## 3.4 학습한 모델을 이용한 추론

그럼 이제 학습한 모델을 실제 서비스에 이용하기 위해서 모델을 이용해 출력되는 추론 결과를 이용해 실제 데이터 반영해 사람들로부터 모델이 작동하고 있구나 하는 것을 보여주어야 합니다. 아래 코드는 학습한 모델에 일반적인 문장을 입력으로 넣어 모델에서 해당 문장에 대한 개체명 태그 결과값을 받고, 문장에서 개체명으로 인식된 부분에 개체명 태그를 달아주는 코드입니다. 우선은 학습하지 않은 모델인 klue/bert-base 모델을 이용해 추론을 진행해 보도록 하겠습니다.

```python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
#MODEL_PATH = "./final_ner_model"  # 학습 완료된 모델 경로

# 토크나이저와 모델 불러오기
model_id = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

# 라벨 맵핑 (모델 config에서 자동으로 가져옵니다)
id2label = model.config.id2label

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==========================================
# 2. 핵심 함수: 문장을 입력받아 태깅된 문장 반환
# ==========================================
def predict_ner(text):
    # 2-1. 입력 텍스트 토크나이징
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        return_offsets_mapping=True, 
        truncation=True, 
        max_length=128
    )
    
    # offset_mapping은 추론에 불필요하므로 별도로 저장 후 제거
    offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
    
    # 데이터를 GPU로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 2-2. 모델 추론
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 2-3. 예측 결과 변환 (Logits -> Tag IDs -> Tag Names)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    pred_tag_ids = predictions[0].cpu().numpy()
    
    # 특수 토큰([CLS], [SEP])을 제외하고 실제 토큰만 필터링
    # offset_mapping을 이용해 실제 글자가 있는 구간만 추출
    valid_tokens = []
    for idx, (offset, tag_id) in enumerate(zip(offset_mapping, pred_tag_ids)):
        start, end = offset
        # [CLS], [SEP] 등은 offset이 (0, 0)
        if start != end:
            label = id2label[tag_id]
            valid_tokens.append((start, end, label))
            
    return decode_ner_tags(text, valid_tokens)

# ==========================================
# 3. 포맷팅 함수: 예측된 태그를 원문에 삽입
# ==========================================
def decode_ner_tags(original_text, valid_tokens):
    """
    valid_tokens: [(start_idx, end_idx, label), ...]
    BIO 태그를 파싱하여 <단어:태그> 형태로 변환
    """
    result_text = ""
    last_idx = 0
    
    # 개체명 묶기 (Chunking) 로직
    current_entity = None # {'start': 0, 'end': 0, 'label': 'QT'}
    
    entities = []
    
    for start, end, label in valid_tokens:
        # BIO 태그 분석
        if label.startswith("B-"):
            # 이전 개체명이 있었다면 저장
            if current_entity:
                entities.append(current_entity)
            
            # 새로운 개체명 시작
            entity_type = label.split("-")[1]
            current_entity = {
                "start": start,
                "end": end,
                "label": entity_type
            }
            
        elif label.startswith("I-"):
            # 현재 진행 중인 개체명이 있고, 타입이 같다면 범위 확장
            if current_entity and label.split("-")[1] == current_entity['label']:
                current_entity['end'] = end
            else:
                # 문법적으로 맞지 않는 I 태그가 나오면(B 없이 I 등), 
                # 이전 개체명 닫고 새로 시작하거나 무시 (여기서는 새로 시작으로 처리)
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label.split("-")[1]
                current_entity = {
                    "start": start,
                    "end": end,
                    "label": entity_type
                }
                
        else: # 'O' 태그
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    # 마지막에 남은 개체명 처리
    if current_entity:
        entities.append(current_entity)

    # 4. 원본 텍스트에 태그 삽입 (뒤에서부터 작업하면 인덱스가 꼬이지 않음 -> 여기선 순차적으로 작성)
    # 순차적으로 문자열 조립
    processed_idx = 0
    final_output = ""
    
    for entity in entities:
        start = entity['start']
        end = entity['end']
        label = entity['label']
        
        # 개체명 앞부분 붙이기
        final_output += original_text[processed_idx:start]
        
        # 개체명 부분 포맷팅
        entity_text = original_text[start:end]
        final_output += f"<{entity_text}:{label}>"
        
        processed_idx = end
        
    # 남은 뒷부분 붙이기
    final_output += original_text[processed_idx:]
    
    return final_output

# ==========================================
# 4. 실행 테스트
# ==========================================
if __name__ == "__main__":
    test_sentences = [
        "특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5㎞ 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.",
        "어제 서울 날씨는 맑았지만, 부산은 비가 왔다.",
        "이순신 장군은 조선 시대의 명장이다."
    ]
    
    print("-" * 50)
    for text in test_sentences:
        result = predict_ner(text)
        print(f"입력: {text}")
        print(f"출력: {result}")
        print("-" * 50)
```

모델이 학습되지 않았다는 문구인 "Some weights of BertForTokenClassification were not initialized from the model checkpoint at klue/bert-base and are newly initialized: ['classifier.bias', 'classifier.weight'] You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference." 문구가 뜨면서 출력값에는 입력값과 동일한 문장이 출력되는 것을 확인할 수 있습니다.

```
Output:
Some weights of BertForTokenClassification were not initialized from the model checkpoint at klue/bert-base and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
--------------------------------------------------
입력: 특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5㎞ 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.
출력: 특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5㎞ 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.
--------------------------------------------------
입력: 어제 서울 날씨는 맑았지만, 부산은 비가 왔다.
출력: 어제 서울 날씨는 맑았지만, 부산은 비가 왔다.
--------------------------------------------------
입력: 이순신 장군은 조선 시대의 명장이다.
출력: 이순신 장군은 조선 시대의 명장이다.
--------------------------------------------------
```

그렇다면 이제 우리가 학습한 모델로 변경한 후에 다시 실행해 보도록 하겠습니다. model_id 값에 `./final_ner_model_paper_ver`을 넣어준 뒤 실행해 보면 출력값을 보면 개체명이 등장한 곳에는 개체명 태그가 부착되어 있는 것을 확인할 수 있습니다.

```
Output:
--------------------------------------------------
입력: 특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5㎞ 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.
출력: 특히 <영동고속도로:LC> <강릉:LC> 방향 <문막휴게소:LC>에서 <만종분기점:LC>까지 <5㎞:QT> 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.
--------------------------------------------------
입력: 어제 서울 날씨는 맑았지만, 부산은 비가 왔다.
출력: <어제:DT> <서울:LC> 날씨는 맑았지만, <부산:LC>은 비가 왔다.
--------------------------------------------------
입력: 이순신 장군은 조선 시대의 명장이다.
출력: <이순신:PS> 장군은 <조선 시대:DT>의 명장이다.
--------------------------------------------------
```

## 3.5 허깅페이스 허브에 모델 저장

이제 마지막으로 학습한 모델을 자신의 허깅페이스 허브에 업로드 해보도록 하겠습니다. `hf_token`에는 본인의 허깅페이스 토큰 값을 넣어주어야 하고, api.upload_folder의 repo_id 값에도 본인의 허깅페이스 사용자 이름 값을 넣어주어야 합니다. 아래 코드를 무작정 실행만 해서는 안됩니다. 자신의 허깅페이스 계정과 관련된 값들을 넣은 후에 허깅페이스를 보면 아래 이미지와 같이 업로드가 잘 된 것을 볼 수 있습니다. 이렇게 업로드를 시켜야 하는 이유는 집에서 자신의 PC로 직접 학습하는 경우에는 모델이 저장장치에 저장이 되어 있지만 RunPod와 같이 환경을 빌려서 진행하는 경우 Pod를 삭제하면 진행했던 결과물들이 모두 사라지기 때문이며, 또한 이렇게 허깅페이스에 자신이 학습 시킨 모델을 업로드하면 추후에 회사에 지원을 할 때에도 직접 학습을 해본 경험이 있다는 것을 간접적으로 어필할 수 있기 때문에 왠만하면 허깅페이스에 업로드 하는 것을 추천드립니다.

```python
from huggingface_hub import login
from huggingface_hub import HfApi

login(token='hf_token')

api = HfApi()
repo_id = "klue-bert-base-klue-ner-finetuned"
api.create_repo(repo_id=repo_id)

api.upload_folder(
    folder_path="./final_ner_model_paper_ver",
    repo_id=f"본인의 허깅페이스 사용자 이름/{repo_id}",
    repo_type="model",
)
```

<div align="center">
  <img src="/assets/images/toy_project/comparison_of_entity_recognition_performance_by_model/1/huggingface_model_upload.png" width="65%" height="40%"/>
</div>

# 마치며

오랫동안 미뤄오던 토이 프로젝트의 첫 발을 떼었습니다. 처음엔 시작을 하면서 과연 잘 될까 하면서 걱정을 많이 했었지만 그래도 막상 시작을 해보니 어떻게든 마무리를 짓긴 한 거 같습니다. 예전 대학원생 때 만들었던 Bi-LSTM-CRF 모델을 이용한 개체명 인식 모델은 이번 포스트를 작성할 때와는 다르게 모델도 직접 구현을 해주어야 했고, 또 대부분의 모델은 영어를 기반으로 나온 모델이라 한국어에 적용시키기 위해선 많은 부분을 손을 봐야 했었습니다. 하지만 BERT 모델은 huggingface에서 많은 지원을 해주어 Trainer와 AutoModel 클래스 덕분에 굉장히 쉽게 이번 과정을 진행할 수 있었던 것 같습니다. 이제 BERT 모델을 이용한 개체명 인식을 만들어 보았으니 다음번엔 동일한 데이터에 Bi-LSTM-CRF 모델을 사용해 어느 정도의 성능이 나오는지 한 번 알아보도록 하겠습니다. 내용이 많이 난잡하긴 하지만 그래도 자신의 토이 프로젝트를 어떻게 시작해야 하나 하는 분들에게 도움이 되었으면 좋겠습니다.

긴 글 읽어주셔서 감사드리며, 본문 내용 중에 잘못된 내용이나 오타, 궁금하신 사항이 있으신 경우 댓글 달아주시기 바랍니다.

# 참조

- <https://huggingface.co/datasets/klue/klue>
- <https://huggingface.co/klue/bert-base>
- <https://arxiv.org/pdf/2105.09680>