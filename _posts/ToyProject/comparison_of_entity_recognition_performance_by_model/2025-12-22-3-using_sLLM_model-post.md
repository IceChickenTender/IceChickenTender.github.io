---
title: "[ToyProject] 모델별 개체명 인식 성능 비교 - 3. sLLM을 이용한 개체명 인식기 구축과 성능 평가"
categories:
  - ToyProject
tags:
  - ToyProject

use_math: true
toc: true
toc_sticky: true
toc_label: "sLLM을 이용한 개체명 인식기 구축과 성능 평가"
---

# 머리말

이전에 진행 했던 BERT와 Bi-LSTM-CRF 모델을 이용한 개체명 인식기 구현에 이어서 sLLM을 이용한 개체명 인식기 구현을 진행했고, 그간의 실험 과정과 시행착오를 기록으로 남기고자 합니다.

코드 실행을 위해서는 우선 아래의 라이브러리들을 먼저 설치해 주어야 합니다.

```bash
pip install transformers peft bitsandbytes trl datasets accelerate huggingface-hub tensorboardX seqeval
```

참고로 저는 sLLM 모델 학습을 위해 RunPod에서 RTX 4090 GPU를 사용할 수 있는 Pod를 빌려서 학습을 진행했으며 학습에 소요된 시간은 대략적으로 1시간 30분 정도 소요되었습니다. 그리고 학습된 모델 평가에는 RTX 3090 GPU를 사용할 수 있는 Pod를 빌려서 평가를 진행했습니다.

# 1. 데이터 전처리

데이터는 이전에 진행했던 BERT와 Bi-LSTM-CRF 모델을 이용한 개체명 인식 모델 구현 때와 동일하게 KLUE NER 데이터를 사용했습니다.

데이터 전처리는 sLLM의 경우 BERT 모델이나 Bi-LSTM-CRF와 같이 시퀀스 데이터를 이용해 학습하는 것이 아니라 프롬프트로 입력 문장과 그 입력 문장에 등장하는 개체명들이 무엇인지를 sLLM에게 알려주는 프롬프트 구조를 따르기 때문에, 학습과 평가를 위해 총 두 단계의 전처리 과정이 필요합니다.

## 1.1 학습 데이터를 위한 데이터 전처리

첫 번째 데이터 전처리는 sLLM 학습 시 사용될 데이터를 만들기 위해 KLUE NER 데이터를 이용해 전처리를 진행하는 것으로 BIO로 태깅되어 있는 시퀀스 데이터를 이용해 원본 문장과 해당 문장에서 각 태그별로 어떤 개체명이 추출이 되는지를 알려주는 형식의 데이터를 구성해야 합니다. 다음은 최종적으로 만들어지는 학습 데이터의 형식입니다.

LLM 모델은 BERT나 Bi-LSTM-CRF와 같은 태스크 특화 모델이 아닌 하나의 모델로 모든 일을 하는 모델로써 LLM 모델을 태스크 특화 모델로 동작하도록 하기 위해선 아래 데이터와 같이 프롬프트로 특정 태스크를 하도록 하는 데이터를 통해 학습을 시켜서 진행하게 됩니다.

```json
{
  "messages": [
    {"role": "system", "content": "당신은 유능한 개체명 인식기입니다. 입력된 문장에서 개체명을 찾아 JSON 형식으로 출력하세요."},
    {"role": "user", "content": "손흥민은 춘천에서 태어났다."},
    {"role": "assistant", "content": "{'PS': ['손흥민'], 'LC': ['춘천']}"}
  ]
}
```

그렇다면 KLUE NER 데이터셋을 이용해 LLM 학습에 사용하기 위한 프롬프트 형태의 json 데이터를 만드는 코드를 한 번 알아보도록 하겠습니다.

우선 BIO 형태의 데이터를 프롬프트 형태로 바꾸는 함수부터 알아보도록 하겠습니다. 저는 bio_to_json 이라는 이름으로 함수를 정의하였습니다. 이 함수는 개체명 태그에서 'B-'로 시작할 때 개체명 태그의 시작으로 인식하여 개체명이 없는 태그인 'O' 태그 혹은 그 다음 개체명 태그인 'B-' 태그가 올 때까지 진행을 하면서 tokens에 있는 음절 정보를 연결하여 하나의 개체명으로 만들고 entity_dict 라는 딕셔너리에 "개체명 태그: 개체명 리스트" 형태로 저장을 합니다. 루프를 통해 한 문장에 해당하는 데이터의 처음부터 끝까지 모두 보고나면 tokens를 모두 합친 full_text와 한 문장에서 등장한 개체명태그: 개체명리스트 딕셔너리를 String 형태로 변환한 결과를 반환합니다.

```python
def bio_to_json(tokens, tags, id2label):
    """
    BIO 태그 리스트를  파싱하여 개체명 딕셔너리(JSON)로 변환하는 핵심 함수
    예: tokens=['손','흥','민','은'], tags=[B-PS, I-PS, I-PS, O] -> {'PS': ['손흥민']}
    """
    entity_dict = {}
    current_entity_text = []
    current_entity_label = None

    for token, tag_id in zip(tokens, tags):
        tag_label = id2label[tag_id]

        # 1. 'B-' 태그를 만났을 때
        if tag_label.startswith("B-"):
            # 이전 개체명이 있었다면 저장
            if current_entity_label:
                if current_entity_label not in entity_dict:
                    entity_dict[current_entity_label] = []
                entity_dict[current_entity_label].append("".join(current_entity_text))

            # 새로운 개체명 시작
            current_entity_label = tag_label.split("-")[1] # "B-PS" -> "PS"
            current_entity_text = [token]
        # 2. 'I-'(Inside) 태그를 만났을 때 (현재 라벨과 같아야 함)
        elif tag_label.startswith("I-") and current_entity_label:
            if tag_label.split("-")[1] == current_entity_label:
                current_entity_text.append(token)
            else:
                # 태그가 꼬인 경우 (B없이 I가 나오거나 다른 I가 나온 경우) - 끊고 새로 시작
                # 여기서는 안전하게 이전 것을 저장하고 초기화
                if current_entity_label not in entity_dict:
                    entity_dict[current_entity_label] = []
                entity_dict[current_entity_label].append("".join(current_entity_text))
                current_entity_label = None
                current_entity_text = []
        # 3. 'O'(Outside) 태그를 만났을 때
        else:
            if current_entity_label:
                if current_entity_label not in entity_dict:
                    entity_dict[current_entity_label] = []
                entity_dict[current_entity_label].append("".join(current_entity_text))
                current_entity_label = None
                current_entity_text = []

    # 마지막에 남은 개체명 처리
    if current_entity_label:
        if current_entity_label not in entity_dict:
            entity_dict[current_entity_label] = []
        entity_dict[current_entity_label].append("".join(current_entity_text))

    # 복원된 문장 (KLUE는 tokens 공백 없이 붙이는게 원본에 가까움 필요 시 " ".join 등 조정)
    full_text = "".join(tokens).replace(" ", " ")

    return full_text, json.dumps(entity_dict, ensure_ascii=False)
```

다음으로는 bio_to_json 함수를 이용해 학습에 사용될 프롬프트를 만드는 함수를 정의해 보도록 하겠습니다. 각 문장마다 하나의 entry를 만들어 system 항목에는 미리 정의한 system_prompt를 user 항목에는 입력 문장을 assistant에는 bio_to_json 함수를 이용해 뽑아낸 개체명 태그와 개체명을 넣어줍니다. 그리고 formatted_data라는 리스트에 entry를 저장하고 formatted_data 리스트를 반환합니다.

```python
def create_chat_dataset(dataset_split, id2label):
    formatted_data = []

    # 시스템 프롬프트: 모델에게 역할을 부여하고 출력 형식을 강제함
    system_prompt = (
        "당신은 유능한 개체명 인식기(NER)입니다. "
        "입력된 문장에서 개체명을 추출하여 JSON 형식으로 출력하세요. "
        "개체명 태그는 다음과 같습니다: PS(사람), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량). "
        "해당하는 개체명이 없으면 빈 JSON '{}'을 출력하세요."
    )

    for item in tqdm(dataset_split, desc="Processing"):
        tokens = item['tokens']
        ner_tags = item['ner_tags']

        # BIO 태그 -> 텍스트 문장 & JSON 정답 변환
        input_text, output_json = bio_to_json(tokens, ner_tags, id2label)

        # LLM 학습용 Chat Format (OpenAI/HuggingFace 표준)
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_json}
            ]
        }
        formatted_data.append(entry)

    return formatted_data
```

마지막으로 이 과정을 다른 파이썬 파일에서 쉽게 사용할 수 있게 이 과정을 한 번에 수행하도록 하는 함수를 정의합니다. 저는 preprocessing_data 라는 이름의 함수로 정의를 했습니다. 학습 데이터는 jsonl 파일로 생성되며, 평가에 사용될 validation 데이터셋은 프롬프트 형식이 아닌 다른 형식으로 평가에 사용되기 때문에 학습 데이터만 프롬프트 형태로 변환하도록 하였습니다.

```python
def preprocessing_data():
    # 1. 데이터셋 로드
    print(">>> KLUE NER 데이터셋 로드")
    dataset = load_dataset("klue", "ner")

    # 2. 라벨 매핑 정보 생성
    label_list = dataset["train"].features["ner_tags"].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}

    # 3. 변환 수행
    train_chat_data = create_chat_dataset(dataset['train'], id2label)

    # 4. JSONL 파일로 저장
    os.makedirs("../data", exist_ok=True)

    with open("../data/train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_chat_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    # 샘플 데이터 1개 확인
    print(json.dumps(train_chat_data[0], indent=4, ensure_ascii=False))
```

아래는 위에서 정의한 함수들을 이용해 얻은 학습에 사용될 데이터입니다.

```json
{
    "messages": [

        {

            "role": "system",

            "content": "당신은 유능한 개체명 인식기(NER)입니다. 입력된 문장에서 개체명을 추출하여 JSON 형식으로 출력하세요.

 개체명 태그는 다음과 같습니다: PS(사람), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량). 해당하는 개체명이 없으면 빈 JSON '{}'을 출력하세요."

        },

        {

            "role": "user",

            "content": "특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5㎞ 구간에는 승용차 전용 임시 갓길차로제 를 운영하기로 했다."

        },

        {

            "role": "assistant",

            "content": "{\"LC\": [\"영동고속도로\", \"강릉\", \"문막휴게소\", \"만종분기점\"], \"QT\": [\"5㎞\"]}"

        }
    ]
}
```

## 1.2 모델 평가를 위한 데이터 전처리

두 번째로 평가를 진행할 때 사용하는 validation 데이터셋에 대한 전처리입니다. 이전에 진행했던 BERT나 Bi-LSTM-CRF 모델과 달리 sLLM 모델은 입력을 프롬프트로 주며 출력도 우리가 정의한 "개체명태그: 개체명리스트"의 형태로 주게 됩니다. 나중에 모델 평가 항목에서 설명을 하겠지만 이전에 진행했던 모델들과 같이 tokenizer 혹은 KLUE NER 데이터셋과 같은 형태인 각 음절에 개체명 태그를 붙인 형태로 결과를 뱉어내게 하면 되지 않냐고 생각하실 수 있으시겠지만 LLM 모델은 모델 내부에 있는 tokenizer에만 기반해 개체명을 추출하고 있으며 그 외의 음절이나 어절 혹은 다른 tokenizer의 token 별 인덱싱은 하지 못하는 치명적인 단점을 가지고 있습니다. 따라서 우리는 sLLM을 학습 시킬 때 사용한 프롬프트의 출력 형태로 데이터를 받게 될 것이고, 이 형태를 기반해서 다시 음절 단위의 BIO 형태로 바꾸어 주는 함수를 정의하여야 이전 모델과 같이 F1-score를 이용해 평가를 진행할 수 있습니다. 그래서 저는 json_to_bio 라는 함수를 정의하여 모델이 내뱉은 json에서 각 음절 별로 개체명 태그를 붙이도록 하여 이를 이용해 F1-score 평가를 진행하였습니다.

```python
def json_to_bio(text, json_str, tokens, id2label):
    """모델이 생성한 JSON을 파싱하여 원본 토큰에 맞는 BIO 태그 리스트 생성"""
    # 1. 초기화 (모두 'O'로 시작)
    predicted_tags = ['O'] * len(tokens)
    
    try:
        # 모델 출력에서 JSON 부분만 추출 (가끔 잡다한 말을 붙일 수 있음)
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if match:
            json_str = match.group()
        
        pred_dict = json.loads(json_str)
    except:
        # 파싱 실패 시 모두 'O' 반환 (형식 불일치 페널티)
        return predicted_tags

    # 2. 텍스트 매칭 및 태그 할당
    # 토큰을 다시 하나의 문자열로 합쳐서 위치를 찾음 (KLUE 토큰 특성 고려)
    # 주의: 토크나이저 방식에 따라 매핑이 복잡할 수 있으나, 여기선 단순 매칭 시도
    
    # 원본 문장 재구성 (토큰 offset 매핑을 위해 필요하지만, 약식으로 진행)
    # 실제로는 char_to_token 매핑이 필요함. 여기서는 Text 매칭 방식으로 근사.
    
    token_str_list = tokens # ['손', '흥', '민', '은']
    
    for label, entity_list in pred_dict.items():
        if not isinstance(entity_list, list): continue
        
        for entity_text in entity_list:
            # entity_text가 토큰 리스트 안에서 어디에 있는지 찾기 (Sliding Window)
            # 예: "손흥민" -> tokens에서 ['손', '흥', '민'] 연속 구간 찾기
            
            # 토큰들을 합친 임시 문자열 생성
            temp_tokens = "".join(tokens) 
            # entity_text의 시작 위치 찾기
            start_idx = temp_tokens.find(entity_text.replace(" ", "")) 
            
            if start_idx == -1: continue # 못 찾으면 패스
            
            # Char Index를 Token Index로 변환 (간략화된 로직)
            current_len = 0
            token_start = -1
            token_end = -1
            
            for i, token in enumerate(tokens):
                token_len = len(token)
                if current_len == start_idx:
                    token_start = i
                if current_len == start_idx + len(entity_text.replace(" ", "")) - 1: # 끝 지점
                    token_end = i
                
                # 범위 안에 있으면
                if token_start != -1 and token_end == -1:
                    # 아직 끝을 못 찾았는데 현재 토큰이 범위 내에 포함되면
                    pass
                elif token_start != -1 and i >= token_start:
                     if current_len + token_len > start_idx + len(entity_text.replace(" ", "")):
                         token_end = i 
                
                current_len += token_len
            
            # 범위 찾았으면 태그 할당
            if token_start != -1:
                if token_end == -1: token_end = token_start # 1글자짜리
                
                predicted_tags[token_start] = f"B-{label}"
                for i in range(token_start + 1, token_end + 1):
                    if i < len(predicted_tags):
                        predicted_tags[i] = f"I-{label}"

    return predicted_tags
```

# 2. 사용할 sLLM 모델

개체명 인식 구축에 사용할 모델로는 이전에 SQL 쿼리를 작성해 주는 sLLM을 학습해보는 실습을 진행할 때 사용했던 `beomi/Yi-Ko-6B` 모델을 사용하고자 하였으나 최근 한국어 벤치마크 기준 6~8B 모델 중 가장 높은 성능을 보이는 `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct` 모델을 사용하고자 합니다. 이 모델은 허깅페이스에 공개가 되어 있습니다. 다만 해당 모델을 사용하려면 허깅페이스에서 로그인을 한 후 <https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct>에 가셔서 약관 동의를 해야 모델을 사용하실 수 있습니다.

# 3. 모델 학습

포스트 초기에 설명드린 것과 같이 학습은 RunPod의 RTX 4090 GPU를 제공하는 Pod를 빌려서 학습을 진행했으며, RTX 4090은 VRAM이 24GB이므로 경량화를 위해 4-bit 양자화를 진행했습니다. 양자화에는 bitsandbytes라는 라이브러리를 활용했습니다.

그리고 LLM 모델의 파인튜닝에 사용되는 기법으로 LLM 모델 전체를 학습시키는 것은 불가능하므로, 작은 어댑터(Adapter) 모듈만 붙여서 학습 시키는 LoRA(Low-Rank Adaption) 기법을 활용했고, 이 기법을 적용하기 위해 peft 라이브러리를 활용했습니다.

마지막으로 사전 학습이 아닌 LLM 모델의 지도 미세 조정 방식을 위해 허깅페이스의 trl(Transformer Reinforcement Learning) 라이브러리의 SFTTrainer를 사용하였습니다. 

전체 학습 코드는 다음과 같습니다.

```python
import torch
import os
import data_processing as pr
from datasets import load_dataset
from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 데이터 전처리를 하지 않았다면 진행
#pr.preprocessing_data()

# 1. 설정 (Configuration)
MODEL_ID = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
NEW_MODEL_NAME = "EXAONE-3.0-7.8B-KLUE-NER-LoRA"
DATA_PATH_TRAIN = "../data/train.jsonl"

# 하이퍼파라미터
BATCH_SIZE = 4          # GPU 메모리에 따라 조절 (VRAM 24GB 기준 2~4 추천)
GRAD_ACCUMULATION = 4   # 실제 배치 효과 = BATCH_SIZE * GRAD_ACCUMULATION (여기선 16)
LEARNING_RATE = 2e-4    # QLoRA 표준 학습률
NUM_EPOCHS = 1          # 1 Epoch만 돌아도 데이터가 충분함 (약 2.1만 개)
MAX_SEQ_LENGTH = 1024   # 입력 문장 최대 길이 (메모리 절약을 위해 1024 설정)

# 2. 데이터셋 로드
print(">>> 데이터셋 로드 중...")
if not os.path.isfile("../data/train.jsonl"):
    pr.preprocessing_data()

dataset = load_dataset("json", data_files={"train":DATA_PATH_TRAIN})

print(">>> 데이터셋 로드 완료!")

# 3. 모델 및 토크나이저 로드(QLoRA 설정)
print(">>> 모델 및 토크나이저 로드 중...")

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 학습 안정성을 위한 설정
model.config.use_cache = False #학습 중엔 캐시 사용을 하지 않음 (Gradient Checkpointing 호환)
model.config.pretraining_tp = 1

# 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "right" # SFTTrainer는 right padding을 선호
tokenizer.pad_token = tokenizer.eos_token # EXAONE은 pad 토큰이 명시되지 않을 수 있어 EOS로 대체

# 4. LoRA (PEFT) 설정
# 모델의 모든 레이어를 학습하는 것이 아니라, 일부 레이어에 어댑터를 붙여 학습
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=32,   # Rank (높을수록 표현력 증가하지만 메모리 사용량 증가)
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 모든 선형 레이어 타겟
)

# 모델을 k-bit 학습 준비 상태로 만듦
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# 5. Trainer 설정 (SFTTrainer)
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=50,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    dataset_text_field="messages",  # 데이터셋의 텍스트 컬럼명
    max_length=MAX_SEQ_LENGTH,  # 최대 길이 설정
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    processing_class=tokenizer, 
    
    args=training_args,
)

# 6. 학습 시작
print(">>> 학습 시작!")
trainer.train()

print(f">>> 모델 저장 중... {NEW_MODEL_NAME}")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
```

그리고 학습은 이전에 실험을 진행했던 모델들과 달리 1 epoch만 진행합니다. 그 이유는 LLM 모델의 경우 이미 사전 학습에서 방대한 양의 데이터를 학습했기 때문에 BERT나 Bi-LSTM-CRF 모델과 달리 이미 방대한 양의 지식이 있는 상태입니다. 이러한 상황에서 우리가 LLM에 학습시키는 것은 BERT나 Bi-LSTM-CRF 모델에 패턴과 특징을 깊이 있게 학습시키는 것이 아닌 말투와 형식(Style & Format)을 교정하는 과정입니다. 또한 LLM 모델의 경우 방대한 데이터와 엄청난 큰 모델의 크기 때문에 이미 굉장히 똑똑한 상태입니다. 이러한 상태에서 3 Epoch을 초과할 경우 모델이 학습 데이터의 형식에 매몰되는 과적합(Overfitting) 현상이 발생합니다. 학습 데이터에 없는 데이터가 들어오게 되면 동작을 하지 못하게 됩니다. 또한 극단적인 과적합은 오히려 파멸적인 망각(Catastrophic Forgetting)을 유발하게 되어 모델이 원래 가지고 있던 일반적인 지식이나 추론 능력, 한국어 문법 능력 등을 까먹어 버리는 현상이 발생합니다. 또한 우리가 사용하는 KLUE NER 데이터셋의 학습 데이터의 양은 21,000개의 문장으로 생각보다 많은 양입니다. 데이터가 1,000개 미만일 경우에는 3~5 epoch 정도를 돌리기도 하지만 이 정도의 양은 1 epoch만 학습시켜도 모델이 똑똑하기 때문에 충분히 학습을 진행합니다.

이제 위 코드를 이용해 학습을 진행하게 되면 다음과 같이 진행 상황이 출력이 되게 됩니다. 학습 시 아래와 같이 loss 값이 2.x 에서 줄어들지 않을 경우에는 제대로 학습이 되지 않고 있는 것이기 때문에 실행 후 어느 정도 지켜봐야 합니다.

```
Output:
>>> 학습 시작!
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 361}.
  0%|                                                                                                                                   | 0/1313 [00:00<?, ?it/s]/usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
{'loss': 2.0249, 'grad_norm': 0.27734375, 'learning_rate': 0.0001999753350574969, 'entropy': 1.2126822836697102, 'num_tokens': 119068.0, 'mean_token_accuracy': 0.6868910838663578, 'epoch': 0.04}
{'loss': 0.7297, 'grad_norm': 0.2138671875, 'learning_rate': 0.0001989418443458192, 'entropy': 0.7228597594797611, 'num_tokens': 238411.0, 'mean_token_accuracy': 0.8619158652424812, 'epoch': 0.08}
{'loss': 0.649, 'grad_norm': 0.208984375, 'learning_rate': 0.00019640378558545487, 'entropy': 0.6465799322724343, 'num_tokens': 357934.0, 'mean_token_accuracy': 0.8703803929686547, 'epoch': 0.11}
{'loss': 0.6301, 'grad_norm': 0.1357421875, 'learning_rate': 0.00019239975399505763, 'entropy': 0.6288354907929897, 'num_tokens': 476933.0, 'mean_token_accuracy': 0.8741539368033409, 'epoch': 0.15}
{'loss': 0.6237, 'grad_norm': 0.1591796875, 'learning_rate': 0.00018699063724087904, 'entropy': 0.6222460491955281, 'num_tokens': 595894.0, 'mean_token_accuracy': 0.8759463322162628, 'epoch': 0.19}
{'loss': 0.613, 'grad_norm': 0.1494140625, 'learning_rate': 0.00018025868954299923, 'entropy': 0.615299743115902, 'num_tokens': 714119.0, 'mean_token_accuracy': 0.877697725892067, 'epoch': 0.23}
{'loss': 0.6052, 'grad_norm': 0.1259765625, 'learning_rate': 0.00017230628086913643, 'entropy': 0.6107596264779568, 'num_tokens': 834188.0, 'mean_token_accuracy': 0.8787398171424866, 'epoch': 0.27}
{'loss': 0.6056, 'grad_norm': 0.1318359375, 'learning_rate': 0.000163254340236532, 'entropy': 0.6093523935973644, 'num_tokens': 953418.0, 'mean_token_accuracy': 0.8791894060373306, 'epoch': 0.3}
{'loss': 0.6092, 'grad_norm': 0.1826171875, 'learning_rate': 0.00015324051679398108, 'entropy': 0.6127409189939499, 'num_tokens': 1071749.0, 'mean_token_accuracy': 0.879370147883892, 'epoch': 0.34}
{'loss': 0.6007, 'grad_norm': 0.1435546875, 'learning_rate': 0.00014241708664767993, 'entropy': 0.602255270332098, 'num_tokens': 1190094.0, 'mean_token_accuracy': 0.8813009190559388, 'epoch': 0.38}
 38%|██████████████████████████████████████████████ 
```

# 4. 허깅페이스에 모델 업로드

이번에 사용한 `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct` 모델의 경우 학습 시간도 이전의 BERT나 Bi-LSTM-CRF와 같이 10분 이내로 끝나는 것이 아니라 최소 1시간 이상 걸리기 때문에 이번에는 학습이 완료된 모델을 허깅페이스 허브에 업로드 하는 작업을 먼저 진행하였습니다. 다음은 학습이 끝나 `EXAONE-3.0-7.8B-KLUE-NER-LoRA` 폴더에 생성된 모델을 허깅페이스 허브에 업로드 하는 코드입니다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login

# 1. 설정
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
ADAPTER_MODEL_PATH = "EXAONE-3.0-7.8B-KLUE-NER-LoRA" # 학습 결과가 저장된 로컬 폴더 이름
HF_USERNAME = "본인의_허깅페이스_계정명"
HF_TOKEN = "본인의_WRITE_권한_토큰"

print(">>> 모델 로드 준비 중...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 베이스 모델 로드 (껍데기)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 3. 학습된 LoRA 어댑터 결합
# PeftModel을 로드하면 push_to_hub 실행 시 자동으로 '어댑터 파일'만 업로드합니다.
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)

# 4. [개선] 토크나이저를 '로컬 학습 폴더'에서 로드
# 학습할 때 저장해둔 설정을 그대로 올리는 것이 가장 안전합니다.
try:
    print(f">>> 로컬 폴더({ADAPTER_MODEL_PATH})에서 토크나이저 로드 시도...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_PATH, trust_remote_code=True)
except:
    print(">>> 로컬 토크나이저 로드 실패, 베이스 모델에서 로드합니다.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

# 5. 허깅페이스 로그인 및 업로드
print(">>> Hugging Face 로그인...")
login(token=HF_TOKEN)

# [수정 2] 레포지토리 ID 포맷 수정 (슬래시 추가)
repo_id = f"{HF_USERNAME}/{ADAPTER_MODEL_PATH}"
print(f">>> 업로드 시작: {repo_id}")

# 어댑터 업로드 (adapter_model.safetensors, adapter_config.json 등)
model.push_to_hub(repo_id, use_temp_dir=False)

# 토크나이저 업로드
tokenizer.push_to_hub(repo_id, use_temp_dir=False)

print(">>> 업로드 완료! 허깅페이스 웹사이트에서 확인하세요.")
```

모델 업로드가 정상적으로 되면 아래 이미지와 같이 허깅페이스에서 자신의 Models 부분에서 확인할 수 있습니다.

<div align="center">
  <img src="/assets/images/toy_project/comparison_of_entity_recognition_performance_by_model/3/huggingface_hub_sLLM_upload.png" width="80%" height="40%"/>
</div>

# 5. 모델 평가

모델의 평가로는 BERT와 Bi-LSTM-CRF 에서 진행했던 것과 같이 seqeval을 이용한 F1-score 방식을 사용하기로 하였습니다. 다만 LLM 모델의 출력 형태는 우리가 미세 조정 시켰던 방식으로 생성하는 방식이기 때문에 이전에 정의 해뒀던 json_to_bio 함수를 이용해 bio 형태로 바꾸는 작업을 진행해 주어야 합니다. 그리고 추가적으로 원래는 평가 데이터인 validation 데이터셋 전체(5,000개)를 사용해 평가를 진행해야 하지만 사용하는 모델이 sLLM 모델이라고 해도 LLM 모델이기 때문에 결과 생성에 시간이 걸려 5,000개의 데이터를 이용한 평가를 진행하면 대략 4시간 정도의 시간이 걸려 랜덤 샘플링한 100개의 데이터만 사용하여 평가를 진행하였습니다. 아래는 모델 평가 코드입니다.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report
import data_processing as pr
import json

# 1. 설정

BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
#ADAPTER_MODEL_PATH = "EXAONE-3.0-7.8B-KLUE-NER-LoRA"
ADAPTER_REPO_ID = "Laseung/EXAONE-3.0-7.8B-KLUE-NER-LoRA"
MAX_NEW_TOKENS = 512



# 시스템 프롬프트 (학습 때와 동일해야 함)
SYSTEM_PROMPT = (
    "당신은 유능한 개체명 인식기(NER)입니다. "
    "입력된 문장에서 개체명을 추출하여 JSON 형식으로 출력하세요. "
    "개체명 태그는 다음과 같습니다: PS(사람), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량). "
    "해당하는 개체명이 없으면 빈 JSON '{}'을 출력하세요."
)

# 2. 모델 로드
print(">>> 모델 로드 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 학습된 LoRA 어댑터 결합
#model = PeftModel.from_pretrained(base_model,ADAPTER_MODEL_PATH)
model = PeftModel.from_pretrained(base_model,ADAPTER_REPO_ID)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# 4. 평가 실행
print(">>> 데이터셋 로드 및 평가 시작...")
dataset = load_dataset("klue", "ner")
val_data = dataset['validation']

# 평가 시간 단축을 위해 500개만 랜덤 추출하여 평가
small_val_data = val_data.shuffle(seed=42).select(range(100))

# 라벨 정보
label_list = dataset['train'].features['ner_tags'].feature.names
id2label = {i: label for i, label in enumerate(label_list)}

true_labels = []
pred_labels = []

# 진행상황 표시
for i in tqdm(range(len(small_val_data))):
    sample = small_val_data[i]
    tokens = sample['tokens']
    original_tags = [id2label[tag] for tag in sample['ner_tags']]
    input_text = "".join(tokens).replace(" ", " ") # 원본 문장 복원

    # 프롬프트 구성
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=terminators,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = outputs[0][input_ids.shape[-1]:]
    decoded_output = tokenizer.decode(response, skip_special_tokens=True)

    # JSON -> BIO 변환
    pred_bio = pr.json_to_bio(decoded_output, tokens)

    true_labels.append(original_tags)
    pred_labels.append(pred_bio)


print(f"총 샘플 수: {len(small_val_data)}, JSON 파싱 에러: {error_count} 건")

# 점수 계산
print("\n>>> 평가 결과 Report:")
print(classification_report(true_labels, pred_labels))
print(f"F1-Score:{f1_score(true_labels, pred_labels):.4f}")
```

## 5.1 F1-score 트러블 슈팅

평가를 진행한 결과 F1-score가 0점인 성능을 얻었습니다. 

```
Output:
>>> 평가 결과 Report:
/usr/local/lib/python3.11/dist-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
              precision    recall  f1-score   support

          DT       0.00      0.00      0.00         6
          LC       0.00      0.00      0.00         8
          OG       0.00      0.00      0.00         1
          PS       0.00      0.00      0.00         7
          QT       0.00      0.00      0.00         7
          TI       0.00      0.00      0.00         1

   micro avg       0.00      0.00      0.00        30
   macro avg       0.00      0.00      0.00        30
weighted avg       0.00      0.00      0.00        30

F1-Score:0.0000
```

예상치 못한 결과에 직면하여, 문제의 근본 원인을 파악하기 위한 정밀 디버깅을 수행했습니다. 우선은 모델이 제대로된 json을 내뱉고 있는지를 한 번 확인해 보았습니다. 다음은 평가를 진행하는 루프안에 넣어줄 디버깅 코드입니다.

```python
f = open("debug_eval_output.txt", "w", encoding="utf-8")
# ... (evaluate_hub.py의 평가 루프 내부) ...
    response = outputs[0][input_ids.shape[-1]:]
    decoded_output = tokenizer.decode(response, skip_special_tokens=True)
    
    # [디버깅 코드 추가] ---------------------------------------
    f.write(f"[Input]: {input_text}\n")
    f.write(f"[Model Output]: {decoded_output}\n")
    f.write(f"[True Tags]: {original_tags}")
    f.write("\n" + "="*50 + "\n\n")
    # --------------------------------------------------------

    pred_bio = json_to_bio(decoded_output, tokens)
    # ...
```

디버깅을 위한 출력 코드를 출력해보니 'Model Output' 에는 어느 정도 잘 출력해 주는 것을 확인할 수 있습니다. 즉 모델 학습은 잘 된 것이라 볼 수 있습니다. 

````
Output:
[Input]: 안현수는 15일 밤(한국시간) 러시아 소치 아이스버그 스케이팅 팰리스에서 열린 2014 소치 동계올림픽 쇼트트랙 남자 1000m결승에서 금메달을 획득했다.
[Model Output]: {"PS": ["안현수"], "DT": ["15일", "2014"], "LC": ["한국", "러시아 소치 아이스버그 스케이팅 팰리스"], "TI": ["밤"], "QT": ["1000m"]}
[True Tags]: ['B-PS', 'I-PS', 'I-PS', 'O', 'O', 'B-DT', 'I-DT', 'I-DT', 'O', 'B-TI', 'O', 'B-LC', 'I-LC', 'O', 'O', 'O', 'O', 'B-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'I-LC', 'O', 'O', 'O', 'O', 'O', 'O', 'B-DT', 'I-DT', 'I-DT', 'I-DT', 'O', 'B-LC', 'I-LC', 'O', 'B-DT', 'I-DT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-QT', 'I-QT', 'I-QT', 'I-QT', 'I-QT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
==================================================

[Input]: 스토리 구성도 괜찮고 영화도 잘찍은 한국영화인데 평점이 너무 낮은거아님??? 9점영화이지만 평점이 낮아서 10점 드림
[Model Output]: {"QT": ["9점", "10점"]}
[True Tags]: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LC', 'I-LC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-QT', 'I-QT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-QT', 'I-QT', 'I-QT', 'O', 'O', 'O']
==================================================

[Input]: 중국 정부는 이번 폭발로 파손되거나 안전 우려가 제기되는 톈진항 주변 주택들을 사고가 나기 전 시장가격으로 사들인다는 방침을 세웠다.
[Model Output]: ```json
{
    "OG": ["중국 정부"],
    "LC": ["톈진항"]
}
```
[True Tags]: ['B-OG', 'I-OG', 'I-OG', 'I-OG', 'I-OG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LC', 'I-LC', 'I-LC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
==================================================

[Input]: 블리스의 성장기. 엔딩크레딧 음악이 좋다.
[Model Output]: {}
[True Tags]: ['B-PS', 'I-PS', 'I-PS', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
==================================================

[Input]: 예를 들어 하나면 얘기하자면 이소룡이 영춘권을 처음 배웠을 때는 14살이었습니다.
[Model Output]: {"PS": ["이소룡"], "QT": ["하나"], "DT": ["14살"]}
[True Tags]: ['O', 'O', 'O', 'O', 'O', 'O', 'B-QT', 'I-QT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PS', 'I-PS', 'I-PS', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-QT', 'I-QT', 'I-QT', 'O', 'O', 'O', 'O', 'O', 'O']
==================================================

...생략...
````

데이터를 봤을 때 json_to_bio의 함수에서 띄어쓰기를 잘 처리하지 못하는 문제 때문인 것 같습니다. 그래서 json_to_bio 함수를 다음과 같이 수정해 보도록 하겠습니다.

```python
def json_to_bio(json_str, tokens):
    """
    공백 무시(Whitespace Invariant) 매칭을 적용한 강화된 파싱 함수
    """
    predicted_tags = ['O'] * len(tokens)
    
    try:
        # JSON 파싱 시도
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if match:
            json_str = match.group()
        pred_dict = json.loads(json_str)
    except:
        # 파싱 실패 시
        return predicted_tags

    # 1. 전체 토큰을 공백 없이 하나의 문자열로 연결
    # KLUE 토큰 예: ['안', '녕', '하', '세', '요'] -> "안녕하세요"
    token_str_full = "".join(tokens)
    
    # 각 토큰이 전체 문자열에서 어디서 시작하고 끝나는지 매핑 정보 생성
    # token_spans[i] = (start_index, end_index)
    token_spans = []
    current_idx = 0
    for token in tokens:
        end_idx = current_idx + len(token)
        token_spans.append((current_idx, end_idx))
        current_idx = end_idx

    for label, entity_list in pred_dict.items():
        if not isinstance(entity_list, list): continue
        
        for entity_text in entity_list:
            if not isinstance(entity_text, str): continue
            
            # [핵심] 검색 대상과 원본 모두 공백 제거 후 매칭
            clean_entity = entity_text.replace(" ", "")
            if not clean_entity: continue
            
            # 원본 문자열(공백제거됨)에서 개체명 찾기
            # find는 첫 번째 등장만 찾으므로, 중복 처리를 위해 loop가 필요할 수 있으나
            # 여기서는 단순화를 위해 첫 번째 매칭만 처리
            start_idx = token_str_full.find(clean_entity)
            
            if start_idx == -1: 
                continue # 못 찾음
            
            end_idx = start_idx + len(clean_entity)
            
            # Character Index -> Token Index 변환
            # 찾은 글자 범위(start_idx ~ end_idx)에 걸쳐 있는 토큰들을 찾음
            t_start = -1
            t_end = -1
            
            for i, (ts_start, ts_end) in enumerate(token_spans):
                # 겹치는 구간이 있는지 확인 (Intersection)
                # max(start, ts_start) < min(end, ts_end)
                if max(start_idx, ts_start) < min(end_idx, ts_end):
                    if t_start == -1: t_start = i
                    t_end = i
            
            # 태그 할당
            if t_start != -1:
                predicted_tags[t_start] = f"B-{label}"
                for i in range(t_start + 1, t_end + 1):
                    predicted_tags[i] = f"I-{label}"
                    
    return predicted_tags
```

json_to_bio 함수를 변경한 후 다음과 같이 F1-score가 68.20으로 상승한 것을 확인할 수 있었습니다.

```
Output:
>>> 평가 결과 Report:
              precision    recall  f1-score   support

          DT       0.83      0.55      0.66        53
          LC       0.81      0.47      0.59        45
          OG       0.67      0.60      0.63        40
          PS       0.96      0.66      0.78        77
          QT       0.82      0.60      0.69        55
          TI       1.00      0.38      0.56        13

   micro avg       0.84      0.58      0.68       283
   macro avg       0.85      0.54      0.65       283
weighted avg       0.85      0.58      0.68       283

F1-Score:0.6820
```

## 5.2 최종 성능 분석 및 한계점

현재 json_to_bio 함수를 수정한 후 위와 같은 평가표를 확인할 수 있었습니다. 위의 평가표를 OG 태그를 제외한 대부분의 태그들은 precision이 높은 것을 확인할 수 있습니다. 즉 모델은 학습 데이터를 잘 학습해 모델이 내뱉은 결과는 얼추 어느 정도 맞추고 있다는 것으로도 볼 수 있습니다. 하지만 recall의 경우 대부분의 태그에서 낮은 것을 확인할 수 있습니다. micro avg 기준으로 보자면 실제 정답 100개 중에서 58개만 찾고 나머지는 찾지 못했다는 것입니다. 그렇다면 recall이 낮은 이유로 가장 먼저 드는 생각은 LLM 모델의 경우 json 형식에 맞춰서 결과를 달라고 했을 때 실제 json 형식과 달라 json 파싱을 하지 못해 그 데이터는 모두 'O' 태그로 처리해 recall이 높을 수 있습니다. 그렇다면 실제로 모델이 json 형식대로 잘 주고 있는지 한 번 검사해 보도록 하겠습니다.

아래 코드를 평가 코드에 적용해 줍니다.

```python
# 평가 루프 내부에 추가
error_count = 0
# ...
    try:
        pred_dict = json.loads(decoded_output)
    except json.JSONDecodeError:
        error_count += 1 # 에러 카운트 증가
        f.write(f"JSON Error: {decoded_output}\n\n") # 에러 난 문장 확인
# ...
print(f"총 샘플 수: {len(val_data)}, JSON 파싱 에러: {error_count} 건")
```

출력된 건수를 보면 100건 중 48건으로 아주 높은 확률로 json 파싱이 되지 않은 것으로 확인이 됩니다. 모델은 어느 정도 결과를 내뱉어 주고 있지만 "```json ... ```" 형식을 출력하고 있어 이로 인해 json 파싱이 되지 않아 recall 점수가 낮은 것으로 추정이 됩니다.

````json
Output:
총 샘플 수: 100, JSON 파싱 에러: 48 건
```

아래는 모델이 뱉은 json 형식의 결과 입니다.

```
JSON Error: ```json
{
    "OG": ["중국 정부"],
    "LC": ["톈진항"]
}
```

JSON Error: ```json
{
    "DT": ["20일"],
    "QT": ["6명"]
}
```

JSON Error: ```json
{
  "TI": ["후반 16분"],
  "QT": ["7.2km"]
}
```

JSON Error: ```json
{
    "LC": ["로마", "포폴로광장"]
}
```

JSON Error: ```json
{
  "DT": ["내년 말", "2016년 말"],
  "QT": ["1%", "2.25%"]
}
```

JSON Error: ```json
{
    "OG": ["유엔"]
}
```

JSON Error: ```json
{
  "PS": ["임권택", "강수연"]
}
```

JSON Error: ```json
{
    "QT": ["1", "2"],
    "PS": ["이지혜"]
}
```

... 생략 ...
````

recall의 성능을 높이기 위해 json 파싱을 줄이기 위해 "```json ... ```"의 형태를 없애주는 코드를 json_to_bio 함수에 추가를 해주고 평가를 진행해 보았습니다. 코드는 다음과 같습니다. 그리고 처리를 못하는 케이스를 확인하기 위해 예외가 발생한 경우 해당 json String을 출력하도록 해보았습니다.

```python
try:
  # (1) 마크다운 태그 무조건 제거
  if json_str:
      json_str = json_str.replace("```json", "").replace("```", "").strip()
  
  # (2) 가장 바깥쪽 중괄호 {} 찾아서 그 안의 내용만 추출
  #     모델이 사족을 달거나 앞뒤에 공백이 있어도 해결됨
  match = re.search(r"\{.*\}", json_str, re.DOTALL)
  if match:
      json_str = match.group()
      
  # (3) JSON 파싱 시도
  pred_dict = json.loads(json_str)
  
except json.JSONDecodeError:
  # (4) 2차 시도: 홑따옴표(')를 쌍따옴표("")로 변경 후 재시도
  print(f"JSON 파싱 실패: {json_str}\n\n")
  try:
      fixed_str = json_str.replace("'", '"')
      pred_dict = json.loads(fixed_str)
  except:
      # 그래도 안 되면 빈 딕셔너리로 진행 (이 경우만 진짜 0점)
      print(f"JSON 파싱 실패: {json_str}\n\n")
      return predicted_tags
except Exception:
  # 기타 에러 발생 시
  print(f"JSON 파싱 실패: {json_str}\n\n")
  return predicted_tags
```

다시 평가를 진행해 보면 예외 발생 시 출력하도록 한 출력 코드는 출력되지 않았고 성능은 이전과 같이 동일하게 출력되는 것을 확인했습니다. 그렇다면 이제 확인해 봐야 할 것은 실제 틀린 케이스를 출력해보고 그 케이스들이 무엇 때문에 틀린지 한 번 확인해 보도록 하겠습니다.

아래는 10개의 틀린 케이스를 출력해보도록 했을 때 출력된 결과입니다. 아래 결과를 보면 여러 원인으로 인해 성능이 내려간 것을 알 수 있습니다. 

우선 3번째 문장과 6번째 문장에서 잡히지 말아야 할 일반명사까지 개체명으로 잡아버리는 문제가 존재합니다.

두 번째로 단위 표기 불일치가 존재합니다 9번째 문장에서 원본 문장에서는 한글 자모/특수문자 단위인 `㎞` 가 사용된 `7.2㎞`를 모델은 단순히 `7.2km` 내 뱉고 있습니다. 이로인해 find 함수로 문자열을 찾게 되면 찾지 못해 개체명 태그를 달지 못하는 문제가 발생합니다.

세 번째로 두 번째 문장에서 띄어쓰기 및 조사 처리가 제대로 되지 않아 개체명 태그가 부착되지 못했습니다. 

```
틀린 케이스:

입력: 안현수는 15일 밤(한국시간) 러시아 소치 아이스버그 스케이팅 팰리스에서 열린 2014 소치 동계올림픽 쇼트트랙 남자 1000m결승에서 금메달을 획득했다.

모델 생성(JSON): {"PS": ["안현수"], "DT": ["15일"], "LC": ["한국", "러시아 소치 아이스버그 스케이팅 팰리스", "소치"], "TI": ["밤", "1000m"], "QT": ["한국", "2014", "동계올림픽", "남자", "1000m", "금메달"]}

실제 정답(Target): ['안(B-PS)', '현(I-PS)', '수(I-PS)', '1(B-DT)', '5(I-DT)', '일(I-DT)', '밤(B-TI)', '한(B-LC)', '국(I-LC)', '러(B-LC)', '시(I-LC)', '아(I-LC)', ' (I-LC)', '소(I-LC)', '치(I-LC)', ' (I-LC)', '아(I-LC)', '이(I-LC)', '스(I-LC)', '버(I-LC)', '그(I-LC)', ' (I-LC)', '스(I-LC)', '케(I-LC)', '이(I-LC)', '팅(I-LC)', ' (I-LC)', '팰(I-LC)', '리(I-LC)', '스(I-LC)', '2(B-DT)', '0(I-DT)', '1(I-DT)', '4(I-DT)', '소(B-LC)', '치(I-LC)', '동(B-DT)', '계(I-DT)', '1(B-QT)', '0(I-QT)', '0(I-QT)', '0(I-QT)', 'm(I-QT)']

내 코드의 변환 결과(Bio): ['안(B-PS)', '현(I-PS)', '수(I-PS)', '1(B-DT)', '5(I-DT)', '일(I-DT)', '밤(B-TI)', '한(B-QT)', '국(I-QT)', '소(B-LC)', '치(I-LC)', '2(B-QT)', '0(I-QT)', '1(I-QT)', '4(I-QT)', '동(B-QT)', '계(I-QT)', '올(I-QT)', '림(I-QT)', '픽(I-QT)', '남(B-QT)', '자(I-QT)', '1(B-QT)', '0(I-QT)', '0(I-QT)', '0(I-QT)', 'm(I-QT)', '금(B-QT)', '메(I-QT)', '달(I-QT)']



입력: 중국 정부는 이번 폭발로 파손되거나 안전 우려가 제기되는 톈진항 주변 주택들을 사고가 나기 전 시장가격으로 사들인다는 방침을 세웠다.

모델 생성(JSON): {"OG": ["중국 정부"], "LC": ["톈진항"], "DT": []}

실제 정답(Target): ['중(B-OG)', '국(I-OG)', ' (I-OG)', '정(I-OG)', '부(I-OG)', '톈(B-LC)', '진(I-LC)', '항(I-LC)']

내 코드의 변환 결과(Bio): ['톈(B-LC)', '진(I-LC)', '항(I-LC)']



입력: 블리스의 성장기. 엔딩크레딧 음악이 좋다.

모델 생성(JSON): {"PS": ["블리스"], "QT": ["엔딩크레딧"]}

실제 정답(Target): ['블(B-PS)', '리(I-PS)', '스(I-PS)']

내 코드의 변환 결과(Bio): ['블(B-PS)', '리(I-PS)', '스(I-PS)', '엔(B-QT)', '딩(I-QT)', '크(I-QT)', '레(I-QT)', '딧(I-QT)']



입력: 예를 들어 하나면 얘기하자면 이소룡이 영춘권을 처음 배웠을 때는 14살이었습니다.

모델 생성(JSON): {"PS": ["이소룡"], "QT": ["14살"]}

실제 정답(Target): ['하(B-QT)', '나(I-QT)', '이(B-PS)', '소(I-PS)', '룡(I-PS)', '1(B-QT)', '4(I-QT)', '살(I-QT)']

내 코드의 변환 결과(Bio): ['이(B-PS)', '소(I-PS)', '룡(I-PS)', '1(B-QT)', '4(I-QT)', '살(I-QT)']



입력: 한편 지난 20일 '비정상회담'은 개편 1주년을 맞아 멤버 6명의 하차 소식을 전하면서 시청자들의 아쉬움을 사고 있다.

모델 생성(JSON): {"DT": ["20일"], "QT": ["1주년", "6명"]}

실제 정답(Target): ['지(B-DT)', '난(I-DT)', ' (I-DT)', '2(I-DT)', '0(I-DT)', '일(I-DT)', '1(B-DT)', '주(I-DT)', '년(I-DT)', '6(B-QT)', '명(I-QT)']



입력: 코난은탐정인데 추리를 안하고 왜액션만하지

모델 생성(JSON): {"PS": ["코난"], "QT": ["탐정"]}

실제 정답(Target): ['코(B-PS)', '난(I-PS)']

내 코드의 변환 결과(Bio): ['코(B-PS)', '난(I-PS)', '탐(B-QT)', '정(I-QT)']



입력: 헤닝은 영국 맨체스터 인근 솔포드에서 부인, 두 십대 자녀와 함께 살며 택시를 몰던 평범한 시민이었지만 고통받는 시리아 사람들의 모습을 보고 구호활동에 뛰어들었다.

모델 생성(JSON): {"PS": ["헤닝"], "LC": ["영국 맨체스터", "솔포드", "시리아"], "QT": ["두 십대"]}

실제 정답(Target): ['헤(B-PS)', '닝(I-PS)', '영(B-LC)', '국(I-LC)', ' (I-LC)', '맨(I-LC)', '체(I-LC)', '스(I-LC)', '터(I-LC)', ' (I-LC)', '인(I-LC)', '근(I-LC)', ' (I-LC)', '솔(I-LC)', '포(I-LC)', '드(I-LC)', '두(B-QT)', ' (I-QT)', '십(I-QT)', '대(I-QT)', '시(B-LC)', '리(I-LC)', '아(I-LC)']

내 코드의 변환 결과(Bio): ['헤(B-PS)', '닝(I-PS)', '솔(B-LC)', '포(I-LC)', '드(I-LC)', '시(B-LC)', '리(I-LC)', '아(I-LC)']



입력: 결국, 정씨는 지난달 말 서울의 한 병원에 찾아가 검사를 받던 중 외상성 경막하출혈 진단을 받았다.

모델 생성(JSON): {"PS": ["정"], "DT": ["지난달 말"], "LC": ["서울"], "QT": ["한"]}

실제 정답(Target): ['정(B-PS)', '지(B-DT)', '난(I-DT)', '달(I-DT)', ' (I-DT)', '말(I-DT)', '서(B-LC)', '울(I-LC)']

내 코드의 변환 결과(Bio): ['정(B-PS)', '서(B-LC)', '울(I-LC)', '한(B-QT)']



입력: 선발로 나와 후반 16분까지 7.2㎞를 뛰어 적지 않은 활동량은 기록했으나 선전으로 평가되기에는 부족한 면이 있었다.

모델 생성(JSON): {"TI": ["후반 16분"], "QT": ["7.2km"]}

실제 정답(Target): ['후(B-TI)', '반(I-TI)', ' (I-TI)', '1(I-TI)', '6(I-TI)', '분(I-TI)', '7(B-QT)', '.(I-QT)', '2(I-QT)', '㎞(I-QT)']

내 코드의 변환 결과(Bio): []



입력: 하루에 한번씩 안빠지고 맨날본다 ㅋㅋ

모델 생성(JSON): {"QT": ["하루"], "TI": ["한번"], "DT": ["매일"]}

실제 정답(Target): ['하(B-DT)', '루(I-DT)', '한(B-QT)', '번(I-QT)']

내 코드의 변환 결과(Bio): ['하(B-QT)', '루(I-QT)', '한(B-TI)', '번(I-TI)']
```

그럼 이제 성능을 높이기 위해서 진행해야할 작업을 정의해보도록 하겠습니다. 첫 째로 모델이 일반 명사에 대해서는 개체명으로 잡지 않도록 하기 위해 프롬프트 엔지니어링을 적용하고자 프롬프트를 다음과 같이 수정하였습니다.

```python
SYSTEM_PROMPT = (
    "당신은 엄격한 기준을 가진 개체명 인식기(NER)입니다.\n"
    "입력 문장에서 다음 태그에 해당하는 단어만 정확히 추출하여 JSON으로 출력하세요.\n"
    "태그: PS(사람), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량)\n\n"
    "*** 주의사항 ***\n"
    "1. '탐정', '엔딩크레딧', '시민', '선수' 같은 일반 명사는 절대 추출하지 마세요.\n"
    "2. 문장에 없는 단어를 만들어내거나 요약하지 마세요.\n"
    "3. 날짜나 수량은 문장에 적힌 그대로(단위 포함) 가져오세요.\n"
    "4. 해당하는 개체명이 없으면 빈 리스트를 반환하세요.\n\n"
    "[예시]\n"
    "입력: 명탐정 코난은 추리를 잘한다.\n"
    "출력: {\"PS\": [\"코난\"], \"QT\": [], \"OG\": [], \"LC\": [], \"DT\": [], \"TI\": []}\n"
    "(설명: '명탐정'은 직업이므로 추출하지 않음)\n\n"
    "이제 분석하세요."
)
```

두 번째로 단위 표기 불일치 문제를 해결하기 위해 모델이 내 뱉는 단위 영어를 한글 자모/특수문자로 변환하도록 하여 실제 데이터셋에 맞추도록 하기 위해 json_to_bio 함수에 다음 코드를 추가하였습니다.

```python
def json_to_bio(json_str, tokens):
    # ... (기존 마크다운 제거 및 JSON 파싱 로직) ...
    
    # [1] 전체 토큰 문자열 생성
    token_str_full = "".join(tokens)
    
    # [2] 정규화 맵 (여기에 계속 추가하면 점수가 오릅니다)
    correction_map = {
        "km": "㎞", "KM": "㎞", "Km": "㎞",
        "m": "ｍ", "M": "ｍ",
        "%": "％", 
        "kg": "㎏", "KG": "㎏",
        "cm": "㎝", "CM": "㎝",
        "mm": "㎜", "MM": "㎜",
        "C": "℃", "F": "℉"
    }

    # ... (매칭 루프 시작) ...
    for label, entity_list in pred_dict.items():
        # ...
        for entity_text in entity_list:
            
            # [3] 공백 제거
            clean_entity = entity_text.replace(" ", "")
            
            # [4] ★ 핵심: 단위 변환 적용 ★
            for eng_unit, special_unit in correction_map.items():
                if eng_unit in clean_entity:
                    clean_entity = clean_entity.replace(eng_unit, special_unit)

            # [5] 이후 find 로직 수행
            start_idx = token_str_full.find(clean_entity)
            # ...
```

마지막으로 띄어쓰기 및 조사 문제는 LLM 모델의 결과를 BIO 형태로 바꿀때 생기는 고질적인 문제라 해결하지 못했습니다. 그래서 우선 단위 표기의 일치와 시스템 프롬프트의 수정으로 일반 명사를 개체명으로 안잡게 하도록하여 다시 평가를 진행했습니다. 다음은 평가 결과입니다.

이전 68.20보다 3퍼센트나 오른 71.29 라는 F1-score 성능을 얻을 수 있었습니다. 다만 이 이상으로는 LLM 모델을 이용한 BIO 형태로 변환해서 점수를 매겨서 성능을 올리는 것에 한계가 있어 이후 시스템 프롬프트를 좀 더 정교하게 바꿔보고 json_to_bio 함수를 좀 더 바꿔보기도 하였지만 71.29%에서 더 이상의 성능 향상은 볼 수 없었습니다.

```
>>> 평가 결과 Report:
              precision    recall  f1-score   support

          DT       0.82      0.60      0.70        53
          LC       0.76      0.58      0.66        45
          OG       0.74      0.65      0.69        40
          PS       0.97      0.81      0.88        77
          QT       0.69      0.53      0.60        55
          TI       0.62      0.38      0.48        13

   micro avg       0.81      0.64      0.71       283
   macro avg       0.77      0.59      0.67       283
weighted avg       0.81      0.64      0.71       283

F1-Score:0.7129
```

# 6. 모델별 성능 비교, 결과 분석 및 엔지니어링 인사이트

다음은 각 모델별 성능을 비교하고 주요 특징을 정리한 표입니다.

|모델 (Model)|Precision (P)|Recall (R)|F1-Score (F1​)|주요 특징|
|:----------|:-----------:|:--------:|:-----------:|:-------:|
|BERT-Base|0.89|0.87|0.88|인코더 기반의 정밀한 토큰 분류, 가장 안정적인 성능|
|Bi-LSTM-CRF|0.86|0.84|0.85|시퀀스 종속성 모델링의 강점, 준수한 성능과 속도|
|EXAONE-3.0 (sLLM)|0.81|0.64|0.71|높은 문맥 이해도, 포맷팅 및 단위 매칭 이슈 존재|

<br>

1. BERT & Bi-LSTM-CRF: 검증된 안정성
  - 장점: 토크나이저와 원본 데이터셋의 인덱스가 1:1로 매칭되는 Classification 방식이므로, 개체명의 경계를 놓치는 실수가 적습니다.
  - 결과: F1-score 80점대 후반의 높은 성능을 보여주며 실무적으로 즉시 활용 가능한 수준임을 입증했습니다.

<br>

2. sLLM(EXAONE 3.0): 지능은 높지만, 규칙은 낯선 모델
  - 장점: 학습하지 않은 복잡한 문장 구조에서도 개체명의 의미를 파악하는 능력이 탁월합니다(예시: 안현수, 헤닝 등의 케이스)
  - 한계: Generation 방식의 한계로 `km` → `㎞`와 같은 미세한 단위 불일치나 JSON 포맷팅 오류가 발생했습니다.
  - 성과: 초기 0점에서 시작해 프롬프트 엔지니어링과 Regex 기반 후처리 파이프라인 구축을 통해 최종적으로 71.29점까지 성능을 끌어올렸습니다.

# 마치며

이번엔 목표로 했던 LLM 모델을 이용한 개체명 인식기 구현을 진행해 보았습니다. BERT와 Bi-LSTM-CRF와는 달리 학습 데이터 구축부터 달랐고, 무엇보다 이전 BERT나 Bi-LSTM-CRF 모델보다는 좀 더 나은 성능을 보여줄 것 같았던 LLM 모델에서 제일 낮은 성능을 확인하여 놀랐습니다. 다만 이는 모델이 학습이 덜 되어 모델의 결과가 안 좋아서 발생한 것이 아닌 이는 모델의 지능 문제라기보다, 전통적인 NER 평가 지표를 생성형 모델에 그대로 적용하면서 발생한 구조적 불일치 때문이었습니다.

특히나 LLM 모델을 이용해 개체명 인식기를 구현하면서 알게된 것은 LLM 모델은 BERT나 Bi-LSTM-CRF와 같이 분류 모델이 아닌 생성 모델이기 때문에 BERT나 Bi-LSTM-CRF는 사용한 tokenizer와 원본 데이터와 비교하여 사람이 직접 보고 어느 정도 매칭이 가능하지만 LLM 모델은 내부의 tokenizer를 기준으로 하기 때문에 원본 데이터의 index를 기준으로 개체명을 뽑아서 원본 데이터에 맞게 가공하는 작업이 되지 않는다는 것을 알게 되었습니다.

또한 LLM 모델은 전처리, 후처리 작업도 중요하지만 무엇보다도 얼마나 정교한 프롬프트를 작성하느냐에 따라 모델이 내뱉는 결과의 성능이 바뀔 수 있다는 것을 보고 프롬프트 엔지니어링이 정말 중요하다는 것을 알게 되었습니다.

마지막으로 이번 포스트에서는 진행하지 못했지만 이번 포스트에서 볼 수 있었던 것 같이 LLM의 출력 결과를 억지로 옛날 방식의 평가 방식에 맞추려고 하다보니 LLM 출력 전처리가 제대로 되지 않아 몇몇 모델은 개체명을 정상적으로 잡았음에도 불구하고 원본 데이터 형식으로 제대로 변환이 되지 않아 F1-Score 특히 recall 점수가 precision에 비해 낮았던 것을 볼 수 있었습니다.다음 포스트에서는 이번 포스트에서 학습을 진행했던 모델을 이용해 LLM 모델을 이용한 개체명 인식기의 평가 방식에 대해서 좀 더 알아보고 알아본 평가 방식대로 진행 했을 때 제가 학습시킨 LLM의 실제 성능이 어느 정도인지를 알아보는 포스트를 준비해 보고자 합니다.

이번 토이 프로젝트를 통해 모델의 체급(Parameter)이 크다고 해서 모든 태스크에서 만능은 아니라는 점을 직접 체감했습니다. NER처럼 **엄격한 규칙(Strict Gold Standard)**이 적용되는 태스크에서는 여전히 BERT 계열의 인코더 모델이 효율적일 수 있으며, sLLM을 활용할 때는 단순한 파인튜닝을 넘어 정교한 프롬프트 설계와 강건한(Robust) 후처리 로직이 성능의 핵심임을 깨달았습니다.

긴 글 읽어주셔서 감사드리며, 본문 내용 중 잘못된 내용이나 오타, 궁금하신 사항이 있으신 경우에는 댓글 달아주시기 바랍니다.

# 참조
- <https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct>