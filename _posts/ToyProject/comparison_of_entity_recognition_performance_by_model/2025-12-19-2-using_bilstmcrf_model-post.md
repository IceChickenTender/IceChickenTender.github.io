---
title: "[ToyProject] 모델별 개체명 인식 성능 비교 - 2. Bi-LSTM-CRF 모델을 이용한 개체명 인식기 구축과 성능 평가"
categories:
  - ToyProject
tags:
  - ToyProject
  - Deeplearning

use_math: true
toc: true
toc_sticky: true
toc_label: "Bi-LSTM-CRF 모델을 이용한 개체명 인식기 구축과 성능 평가"
---

이전에 진행했던 "BERT 모델을 이용한 개체명 인식기 구축과 성능 평가"에 이어서 이번엔 Bi-LSTM-CRF 모델을 이용한 개체명 인식기 성능 평가를 진행해 보도록 하겠습니다.

실습 진행에 있어 필요한 라이브러리들 설치를 먼저 진행해 줍니다. 실습은 코랩과 같은 환경을 기준으로 하므로 pytorch, numpy 등과 같은 라이브러리는 기본적으로 설치되어 있다는 것을 전제로 하겠습니다.

참고로 저는 RunPod에서 3090 서버를 빌려 학습을 진행했습니다.

```bash
pip install transformers huggingface_hub seqeval evaluate datasets pytorch-crf
```

# 1. 데이터

데이터는 이전에 진행했던 "BERT 모델을 이용한 개체명 인식기 구축과 성능 평가"에서와 동일한 KLUE NER 데이터를 사용했습니다. 그리고 BERT 모델과의 비교를 위해 형태소 분석기를 사용하지 않고, BERT 모델에 있는 Tokenizer를 사용해 전처리가 된 데이터를 동일하게 사용했습니다. 사실 Bi-LSTM-CRF를 이용한 여러 연구들에서는 개체명 인식기의 학습 데이터는 형태소 분석기로 형태소 분석이 된 데이터를 사용하는 것이 필수적이었으나 Tokenizer가 일종의 형태소 분석기와 동일한 작업을 진행하고, 또한 이전 실험에 진행했던 것과 동일한 데이터를 사용해야 해서 BERT 모델에 있는 Tokenizer를 이용해 전처리된 데이터를 사용했습니다. 데이터 전처리 과정은 다음과 같습니다.

우선 이전 과정에서 진행했던 KLUE NER 데이터셋을 klue/bert-base tokenizer의 token에 맞게 개체명 태그를 달아주는 과정을 진행해 줍니다.

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

def tokenize_and_align_labels(examples):
    raw_inputs = ["".join(x) for x in examples["tokens"]]
    tokenized_inputs = tokenizer(
        raw_inputs,
        truncation=True,
        return_offsets_mapping=True,
        padding="max_length", # 편의상 max_length로 맞추되, 뒤에서 다시 처리 가능
        max_length=128
    )

    labels = []
    for i, (doc_tags, offset_mapping) in enumerate(zip(examples["ner_tags"], tokenized_inputs["offset_mapping"])):
        encoded_labels = []
        for offset in offset_mapping:
            start, end = offset
            if start == end:
                encoded_labels.append(-100) # 특수 토큰
                continue
            origin_char_idx = start
            if origin_char_idx < len(doc_tags):
                encoded_labels.append(doc_tags[origin_char_idx])
            else:
                encoded_labels.append(-100)
        labels.append(encoded_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

model_id = "klue/bert-base"

dataset = load_dataset("klue", "ner")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 라벨 맵핑
label_list = dataset["train"].features["ner_tags"].feature.names
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}
pad_tag_id = label2id['O']

# 전처리 수행
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
# 필요한 컬럼만 남기기 (CRF 모델에는 input_ids와 labels만 필요)
tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)
tokenized_datasets = tokenized_datasets.remove_columns(["attention_mask", "offset_mapping"]) # CRF는 별도 마스크 생성 예정
tokenized_datasets.set_format("torch")
```

Bi-LSTM-CRF 모델은 BERT 모델과는 다르게 뒤에 CRF 층이 있어 BERT 모델에서 사용하던 mask에 별도의 작업을 진행해 주어야 합니다. tokenizer를 통해 추출한 mask에는 0과 1의 값으로 이루어져 있지만 CRF 층에는 이를 True와 False인 boolean 값을 이용합니다. 그러므로 CRF 층에서 사용할 수 있게 mask의 값을 바꾸어 주어야 합니다. 이를 위한 함수 정의와 정의한 함수를 사용해 데이터셋을 batch_size로 나누는 작업을 진행해 보도록 하겠습니다.

```python
def collate_fn(batch):
    # 1. 데이터 추출
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 2. 패딩(Pytorch 기본 pad_sequence 활용
    # input_ids 패딩(tokenizer.pad_token_id로 채움)
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    # labels 패딩 (일단 -100이나 'O' 태그로 채움)
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    # 3. CRF용 마스크 생성
    # labels가 -100인 곳(특수 토큰, 패딩, 서브워드 뒷부분)은 False, 나머지는 True
    # 이렇게 하면 CRF는 해당 지점의 Loss를 계산하지 않음
    mask = (labels_padded != -100)

    # BERT의 첫 번째 토큰은 무조건 [CLS]이므로 인덱스 0입니다.
    # 이 부분의 마스크를 강제로 True로 바꿔줍니다.
    mask[:, 0] = True
    
    # [CLS]의 라벨이 -100이면 에러가 나므로, 의미 없는 태그인 'O' (pad_tag_id)를 할당합니다.
    # (학습에 큰 영향 없이 CRF가 시작점을 잡을 수 있게 해줍니다.)
    labels_padded[:, 0] = pad_tag_id

    # 4. -100 값 치환 (CRF 에러 방지)
    # 마스크가 False인 곳은 계산 안하겠지만, 인덱스 에러 방지를 위해 'O' 태그 ID로 덮어씀

    # 4. 나머지 -100 값 치환
    labels_padded[labels_padded == -100] = pad_tag_id

    return input_ids_padded, labels_padded, mask

batch_size = 32

train_dataloader = DataLoader(
    tokenized_datasets['train'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4, # CPU 병렬 로딩
    pin_memory=True # CPU에서 GPU로의 전송 가속
)

val_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4, # CPU 병렬 로딩
    pin_memory=True # CPU에서 GPU로의 전송 가속
)
```

추가로 위 코드들은 나중에 train.py 라는 파일에 합칠 예정입니다.

# 2. Bi-LSTM-CRF 모델 정의

Bi-LSTM-CRF 모델은 워낙 유명한 모델이라 Pytorch 공식 사이트에서 제공하고 있는 것을 가져왔습니다. 저는 model.py 라는 파일에다가 모델만 따로 분리를 해주었습니다.

```python
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.target_size = len(tag_to_ix)

        # 1. 임베딩 층 (Word Embedding)
        # BERT와 달리 처음부터 학습하거나, Word2Vec/FastText 등을 로드해서 쓸 수 있습니다.
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        # 2. Bi-LSTM 층
        # batch_first=True로 설정해야 (Batch, Seq, Feature) 순서로 처리됩니다.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # 3. Hidden Layer to Tag Space
        # LSTM의 출력(hidden_dim)을 태그 개수(target_size)로 변환
        self.hidden2tag = nn.Linear(hidden_dim, self.target_size)

        # 4. CRF 층 (pytorch-crf 라이브러리 활용)
        self.crf = CRF(self.target_size, batch_first=True)

    def forward(self, input_ids, tags, mask):
        """
        학습(Training) 시 사용: Loss(Negative Log Likelihood) 반환
        """
        # 1. 임베딩
        embeds = self.word_embeds(input_ids)

        # 2. LSTM 통과
        lstm_out, _ = self.lstm(embeds)

        # 3. 태그 공간으로 투영 (Emissions)
        emissions = self.hidden2tag(lstm_out)

        # 4. CRF Loss 계산 (마스크 적용 필수!)
        # -log_likelihood를 반환하므로, 이를 최소화(minimize)하는 방향으로 학습하면 됩니다.
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss

    def decode(self, input_ids, mask):
        """
        추론(Inference) 시 사용: 가장 높은 확률의 태그 시퀀스 반환
        """
        embeds = self.word_embeds(input_ids)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)

        # Viterbi 알고리즘을 통해 가장 최적의 경로(Best Path) 추출
        best_paths = self.crf.decode(emissions, mask=mask)
        return best_paths
```

# 3. 평가 함수 정의

데이터 전치리와 모델 정의까지 완료 했습니다. 그렇다면 이제 학습 시 마다 모델의 성능이 어떻게 달라지는지 보기 위해 성능 평가 함수를 정의해 보도록 하겠습니다. 성능 평가 함수로는 이전 과정에서 사용했던 seqeval을 사용해 진행합니다.

```python
import torch
from seqeval.metrics import f1_score, classification_report

# 평가 함수
def evaluate(model, dataloader, id2label):
    model.eval() # 평가 모드로 전환

    true_labels = []
    pred_labels = []

    with torch.no_grad(): # 기울기 계산 끄기
        for batch in dataloader:
            input_ids, tags, mask = batch
            input_ids = input_ids.to(device)
            tags = tags.to(device)
            mask = mask.to(device)

            # 모델 추론
            # CRF의 decode는 Loss가 아니라 Best Tag Sequence(List[List[int]])를 반환합니다.
            predictions = model.decode(input_ids, mask)

            # 4. 정답(Tags)과 예측값(Predictions)을 문자열로 변환
            # predictions는 가변 길이 리스트이고, tags는 패딩된 텐서이므로 주의해서 매핑
            for pred_seq, true_seq, mask_seq in zip(predictions, tags, mask):
                # pred_seq: [2, 3, 3] (예측된 id 리스트)
                # true_seq: [2, 3, 3, 0, 0] (패딩된 정답 ID 텐서)
                # mask_seq: [True, True, True, False, False]

                # 마스크가 True인 부분만 실제 정답
                valid_true = true_seq[mask_seq].cpu().numpy()

                # ID -> Label 변환 ('2' -> 'B-PS')
                # 예측값 변환
                pred_labels_str = [id2label[tag_id] for tag_id in pred_seq]

                # 정답 변환
                true_labels_str = [id2label[tag_id] for tag_id in valid_true]

                pred_labels.append(pred_labels_str)
                true_labels.append(true_labels_str)

    # seqeval을 이용한 성능 계산
    f1 = f1_score(true_labels, pred_labels)

    print(classification_report(true_labels, pred_labels))

    return f1
```

# 4. 모델 학습 진행

이제 평가 함수까지 정의를 했으니 모델 학습 루프를 구현해 학습을 진행해 보도록 하겠습니다.

```python
import torch
import time
import os
from model import BiLSTM_CRF
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.vocab_size
tag_to_ix = label2id
EMBEDDING_DIM = 768
HIDDEN_DIM = 256

model = BiLSTM_CRF(
    vocab_size = vocab_size,
    tag_to_ix = tag_to_ix,
    embedding_dim = EMBEDDING_DIM,
    hidden_dim = HIDDEN_DIM
)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
path = "./models/klue_ner_bi_lstm_crf.pth"
os.makedirs("./models", exist_ok=True)
best_f1 = 0.0
f1_list = []

print(">>> 학습 시작")
total_start = time.time()
for epoch in range(10):
    model.train()
    total_loss = 0.0
    epoch_start = time.time()
    
    for batch in train_dataloader:
        input_ids, tags, mask = batch
        input_ids = input_ids.to(device)
        tags = tags.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # 속도 증가를 위해 FP16 연산 적용
        loss = model(input_ids, tags, mask)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    score = evaluate(model, val_dataloader, id2label)
    f1_list.append(score)
    print(f"{epoch+1} Loss : {total_loss/len(train_dataloader)}, F1-score : {score:.4f}")
    if score > best_f1:
        best_f1 = score
        print(f"Best F1-score was changed, Best F1-score is {best_f1:.4f}")
        print("Save Best model...")
        torch.save(model.state_dict(), path)
    epoch_end = time.time()
    print(f"{epoch+1} train time : {(epoch_end - epoch_start)}s")

# 전체 학습 종료 리포트
total_end = time.time()
print("\n" + "="*30)
print(f" 학습 완료 리포트")
print(f"="*30)
print(f" - 최고(Best) F1-score : {best_f1:.4f}")
print(f" - 전체 평균 F1-score  : {sum(f1_list)/len(f1_list):.4f}")
print(f" - 총 소요 시간        : {(total_end - total_start):.2f}s")
```

10 에폭 기준 최고 F1-score 성능은 75.82% 이고 전체 평균 F1-score는 74.44%로 이전 BERT 모델의 성능인 최고 F1-score인 88.97%, 평균 F1-score는 88.22%와 비교하면 최고 F1-score는 13.15%, 평균 F1-score는 13.78% 정도 차이가 나는 것으로 굉장한 성능 차이가 나는 것을 확인할 수 있습니다. 또한 학습 소요 시간에서도 BERT 모델의 경우 한 에폭당 대략 55초 정도 걸리는 반면 Bi-LSTM-CRF 모델의 경우 BERT 모델 처럼 병렬 연산이 불가능해서 그런지 한 에폭당 대략 82초 정도가 걸려 Bi-LSTM-CRF 모델이 느리면서 성능이 더 낮은 것을 확인할 수 있었습니다.  

```
Output:
==============================
 학습 완료 리포트
==============================
 - 최고(Best) F1-score : 0.7582
 - 전체 평균 F1-score  : 0.7444
 - 총 소요 시간        : 819.67s
```

# 5. 허깅페이스 허브에 모델 업로드

그럼 학습한 모델을 허깅페이스 허브에 업로드 해보도록 하겠습니다. 이번 Bi-LSTM-CRF 모델은 허깅페이스에서 지원하는 transfomer 모델이 아니기 때문에 모델과 모델의 뼈대인 config.json 파일 그리고 데이터 전처리에 사용했던 tokenizer도 따로 업로드를 해주어야 합니다. 그러기 위해선 config.json 파일에 들어갈 내용과 tokenizer를 명시해 주어야 합니다.

config.json 파일에는 모델에서 사용하는 vocab_size, embedding_dim, hidden_dim, tag_to_ix, model_type을 정의해 주어야 합니다.

학습 시 저장되는 모델의 확장자는 pth인데 허깅페이스는 일반적으로 bin 확장자를 사용합니다. 그러므로 pth 파일을 bin 파일로 바꾸어 주는 과정도 필요합니다.

```python
import torch
import json
import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer
from datasets import load_dataset

# 1. 설정 및 정보
repo_id = "본인의 허깅페이스 이름/klue-ner-bi-lstm-crf"
token = "본인의 허깅페이스 token 값"
save_dir = "./upload_pack"

# 폴더 생성
os.makedirs(save_dir, exist_ok=True)

# 라벨 맵핑
dataset = load_dataset("klue", "ner")
label_list = dataset["train"].features["ner_tags"].feature.names
label2id = {label: i for i, label in enumerate(label_list)}

# 2. 필수 파일 준비

# Config 파일 저장 (모델 구조 복원을 위해 필수)
config = {
    "vocab_size": 32000, # tokenizer.vocab_size
    "embedding_dim": 768, # 사용한 값
    "hidden_dim": 256, # 사용한 값
    "tag_to_ix" : label2id, # 라벨 맵핑 정보
    "model_type": "BiLSTM_CRF" # 식별용
}

with open(f"{save_dir}/config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

# 학습된 모델이 메모리에 있다면:
# torch.save(model.state_dict(), f"{SAVE_DIR}/pytorch_model.bin") 
# *Tip: 허깅페이스는 보통 .pth 대신 pytorch_model.bin 이라는 이름을 씁니다.

# 모델 가중치 복사 혹은 저장
import shutil
shutil.copy("./models/klue_ner_bi_lstm_crf.pth", f"{save_dir}/klue_ner_bi_lstm_crf.bin")

# 토크나이저 저장
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
tokenizer.save_pretrained(save_dir)

# 3. 허깅페이스 허브에 업로드
print(f">>> Uploading to {repo_id}...")

try:

    #레포지토리 생성
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    # 폴더 내 모든 파일 업로드
    api = HfApi()
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    print(">>> 업로드 성공! 아래 주소에서 확인해보세요.")
    print(f"https://huggingface.co/{repo_id}")
except Exception as e:
    print(f">>> 업로드 실패: {e}")
```

실행해서 성공적으로 업로드가 되면 아래와 같은 결과가 뜨며, 자신의 저장소에 가보면 저장이 된 것을 확인할 수 있습니다.

```
Output:
>>> Uploading to Laseung/klue-ner-bi-lstm-crf...
Processing Files (1 / 1)      : 100%|█████████████████████████████████████████████████████████|  102MB /  102MB, 43.3MB/s  
New Data Upload               : |                                                             |  0.00B /  0.00B,  0.00B/s  
  .../klue_ner_bi_lstm_crf.bin: 100%|█████████████████████████████████████████████████████████|  102MB /  102MB            
>>> 업로드 성공! 아래 주소에서 확인해보세요.
```

<div align="center">
  <img src="/assets/images/toy_project/comparison_of_entity_recognition_performance_by_model/2/huggingface_upload_bi_lstm_crf.png" width="65%" height="40%"/>
</div>

# 6. 학습한 모델을 이용한 추론

이제 학습까지 완료했고, 허깅페이스에 업로드까지 완료 했습니다. 그렇다면 학습한 모델을 이용해 실제 서비스를 할 수 있는지 모델을 이용한 추론을 진행해 보고 추론 결과가 어떤지도 한 번 보도록 하겠습니다. 코드는 BERT 모델의 추론 코드와 비슷합니다. 다만 허깅페이스에서 지원하지 않는 Bi-LSTM-CRF 모델이라 config.json 파일을 이용한 모델의 뼈대를 구성해 주어야 합니다.

```python
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import torch.nn as nn
from model import BiLSTM_CRF

# 모델 로드 함수 (huggingface hub 에서 다운로드)
def load_model_from_hub(repo_id, device):
    print(f">>> HuggingFace Hub에서 모델 다운로드 중... {repo_id}")

    # 1. 파일 다운로드
    model_path = hf_hub_download(repo_id=repo_id, filename="klue_ner_bi_lstm_crf.bin")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

    # 2.config 로드
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 3. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    # 4. 모델 뼈대 생성
    # config에 저장된 tag_to_ix(label2id)를 가져옴
    tag_to_ix = config["tag_to_ix"]
    label2id = {int(v): k for k, v in tag_to_ix.items()}

    model = BiLSTM_CRF(
        vocab_size=config['vocab_size'],
        tag_to_ix=tag_to_ix,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    )

    # 5. 가중치(State Dict) 로드
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(">>> 모델 로드 완료!")
    return model, tokenizer, label2id

def predict_ner(text, model, tokenizer, label2id, device):
    # 전처리
    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=128
    )

    input_ids = inputs["input_ids"].to(device)

    # attention_mask는 0,1로 되어 있으므로 Boolean(True/False)로 변환
    mask = inputs["attention_mask"].bool().to(device)

    # Offset Mapping 추출 (후처리에 사용)
    offset_mapping = inputs["offset_mapping"][0].cpu().numpy()

    # 모델 추론
    with torch.no_grad():
        # returns List[List[int]]
        predictions = model.decode(input_ids, mask)
        pred_tags_ids = predictions[0]
    
    # 결과 매핑 (ID -> Label) & 특수 토큰 제외
    valid_tokens = []

    # input_ids[0]과 pred_tags_ids의 길이는 같습니다 (마스크 된 부분 제외)
    # 하지만 tokenizer는 [CLS], [SEP]을 포함하므로 인덱스를 주의해야 합니다.

    for idx, (offset, tag_id) in enumerate(zip(offset_mapping, pred_tags_ids)):
        start, end = offset

        # [CLS], [SEP] 등은 offset이 (0, 0)입니다. 이는 건너뜁니다.
        if start != end:
            label = label2id[tag_id]
            valid_tokens.append((start, end, label))
    return format_result(text, valid_tokens)

def format_result(original_text, valid_tokens):
    """
    BIO 태그를 파싱하여 <단어:태그>형태로 변환
    """

    result_text = ""
    processed_idx = 0

    current_entity = None
    entities = []

    for start, end, label in valid_tokens:
        if label.startswith("B-"):
            if current_entity: entities.append(current_entity)
            current_entity = {"start":start, "end":end, "label":label.split("-")[1]}
        elif label.startswith("I-"):
            if current_entity and label.split("-")[1] == current_entity['label']:
                current_entity['end'] = end
            else:
                if current_entity: entities.append(current_entity)
                current_entity = {"start":start, "end":end, "label":label.split("-")[1]}
        else: # 'O' 태그
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity: entities.append(current_entity)

    # 문자열 조립
    for entity in entities:
        start, end, label = entity['start'], entity['end'], entity['label']
        result_text += original_text[processed_idx:start]
        result_text += f"<{original_text[start:end]}:{label}>"
        processed_idx = end
    
    result_text += original_text[processed_idx:]
    return result_text

if __name__ == "__main__":
    # 설정
    repo_id = "본인의 허깅페이스 이름/klue-ner-bi-lstm-crf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model, tokenizer, label2id = load_model_from_hub(repo_id, device)

    # 테스트 문장
    test_sentences = [
        "특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5㎞ 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.",
        "손흥민은 1992년 7월 8일 춘천에서 태어났다.",
        "SK하이닉스는 경기도 이천에 본사를 두고 있다."
    ]

    print("\n" + "="*60)
    for text in test_sentences:
        result = predict_ner(text, model, tokenizer, label2id, device)
        print(f"입력: {text}")
        print(f"결과: {result}")
        print("-" * 60)
```

모델 추론 결과를 보면 예제 문장에서는 개체명 인식이 잘 되는 것을 확인할 수 있습니다.

```
Output:
============================================================
입력: 특히 영동고속도로 강릉 방향 문막휴게소에서 만종분기점까지 5㎞ 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.
결과: 특히 <영동고속도로:LC> <강릉:LC> 방향 <문막휴게소:LC>에서 <만종분기점:LC>까지 <5㎞:QT> 구간에는 승용차 전용 임시 갓길차로제를 운영하기로 했다.
------------------------------------------------------------
입력: 손흥민은 1992년 7월 8일 춘천에서 태어났다.
결과: <손흥민:PS>은 <1992년 7월 8일:DT> <춘천:LC>에서 태어났다.
------------------------------------------------------------
입력: SK하이닉스는 경기도 이천에 본사를 두고 있다.
결과: <SK하이닉스:OG>는 <경기도 이천:LC>에 본사를 두고 있다.
------------------------------------------------------------
```

# 7. 성능 향상을 위한 고찰

마지막으로 Bi-LSTM-CRF 모델을 이용한 개체명 인식기의 성능 향상 방법에 대한 고찰을 해보았습니다. BERT가 등장하기 전에 자연어 처리 분야에서는 Bi-LSTM-CRF 모델을 가장 많이 사용했고, 특히나 Bi-LSTM-CRF 모델의 성능 향상에는 사전 학습된 임베딩이 성능 향상에 가장 중요한 요소였습니다. 하지만 이번 실험에서는 사전 학습된 임베딩을 사용하지 않고 임베딩의 값이 랜덤한 값으로 정해지도록 하였습니다. 제가 대학원생 일 때에도 Word2Vec, GloVe, FastText 등 여러 임베딩 모델과 대용량 학습 데이터를 이용해 만든 임베딩을 이용해 실험을 진행해보았고, 그 중에서도 GloVe가 가장 성능이 좋았었습니다. 또한 이와 함께 BERT의 언어 모델 아이디어 기반이 되는 ELMo 모델을 이용해 추출한 임베딩을 기존 임베딩에 결합해 모델 학습을 했을 때 아주 큰 폭으로 성능 향상이 되었던 기억이 있습니다. 그래서 여태까지 진행한 실험에 초기 임베딩 값으로 랜덤 임베딩이 아닌 사전 학습된 임베딩을 사용했을 때 실제로 성능 향상이 이루어지는지 한 번 실험해 보도록 하겠습니다. 사용할 임베딩은 klue/bert-base 모델의 임베딩을 사용할 것이며, 허깅페이스에 등록된 transformers 기반 bert 모델들은 모두 임베딩을 제공합니다. 아래 코드를 학습 루프 이전 모델 정의하는 부분에 넣어 주도록 하겠습니다.

```python
from transformers import AutoModel

# 1. BERT 모델 로드 (가중치를 꺼내기 위함)
print(">>> BERT 임베딩 로드 중...")
bert_model = AutoModel.from_pretrained("klue/bert-base")

# 2. BERT의 임베딩 가중치 추출
# model.embeddings.word_embeddings.weight에 (32000, 768) 크기의 텐서가 있습니다.
bert_embedding_weight = bert_model.embeddings.word_embeddings.weight

# 3. BiLSTM-CRF 모델 초기화
# (vocab_size와 embedding_dim은 BERT와 맞춰줘야 합니다!)
vocab_size = tokenizer.vocab_size  # 32000
embedding_dim = 768                # roberta-base의 차원

model = BiLSTM_CRF(
    vocab_size=vocab_size,
    tag_to_ix=tag_to_ix,
    embedding_dim=embedding_dim,
    hidden_dim=256 # hidden_dim은 자유롭게 설정 가능 (보통 256 or 512)
)

# 4. 가중치 덮어씌우기 (핵심!)
# 우리 모델의 word_embeds에 BERT의 가중치를 복사합니다.
model.word_embeds.weight = nn.Parameter(bert_embedding_weight.clone())

# 5. (선택 사항) 임베딩 층 얼리기 vs 학습하기
# True로 설정하면 BERT가 배운 그대로 고정되고, False면 NER 데이터에 맞춰 미세 조정됩니다.
# 데이터가 충분한 KLUE NER의 경우 False(학습 허용)가 성능이 더 좋습니다.
model.word_embeds.weight.requires_grad = False 

print(">>> BERT 임베딩 이식 완료!")
```

간혹 실행하면 다음과 같은 문구가 뜹니다. 이 문구는 데이터셋을 준비하는 코드를 보시면 DataLoader에 `num_workers=4`로 되어 있는데 이 설정은 CPU가 데이터를 더 빨리 실어나르기 위해 프로세스를 여러 개 복제해서 사용하겠다는 것인데 허깅페이스의 Tokenizer도 기본적으로 속도를 높이기 위해 자체적인 병렬처리 기능을 가지고 있습니다. 하지만 이렇게 병렬처리를 하다보면 교착상태에 빠질 수 있어 아래와 같은 문구를 출력하면서 Tokenizer의 병렬처리를 끄겠다고 사용자에게 알려줍니다.

```
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...

To disable this warning, you can either:

        - Avoid using `tokenizers` before the fork if possible

        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
```

만약 위 문구를 보기 싫으시다면 아래 코드를 추가해 주시면 됩니다.

```python
import os

# 토크나이저의 병렬 처리를 강제로 끕니다. (Deadlock 방지 및 경고 제거)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 그 다음 다른 라이브러리들을 import 합니다.
import torch
from transformers import AutoTokenizer
# ...
```

그럼 이제 학습을 진행해 보고 실제로 성능이 향상되었는지 확인을 해보도록 하겠습니다. 랜덤 임베딩 값을 사용하지 않고 klue/bert-base 임베딩을 사용하였을 때 최고 F1-score 기준 83.30%으로 7.48%나 성능이 향상된 것을 확인할 수 있었습니다. 즉 사전 학습된 임베딩이 얼마나 중요한지 다시 한 번 알게 되었습니다.

```
Output:
==============================
 학습 완료 리포트
==============================
 - 최고(Best) F1-score : 0.8330
 - 전체 평균 F1-score  : 0.8094
 - 총 소요 시간        : 843.88s
```

찐막으로 `model.word_embeds.weight.requires_grad`의 값이 False일 경우 성능이 좀 더 높다고 하는데 혹시나 해서 True로 변경해서 학습을 진행해 보았습니다. 사실 사용하는 임베딩이 정말로 대용량의 학습 데이터를 이용해 잘 사전 학습된 것이 아니라면 임베딩 값도 같이 데이터에 맞게 학습하는 것이 일반적으로는 성능이 더 좋다고 알고 있었습니다. 혹시나 해서 임베딩은 학습을 하지 않도록 하고 학습을 진행했을 때 임베딩도 같이 학습하도록 한 것에 비해서는 성능이 랜덤 임베딩을 사용했을 때 보다는 좋았지만, 엄청나게 좋아지지는 않았습니다.

```
Output:
==============================
 학습 완료 리포트
==============================
 - 최고(Best) F1-score : 0.7811
 - 전체 평균 F1-score  : 0.7690
 - 총 소요 시간        : 843.71s
```

# 마치며

Bi-LSTM-CRF 모델을 이용한 개체명 인식기 구축을 진행해 보았습니다. BERT 모델의 tokenizer를 이용해 처리된 데이터로도 Bi-LSTM-CRF 모델의 학습이 가능하다는 것을 확인할 수 있었고, CRF 층에 적용하기 위해 어떤 처리를 해야하는지도 자세히 알 수 있었습니다. 특히 이번 실험으로 동일한 데이터에서 BERT 모델과 달리 학습 속도도 느리고, 성능도 BERT 모델과 차이가 많이 나는 것을 확인할 수 있어 많이 뜻 깊은 실험이었고, 이를 블로그로도 정리하게 되어 여러모로 많은 것을 또 배운 시간이었습니다.

긴 글 읽어주셔서 감사드리며, 본문 내용 중에 잘못된 내용이나 오타, 궁금하신 사항이 있으신 경우 댓글 달아주시기 바랍니다.