---
title: "[LLM/RAG] LLM Appliation - 2. sLLM 학습 시켜보기"
categories:
  - LLM/RAG

tags:
  - LLM/RAG
  
use_math: true  
toc: true
toc_sticky: true
toc_label: "sLLM 학습 시켜 보기"
---

Chat-GPT가 크게 히트를 치면서 인공지능을 전공 분야로 공부하지 않은 사람들도 인공지능에 관심을 가지게 되었고, 특히 Chat-GPT에 사용된 LLM 모델에 많은 사람들이 주목하고 있습니다. 저 또한 고전적인 사전과 규칙 기반의 자연어처리 기반 정보 추출기로 제품을 만들던 회사에서 LLM 시대에 뒤처지겠단 생각이 들어 회사를 나오게 되었고, 대학원생 때 했던 딥러닝 공부를 다시 시작하게 되면서 LLM 에 대해서 공부도 하게 되었습니다. 하지만 LLM 을 공부하면 할 수록 예전엔 그래도 몇 백만원만 들이면 시간은 좀 걸릴지언정 대부분의 딥러닝 모델을 구축하고 돌릴 수 있었지만 최근에 여러 글로벌 기업들이 출시하는 LLM 모델은 이제 개인이 이론적으로만 공부할 수 있고, 전혀 돌려볼 수 없다고 생각했습니다. 하지만 이런 문제는 저와 같은 개인뿐만 아니라 중소기업도 가지고 있는 문제 였습니다. 그러다 최근 대기업의 LLM 도 학습 데이터에 없는 요청을 받을 경우 엉뚱한 대답을 뱉거나, 대답하지 못하는 문제가 발생했고, 기업들은 굳이 대규모 LLM 보다는 오히려 사전 학습 데이터와 모델의 규모가 작지만 파인 튜닝용 데이터로 미세 조정을 해 도메인에 좀 더 잘 적응하는 sLLM 에 주목하고 있다는 것을 제가 참고하고 있는 책을 통해 알게 되었습니다. 그래서 이번 포스트는 그런 sLLM 에 대해서 알아보고 직접 학습과 추론까지 진행해 보고자 합니다.

# 개요

범용적인 성능이 뛰어난 상업용 LLM에 비해 비용 효율적이면서 특정 작업 또는 도메인에 특화된 sLLM이 주목받고 있습니다. 그래서 이번에는 자연어 요청으로부터 적합한 SQL을 생성하는 Text2SQL 태스크를 잘 하도록 하는 sLLM 을 직접 학습시켜 보도록 하겠습니다.

Text2SQL은 사용자가 얻고 싶은 데이터에 대한 요청을 자연어로 작성하면 LLM이 요청에 맞는 SQL을 생성하는 작업을 말합니다. 기업 내에서는 일반적으로 데이터 팀이나 팀 내에 있는 데이터 인력이 마케팅, 운영, 사업 등 여러 부서에서 알고 싶은 정보를 추출해서 전달하는 방식으로 데이터를 활용하고 있습니다. 이러한 Text2SQL 태스크는 SQL 생성을 LLM이 보조할 수 있다면 데이터 인력의 생산성을 높일 수 있을 뿐만 아니라 현업 구성원의 SQL 진입 장벽을 낮출 수 있다는 점에서 중요한 태스크라고 볼 수 있습니다.

우선 Text2SQL sLLM의 미세 조정에 사용할 수 있는 데이터셋을 살펴보고, 다음으로 모델이 잘 학습되고 있는지 평가할 때 사용할 평가 파이프라인을 만들어 보도록 하겠습니다. 마지막으로 준비한 학습 데이터로 sLLM의 미세 조정을 수행해 기초 모델과 학습한 모델간의 차이가 얼마나 나는지 비교해 보도록 하겠습니다.

# 1. Text2SQL 데이터셋

Text2SQL 작업을 위해 구축된 대표적인 데이터셋을 살펴보도록 하겠습니다. 먼저 대표적인 영어 데이터셋과 그 형태를 살펴보고, 사용할 수 있는 한국어 데이터셋이 있는지 알아보도록 하겠습니다. 그리고 사용할 수 있는 한국어 데이터셋을 이용해 학습을 진행해 보고, "LLM 을 활용한 실전 AI 어플리케이션 개발" 의 저자가 GPT 를 이용해 만든 합성 데이터를 이용한 학습도 진행해 보도록 하겠습니다.

## 1.1 대표적인 Text2SQL 데이터셋

대표적인 Text2SQL 데이터셋으로는 WikiSQL과 Spider가 있습니다. SQL을 생성하기 위해서는 크게 두 가지 데이터가 필요합니다. 먼저, 어떤 데이터가 있는지 알 수 있는 데이터베이스 정보(테이블과 컬럼)가 필요합니다. 다음으로 어떤 데이터를 추출하고 싶은지 나타낸 요청사항(request 또는 question)이 필요합니다. WikiSQL 데이터를 예시로 살펴보면 아래와 같습니다. 테이블(table) 항목에 테이블 이름, 컬럼 이름(header), 컬럼 형식(types)이 있고 요청(question) 항목으로 어떤 데이터가 필요한지 요청사항이 적혀 있습니다. 마지막으로 SQL 항목에 정답 SQL 데이터가 있습니다.

```json
{
    "phase": 1,
    "question": "How would you answer a second test question?",
    "sql": {
        "agg": 0,
        "conds": {
            "column_index": [2],
            "condition": ["Some Entity"],
            "operator_index": [0]
        },
        "human_readable": "SELECT Header1 FROM table WHERE Another Header = Some Entity",
        "sel": 0
    },
    "table": "{\"caption\": \"L\", \"header\": [\"Header1\", \"Header 2\", \"Another Header\"], \"id\": \"1-10015132-9\", \"name\": \"table_10015132_11\", \"page_i..."
}

```

WikiSQL은 하나의 테이블만 사용하고 SELECT 문에 컬럼을 1개만 사용하거나 WHERE절에 최대 3개의 조건만 사용하는 등 비교적 쉬운 SQL 문으로 구성돼 있습니다.

Spider는 좀 더 현실적인 문제 해결을 위해 구축된 데이터셋으로 ORDER BY, GROUP BY, HAVING, JOIN 등 비교적 복잡한 SQL 문도 포함하고 있습니다. 최근에는 한국어로 번역된 Spider-KO 가 출시되었습니다. 

## 1.2 한국어 데이터셋

한국어 Text2SQL 데이터셋으로는 좀 전에 말했던 Spider-KO 가 있습니다. Spider-KO는 Yale University의 Spider 데이터셋을 한국어로 번역한 Text2SQL 변환 데이터셋입니다. 원본 Spider 데이터셋의 자연어 질문을 한국어로 번역해 구성하였습니다. 번역 과정은 LLM(Claude 3.5 Sonnet) 기반 초기 번역을 수행 후 LLM(Gemini 2.0 Flash + Claude 3.5 Sonnet) 활용 자동 검증 및 개선 프로세스 적용 후 인간 전문가에 의해 최종 검수를 진행했다고 합니다. 아래는 Spider-KO 데이터셋의 예시입니다.

```json
[
  {
    "db_id": "college_1",
    "query": "SELECT count(*) FROM student WHERE age  >  20",
    "question": "나이가 20세가 넘는 학생 수를 세어주세요.",
    "query_toks": ["SELECT", "count", "(", "*", ")", "FROM", "student", "WHERE", "age", ">", "20"],
    "question_toks": ["나이", "가", "20", "세", "가", "넘", "는", "학생", "수", "를", "세어", "주세요", "."],
    "sql": {...} // 구조화된 SQL 구문 정보 (필요시)
  },
  // ... 다른 데이터 쌍
]
```

Spider-KO 다음으로 AI Hub 에서 구축하고 제공하는 NL2SQL 데이터셋이 있습니다. NL2SQL 데이터셋은 데이터베이스에 대해 데이터를 검색하는 자연어 질문과 그와 의미가 동일한 SQL 질의의 쌍으로 구성된 데이터셋으로, 공공기관 데이터 플랫폼에서 수집한 데이터베이스를 활용하여 다양한 분야의 자연어 질문을 SQL 질의로 변환할 수 있는 NL2SQL 모델 개발을 위한 데이터셋입니다. 아래는 NL2SQL 데이터셋의 예시입니다.

```json
	"data": [
		{
			"db_id": "seouldata_healthcare_733",
			"utterance_id": "Whr_3005",
			"hardness": "easy",
			"utterance_type": "BR03",
			"query": "SELECT DUTYADDR FROM TB_PHARMACY_OPERATE_INFO WHERE DUTYNAME = '3층메디칼약국'",
			"utterance": "3층메디칼약국의 주소를 알려줘",
			"values": [
				{
					"token": "3층메디칼약국",
					"start": 0,
					"column_index": 3
				}
			],
			"cols": [
				{
					"token": "주소",
					"start": 9,
					"column_index": 2
				}
			]
		},
    // ... 다른 데이터 쌍
```

## 1.3 합성 데이터 활용

이 포스트는 "LLM을 활용한 실전 AI 어플리케이션 개발" 책을 참조하였고, 해당 책에서 알려주는대로 진행하기 위해 책 저자가 GPT-3.5와 GPT-4를 활용해 생성한 데이터를 사용했습니다(https://huggingface.co/datasets/shangrilar/ko_text2sql).

# 2. 성능 평가 파이프라인 준비하기

## 2.1 Text2SQL 평가 방식

Text2SQL 평가에 사용되는 방식은 생성한 SQL이 문자열 그대로 동일한지 확인하는 EM(Exact Match) 방식과 쿼리를 수행할 수 있는 데이터베이스를 만들고, 프로그래밍 방식으로 SQL 쿼리를 수행해 정답과 일치하는지 확인하는 실행 정확도(Execution Accuracy, EX) 방식이 있습니다. 하지만 EM 방식은 의미상 동일한 SQL 쿼리가 다양하게 나올 수 있는데 문자열이 동일하지 않으면 다르다고 판단한다는 문제가 있고, 실행 정확도 방식은 쿼리를 실행할 수 있는 데이터베이스를 추가로 준비해야 하기 때문에 실제로 Text2SQL 모델이 사용되는 실무 환경이 아닌 간단한 실습 환경에서는 제약이 큽니다.

최근에는 LLM을 이용해 생성 결과를 평가하는 방식이 활발히 연구되고 있습니다. Spider-KO 의 번역도 검수만 사람이 했을뿐 번역과 중간 검토는 LLM을 이용했습니다. 이번 sLLM 실습에서는 GPT-4를 사용해 sLLM이 생성한 결과 평가를 진행합니다.

GPT를 활용한 성능 평가 파이프라인을 준비하기 위해서는 세 가지가 필요합니다. 첫째 평가 데이터셋을 구축해야 합니다. 다음으로 LLM이 SQL을 생성할 때 사용할 프롬프트를 준비합니다. 마지막으로 GPT 평가에 사용할 프롬프트와 GPT-4 API 요청을 빠르게 수행할 수 있는 코드를 작성합니다.

## 2.2 평가 데이터셋 

평가 데이터셋은 포스트를 작성하는데 참조한 "LLM을 활용한 실전 AI 어플리케이션 개발" 책의 저자가 구축한 데이터를 활용했습니다. 이 데이터는 8개 데이터베이스에 대해 생성했으며, 모델의 일반화 성능을 확인하기 위해 7개의 데이터베이스 데이터는 학습에 사용하고, 1개의 데이터 베이스는 평가에 사용했다고 합니다. 이 때 평가에 사용한 데이터베이스는 게임 도메인을 가정하고 만든 데이터베이스이며, 게임 도메인의 경우 테이블 이름이 다른 도메인과 달리 플레이어(players), 퀘스트(quests), 장비(equipments)와 같이 특화된 이름을 사용해서 평가 데이터로 사용했다고 합니다.

## 2.3 라이브러리 설치

참고한 책이 2024년에 출판된 책이라 당시 라이브러리들과 코랩의 버전이 포스트를 작성하는 현재(2025-11-13) 시점과의 버전 차이가 있어 책에서 안내하는 라이브러리를 이용하면 에러가 발생할 수 있습니다. 제가 주로 겪은 에러는 "No module named 'triton.ops'" 에러로 코랩에 설치된 triton과 "bitsandbytes"에서 요구하는 triton과 버전 차이로 인해 발생하는 오류였습니다. 언뜻 보기엔 그냥 triton을 재설치하면 해결되는 문제가 아닌가 하지만 코랩에 설치된 torch와 torchvision 때문에 triton을 재설치해도 코랩에 적용된 torch와 torchvision 때문에 triton을 언인스톨하고 다시 설치를 해봐도 코랩에 적용 중인 torch 버전에 맞는 triton이 강제로 설치되기 때문에 다른 해결 방법이 필요했습니다.

문제를 해결하기 위해 찾아보니 코랩에 설치된 torch, torchvision 을 삭제하고 우리가 돌리고자하는 코드에 맞는 버전을 새로 설치하고, 필요한 라이브러리들을 모두 다 재설치 해야 한다고 해서 다음과 같이 진행해 보았습니다. 우선 기존 설치된 라이브러리와 torch, torchvision까지 모두 언인스톨을 진행했습니다. 그리고 우리가 사용하고자 하는 라이브러리(bitsandbytes)에서 사용하는 triton에 맞는 torch와 torchvision 설치를 진행합니다. 그리고 우리가 사용하고자 하는 라이브러리 설치를 진행합니다.

```python
# 이전에 설치했던 라이브러리들과 코랩에 설치되어 있는 torch 와 torchvision 삭제를 진행
!pip uninstall -y torch torchvision torchaudio bitsandbytes triton transformers accelerate
```

```python
# 현재 사용하고자 하는 라이브러리의 triton 버전에 맞는 torch와 torchvision을 직접 설치 진행
!pip install --no-cache-dir torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
```

```python
# 우리가 사용하고자 하는 라이브러리들 설치를 진행
!pip install --no-cache-dir \
  "transformers==4.48.3" \
  "accelerate>=1.2.0" \
  "bitsandbytes>=0.45.5" \
  "triton==3.1.0" \
  "peft>=0.13.0" \
  "datasets>=2.20.0" \
  "tiktoken"

!pip install --no-cache-dir -U autotrain-advanced
```

저는 위와 같이 진행 후에 예제 코드들 정상적으로 실행되었습니다. 아마도 딥러닝 공부하시는 분들은 자주 겪으시겠지만 딥러닝 모델, 라이브러리 등의 발전이 매우 빨라 버전 호환이 안되는 경우가 잦아 이러한 문제가 많이 발생합니다. 일단 저는 위와 같은 방법으로 해결을 했지만 위와 같이 진행해도 안될 가능성이 큽니다. 그럴 경우에는 다른 방법들도 한 번 찾아보시는 것을 추천드리며, 에러가 발생했을 때 에러를 어떻게 해결했는지도 잘 정리해 놓으면 추후에 많은 도움이 되니 꼭 시간 내서 정리를 해놓으시길 바랍니다.

## 2.4 SQL 생성 프롬프트

LLM이 SQL을 생성하도록 하기 위해서는 지시사항과 데이터를 포함한 프롬프트(Prompt)를 준비해야 합니다.

프롬프트는 먼저 SQL을 생성하라는 요청을 작성하고 필요한 데이터(DDL과 Question)을 입력한 후 정답에 해당하는 SQL을 마지막에 채워 넣습니다. 학습 데이터에서는 정답이 채워진 형태로 사용하고 SQL을 생성할 때는 query를 입력하지 않아 기본값인 빈 문자열이 들어가도록 합니다.

```python
# LLM 에 사용할 prompt 를 만드는 함수
def make_prompt(ddl, question, query=''):
    prompt = f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.
    ### DDL:
    {ddl}

    ### Question:
    {question}

    ### SQL:
    {query}
    """
    return prompt
```

학습에 사용할 데이터 예시는 다음과 같습니다.

```python
"""당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

### DDL:
CREATE TABLE players (
  player_id INT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(255) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  date_joined DATETIME NOT NULL,
  last_login DATETIME
);

### Question:
사용자 이름에 'admin'이 포함되어 있는 계정의 수를 알려주세요.

### SQL:
"""
```

## 2.5 GPT-4 평가 프롬프트와 코드 준비

GPT-4를 사용해 평가를 수행한다면 반복적으로 GPT-4에 API 요청을 보내야 합니다. 평가 시에는 112개의 평가 데이터를 사용하므로 for문을 통해 반복적인 요청을 수행해도 시간이 오래 걸리지 않습니다. 하지만 데이터셋을 더 늘린다면 단순 for문으로는 시간이 오래 걸리므로 이럴 때는 OpenAI가 openai-cookbook 깃허브 저장소에서 제공하는 코드(<https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py>)를 활용하면 요청 제한을 관리하면서 비동기적으로 요청을 보낼 수 있습니다. 이 코드는 요청을 보낼 내용을 저장한 jsonl 파일을 읽어 순차적으로 요청을 보냅니다. 중간에 에러가 발생하거나 요청 제한에 걸리면 다시 요청을 보내서 결과의 누락도 막아줍니다.

아래 예제는 평가 데이터셋에서 요청 jsonl 파일을 생성하는 예제입니다. 이 코드에서는 입력한 데이터프레임을 순회하면서 평가에 사용할 프롬프트를 생성하고 jsonl 파일에 요청할 내용을 기록합니다. 프롬프트에서는 DDL과 Question을 바탕으로 LLM이 생성한 SQL이 정답 SQL과 동일한 기능을 하는지 평가하도록 했습니다. 판단 결과는 resolve_yn이라는 키에 "yes"또는 "no"의 텍스트가 있는 JSON 형식으로 반환하도록 했습니다.

```python
import json
import pandas as pd
from pathlib import Path

# 평가를 위한 요청 jsonl 작성 함수
def make_requests_for_gpt_evaluation(df, filename, dir='requests'):
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True)
    prompts = []
    for idx, row in df.iterrows():
        prompts.append("""Based on below DDL and Question, evaluate gen_sql can resolve Question. If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}""" + f"""

        DDL: {row['context']}
        Question: {row['question']}
        gt_sql: {row['answer']}
        gen_sql: {row['gen_sql']}""")

        jobs = [{"model":"gpt-4-turbo-preview", "response_format":{"type":"json_object"}, "messages":[{"role":"system", "content": prompt}]} for prompt in prompts]

        with open(Path(dir, filename), "w") as f:
            for job in jobs:
                json_string = json.dumps(job)
                f.write(json_string+"\n")
```

다음으로 jsonl 파일에 있는 내용을 csv파일로 변환하는 함수를 정의합니다. 밑에서 다루겠지만, GPT-4 모델에 평가를 하도록 하기 위해 만든 jsonl 파일을 OpenAI API에 요청하면 결과값을 jsonl 파일에 담아서 보내줍니다. 이때 jsonl 파일에 있는 내용을 좀 더 읽기 쉽게 csv파일로 변환해서 필요한 열만 읽어오도록 합니다.

```python
def change_jsonl_to_csv(input_file, output_file, prompt_column="prompt", response_column="response"):
    prompts = []
    responses = []
    with open(input_file, 'r') as json_file:
        for data in json_file:
            prompts.append(json.loads(data)[0]['messages'][0]['content'])
            responses.append(json.loads(data)[1]['choices'][0]['message']['content'])
    df = pd.DataFrame({prompt_column: prompts, response_column: responses})
    df.to_csv(output_file, index=False)
    return df
```

# 3. 평가와 미세 조정(학습) 수행하기

이제 실습에 사용할 기초 모델을 선택하고, 앞에서 준비한 평가 데이터로 평가를 진행하고, 학습 데이터로는 미세 조정을 수행합니다. 평가와 학습 예제에 사용하는 모델은 beomi/Yi-Ko-6B 모델을 사용합니다. 이 모델은 중국의 01.AI가 발표한 영어-중국어 모델인 01-ai/Yi-6B를 한국어에 확장한 모델입니다. 현재 한국어 LLM 리더보드에 등록된 모델들의 경우 상위 모델 대부분이 파인튜닝 난이도가 매우 높거나(모델 크기), GPU 메모리 요구량이 커서 코랩에서 제공하는 A100으로도 학습이 불가능한 모델이 많습니다. 또한 기업/기관용 모델은 weights 공개가 되어 있지 않거나, 한국어 LLM 모델의 경우 태스크별 성능 편차가 심합니다. 하지만 beomi/Yi-Ko-6B 모델은 성능/안전성/속도 모두에서 균형이 잡혀있고, 개인 사용자도 A100을 1시간 정도 사용해서 학습을 시킬 수 있습니다. 다만 다른 상위 모델에 비해선 모델의 크기도 작고 아주 높은 성능을 내기 어렵다는 단점이 있습니다만 우리는 실전에 사용하기 보단 공부를 위한 것이기 때문에 해당 모델을 사용하기로 하였습니다.

## 3.1 기초 모델로 평가하기

먼저 예시 데이터를 입력했을 때 미세 조정을 하지 않은 기초 모델이 어떤 결과를 생성하는지 확인을 해보도록 하겠습니다. 아래 코드를 실행하면 기초 모델을 불러와 프롬프트에 대한 결과를 생성합니다. 

```python
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def make_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # BitsAndBytesConfig 객체를 생성하여 전달
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", # 최신 NF4 양자화 타입 사용
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


model_id = "beomi/Yi-Ko-6B"
hf_pipe = make_inference_pipeline(model_id)

example = """당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

### DDL:
CREATE TABLE players (
  player_id INT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(255) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  date_joined DATETIME NOT NULL,
  last_login DATETIME
);

### Question:
사용자 이름에 'admin'이 포함되어 있는 계정의 수를 알려주세요.

### SQL:
"""

#result = hf_pipe(example, do_sample=False, return_full_text=False, max_length=256, truncation=True)
result = hf_pipe(example, do_sample=False, return_full_text=False, max_new_tokens=256, truncation=True)
print(result)
```

생성된 결과를 보면 SQL 생성은 잘 된 것을 확인할 수 있지만, 그 뒤에 "SQL 봇은"으로 시작하는 추가 답변이 있는 것을 볼 수 있습니다. 이를 통해 기초 모델도 요청에 따라 SQL을 생성할 수 있는 능력이 있지만 형식에 맞춰 답변하도록 하기 위해서는 추가적인 학습이 필요하다는 것을 확인할 수 있습니다.

```
Output:
[{'generated_text': "SELECT COUNT(*) FROM players WHERE username LIKE '%admin%';\n\n### SQL 봇의 장점:\n- SQL 봇은 SQL을 생성하기 때문에, SQL 봇을 사용하면 SQL을 생성하는 시간을 절약할 수 있습니다.\n- SQL 봇은 SQL을 생성하기 때문에, SQL 봇을 사용하면 SQL을 생성하는 시간을 절약할 수 있습니다.\n- SQL 봇은 SQL을 생성하기 때문에, SQL 봇을 사용하면 SQL을 생성하는 시간을 절약할 수 있습니다.\n- SQL 봇은 SQL을 생성하기 때문에, SQL 봇을 사용하면 SQL을 생성하는 시간을 절약할 수 있습니다.\n- SQL 봇은 SQL을 생성하기 때문에, SQL 봇을 사용하면 SQL을 생성하는 시간을 절약할 수 있습니다.\n- SQL 봇은 SQL을 생성하기 때문에, SQL 봇을 사용하면 SQL을 생성하는 시간을 절약할 수 있습니다.\n- SQL 봇은 SQL을 생성하기 때문에, SQL 봇을 사용하면 SQL을 생성하는 시간을 절약할 수 있습니다.\n- SQL 봇은 SQL을 생성하기 때문에, SQL 봇"}]
```

그럼 이제 기초 모델과 평가 데이터를 이용해 평가를 진행해 보도록 하겠습니다. load_datasets을 사용해 허깅페이스에서 데이터셋을 내려받고, make_promprt 함수를 사용해 LLM 추론에 사용할 프롬프트를 생성합니다. 다음으로 모델과 토크나이저를 이용해 만든 pipeline 변수에 생성한 프롬프트를 입력으로 주어 SQL을 생성하고, make_requests_for_gpt_evaluation 함수를 사용해 평가에 사용할 jsonl 파일을 만들고 GPT-4 API에 평가 요청을 전달합니다. 그리고 작아도 LLM 모델인지라 평가 데이터를 이용해 SQL을 생성할 때에도 생각보다 시간이 걸립니다. 코랩 환경 기준 T4 GPU를 사용하면 40분 정도 걸리고, A100을 사용하면 15~20분 정도 걸렸습니다. 그러니 아래 코드를 실행했는데 아무것도 안뜨고 계속 실행만 된다고 해서 멈추지 마시고 일단 계속 기다려 보시기 바랍니다.

```python
# 기초 모델의 평가를 위한 jsonl 생성

# sql 생성
gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False, return_full_text=False, max_new_tokens=256, truncation=True)
#gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False, return_full_text=False, max_new_tokens=1024, truncation=True)
gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
print(gen_sqls)
df['gen_sql'] = gen_sqls

# 평가를 위한 requests.jsonl 생성
base_eval_filepath = "/content/drive/MyDrive/LLM_RAG_Application/sLLM/requests/text2sql_evaluation_base.jsonl"
make_requests_for_gpt_evaluation(df, base_eval_filepath)
```

생성한 json 파일은 파라미터로 지정한 {dir}/{filename} 위치에 저장됩니다. 그리고 저장한 파일과 OpenAI API 요청 코드를 코랩에서 실행하도록 해서 평가를 진행합니다. 저는 평가에 gpt-4-preview 모델을 사용했습니다.

```python
# GPT-4-preview 모델을 이용해 평가 진행

base_result_filepath = "/content/drive/MyDrive/LLM_RAG_Application/sLLM/results/text2sql_evaluation_result_base.jsonl"

!python "/content/drive/MyDrive/LLM_RAG_Application/sLLM/api_request_parallel_processor.py" \
--requests_filepath {base_eval_filepath}  \
--save_filepath {base_result_filepath} \
--request_url https://api.openai.com/v1/chat/completions \
--max_requests_per_minute 2500 \
--max_tokens_per_minute 100000 \
--token_encoding_name cl100k_base \
--max_attempts 5 \
--logging_level 20
```

```
Output:
INFO:root:Starting request #0
INFO:root:Starting request #1
INFO:root:Starting request #2
... 생략 ...
INFO:root:Starting request #111
INFO:root:Parallel processing complete. Results saved to /content/drive/MyDrive/LLM_RAG_Application/sLLM/results/text2sql_evaluation_result_base.jsonl
```

그리고 코랩의 특성상 세션이 종료되면 그 세션에 생성되었던 파일도 삭제가 됩니다. 그러므로 이렇게 파일을 이용해 작업을 할 때에는 되도록이면 구글 드라이브 마운트를 진행해서 하시길 바랍니다. 거기다 이번 예제는 평가를 진행할 때 OpenAI의 API를 사용하는 것이기 때문에 조금이지만 그래도 돈을 쓰게 됩니다. 그러므로 본인이 돈이 많으신 분들은 귀찮은데 그냥하지 뭐 하시는 분들을 제외하고 구글 드라이브에 저장해서 최대한 OpenAI API 콜을 줄이시길 바랍니다. 제가 예제 코드를 실행했을 때 평가 한 번 하면 보통 0.5$ 정도 쓰는 것 같습니다. 참고하시길 바랍니다.

마지막으로 GPT-4로부터 받은 평가 결과 데이터에서 기초 모델이 생성한 대답을 "yes"로 판단한 개수를 세어 정확도로 계산을 진행해 기초 모델의 평가를 진행해 봅니다.

```python
# 미세 조정을 하지 않은 기본 모델로 성능 평가 진행

base_result_csv_filepath = "/content/drive/MyDrive/LLM_RAG_Application/sLLM/results/base_yi_ko_6b_eval.csv"

base_eval = change_jsonl_to_csv(base_result_filepath, base_result_csv_filepath, "prompt", "resolve_yn")
base_eval['resolve_yn'] = base_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]
print(f"맞은 개수 : {num_correct_answers}, 성능 : {num_correct_answers/112*100:.2f}%")
```

총 112개의 데이터 중에서 기초 모델이 생성한 SQL 중에서 GPT-4 는 10개만 정답이라고 응답을 보내주었고, 정확도로 백분율로 나타내 보면 8.93% 정도의 정확도를 보입니다. 일단 생각보다 굉장히 낮은 정확도를 보이고 있습니다.

```
Output:
맞은 개수 : 10, 성능 : 8.93%
```

## 3.2 미세 조정 수행

이제 학습 데이터로 미세 조정을 수행해서 실제로 미세 조정을 진행하고 평가를 진행했을 때 성능이 오르는지 확인을 해볼 차례입니다. 미세 조정에는 autotrain-advanced 라이브러리를 사용합니다. autotrain-advanced 라이브러리는 허깅페이스에서 trl 라이브러리를 한 번 더 추상화해 개발한 라이브러리입니다. 이 라이브러리에 대해서는 추후에 따로 좀 더 자세히 다뤄 보도록 하겠습니다. 우선 학습 데이터도 이전의 평가 데이터와 마찬가지로 프롬프트 형식으로 바꿔줍니다.

```python
# 학습 데이터 불러오기
from datasets import load_dataset

df_sql = load_dataset("shangrilar/ko_text2sql", "origin")["train"]
df_sql = df_sql.to_pandas()
df_sql = df_sql.dropna().sample(frac=1, random_state=42)
df_sql = df_sql.query("db_id != 1")

for idx, row in df_sql.iterrows():
    df_sql.loc[idx, 'text'] = make_prompt(row['context'], row['question'], row['answer'])

!mkdir data
df_sql.to_csv('data/train.csv', index=False)
```

이제 autotrain-advanced 라이브러리를 사용해 지도 미세 조정을 수행합니다. 설정인자에 대한 자세한 사항은 autotrain 라이브러리의 LLM 미세 조정 코드 링크(<https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/cli/run_llm.py>)에서 확인할 수 있습니다.

```python
# 미세 조정 명령어
base_model = 'beomi/Yi-Ko-6B'
finetuned_model = 'yi-ko-6b-text2sql'

!autotrain llm \
--train \
--model {base_model} \
--project-name {finetuned_model} \
--data-path data/ \
--text-column text \
--lr 2e-4 \
--batch-size 8 \
--epochs 1 \
--block-size 1024 \
--warmup-ratio 0.1 \
--lora-r 16 \
--lora-alpha 32 \
--lora-dropout 0.05 \
--weight-decay 0.01 \
--gradient-accumulation 8 \
--mixed-precision fp16 \
--peft \
--quantization int4 \
--trainer sft
```

모델 학습 과정에서 메모리 에러가 발생할 수 있습니다 이럴 경우 batch_size를 줄여서 다시 시도해 보시길 바랍니다. 그리고 모델 학습에는 구글 코랩의 A100 GPU 기준으로 저는 1시간 15분 정도 소요되었습니다. 무료로 사용할 수 있는 T4 GPU를 사용하는 경우 A100 대비 약 8~10배의 시간이 소요되니 학습에는 웬만하면 A100을 사용하시기 바랍니다. 학습을 마친 후에는 LoRa 어댑터와 기초 모델을 합치고 허깅페이스 허브에 모델을 저장합니다. 일단 무조건 허깅페이스 허브에 모델을 저장하시는걸 추천드립니다. 학습 시간이 길다보니 돌려 놓고 다른걸 하다가 세션이 끊기면 다시 학습을 해야 하므로 무조건 저장해 두시길 바랍니다.

```python
# LoRA 어댑터 결합 및 허깅페이스 허브 업로드
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel

model_name = base_model
device_map = {"":0}

# LoRA와 기초 모델 파라미터 합치기
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

model = PeftModel.from_pretrained(base_model, finetuned_model)
model = model.merge_and_unload()

# 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

```python
# 허깅페이스 허브에 모델 및 토크나이저 저장
from huggingface_hub import login

login(token="본인의 허깅페이스 토큰")
repo_id = f"본인의 허깅페이스 계정 이름/"+finetuned_model
model.push_to_hub(repo_id, use_temp_dir=False)
tokenizer.push_to_hub(repo_id, use_temp_dir=False)
```

미세 조정을 진행한 모델로 이전에 기초 모델에 넣었던 예제 데이터를 넣어 보도록 하겠습니다.

```python
# 미세 조정한 모델로 예시 데이터에 대한 SQL 생성

model_id = "본인의 허깅페이스 계정 이름/yi-ko-6b-text2sql"
hf_pipe = make_inference_pipeline(model_id)

example = """당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

### DDL:
CREATE TABLE players (
  player_id INT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(255) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  date_joined DATETIME NOT NULL,
  last_login DATETIME
);

### Question:
사용자 이름에 'admin'이 포함되어 있는 계정의 수를 알려주세요.

### SQL:
"""

result = hf_pipe(example, do_sample=False, return_full_text=False, max_new_tokens=256, truncation=True)
print(result)
```

생성한 결과를 보면 기초 모델과는 달리 뒤에 추가적인 답변 없이 SQL만 생성해 주는 것을 확인할 수 있습니다.

```
Output:
[{'generated_text': "SELECT COUNT(*) FROM players WHERE username LIKE '%admin%';\n"}]
```

그럼 마지막으로 기초 모델로 진행했던 평가 데이터로 평가를 진행해 보도록 하겠습니다.

```python
# 미세 조정한 모델로 성능 측정

# sql 생성
gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False, return_full_text=False, max_new_tokens=256, truncation=True)
gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
df['gen_sql'] = gen_sqls

# 평가를 위한 requests.jsonl 생성
eval_filepath = "/content/drive/MyDrive/LLM_RAG_Application/sLLM/requests/text2sql_evaluation_finetuned.jsonl"
make_requests_for_gpt_evaluation(df, eval_filepath)
result_filepath = "/content/drive/MyDrive/LLM_RAG_Application/sLLM/results/text2sql_evaluation_result_finetuned.jsonl"
```

```python
# GPT-4 평가 수행
!python "/content/drive/MyDrive/LLM_RAG_Application/sLLM/api_request_parallel_processor.py" \
--requests_filepath {eval_filepath}  \
--save_filepath {result_filepath} \
--request_url https://api.openai.com/v1/chat/completions \
--max_requests_per_minute 2500 \
--max_tokens_per_minute 100000 \
--token_encoding_name cl100k_base \
--max_attempts 5 \
--logging_level 20
```

```python
# 미세 조정을 하지 않은 기본 모델로 성능 평가 진행

result_filepath = "/content/drive/MyDrive/LLM_RAG_Application/sLLM/results/text2sql_evaluation_result_finetuned.jsonl"

result_csv_filepath = "/content/drive/MyDrive/LLM_RAG_Application/sLLM/results/finetuned_yi_ko_6b_eval.csv"

finetuned_eval = change_jsonl_to_csv(result_filepath, result_csv_filepath, "prompt", "resolve_yn")
finetuned_eval['resolve_yn'] = finetuned_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
num_correct_answers = finetuned_eval.query("resolve_yn == 'yes'").shape[0]
print(f"맞은 개수 : {num_correct_answers}, 성능 : {num_correct_answers/112*100:.2f}%")
```

미세 조정한 모델의 성능을 보면 맞은 개수 62개로 정확도 55.36%로 대략 6배 가량 성능이 향상된 것을 확인할 수 있습니다. 

```
Output:
맞은 개수 : 62, 성능 : 55.36%
```

# 정리

이번엔 Text2SQL 작업을 위한 sLLM을 학습시키는 아주 기본적인 실습을 진행했습니다. 평가 데이터와 gpt-4 api 를 이용해 평가를 진행해 보았으며, 학습 데이터를 이용해 sLLM의 미세 조정까지 진행해 보았습니다. 그리고 최종적으로 미세 조정을 했을 때와 하지 않았을 때 우리가 원하는 태스크에서 아주 극명하게 성능 차이가 나는 것을 확인할 수 있었습니다.

# 마치며

이번 포스트를 정리하면서 사전 학습은 아니지만 그래도 실제 LLM 모델을 직접 미세 조정까지 진행해 성능 평가까지 진행하여 LLM 모델을 다룬 경험을 쌓았고, LLM에 있어 미세 조정이 얼마나 중요한지도 알 수 있었습니다. 다만 이번 포스트에서 사용한 데이터는 사람이 만든 데이터가 아닌 LLM을 이용해 만든 합성 데이터를 이용한 데이터였고, 실제 사람이 만든 데이터를 이용해 학습하면 어떨지도 진행해 보고 싶었으나 시간 여유가 없어 진행해 보지 못했습니다. 우선 제가 목표로 하는 과정들과 목표 일정이 있기 때문에 이 일정들을 마무리 한 후에 바로 데이터 소개에서 말씀드렸던 Spider-Ko 데이터와 AI Hub 에서 배포한 NL2SQL 데이터를 이용한 미세 조정도 진행해서 성능이 어떻게 달라지는지 한 번 비교해 보도록 하겠습니다. 또한 학습 데이터를 보고 학습 데이터를 정제한 데이터를 이용했을 때와 기초 모델을 바꾸었을 때의 성능 측정도 진행해 보도록 하겠습니다.

긴 글 읽어주셔서 감사드리며, 본문 내용 중에 잘못된 내용이나 오타, 궁금하신 것이 있을 경우 댓글 달아주시 바랍니다.

# 참조

- 허정준 저, LLM 을 활용한 실전 AI 어플리케이션 개발
- [허깅페이스 Spider-KO](https://huggingface.co/datasets/huggingface-KREW/spider-ko)
- [AI hub NL2SQL](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=NL2SQL&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71351)
