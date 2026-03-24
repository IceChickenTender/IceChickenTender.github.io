---
title: "[ToyProject] 블로그에 RAG를 적용한 AI 검색 기능 만들기 - 2. RAG 시스템 평가 방법 및 평가 시스템 구축"
categories:
  - ToyProject
tags:
  - ToyProject

use_math: true
toc: true
toc_sticky: true
toc_label: "RAG 시스템 평가 방법 및 평가 시스템 구축"
---

# 1. 개요

저번 포스트에서는 아주 간단한 Naive RAG 시스템을 만들었고, 이를 블로그에 적용하는 내용을 다루었습니다. 사실 RAG 시스템의 기본적인 틀은 "문서를 이용한 벡터 DB 구축 -> 입력 쿼리를 이용해 벡터 DB로 부터 문서 검색 -> 입력 쿼리와 문서 검색을 LLM에게 같이 전달 -> LLM으로부터 받은 답변 출력" 입니다. RAG를 공부해 보시면 아시겠지만 RAG의 기본적인 틀 자체는 매우 간단합니다. 문제는 RAG 시스템의 성능을 어떻게 끌어 올리느냐 입니다. 사실 이러한 모든 모듈의 문제는 성능의 고도화이며, 이를 위해선 정확한 성능을 측정하는 방법과 시스템을 구축하는 것에 있습니다. 이번 포스트에서는 현재 블로그에 구축되어 있는 RAG 시스템의 성능을 측정하는 여러 방법들에 대해서 알아보고, 제가 알아보고 진행했던 평가 데이터 구축, 평가 방법 구현들을 포스트에 정리해 보고자 합니다.

# 2. 평가 데이터가 없을 때의 RAG 시스템 평가 방법 

사실 RAG 시스템도 어떻게 보면 정보 검색기로 볼 수 있기 때문에 평가 데이터만 구축되어 있다면 기존의 정보 검색기에서 사용되는 평가 방법과 함께 RAG 시스템의 최종 결과물의 성능을 측정하는 방법인 "Faithfulness"와 "Answer Relevancy"를 측정하기만 하면 됩니다. 다만 문제는 제가 블로그에 구축한 RAG 시스템과 같이 평가 데이터가 없을 경우입니다. 이번 항목에서는 제가 알아본 평가 데이터가 없을 때의 RAG 시스템 평가 방법들과 제가 사용한 평가 방법들에 대해서 정리하고, 어떻게 진행을 하였는지 정리하도록 하겠습니다.

## 2.1 RAGAS(RAG Assessment)

### 2.1.1 RAGAS란?

RAGAS란? RAG 파이프라인의 성능을 데이터 기반으로 측정하기 위한 오픈소스 프레임워크입니다.

기존의 LLM 평가는 사람이 직접 검토하거나 정답 셋(Ground Truth)이 반드시 필요했다면, RAGAS는 **'참조 데이터 없이도'** LLM을 활용해 RAG 시스템의 각 구성 요소를 독립적으로 평가할 수 있는 지표를 제공합니다. 즉 LLM이 생성한 답변뿐만 아니라, 그 답변의 근거가 된 '검색된 문서(Context)'의 품질까지 수치화합니다.

### 2.1.2 RAGAS를 사용하는 이유

RAG 시스템은 단순히 "답변이 좋은가?"를 넘어, "문서를 잘 찾아왔는가?"와 "찾아온 문서에 기반해 답변했는가?"라는 두 가지 복합적인 문제를 안고 있습니다. RAGAS는 다음 이유로 필수적입니다.

- 컴포넌트별 진단: 전체 시스템을 통째로 평가하는 것이 아니라, 검색(Retrieval) 단계와 생성(Generation) 단계를 분리하여 어디에서 병목이 발생하는지 정확히 짚어냅니다.
- 비용 및 시간 절감: 수천 개의 테스트 케이스를 사람이 일일이 검토하는 것은 불가능합니다. RAGAS는 LLM이 평가자가 되어 자동화된 평가(LLM-as-a-judge)를 수행하므로 개발 주기를 획기적으로 단축합니다.
- 정답 데이터의 부재 해결: 실무에서는 질문에 대한 정답(Ground Truth)이 없는 경우가 많습니다. RAGAS는 질문, 컨텍스트, 답변 간의 논리적 일관성만을 가지고도 품질을 측정할 수 있는 지표를 제공합니다.

### 2.1.3 RAGAS의 핵심 평가 척도(RAG Triad)

RAGAS는 시스템의 신뢰도를 평가하기 위해 **'RAG Triad(RAG 삼각구도)'**라는 개념을 중심으로 지표를 구성합니다.

1. 생성(Generation) 관련 지표

    - 충실도(Faithfulness): 모델이 답변을 할 때, 검색된 문서(Context) 내의 정보만을 사용했는지 측정합니다. 이 수치가 낮으면 모델이 환각을 일으키고 있다는 뜻입니다.

      답변(A)이 검색된 컨텍스트(C)에 얼마나 근거하고 있는지를 측정합니다. 환각 유무를 판별하는 가장 중요한 지표입니다.

      - 계산법:
          1. LLM이 답변(A)에서 독립적인 명제들을 추출합니다.
          2. 각 명제가 컨텍스트(C)로부터 추론 가능한지 확인합니다.

          $$Faithfulness = \frac{|\text{컨텍스트로 검증된 명제 수}|}{|\text{답변에서 추출된 전체 명제 수}|}$$

    - 답변 관련성(Answer Relevance): 질문에 대해 답변이 얼마나 직접적이고 적절한지를 평가합니다. 문서 내용은 맞더라도 질문의 의도와 동떨어진 답변을 하면 점수가 낮아집니다.

      생성된 답변(A)이 원래의 질문(Q)에 얼마나 부합하는지를 측정합니다. 답변의 정확성보다는 '질문의 의도'를 잘 파악했는지에 집중합니다.

      - 계산법:
        1. 생성된 답변(A)을 바탕으로 LLM이 역으로 질문($q_i$)들을 역생성합니다.
        2. 원래의 질문(Q)과 생성된 질문($q_i$) 사이의 코사인 유사도를 계산하여 평균을 구합니다.

        $$Answer\ Relevance = \frac{1}{n} \sum_{i=1}^{n} \text{sim}(Q, q_i)$$

2. 검색(Retrieval) 관련 지표

    - 컨텍스트 정밀도(Context Precision): 질문과 관련된 정보가 검색된 경과 상위권에 잘 포함되어 있는지 측정합니다. 검색 엔진의 순위 지정 능력을 평가합니다.

      검색된 컨텍스트(C) 중에서 정답과 관련된 정보가 상위 순위에 잘 배치되었는지를 측정합니다.
    
      - 계산법:
        1. 검색된 각 청크(Chunk)가 유용한지(k) 여부를 판단합니다.
        2. Precision@k를 활용하여 가중치를 부여합니다.

        $$Context\ Precision = \frac{\sum_{k=1}^{K} (Precision@k \times v_k)}{\text{컨텍스트 내 관련 문장의 총 합}}$$

        $v_k$는 $k$번째 결과가 관련이 있으면 $1$, 없으면 $0$인 지시 함수입니다.

    - 컨텍스트 재현율(Context Recall): 질문에 답하기 위해 필요한 모든 정보가 검색된 문서 안에 포함되어 있는지를 평가합니다. 실제 정답과 컨텍스트를 비교하여 측정합니다.

      실제 정답(Ground Truth, GT)을 작성하기 위해 필요한 정보가 검색된 컨텍스트(C) 내에 모두 포함되어 있는지를 측정합니다.

      - 계산법:
        1. 정답(GT)을 여러 개의 문장으로 나눕니다.
        2. 각 문장이 컨텍스트(C)에 포함되어 있는지 확인합니다.

        $$Context\ Recall = \frac{|\text{컨텍스트로 설명 가능한 정답 내 문장 수}|}{|\text{정답 내 전체 문장 수}|}$$

3. 기타 확장 지표
    - Context Entities Recall: 검색된 내용과 정답 간의 개체(Entity) 일치 여부를 측정합니다.
    - Answer Semantic Similarity: 정답과 생성된 답변 간의 의미론적 유사성을 벡터 기반으로 측정합니다.

### 2.1.4 RAGAS를 사용해본 결과

처음에는 RAGAS를 이용해 자동으로 평가 데이터를 구축하고 이 데이터를 기반으로 평가 시스템을 구축하고자 하였습니다. 하지만 RAGAS를 직접 사용해본 결과 RAGAS의 경우 다음과 같은 문제가 있습니다.

1. 데이터 생성이 잘 되지 않는 문제

    - Knowledge Graph를 적용한 최신 버전 RAGAS(v0.4.3 이후)버전의 경우 Knowledge Graph를 적용하여 더욱 세밀하고 정확한 평가 데이터를 자동으로 생성할 수도 있으나 Knowledge Graph를 초기 설정이 매우 까다롭고, Knowledge Graph 구축에 성공한다고 하더라도 문서가 다양하지 않거나, 생성된 Knowledge Graph의 Node들 간의 관계 형성이 되지 않으면 평가 데이터 생성이 되지 않음
    
    - 구버전 RAGAS의 경우에도 원하는 양 만큼의 데이터를 얻기 힘듬

2. 비용 문제

    - 최신 버전 RAGAS의 경우 Knowledge Graph 구축에도 LLM을 사용하기 때문에 LLM API 비용이 상당히 많이 발생함

위와 같은 RAGAS의 문제가 발생하여 다른 대안을 없을까하여 찾아 보다 RAGAS와는 비슷하지만 사용이 편한 Deepeval에 대해서 알게 되었습니다.


## 2.2 DeepEval

### 2.2.1 DeepEval이란 무엇인가?

DeepEval은 LLM 애플리케이션의 성능을 테스트하기 위한 오픈소스 유닛 테스트 프레임워크입니다. 가장 큰 특징은 우리가 흔히 소프트웨어 개발에서 사용하는 Pytest와 매우 유사한 방식으로 동작한다는 점입니다. "Vibe Check(느낌적인 느낌)"로 모델을 평가하는 것이 아니라, 수치화된 지표를 기반으로 모델의 답변이 기준치(Threashold)를 넘지 못하면 테스트를 실패(Fail) 시키는 엄격한 평가 체계를 제공합니다.

### 2.2.2 DeepEval을 사용하는 이유

단순히 "답변이 좋다"라고 말하는 것과 "이 모델의 충실도는 0.85이므로 배포 가능하다"라고 말하는 것은 차원이 다릅니다. DeepEval을 사용하는 이유는 다음과 같습니다.

- 자동화된 단위 테스트: 매번 사람이 검수할 필요 없이, 코드를 수정할 때마다 자동으로 성능 변화를 체크할 수 있습니다.
- CI/CD 통합: Github Actions와 같은 파이프라인에 통합하여, 평가 점수가 기준 미달일 경우 배포를 자동으로 차단할 수 있습니다.
- 다양한 평가 지표: RAG 전용 지표뿐만 아니라 할루시네이션(Hallucination), 독성(Toxicity), 편향성(Bias) 등 일반적인 LLM 안정성 지표도 함께 제공합니다.
- 결과 시각화: 'Confident AI'라는 대시보드와 연동하여 테스트 결과를 한눈에 모니터링하고 관리할 수 있습니다.

### 2.2.3 RAGAS vs DeepEval: DeepEval만의 독보적인 장점

RAGAS가 RAG 파이프라인의 수학적/이론적 지표에 집중한다면, DeepEval은 실무 개발 환경에서의 유연성과 확장성에 강점이 있습니다.

1. G-Eval 기반의 고도화된 평가

    DeepEval은 G-Eval이라는 프레임워크를 사용합니다. 이는 LLM이 단순히 점수를 매기는 것을 넘어, **Chain of Thought(생각의 사슬)**를 통해 평가 근거를 먼저 생성한 뒤 점수를 산출하는 방식입니다. 덕분에 인간의 판단과 가장 유사한 평가 결과를 얻을 수 있습니다.

2. 테스트 프레임워크와의 완벽한 결합(Pytest)

    RAGAS는 보통 Jupyter Notebook에서 데이터프레임 형태로 결과를 확인하지만, DeepEval은 터미널에서 `pytest` 명령어로 테스트를 실행합니다. 개발자들에게 있어 익숙한 UI와 워크플로우를 그대로 사용할 수 있다는 것은 엄청난 메리트입니다.

3. RAG를 넘어선 범용성

    RAGAS는 이름에서 알 수 있듯 RAG에 특화되어 있습니다. 반면 DeepEval은 요약(Summarization), 대화의 일관성(Conversational Continuity), 독성 테스트 등 모든 종류의 LLM 애플리케이션에 범용적으로 사용가능합니다.

저는 RAGAS가 아닌 DeepEval을 이용해 평가를 진행했는데 우선 RAGAS보다 저렴한 평가 데이터 생성 비용때문에 DeepEval을 채택했고, 두 번째로는 RAGAS와 비교했을 때 사용하기 간단하다는 점 마지막으로 별도의 평가 데이터가 존재한다면 이 평가 데이터와 LLM을 이용해 "faithfulness", "answer relevancy", "contextual relevancy"와 같은 평가 척도를 측정해 준다는 점에서 DeepEval을 채택하였습니다. 그렇다면 이제 DeepEval을 이용해 제가 진행했던 제 블로그에 있는 RAG 시스템의 평가 과정을 정리해 보도록 하겠습니다.

# 3. DeepEval을 이용한 평가 과정

## 3.1 DeepEval을 이용한 평가 데이터 생성

처음엔 DeepEval을 이용해 자동으로 평가 데이터를 생성하고 생성된 평가 데이터를 이용하고자 하였습니다. 하지만 테스트용으로 어떤 PDF 문서를 이용해 평가 데이터를 생성했을 때의 결과가 만족스럽지 못했고, 평가 데이터의 품질 향상을 위해선 프롬프트를 수정하면서 매번 데이터를 생성해야 하므로 RAGAS보다는 비용이 덜 들긴 하지만 그래도 생각보다 많은 비용이 발생할 것 같았습니다. 그리고 무엇보다도 제 블로그에 있는 RAG 시스템에 사용되는 데이터는 제가 작성한 포스트들인데 이 포스트들마다 질문-답변 쌍의 데이터를 추출해야 하고, 또 제가 작성한 포스트들 개수 차제는 많지는 않지만 기술 블로그의 특성상 포스트의 전체 토큰 자체가 많아 데이터 생성과 이를 위한 테스트를 무작정 진행하기에는 무리가 있다고 생각하였습니다.

그래서 일단은 20~30개 정도의 간단한 데이터를 이용해 DeepEval에서 제공하는 LLM을 이용한 평가를 진행해보고 또한 제가 구축한 벡터 DB의 검색 품질도 측정하여 어느 정도 기준이 통과 되면 DeepEval을 이용한 평가 데이터 생성을 진행해보기로 하였습니다. 아래는 DeepEval을 이용해 평가 데이터를 생성하는 예시 코드와 생성된 데이터입니다.

```python
import shutil
import os
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI

# 한글 출력을 강제하는 커스텀 LLM 클래스 정의
class KoreanEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name="gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        # 모든 프롬프트 끝에 한국어 지시어를 강제로 주입합니다.
        korean_prompt = f"{prompt}\n\nIMPORTANT: Please write the 'question' and 'expected_output' in Korean only."
        response = self.model.invoke(korean_prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        korean_prompt = f"{prompt}\n\nIMPORTANT: Please write the 'question' and 'expected_output' in Korean only."
        response = await self.model.ainvoke(korean_prompt)
        return response.content

    def get_model_name(self):
        return "Custom Korean GPT Model"


# 1. 제너레이터 초기화
korean_model = KoreanEvalModel()
synthesizer = Synthesizer(model=korean_model)

context_config = ContextConstructionConfig(
    max_contexts_per_document=12,
    chunk_size=1024
)

# 2. PDF에서 바로 질문-정답 쌍 생성
# num_golends: 생성할 데이터 개수

original_path = "/content/drive/MyDrive/LangChain/pdf_data/SPRI_AI_Brief_2023년12월호_F.pdf"
temp_path = "/content/spri_repot_temp.pdf"
shutil.copy(original_path, temp_path)

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[temp_path],
    max_goldens_per_context=1,
    context_construction_config=context_config,
    include_expected_output=True
)

print(f"생성된 테스트 케이스 수: {len(goldens)}")

# 임시 파일 삭제
if os.path.exists(temp_path):
    os.remove(temp_path)
  
# 4. 데이터셋 저장 (최신 메서드 적용: save_as_json -> save)
if len(goldens) > 0:
    dataset = EvaluationDataset(goldens=goldens)

    if os.path.exists("/content/drive/MyDrive/LangChain/SPRI_AI_Brief_testset.json"):
        os.remove("/content/drive/MyDrive/LangChain/SPRI_AI_Brief_testset.json")

    dataset.save_as(
        file_type="json",
        directory="/content/drive/MyDrive/LangChain/",
        file_name="SPRI_AI_Brief_testset.json"
    )
    #(file_path="/content/drive/MyDrive/LangChain/SPRI_AI_Brief_testset.json")
    print("✅ 데이터셋 저장 완료!")
else:
    print("⚠️ 생성된 데이터가 없습니다. PDF 텍스트 추출 상태를 확인하세요.")
```

다음은 생성된 데이터 중의 하나입니다. 생성된 데이터를 보면 깔금하게 정제되지 않은 것을 볼 수 있습니다. `input`과 `expected_output`을 보면 입력 쿼리에도 "기대 출력"이 포함되어 있고, 예상 답변에는 "질문"이 포함되어 있습니다. 이를 바로 잡으려면 프롬프트 엔지니어링 과정을 거치고 프롬프트 엔지니어링을 했음에도 개선되지 않는다면 그 때는 규칙 기반으로 정제하는 작업을 해야 합니다. 프롬프트 엔지니어링을 하면서 데이터를 생성하는 것은 많은 비용이 발생하며, 규칙 기반으로 정제하는 작업은 모든 생성된 평가 데이터가 아래와 똑같은 패턴을 가지지 않을 수도 있기 때문에 두 과정을 모두 한 번에 진행해야 합니다. 저는 최대한 비용을 아끼기 위해 일단은 DeepEval을 이용한 평가 데이터 생성 방법은 최대한 마지막에 사용하고자 합니다.

```json
{
        "input": "질문: 영국 AI 안전 연구소는 첨단 AI 시스템 평가와 AI 안전 연구 외에 어떤 기능을 갖추고 있나요?  \n기대 출력: AI 안전성 증진 위한 정보교류 활성화 기능도 포함됨.",
        "actual_output": null,
        "expected_output": "질문: 영국 AI 안전 연구소는 첨단 AI 시스템 평가와 AI 안전 연구 외에 어떤 기능을 갖추고 있나요?  \n기대 출력: AI 안전성 증진을 위한 정보 교류 활성화 기능도 포함됨.",
        "retrieval_context": null,
        "context": [
            "1. 정책/법제  2. 기업/산업 3. 기술/연구  4. 인력/교육영국 과학혁신기술부, AI 안전 연구소 설립 발표n영국 과학혁신기술부가 첨단 AI 시스템에 대한 평가를 통해 안전성을 보장하기 위한 AI 안전 연구소를 설립한다고 발표nAI 안전 연구소는 핵심 기능으로 첨단 AI 시스템 평가 개발과 시행, AI 안전 연구 촉진, 정보교류 활성화를 추진할 계획\nKEY Contents\n£영국 AI 안전 연구소, 첨단 AI 시스템 평가와 AI 안전 연구, 정보 교류 추진n영국 과학혁신기술부가 2023년 11월 2일 첨단 AI 안전에 중점을 둔 국가 연구기관으로 AI 안전 연구소(AI Safety Institute)를 설립한다고 발표∙AI 안전 연구소는 첨단 AI의 위험을 이해하고 거버넌스 마련에 필요한 사회·기술적 인프라 개발을 통해 영국을 AI 안전 연구의 글로벌 허브로 확립하는 것을 목표로 함∙영국 정부는 향후 10년간 연구소에 공공자금을 투자해 연구를 지원할 계획으로, 연구소는 △첨단 AI 시스템 평가 개발과 시행 △AI 안전 연구 촉진 △정보 교류 활성화를 핵심 기능으로 ",
            "�각국은 AI 안전 보장을 위해 첨단 AI 개발기업의 투명성 향상, 적절한 평가지표와 안전 테스트 도구 개발, 공공부문 역량 구축과 과학 연구개발 등의 분야에서 협력하기로 합의£영국 총리, 정부 주도의 첨단 AI 시스템 안전 테스트 계획 발표n리시 수낙 영국 총리는 AI 안전성 정상회의를 마무리하며 첨단 AI 모델에 대한 안전성 시험 계획 수립과 테스트 수행을 주도할 영국 AI 안전 연구소의 출범을 발표∙첨단 AI 모델의 안전 테스트는 국가 안보와 안전, 사회적 피해를 포함한 여러 잠재적 유해 기능에 대한 시험을 포함하며, 참석자들은 정부 주도의 외부 안전 테스트에 합의∙각국 정부는 테스트와 기타 안전 연구를 위한 공공부문 역량에 투자하고, 테스트 결과가 다른 국가와 관련된 경우 해당 국가와 결과를 공유하며, 적절한 시기에 공동 표준 개발을 위해 노력하기로 합의 n참가국들은 튜링상을 수상한 AI 학자인 요슈아 벤지오 교수가 주도하는 ‘과학의 현황(State of the Science)’ 보고서 작성",
            "1. 정책/법제  2. 기업/산업 3. 기술/연구  4. 인력/교육영국 AI 안전성 정상회의에 참가한 28개국, AI 위험에 공동 대응 선언n영국 블레츨리 파크에서 개최된 AI 안전성 정상회의에 참가한 28개국들이 AI 안전 보장을 위한 협력 방안을 담은 블레츨리 선언을 발표n첨단 AI를 개발하는 국가와 기업들은 AI 시스템에 대한 안전 테스트 계획에 합의했으며, 영국의 AI 안전 연구소가 전 세계 국가와 협력해 테스트를 주도할 예정 \nKEY Contents\n£AI 안전성 정상회의 참가국들, 블레츨리 선언 통해 AI 안전 보장을 위한 협력에 합의n2023년 11월 1~2일 영국 블레츨리 파크에서 열린 AI 안전성 정상회의(AI Safety Summit)에 참가한 28개국 대표들이 AI 위험 관리를 위한 ‘블레츨리 선언’을 발표 ∙선언은 AI 안전 보장을 위해 국가, 국제기구, 기업, 시민사회, 학계를 포함한 모든 이해관계자의 협력이 중요하다고 강조했으며, 특히 최첨단 AI 시스템 개발 기업은 안전 평가를 비롯한 적절한 조치를 취하여 AI 시스템의 안전을 보장할 책임이 있다고 지적�"
        ],
        "name": null,
        "comments": null,
        "source_file": "/content/spri_repot_temp.pdf",
        "tools_called": null,
        "expected_tools": null,
        "additional_metadata": {
            "evolutions": [
                "Constrained"
            ],
            "synthetic_input_quality": 0.6
        },
        "custom_column_key_values": null
    }
```

## 3.2 DeepEval을 이용한 평가 진행

위에서 본 것과 같이 DeepEval을 이용한 평가 데이터 생성도 많은 비용과 노력이 들어가기 때문에 일단은 최대한 비용이 들지 않는 방식으로 평가를 진행해 보고자 합니다.

찾아보니 DeepEval에는 평가 데이터가 있을 경우 "faithfulness"와 "answer relevancy", "contextual relevancy" 를 측정해주는 기능이 있다고 합니다. 이를 위해서 간단히 몇 개의 데이터만 만들어서 DeepEval을 이용한 평가를 진행해 보도록 하겠습니다.

## 3.2.1 평가 데이터 생성

평가 데이터는 OpenAI나 Google 등의 플랫폼에서 제공하는 API를 사용하기 보다는 간단히 20~30개 정도의 데이터만 생성해서 사용할 것이기 때문에 현재 사용하고 있는 Google Gemini Pro를 사용하여 평가 데이터를 생성했습니다.

Gemini에게 제 블로그의 url을 주고 제 블로그를 보고 평가 데이터를 생성해 달라고 하였습니다. 아래는 Gemini가 생성해준 평가 데이터입니다.

```json
[
  {
    "input": "LangChain Expression Language(LCEL)의 선언적 구문이 복잡한 LLM 애플리케이션 구축에 주는 이점은 무엇인가요?",
    "expected_output": "LCEL은 복잡한 로직을 간결하고 읽기 쉬운 방식으로 표현할 수 있게 해주며, 다양한 컴포넌트를 유연하게 조합하여 유지보수와 확장이 용이한 시스템을 구축할 수 있게 돕습니다.",
    "category": "LangChain"
  },
  {
    "input": "Maven 3.8.1 버전부터 HTTP 리포지토리 연결이 차단되는 에러가 발생하는 근본적인 이유는?",
    "expected_output": "보안 강화를 위해 Maven 3.8.1 버전부터 HTTP 프로토콜을 이용한 외부 리포지토리 연결이 기본적으로 차단되도록 설정이 변경되었기 때문입니다.",
    "category": "Maven"
  },
  {
    "input": "Jenkins 빌드 시 'cannot find symbol' 에러가 발생했을 때, 로컬 저장소의 라이브러리를 삭제하는 구체적인 명령어는?",
    "expected_output": "Execute Shell 단계에서 'rm -rf' 명령어를 사용하여 에러가 발생하는 해당 모듈의 라이브러리 경로를 강제로 삭제합니다.",
    "category": "Jenkins"
  },
  {
    "input": "LangGraph 구조에서 노드(Node)는 어떤 단위의 작업을 의미합니까?",
    "expected_output": "노드는 LLM 호출이나 도구 실행과 같은 특정 기능을 담고 있는 작업의 최소 단위이자 에이전트를 나타내는 모듈입니다.",
    "category": "LangChain"
  },
  {
    "input": "GitHub 블로그 운영을 위해 Ruby를 설치할 때 (x64)가 아닌 (x86) 버전을 권장하는 이유는 무엇인가요?",
    "expected_output": "Jekyll이 32비트(x86) 환경을 기준으로 최적화되어 설계되었기 때문에 x64 버전 설치 시 발생할 수 있는 환경 설정 오류를 방지하기 위함입니다.",
    "category": "Github_Blog"
  },
  {
    "input": "LangChain의 RunnableSequence는 여러 컴포넌트를 어떤 방식으로 실행합니까?",
    "expected_output": "RunnableSequence는 등록된 여러 Runnable 컴포넌트들을 시퀀스(순서)대로 순차적으로 연결하여 실행합니다.",
    "category": "LangChain"
  },
  {
    "input": "Maven Wrapper를 사용하면 프로젝트 환경에서 어떤 장점을 얻을 수 있나요?",
    "expected_output": "사용자의 PC에 Maven이 설치되어 있지 않거나 프로젝트마다 요구하는 버전이 다를 때, 자동으로 최적의 버전을 유지하여 빌드 일관성을 보장해 줍니다.",
    "category": "Maven"
  },
  {
    "input": "LangGraph의 에지(Edge)가 노드 간 상호작용에서 담당하는 역할은 무엇인가요?",
    "expected_output": "에지는 노드 간의 연결을 나타내며, 데이터의 전달 경로와 작업의 흐름, 그리고 조건부 분기 로직을 정의합니다.",
    "category": "LangChain"
  },
  {
    "input": "IntelliJ 업데이트 후 Maven 빌드 시 http 관련 에러가 발생한다면 가장 먼저 확인해야 할 설정 경로는?",
    "expected_output": "Settings > Build, Execution, Deployment > Maven 메뉴의 'Maven home path'가 'Use Maven wrapper'로 설정되어 있는지 확인해야 합니다.",
    "category": "Maven"
  },
  {
    "input": "DFS(깊이 우선 탐색) 알고리즘이 그래프를 탐색할 때 사용하는 주요 자료구조는 무엇인가요?",
    "expected_output": "DFS는 최대한 깊숙이 노드를 방문한 후 돌아오는 방식을 취하므로 스택(Stack) 자료구조를 주로 이용합니다.",
    "category": "Algorithm"
  },
  {
    "input": "LangGraph에서 '노드별 스트리밍 출력' 기능이 사용자 경험(UX) 측면에서 중요한 이유는?",
    "expected_output": "전체 결과가 나올 때까지 기다리는 대신 현재 에이전트가 어떤 작업을 수행 중인지 실시간 피드백을 주어 체감 대기 시간을 줄여주기 때문입니다.",
    "category": "LangChain"
  },
  {
    "input": "메이븐 저장소에서 특정 라이브러리를 삭제한 후 Jenkins 빌드를 다시 실행하면 어떤 현상이 일어납니까?",
    "expected_output": "Nexus 등 원격 저장소로부터 최신 버전의 라이브러리를 강제로 다시 다운로드하여 'Cannot Find Symbol'과 같은 인식 오류를 해결합니다.",
    "category": "Jenkins"
  },
  {
    "input": "Minimal Mistakes 테마에서 '_config.yml' 파일을 수정하는 목적은 무엇인가요?",
    "expected_output": "블로그의 메타데이터, 폰트, 이미지 경로, SEO 옵션 등 전반적인 테마 설정을 커스터마이징하기 위함입니다.",
    "category": "Github_Blog"
  },
  {
    "input": "알고리즘에서 '인접 리스트' 방식이 '인접 행렬' 방식보다 메모리 측면에서 유리한 경우는 언제인가요?",
    "expected_output": "노드의 개수가 많지만 연결된 에지가 적은 경우, 연결된 정보만 저장하는 인접 리스트 방식이 메모리 낭비를 줄여줍니다.",
    "category": "Algorithm"
  },
  {
    "input": "LangChain의 RunnableLambda는 어떤 상황에서 주로 사용됩니까?",
    "expected_output": "사용자가 직접 정의한 일반 파이썬 함수를 LangChain의 인터페이스인 Runnable 객체로 래핑하여 체인에 통합할 때 사용합니다.",
    "category": "LangChain"
  },
  {
    "input": "Jenkins 파이프라인에서 'Execute Shell' 단계를 Maven 빌드 실행 전으로 배치하는 이유는?",
    "expected_output": "빌드가 시작되기 전에 문제가 되는 기존 라이브러리를 미리 제거하여 깨끗한 상태에서 소스가 업데이트됨을 보장하기 위해서입니다.",
    "category": "Jenkins"
  },
  {
    "input": "Mac 환경에서 Jekyll 블로그를 로컬 서버로 실행하기 위해 사용하는 명령어는 무엇인가요?",
    "expected_output": "'bundle exec jekyll serve' 명령어를 통해 로컬호스트 환경에서 블로그를 기동시킬 수 있습니다.",
    "category": "Github_Blog"
  },
  {
    "input": "LangChain의 LCEL 인터페이스가 제공하는 비동기 처리 기능의 명칭은?",
    "expected_output": "비동기 방식으로 체인을 실행할 수 있도록 돕는 'astream' 또는 'ainvoke' 등의 비동기 메서드를 제공합니다.",
    "category": "LangChain"
  },
  {
    "input": "RAG 시스템에서 청킹(Chunking) 사이즈를 너무 크게 잡았을 때 발생할 수 있는 부작용은?",
    "expected_output": "검색 결과에 질문과 상관없는 노이즈 정보가 많이 섞여 들어가 'Contextual Relevancy' 점수가 낮아질 수 있습니다.",
    "category": "NLP_RAG"
  },
  {
    "input": "Maven 3.8.1 에러 해결을 위해 settings.xml을 수정할 때 어떤 태그를 주로 관리합니까?",
    "expected_output": "차단된 HTTP 리포지토리의 미러링 정책을 변경하기 위해 <mirrors> 및 <mirror> 태그를 관리합니다.",
    "category": "Maven"
  },
  {
    "input": "LangGraph의 'State' 객체(예: TypedDict)는 어떤 역할을 수행하나요?",
    "expected_output": "그래프가 실행되는 동안 노드 사이에서 공유되는 데이터를 저장하고 유지하는 메모리 역할을 합니다.",
    "category": "LangChain"
  },
  {
    "input": "알고리즘의 '동적 계획법(DP)'에서 하위 문제의 결과를 저장하여 재사용하는 기법의 명칭은?",
    "expected_output": "이미 계산된 값을 저장해 두었다가 다시 사용하는 '메모이제이션(Memoization)' 기법입니다.",
    "category": "Algorithm"
  },
  {
    "input": "Jekyll 설치 확인을 위해 터미널에서 입력해야 하는 버전 확인 명령어는?",
    "expected_output": "'ruby -v'와 'jekyll -v' 명령어를 통해 정상 설치 여부와 버전을 확인할 수 있습니다.",
    "category": "Github_Blog"
  },
  {
    "input": "LangChain의 LCEL을 사용했을 때 얻을 수 있는 주요 장점은 무엇인가요?",
    "expected_output": "LCEL은 복잡한 체인을 효율적으로 구축할 수 있게 하며, 스트리밍 지원, 비동기 연산, 병렬 실행 등을 기본 제공하여 효율성을 높여줍니다.",
    "category": "LangChain"
  },
  {
    "input": "Maven 3.8.1 버전부터 발생하는 'http repositories are blocked' 에러의 원인과 해결 방법은 무엇인가요?",
    "expected_output": "보안을 위해 HTTP 연결이 차단되어 발생하며, settings.xml에서 미러 설정을 수정하거나 HTTPS로 변경하여 해결합니다.",
    "category": "Maven"
  },
  {
    "input": "LangGraph에서 노드(Node)와 에지는 각각 어떤 역할을 수행하나요?",
    "expected_output": "노드는 특정 작업을 수행하는 단위이고, 에지는 작업의 흐름이나 조건부 분기를 정의하여 상태 중심의 순환 구조를 구현합니다.",
    "category": "LangChain"
  },
  {
    "input": "Jenkins에서 Maven 빌드 시 'cannot find symbol' 에러 해결 절차는 어떻게 되나요?",
    "expected_output": "Execute Shell 단계에서 에러가 발생하는 모듈 라이브러리를 삭제(rm -rf)한 후 재빌드하여 최신 라이브러리를 다시 내려받습니다.",
    "category": "Jenkins"
  }
]
```

## 3.2.2 평가 데이터와 DeepEval을 이용한 평가 실행

대략적인 평가 데이터는 구축이 되었으며, 이제 DeepEval을 이용한 평가를 진행해 보도록 하겠습니다. 하지만 그 전에 먼저 원활한 평가 진행을 위해 Supabase에 구축되어 있는 벡터 DB를 faiss를 이용한 로컬 벡터 DB를 구축하도록 하겠습니다. 벡터 DB 생성 및 로컬 저장은 `create_local_faiss`이고, 생성된 faiss 로컬 벡터 DB를 로드하는 함수는 `load_local_faiss` 로 정의해서 구현했습니다.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from chunk_post import load_and_chunk_posts
import os

# 1. 벡터 DB 생성 및 로컬 저장
def create_local_faiss(save_path):
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small"
    )

    posts_path = r"C:\Users\ssclu\Desktop\icechickentender.github.io\_posts"
    chunks = load_and_chunk_posts(posts_path)

    # 문서를 벡터화하여 인덱스 생성
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 로컬 폴더에 저장 (index.faiss, index.pkl 파일 생성됨)
    vectorstore.save_local(save_path)
    print(f"✅ FAISS 인덱스가 {save_path}에 저장되었습니다.")

# 2. 저장된 벡터 DB 로드
def load_local_faiss(save_path="faiss_index"):
    embeddings = OpenAIEmbeddings(
        model ="text-embedding-3-small"
    )
    # 저장된 파일을 불러와 vectorstore 객체 반환
    # allow_dangerous_deserialization=True는 로컬 신뢰 파일 로드 시 필수입니다.
    vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

if __name__ == "__main__":
    create_local_faiss("../faiss_recursive_chunk1000_overlap150")
```

그럼 이제 DeepEval을 이용한 평가를 수행하는 코드를 구현해서 평가를 진행해 보도록 하겠습니다.

추가적으로 벡터 DB에 적재된 문서들의 chunking 정보는 다음과 같습니다.

```
chunk parameter
test_splitter = RecursiveCharacterTestSplitter
chunk_size = 1000
chunk_overlab = 150
```

```python
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
import openai
from local_vector_db_generate import load_local_faiss
from langchain_openai import OpenAIEmbeddings
import os
from deepeval.evaluate import AsyncConfig
from load_data import load_and_blog_data

os.environ["DEEPEVAL_MAX_CONCURRENT_TASKS"] = "3"

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    openai_api_key = OPENAI_API_KEY
)

client = openai.OpenAI(api_key=OPENAI_API_KEY)

vectorstore = load_local_faiss("../faiss_recursive_chunk1000_overlap150")
retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={
        'score_threshold':0.5,
        'k':3
}
)

def generate_answer(query):
    # 1. 검색 수행 (앞서 성공한 RPC 호출 로직)
    #query_vector = embeddings.embed_query(query)

    response = retriever.invoke(query)

    if not response:
        return "관련 내용을 찾을 수 없습니다.", []

    # 2. 검색 결과(Context) 정리
    context_text = ""
    context_list = []
    for doc in response: # response 자체가 리스트입니다.
        context_text += f"\n---\n내용: {doc.page_content}\n"
        context_list.append(doc.page_content)

    # 3. LLM에게 답변 요청(프롬프트 주입)
    messages = [
        {"role": "system", "content": "블로그 내용을 바탕으로 답변하고 출처를 명시해줘."},
        {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {query}"}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return completion.choices[0].message.content, context_list

# 1. 골든 데이터셋 정의
golden_data = load_and_blog_data("../data/blog_30_evaluate_data.json")

# 2. 평가 지표 설정 (비용 효율을 위해 gpt-4o-mini 사용)
eval_model = "gpt-4o-mini"
metrics = [
    FaithfulnessMetric(threshold=0.5, model=eval_model),
    AnswerRelevancyMetric(threshold=0.5, model=eval_model),
    ContextualRelevancyMetric(threshold=0.5, model=eval_model)
]

# 3. 자동 평가 루프 실행
test_cases = []

for data in golden_data:

    actual_answer, retrieved_docs = generate_answer(data["input"])

    test_case = LLMTestCase(
        input=data["input"],
        actual_output=actual_answer,
        expected_output=data["expected_output"],
        retrieval_context=retrieved_docs
    )

    test_cases.append(test_case)

# DeepEval의 evalute에서 LLM 요청 시 비동기로 요청할 것인지 동기로 요청할 것인지를 결정하도록 하는 Config 객체
async_config = AsyncConfig(run_async=False)

# 4. DeepEval 통합 평가 실행
# verbose=True로 설정하면 각 케이스별 상세 이유(Reason)을 볼 수 있습니다.
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config = async_config
)
```

위 코드를 이용해 평가를 진행해 보았습니다. DeepEval을 이용한 평가를 진행하면 각 평가 데이터별로 어떠한 원인으로 이러한 평가를 진행했는지 상세히 알려줍니다만 이러한 평가 결과를 모두다 적기에는 내용이 많은 관계 평가 결과를 요약해서 보여주는 `Overall Metric Pass Rates`만 보도록 하고, 상세한 것들은 Gemini를 통해 분석을 진행해 보았습니다.

평가 결과를 보면 전체적인 성능이 굉장히 낮게 나오는 것을 확인할 수 있습니다. 제 나름대로 이렇게 성능이 낮게 나오는 이유를 분석해 보았습니다.

1. 우선 문서 검색 성능인 `Contextual Relevancy`가 낮은 이유를 분석해 보았습니다. 이 수치가 낮은 이유는 벡터 DB의 문서 검색 옵션 때문인 것으로 확인되었습니다. 문서 검색 시 설정한 threshold 보다 낮은 문서들은 모두 버리도록 했기 때문에 이로인해 검색된 문서가 거의 없었습니다. 

2. 검색된 문서가 없다보니 코드에서와 같이 단순히 쿼리를 LLM에 던져서 대략적인 질문을 받는 것이 아니라 "관련 내용을 찾을 수 없습니다."라는 답변을 반환하도록 했고 이로 인해 답변의 성능인 `faithfulness`와 `answer relevancy` 성능이 낮게 나왔습니다.

```
Overall Metric Pass Rates

Faithfulness: 20.00% pass rate
Answer Relevancy: 20.00% pass rate
Contextual Relevancy: 13.33% pass rate
```

문서 검색의 성능이 문제이고, `search_type='similarity_score_threshold'`으로 인해 유사도가 0.5미만인 문서들은 모두 검색이 안되기 때문에 단순히 임계값 없이 유사도 순으로 검색 되도록 검색기의 옵션을 다음과 같이 변경해 보았습니다.

```python
retriever = vectorstore.as_retriever(
    search_type='similarity', # 임계값 없이 유사도 순으로 추출
    search_kwargs={'k': 3}
)
```

검색기의 옵션 변경 후 평가 결과는 다음과 같습니다.

유사도 값이 낮긴 하지만 낮은 값을 가지는 문서들 중에서 가장 높은 3개를 추출하도록 하여 문서로 제공하도록 하니 답변 성능을 나타내는 `faithfulness`와 `contextual relevancy` 의 값이 거의 완벽에 가까울 정도로 상승한 것을 확인할 수 있습니다.

하지만 여전히 문서 검색의 성능을 나타내는 `Contextual Relevancy`의 수치는 이전 보다는 높긴 하지만 그래도 여전히 낮은 값을 나타내고 있습니다. 그리고 이렇게 문서 검색 성능이 낮게 나오는 이유는 무엇일까 생각을 해보았는데 첫째로는 실제 문서 검색의 성능이 낮기 때문이다. 둘째 문서 검색의 성능 자체는 문제가 없지만 DeepEval의 `Contextual Relevancy`를 측정할 때의 방식 때문에 낮게 나오는 것이다. 라는 생각을 하게 되었습니다. DeepEval의 `Contextual Relevancy`는 검색된 문서와 입력 질문과의 연관 관계를 나타내는 수치입니다. 현재 평가 데이터를 보면 입력 질의와 그에 대한 답변이 길지 않은 문장으로 이루어져 있습니다. 하지만 위에 나와 있듯이 현재 문서 청킹에 적용된 chunk_size와 chunk_overlap은 1000과 150으로 설정되어 있어 하나의 청킹된 문서의 크기는 굉장히 크다고 볼 수 있습니다. 이렇게 청킹을 한 이유는 제 블로그 포스트들의 경우 간단한 내용이 아니라 컴퓨터 과학이나 인공지능과 관련된 기술에 대한 내용이 많기 때문에 chunk_size를 짧게 잡으면 청킹된 문서에 필요한 정보들이 담기지 않을 것 같다는 생각이 들어 chunk_size를 크게 잡았습니다.

```
Overall Metric Pass Rates

Faithfulness: 96.67% pass rate
Answer Relevancy: 100.00% pass rate
Contextual Relevancy: 26.67% pass rate
```

평가를 진행하면서 알게된 내용으로 평가 마다 성능이 다르게 나오는 것을 확인하였습니다. 아마도 LLM을 이용한 평가이다 보니 매 실행마다 다른 성능이 나올 수 있다고 합니다. 그래서 보통 3~5번 정도 평가를 진행한 후에 이에 대한 평균 값을 성능값으로 사용한다고 합니다. 그래서 저 또한 마찬가지로 다시 3번의 평가를 진행한 후의 평균값을 사용하기로 하였습니다. 추가적으로 Contextual Relevancy의 성능이 낮게 나오는 원인을 찾아보니 DeepEval을 이용한 평가를 진행할 때에는 문서 검색기가 찾아준 문서들을 모두 사용하기 때문인 것으로 확인하였습니다. 이에 따라 좀 더 명확한 Contextual Relevancy의 성능 평가를 진행하기 위해 평가를 할 때에는 최상위 문서 하나만을 이용하도록 하였습니다. 이에 따른 성능 평가 수치는 다음과 같습니다.

```
Faithfulness: 98.88%
Answer Relevancy: 100.00% pass rate
Contextual Relevancy: 35.87% pass rate
```

위와 같은 제 생각에 대해서 검증을 하기 위해 저는 전통적인 문서 검색의 성능 평가 척도로 사용되는 Recall@k와 MRR를 사용해 실제 벡터 DB 검색이 잘 되는지 아닌지를 확인해 보려고 합니다.


## 3.2.3 벡터 DB 문서 검색 성능 측정

Gemini가 생성한 평가 데이터와 DeepEval을 이용해 성능 측정을 진행했지만 생각보다 굉장히 낮은 문서 검색 성능에 실제로 문서 검색의 성능이 낮은지를 확인해보고자 전통적인 문서 검색 성능 평가를 진행해 보도록 하겠습니다.

일단 이전에 문제가 됐던 입력 쿼리와 청킹된 문서와의 사이즈 문제가 존재하기 때문에 우선은 입력 쿼리와 검색된 문서의 문자열을 이용한 평가 방식은 차후에 고민을 해보도록 하겠습니다. 그래서 이번 평가에는 메타데이터를 이용해 보고자 합니다. 사용할 메타 데이터는 문서를 특정할 수 있는 제 포스트의 파일 이름을 사용하기로 결정했습니다. 그래서 사용하는 평가 데이터에 다음과 같이 파일 이름 메타 데이터를 추가해 주었습니다.

```json
[
  {
    "input": "LangChain Expression Language(LCEL)의 선언적 구문이 복잡한 LLM 애플리케이션 구축에 주는 이점은 무엇인가요?",
    "expected_output": "LCEL은 복잡한 로직을 간결하고 읽기 쉬운 방식으로 표현할 수 있게 해주며, 다양한 컴포넌트를 유연하게 조합하여 유지보수와 확장이 용이한 시스템을 구축할 수 있게 돕습니다.",
    "category": "LangChain",
    "filename": "2026-01-20-LangChain-12-LangChain_Expression_Language.md"
  },
  생략...
]
```

이번 문서 검색 성능 측정에는 Recall@k와 MRR을 사용하고자 합니다.

Recall@k는 시스템이 전체 정답(Relevant Documents) 중 얼마나 많은 비율을 상위 k개의 결과 안에 포함시켰는지를 측정합니다. RAG 시스템에서는 보통 "사용자의 질문에 답할 수 있는 근거 문서가 검색 결과 k개 안에 들어 왔는가?"를 확인하는 용도로 사용됩니다. Recall@k의 수식은 다음과 같습니다.

$$\text{Recall@k} = \frac{|\text{검색된 상위 } k \text{개 문서 중 관련 문서의 수}|}{|\text{전체 관련 문서의 수}|}$$

MRR(Mean Reciprocol Rank)은 시스템이 가장 관련 있는 문서를 얼마나 상단에 배치했는지를 평가합니다. 여러 개의 쿼리에 대해 **"첫 번째 정답 문서의 순위"**의 역수를 취해 평균을 낸 값입니다. 예를 들면 어떤 질문에 대해서 총 5개의 문서 검색 결과 중 관련 있는 문서가 3번째에 위치한다고 하면 그 랭크의 역수 값을 취해줍니다. 그리고 모든 입력 질문에 대해서 이러한 계산을 진행하고 각 쿼리마다 계산된 순위의 역수 값을 평균을 내주면 이 값이 MRR 값이 되는 것입니다. 수식은 다음과 같습니다.

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$


그리고 청킹을 수행하는 함수에 파일 이름을 메타 데이터로 넣게 수정을 해준 뒤 이전에 사용하던 벡터 DB를 지우고 새로운 벡터 DB를 만들어 주었습니다. 다음은 메타 데이터에 파일명을 추가 시키도록 한 청킹 함수입니다.

```python
import os
import frontmatter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

def load_and_chunk_posts(posts_dir):
    documents = []

    # 1. 파일 순회 (운영체제 독립적 경로 처리)
    for root, dirs, files in os.walk(posts_dir):
        for filename in files:
            if filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                if "sample" in filename:
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)

                    # 카테고리 경로 생성 (예: llm/rag/)
                    categories = post.get("categories", [])
                    dir_name = "".join([f"{c.lower()}/" for c in categories])

                    # 파일명에서 날짜 제거 (2025-12-30-title.md -> title.md)
                    url_name = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", filename)
                    slug = url_name.replace('.md', '')

                    # 최종 메타데이터 구성
                    metadata = {
                        "title": post.get("title", "Untitled"),
                        "category": categories,
                        "tag": post.get("tags", []),
                        "url": f"https://icechickentender.github.io/{dir_name}{slug}/",
                        "filename": filename
                    }

                    # 2. Chunking 설정
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150,
                        separators=["\n\n", "\n", " ", ""]
                    )

                    chunks = text_splitter.split_text(post.content)

                    # 3. Document 객체 생성
                    for chunk in chunks:
                        documents.append(Document(page_content=chunk, metadata=metadata))

    print(f"✅ 총 {len(documents)}개의 청크가 생성되었습니다.")
    return documents
```

벡터 DB 생성은 위에 있는 코드를 그대로 실행 했으며, 아래는 평가 데이터와 청킹된 문서를 이용한 Recall@k, MRR을 측정하는 코드입니다.

```python
import numpy as np
from load_data import load_and_blog_data
from local_vector_db_generate import load_local_faiss

def evaluate_retrieval_by_metadata(golden_data, retriever, k=5):
    """
    텍스트 비교 없이 메타데이터 ID(source_id) 일치 여부로 Recall과 MRR을 계산합니다.
    :param golden_data:
    :param retriever:
    :param k:
    :return:
    """

    hits = 0
    reciprocal_ranks = []

    print(f"메타데이터 기반 검색 평가 시작 ... (k={k})")

    for item in golden_data:
        query = item["input"]
        target_source = item["filename"]

        # 1. FAISS 검색 수행
        retrieved_docs = retriever.invoke(query)

        found_at_rank = 0

        for i, doc in enumerate(retrieved_docs[:k]):
            retrieved_source = doc.metadata.get("filename", "")

            if target_source in retrieved_source:
                hits+=1
                found_at_rank = i + 1
                break
        # 3. MRR 점수 누적
        if found_at_rank > 0:
            reciprocal_ranks.append(1.0 / found_at_rank)
        else:
            reciprocal_ranks.append(0.0)

    recall_at_k = (hits / len(golden_data)) * 100
    mrr = np.mean(reciprocal_ranks)

    return recall_at_k, mrr

if __name__ == "__main__":

    golden_data = load_and_blog_data("../data/blog_30_evaluate_data.json")

    vectorstore = load_local_faiss("../faiss_recursive_chunk1000_overlap150")
    retriever = vectorstore.as_retriever(
        search_type='similarity', # 임계값 없이 유사도 순으로 추출
        search_kwargs={'k': 3}
    )

    k = 5

    recall_at_k, mrr = evaluate_retrieval_by_metadata(golden_data, retriever, k)
    print(f"recall at {k} : {recall_at_k:.1f}%")
    print(f"mrr : {mrr:.3f}%")
```

평가 결과 Recall@k와 MRR의 값은 생각보다 높은 값을 보여주고 있는 것을 확인할 수 있습니다. 즉 DeepEval의 `Contextual Relevancy`는 문서 검색은 잘 되나 입력 쿼리와 검색된 문서의 내용과의 괴리로 인해 낮은 성능이 나온다는 것을 확인할 수 있습니다. 

```
총 27개의 테스트 데이터를 로드했습니다.
메타데이터 기반 검색 평가 시작 ... (k=5)
recall at 5 : 92.6%
mrr : 0.852%
```

## 3.2.4 벡터 DB 문서 검색 성능 향상 

그럼 이제 문서 검색의 성능을 높이기 위한 여러 가지 실험을 한 번 진행해 보도록 하겠습니다.

### 3.2.4.1 청킹 전략에 따른 성능 비교

우선 청킹 크기나 청크 오버랩 크기를 조절했을 때의 성능 비교를 한 번 진행해 보도록 하겠습니다. 현재는 제 블로그의 포스트 글들이 모두 기술 설명에 대한 내용들이기 때문에 많은 정보를 포함시켜야 한다는 생각에 청킹 사이즈를 크게 잡았습니다. 하지만 DeepEval의 `Contextual Relevancy`의 성능이 낮게 측정이 되는데 이유로 청킹 사이즈가 너무 큰 청킹 문서들에 필요한 정보도 많지만 그 만큼 노이즈가 많아서 낮게 나오는 현상이 있습니다. 그래서 우선 청킹 사이즈를 절반인 500으로 낮춰서 성능 측정을 진행해 보도록 하겠습니다.

다음은 청킹 사이즈를 500으로 낮춘 이후의 Recall@k와 MRR 그리고 Contextual Relevancy 성능입니다. 청킹 사이즈를 절반으로 줄이니 성능이 더 낮아진 것을 확인할 수 있습니다. 이는 청킹 사이즈가 줄어들면서 오히려 필요한 주변 문맥이 사라져 진따 정답 청크 대신 단어만 겹치는 가짜 청크를 상위권으로 더 많이 가져와서 성능이 낮아지는 듯 합니다. 그렇다면 청킹 사이즈는 500으로 유지하되 청크 오버랩 사이즈를 좀 더 늘려서 성능 평가를 진행해 보도록 하겠습니다.

```
recall at 5 : 92.6%
mrr : 0.790
contextual relevancy: 14.8%
```

청크 사이즈는 500 청크 오버랩을 300으로 했을 때의 Recall@k와 MRR 성능입니다. 청크 사이즈 500, 청크 오버랩을 150으로 했을 때보다는 MRR 성능이 살짝 올랐지만 청크 사이즈 1000, 청크 오버랩을 150 했을 때보다는 여전히 낮은 성능을 보이고 있습니다. 그렇다면 청크 사이즈를 조금 늘리고 청크 오버랩을 조금 줄여서 실험을 진행해 보도록 하겠습니다.

```
recall at 5 : 92.6%
mrr : 0.818%
```

chunk_size를 750 chunk_overlap을 250으로 설정한 실험을 진행해 보았습니다. 성능의 결과는 아래와 같이 Recall@k는 동일하고, mrr은 바로 직전의 실험 때보다는 오르긴 했지만 Baseline인 chunk_size 1000 chunk_overlap 150일 때의 성능인 0.85에는 미치지 못하는 것을 확인할 수 있습니다.

```
recall at 5 : 92.6%
mrr : 0.827
```

이제 마지막으로 chunk_size를 오히려 올려보는 실험을 진행해 보도록 하겠습니다. chunk_size를 1500 chunk_overlap을 150으로 했을 때의 성능은 다음과 같습니다. 이전의 chunk_size 1000으로 했을 때와 동일한 성능을 보여주고 있지만 성능이 향상된 것은 볼 수 없습니다.

```
recall at 5 : 92.6%
mrr : 0.855%
```

혹시 모르니 chunk_size를 2000으로 한 실험도 진행해 보았습니다. 실험 결과 오히려 성능이 많이 떨어지는 것을 확인할 수 있습니다.

```
recall at 5 : 85.2%
mrr : 0.747%
```

청킹 전략에 따른 성능 비교 실험을 정리해보자면 제 블로그 포스트 데이터는 기술에 대한 설명 혹은 개인적으로 공부했던 내용들을 정리한 내용들이 많은 데이터기 때문에 청크 사이즈를 작게 잡기보다는 최소 1000으로 잡았을 때 문서 검색의 성능이 어느 정도 보장 되는 것을 확인했습니다. 특히나 청크 사이즈를 1000보다 조금 큰 1500과 같은 값을 했을 경우 성능 차이는 없었지만, 2000과 같이 2배 정도 차이가 나는 수치로 잡았을 때는 오히려 성능이 떨어지는 것을 확인하였습니다. 즉 청크 사이즈 1000을 기준으로 더 작은 사이즈는 각 청킹 문서에 오히려 필요한 정보들이 잘려나가게 되어 성능이 낮아졌고, 성능을 너무 크게 했을 때는 각 청킹 문서에 너무 많은 노이즈가 포함되어 성능이 더 큰 폭으로 떨어지는 것을 확인할 수 있었습니다. 그래서 저는 Baseline으로 잡았던 chunk_size 1000, chunk_overlap 150을 사용하기로 하였습니다.

### 3.2.4.2 Text Splitter에 따른 성능 비교

chunk_size와 chunk_overlap 크기를 비교하면서 진행한 실험에서는 두 수치의 변경에 따른 성능 향상을 볼 수 없었습니다. 그래서 이번엔 청킹을 진행하는 Text Splitter에 따른 성능 비교 실험을 진행해 보고자 합니다. 제가 블로그에 구축해 놓은 RAG 시스템에 사용되는 데이터는 이전에도 말씀드렸지만 블로그 포스트 데이터이며, 포스트 작성에 Markdown 형식으로 작성하고 있습니다. 따라서 저는 현재 사용하고 있는 RecursiveCharacterTextSplitter 대신 MarkdwonHeaderTextSplitter를 사용해 청킹된 문서들로 벡터 DB를 구축하고 문서 검색의 성능 평가를 진행해 보고자 합니다.

RecursiveChracterTextSplitter 대신 사용하는 MarkdownHeaderTextSplitter는 `headers_to_split_on` 이라는 변수에 정해준 Header를 기준으로 문서를 청킹하게 됩니다. 하지만 여기서 문제가 있는데 Header 기준으로 문서를 청킹하게 되면 각 헤드별로 어떤 헤드에는 정보가 집중되어 있어서 문서의 길이가 굉장히 길어지는 반면 `#`과 같은 큰 제목을 나타내는 헤더의 경우에는 간략한 개요 정보만 있어 해당 Header로 잘린 문서에는 아주 적은 양의 정보만 담기는 문제가 존재하게 됩니다. 따라서 MarkdownHeaderTextSplitter 단독으로 사용하지 않고 MarkdownHeaderTextSplitter로 잘린 문서들을 RecursiveCharacterTextSplitter를 이용해 한 번더 청킹하는 Hybrid 방식을 사용하고자 합니다. 아래는 MarkdownHeaderTextSplitter와 RecursiveCharacterTextSplitter를 결합한 문서 청킹 코드입니다.

```python
import os
import frontmatter
import re
from dataclasses import dataclass
from typing import List, Literal
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

@dataclass
class ChunkConfig:
    posts_dir: str = "본인의 .md 파일들이 있는 디렉토리 경로"
    chunk_size: int = 1000
    chunk_overlap: int = 150 # 오타 수정: overlab -> overlap
    # splitter_type을 통해 전략 선택 가능
    splitter_type: Literal["recursive", "markdown_hybrid"] = "markdown_hybrid"

def load_and_chunk_posts(config: ChunkConfig):
    documents = []

    # 1. 파일 순회
    for root, dirs, files in os.walk(config.posts_dir):
        for filename in files:
            if filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                if "sample" in filename:
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)

                    # 메타데이터 추출 로직 (기존 유지)
                    categories = post.get("categories", [])
                    dir_name = "".join([f"{c.lower()}/" for c in categories])
                    url_name = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", filename)
                    slug = url_name.replace('.md', '')

                    base_metadata = {
                        "title": post.get("title", "Untitled"),
                        "category": categories,
                        "tag": post.get("tags", []),
                        "url": f"https://icechickentender.github.io/{dir_name}{slug}/",
                        "filename": filename
                    }

                    # 2. Splitter 전략 선택
                    if config.splitter_type == "markdown_hybrid":
                        # [Step A] Markdown Header Splitter 적용
                        headers_to_split_on = [
                            ("#", "Header 1"),
                            ("##", "Header 2"),
                            ("###", "Header 3"),
                        ]
                        md_splitter = MarkdownHeaderTextSplitter(
                            headers_to_split_on=headers_to_split_on,
                            strip_headers=False # 본문 내 헤더 텍스트 유지 여부
                        )
                        md_header_splits = md_splitter.split_text(post.content)

                        # [Step B] Recursive Splitter로 보완 (2단계 분할)
                        # 헤더로 나뉜 덩어리가 여전히 chunk_size보다 클 경우를 대비합니다.
                        recursive_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=config.chunk_size,
                            chunk_overlap=config.chunk_overlap,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        
                        # split_documents를 사용하면 md_header_splits의 메타데이터가 자동 계승됩니다.
                        final_chunks = recursive_splitter.split_documents(md_header_splits)

                        # 기본 메타데이터 결합
                        for doc in final_chunks:
                            doc.metadata.update(base_metadata)
                            documents.append(doc)

                    else:
                        # 기존 Recursive 전략 (기본값)
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=config.chunk_size,
                            chunk_overlap=config.chunk_overlap,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        chunks = text_splitter.split_text(post.content)
                        for chunk in chunks:
                            documents.append(Document(page_content=chunk, metadata=base_metadata))

    print(f"✅ 전략 [{config.splitter_type}] 적용: 총 {len(documents)}개의 청크가 생성되었습니다.")
    return documents

# 실행부
if __name__ == "__main__":
    # Markdown 하이브리드 전략으로 실행
    config = ChunkConfig(
        chunk_size=1200,
        chunk_overlap=150,
        splitter_type="markdown_hybrid"
    )

    all_chunks = load_and_chunk_posts(config)
    
    if all_chunks:
        print("\n--- 첫 번째 청크 상세 정보 ---")
        print(f"Content:\n{all_chunks[0].page_content[:200]}...")
        print(f"Metadata: {all_chunks[0].metadata}")
```

이제 MarkdownHeaderTextSplitter와 RecursiveCharacterTextSplitter를 합친 hybrid 청킹 함수를 이용해 벡터 DB를 생성하고, 생성된 벡터 DB를 이용해 문서 검색 성능 평가를 진행해 보도록 하겠습니다. 성능 평가 결과 생각한 것과는 다르게 오히려 성능이 많이 떨어진 것을 확인할 수 있습니다. 이에 대한 원인으로 MarkdownHeaderTextSplitter로 나누게 되면 어떤 섹션은 200자, 어떤 섹션은 2000자가 되게 됩니다. 이렇게 되면 최종적으로 RecursiveCharacterTextSplitter에서 다시 한 번 청킹을 진행하게 되는데 이 때 기존 1200자 보다 작은 크기의 청크들은 그대로 남게 되어 이러한 청크들은 정보량이 적은 나머지 검색 경쟁력을 잃게되며 이러한 이유로 성능이 하락하는 듯 합니다.

```
recall at 5 : 85.2%
mrr : 0.735%
```

MarkdownHeaderTextSplitter로 인해 정보량이 적은 청크들에 좀 더 정보량을 넣어주도록 하기 위해 각 청크에 헤더 정보를 주입하는 실험을 진행해 보고자 합니다. 청크 내용 앞에 `[카테고리 > 상위헤더 > 현재헤더]` 정보를 추가해 주어 정보량은 적지만 그래도 각 청크들이 어느 카테고리의 어떤 헤더의 정보인지를 명확히 알려주어 검색 경쟁력을 좀 더 실어 주도록 하는 것입니다. 이 방식이 검색 경쟁력을 갖게 하는 이유는 제가 사용하는 임베딩 모델인 OpenAI의 `text-embedding-3-small` 모델은 텍스트의 앞부분에 더 큰 가중치를 두는 경향이 있씁니다. 청크 맨 앞에 "어떤 주제인지"를 명시하면 검색 유사도 점수가 훨씬 더 정교해질 수 있습니다. 다음은 기존 청크 함수에 각 청크에 카테고리와 헤더 정보를 추가하는 코드입니다.

```python
for doc in final_chunks:
    header_list = metadata["category"] + [doc.metadata.get(f"Header {i}") for i in range(1, 4)]
    breadcrumb = " > ".join(h for h in header_list if h)
    doc.page_content = f"[{breadcrumb}]\n{doc.page_content}"
    doc.metadata.update(metadata)
    documents.append(doc)
```

각 청크에 카테고리와 헤더 정보를 추가한 청크를 이용한 벡터 DB를 만들어 문서 검색 성능 평가를 진행해 보았습니다. 평가 결과 확실히 이전 보다는 성능이 향상된 것을 확인할 수 있습니다. 하지만 여전히 RecursiveCharacterTextSplitter를 사용하고 chunk_size 1000, chunk_overlap 150일 때보다 성능이 낮은 것을 확인할 수 있습니다. 그렇다면 여기서 다른 것은 chunk_size가 다른데 chunk_size도 똑같이 1000으로 하고 평가를 다시 진행해 보도록 하겠습니다.

```
recall at 5 : 92.6%
mrr : 0.809%
```

다음은 hybrid 방식 Text Splitter에 chunk_size 1000을 적용시켰을 때의 성능은 다음과 같습니다. MRR 값이 살짝 올랐지만 여전히 Baseline에는 미치지 못하고 있습니다. 이제 여기서 더 이상 Text Splitter를 이용한 실험은 큰 의미가 없으며 다음 실험을 진행해 보도록 하겠습니다.

```
recall at 5 : 92.6%
mrr : 0.818%
```

### 3.2.4.3 Reranker를 활용한 성능 비교

여태까지 청킹과 관련된 파라메터 그리고 Text Splitter에 따른 문서 성능 평가를 진행했습니다. 하지만 Baseline인 RecursiveCharacterTextSplitter를 사용하고 chunk_size 1000, chunk_overlap 150을 뛰어넘는 성능은 보이지 않고 있습니다. 그렇다면 이제 남은 실험은 Reranker를 활용한 실험과 Sparse Vector와 Dense Vector를 함께 사용하는 Hybrid 방식을 이용한 실험만 남아있습니다. 저는 우선 Reranker를 이용한 실험부터 진행해 보고자 합니다.

저는 허깅페이스에 예전에 Bi-Encoder와 Cross-Encoder를 이용한 성능 향상 실험을 위해 KLUE MRC 데이터로 미세 조정을 진행한 Cross-Encoder 모델이 있습니다. 사실 사용하는 평가 데이터를 이용해 Cross-Encoder를 미세 조정하면 더욱 좋겠지만 현재 가지고 있는 평가 데이터의 양이 적어 미세 조정까지는 힘든 상황이고, 우선 KLUE MRC 데이터로 미세 조정된 Cross-Encoder를 사용했을 때 성능이 얼마나 증가하는지만 한 번 살펴볼 예정이고, 추후에 Runpod를 이용해 GPU 환경에서 여러 데이터로 학습된 Cross-Encoder 혹은 또 다른 Rerank 방식을 적용해 평가를 진행해 보도록 하겠습니다.

아래는 Reranker 모델을 이용한 평가 코드입니다.

```python
import time
import numpy as np
from load_data import load_and_blog_data
from local_vector_db_generate import load_local_faiss
from sentence_transformers import CrossEncoder

def evaluate_with_reranker(golden_data, retriever, reranker_model, k=5, rerank_top_n=10):
    """

    :param golden_data:
    :param retriever:
    :param reranker_model:
    :param k:
    :param rerank_top_n:
    :return:
    """

    hits = 0
    reciprocal_ranks = []

    print(f"리 랭커 기반 검색 평가 시작 ... (Retrieved: {rerank_top_n} -> Reranked:{k})")

    for item in golden_data:
        query = item["input"]
        target_source = item["filename"]

        # 1. Retriever로 넉넉하게 n개 추출
        initial_docs = retriever.invoke(query)

        # 2. Cross-Endoer를 위한 (Query, Doc) 쌍 생성
        # 하이브리드 청킹 시 주입한 (Breadcrumb] 정보가 포함된 page_content를 사용
        pairs = [[query, doc.page_content] for doc in initial_docs]

        # 3. 점수 예측 (Logit 혹은 유사도 점수 반환)
        scores = reranker_model.predict(pairs)

        # 4. 점수 기준으로 재정렬
        # (score, doc) 쌍을 만들어 점수 내림차순 정렬
        reranked_results = sorted(zip(scores, initial_docs), key=lambda x : x[0], reverse=True)

        found_at_rank = 0

        # 5. 최종 상위 k개에 대해 메타데이터 비교
        for i, (score, doc) in enumerate(reranked_results[:k]):
            retrieved_source = doc.metadata.get("filename", "")

            if target_source in retrieved_source:
                hits += 1
                found_at_rank = i + 1
                break

        # 6. MRR 점수 계산
        if found_at_rank > 0:
            reciprocal_ranks.append(1.0 / found_at_rank)
        else:
            reciprocal_ranks.append(0.0)
    recall_at_k = (hits / len(golden_data)) * 100
    mrr = np.mean(reciprocal_ranks)

    return recall_at_k, mrr

if __name__ == "__main__":

    golden_data = load_and_blog_data("../data/blog_30_evaluate_data.json")

    k = 5
    vectorstore = load_local_faiss("../faiss_recursive_chunk1000_overlap150")
    cross_model = CrossEncoder('Laseung/klue-roberta-small-klue-mrc-cross-encoder-finetuned')

    k = 5
    rerank_top_n = 10

    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': rerank_top_n}
    )

    start = time.time()
    recall_at_k, mrr = evaluate_with_reranker(golden_data, retriever, cross_model, k, rerank_top_n)
    end = time.time()

    print("\n" + "="*50)
    print(f"📊 최종 평가 결과 (with Cross-Encoder Reranker)")
    print(f"   - Recall@{k} : {recall_at_k:.1f}%")
    print(f"   - MRR@{k}    : {mrr:.3f}")
    print(f"   - 평가 소요 시간 : {(end-start):.2f}초")
    print("="*50)
```

다음은 Baseline에 Reranker를 도입했을 때의 성능입니다. 성능 비교를 위해 Baseline의 성능도 같이 첨부했습니다. 성능 평가 결과를 보면 Baseline에 Reranker를 도입했을 때 Recall@k의 성능이 많이 향상된 것을 확인할 수 있고, MRR또한 0.1가량 향상된 것을 확인할 수 있습니다. 하지만 성능 평가에 소요된 시간을 비교하면 Baseline은 4.37초 Reranker를 도입했을 때는 76.82초로 대략 20배 가량 차이나는 것을 확인할 수 있습니다. 

```
리랭커를 도입하지 않은 기존의 평가 방식의 성능 평가 결과:
recall at 5 : 92.6%
mrr : 0.852%
성능 평가 소요 시간: 4.37 초

리랭커를 도입했을 때의 성능 평가 결과:
Recall@5 : 96.3%
MRR@5    : 0.861
평가 소요 시간 : 76.82초
```

위 Reranker의 도입에 따른 성능 평가는 Reranker를 도입했을 때 극적인 성능 변화가 있는 것이 아니며, 또한 Baseline 대비 소요 되는 시간이 20배 가량 됩니다. 그리고 무엇보다도 사용된 Cross-Encoder 모델은 CPU 환경에서도 동작하긴 하지만 GPU 환경이 아닌 곳에서는 더 많은 시간이 소요되기 때문에 이번에는 간단히 Reranker 모델을 적용해본 정도로만 정리하고 넘어가고자 합니다.

### 3.2.4.4 BM25 Sparse Vector를 적용한 Hybrid 검색 방식을 적용한 성능 평가

이제 마지막으로 BM25 Sparse Vector를 적용한 Hybrid 검색 방식을 적용한 성능 평가를 진행해 보도록 하겠습니다. Baseline 평가 코드에 LangChain에서 제공하는 BM25Retriever 와 EnsembelRetriever를 적용하여 평가를 진행해 보았습니다. BM25Retreiver와 EnsembeRetriever의 코드는 다음과 같습니다.

```python
from langchain_community.retrievers import BM25Retriever
from langchain_teddynote.retrievers import EnsembleMethod

# 1. BM25 리트리버 생성 (전체 청크 데이터 기반)
bm25_retriever = BM25Retriever.from_documents(all_chunks)
bm25_retriever.k = 5 # 상위 5개 추출

# 2. 기존 FAISS 리트리버 (이미 생성된 것 사용)
faiss_retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

# 3. 앙상블 리트리버 구성 (하이브리드)
# weights=[0.5, 0.5]는 벡터와 키워드 검색의 중요도를 1:1로 둡니다.
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)
```

다음은 BM25Retriever를 적용한 Hybrid 검색을 했을 시의 성능입니다. Recall@k의 성능은 Baseline보다는 높은 것을 확인할 수 있습니다. 하지만 MRR의 성능은 오히려 떨어지는 것을 확인할 수 있습니다. 이는 검색된 5개의 청크 중에서 실제 정답 청크가 뒤로 밀려나 있는 상황이며 Baseline보다 Recall@k는 높지만 MRR이 낮은 이유는 BM25Retriever의 검색 결과로 인해 기존 Dense Vector 의 결과도 함께 뒤로 밀려나는 것이 원인일 수 있습니다. 또한 현재 사용 중인 BM25Retriever는 toeknizer로 공백 단위로 token으로 나누고 있으며, 이로 인해 명사들이 조사와 분리 되지 않아 BM25Retriever의 성능이 낮아 MRR의 성능이 낮아지는 것일 수도 있습니다.

```
recall at 5 : 96.3%
mrr : 0.747%
성능 평가 소요 시간: 4.00 초
```

우선 Hybrid 방식에서 Sparse Vecotr와 Dense Vector의 비율 조정에 따른 성능 비교를 먼저 진행해 보도록 하겠습니다. Sparse Vector의 비율을 0.3, Dense Vector의 비율을 0.7을 했을 때의 성능입니다. Baseline 대비 소폭이지만 MRR의 성능이 향상된 것을 확인할 수 있습니다. 

```
sparse vector : 0.3
dense vector : 0.7
recall at 5 : 92.6%
mrr : 0.864%
성능 평가 소요 시간: 3.84 초
```

위 실험을 통해 키워드 기반 검색기의 비율을 조금만 적용하도록 하니 의미 기반 검색인 Dense Vector의 한계점을 어느 정도 보완해주고 있는 것을 확인할 수 있었습니다. 하지만 그래도 아직 목표치인 MRR의 수치가 0.9까지는 도달하지 못했습니다. 그럼 이번엔 BM25Retriever의 tokenizer를 형태소 분석기 기반인 BM25Retriever를 사용한 실험을 진행해 보도록 하겠습니다. 형태소 분석기는 오픈소스 기반인 Kiwi 형태소 분석기를 사용하였고, 다음은 Kiwi 형태소 분석기가 적용된 BM25Retriever 클래스를 직접 정의하였습니다.

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

다음은 Kiwi 형태소 분석기를 이용한 BM25Retriever를 적용한 hybrid 검색기의 성능입니다. Recall@k는 Baseline보다 높으며 MRR도 Baseline가 엇비슷합니다. 그렇다면 최적의 비율을 한 번 찾아보도록 하겠습니다. 일단 Dense Vector 자체의 성능이 괜찮으니 BM25Retriever의 비율을 조금 낮춰보도록 하겠습니다.

```
recall at 5 : 96.3%
mrr : 0.855%
성능 평가 소요 시간: 4.25 초
```

다음은 Kiwi 형태소 분석기를 이용한 BM25Retriever의 비율을 0.4로 하고 Dense Vecotr의 비율을 0.6으로 했을 때의 성능입니다. 그 외에 여러 비율로 실험을 진행해 보았지만 이 비율로 실험을 진행했을 때 가장 높은 성능을 보이는 것을 확인했습니다.

```
sparse vector : 0.4
dense vector: 0.6
kiwi 사용 비율 0.4 0.6
recall at 5 : 92.6%
mrr : 0.889%
```

# 4. 실험 마무리

여태까지 진행한 실험 외에도 여러가지 더 진행해 보고 싶은 실험들이 많지만 이번에는 여기까지만 진행하고, 추후에 성능 고도화를 진행할 때 이번에 하지 못했던 실험들을 진행해 보고자 합니다. 이제 마무리로 가장 성능이 좋았던 하이브리드 검색기를 이용한 DeepEval 평가를 진행해 보고자 합니다. 다음은 Baseline으로 진행했던 DeepEval 성능입니다.

```
Faithfulness: 98.88%
Answer Relevancy: 100.00% pass rate
Contextual Relevancy: 35.87% pass rate
```

다음은 Text Splitter와 chunk_size, chunk_overlap은 Baseline과 동일하지만 문서 검색기에 Kiwi를 적용한 BM25Retriever를 혼합한 Hybrid 문서 검색기를 사용했을 때의 DeepEval 성능 평가 결과입니다. 평가 결과를 보면 Baseline 대비 Contextual Relevancy의 성능이 6.67%가량 향상된 것을 확인할 수 있습니다. 

```
Faithfulness: 96.66%
Answer Relevancy: 100.00% pass rate
Contextual Relevancy: 42.54% pass rate
```

이번에 진행한 실험 내용들을 최종적으로 정리하면 다음과 같습니다.

1. 문서 검색기의 성능 지표는 DeepEval의 성능 지표와 관련성이 있는 것을 확인할 수 있었습니다. 즉 문서 검색기의 성능이 높다는 것은 최상위로 뽑힌 문서가 입력 쿼리와 연관된 문서일 가능성이 높다는 것이므로 최종적으로 `Contextual Relevancy`의 성능도 함께 향상된다는 것을 확인할 수 있었습니다.
2. DeepEval의 `Contextual Relevancy`는 문서 검색기가 뽑아준 상위 k개의 문서를 모두 사용합니다. 특히나 청크 문서를 데이터의 특성상 chunk_size를 크게 잡을 경우에는 상위 k개의 문서에 많은 노이즈가 포함될 수 있으니 정확한 성능 평가를 위해선 k의 값을 조절해주어야 합니다. 하지만 실제로 서비스될 때에는 최상위 한 개의 문서만을 이용하도록 하기에는 위험성이 있으므로 평가를 진행할 때에는 최상위 1개를 사용하도록 하고 실제 서비스 시에는 상위 3~5개의 문서를 제공해 주도록 해야 합니다.
3. RAG 시스템에 사용하는 데이터의 특성에 따른 Text Splitter, 청크 사이즈, 청크 오버랩에 따른 문서 검색기의 성능 평가를 꼭 진행해야 하며 데이터 특성에 맞는 알맞는 하이퍼 파라매터를 찾는 과정이 필수적이라는 것을 알게 되었습니다. 이와 함께 단순히 의미 검색인 Dense Vector만을 사용하기 보단 키워드 검색기를 혼합한 Hybrid 문서 검색기가 오히려 문서 검색기로써 좋은 성능을 보여준다는 것을 확인하였습니다.
4. Reranker는 RAG 시스템에 사용되는 문서의 개수가 많을 수록 무조건 사용되어야 하는 모델인 것을 알 수 있었습니다. 다만 이번엔 Reranker에 초점을 두지 않아서 여러 Reranker 모델을 이용한 실험을 진행하지 못한 것이 아쉬웠으며, Reranker를 적용할 때에는 성능과 속도의 트레이드오프를 많이 고려해야 한다는 것을 이번에 다시 한 번 깨달았습니다.
5. Hybrid 문서 검색기를 사용할 때에는 Tokenizer가 굉장히 중요하다는 것을 알게 되었습니다. 특히나 한국어의 경우 단순히 공백을 기준으로 Tokenizing을 진행하게 될 경우 실제로 의미를 가지는 명사들을 뽑아내지 못해 오히려 성능이 떨어지는 것을 확인했고, 형태소 분석기를 Tokenizer로 사용한 BM25Retriever를 적용했을 때 성능이 많이 향상되는 것을 확인하여 한국어에는 아직 형태소 분석기가 필수적이라는 것을 알게 되었습니다.

# 마치며

이전 포스트는 단순히 대학생의 과제 수준과 같이 Naive RAG 시스템의 구축에 대해서 진행했고 그 내용들을 정리했었습니다. 이번엔 구축한 RAG 시스템이 실제로 상용 시스템으로써 가치를 가지도록 하기 위한 정량적 평가 방법들에 대해서 알아보았고 알아본 평가 방법들로 실제 평가도 진행해 보았습니다.

우선 RAGAS나 DeepEval과 같은 LLM을 이용한 데이터 생성부터 평가까지 모두 자동화 되어 있는 프레임워크에 대해서도 알게되었습니다. 하지만 제가 사용해본 경험이 없고, 제가 느끼기엔 불안정한 것들도 많아서 저처럼 처음 사용하는 사람들에게 사용하기 많이 불편했습니다. 그리고 데이터 생성도 비용 때문에 제가 성능이 낮은 LLM 모델을 사용해서 그런진 모르겠지만 생성된 데이터의 품질이 좀 많이 떨어지는 느낌이 들었습니다. 그렇다고 좋은 모델을 사용하기에는 LLM 출력 비용이 만만치 않아서 이 또한 많은 부담으로 다가왔습니다. 사실 비용 때문에 프롬프트 엔지니어링을 적용한다던지 좀 더 좋은 모델을 사용했을 때의 데이터 품질이 어떤지 등을 비교 해봤어야 했지만 해보지 못한 것이 많은 아쉬움으로 남아 있습니다. 추후에 LLM을 사용해 성능이 높고 더욱 더 경량화된 Reranker 모델을 사용한 RAG 시스템의 성능 고도화도 진행해볼 예정인데 그 때 다시 한 번 평가 데이터 생성을 진지하게 해보고자 합니다.

두 번째로 문서 검색기와 관련한 평가를 진행한 것과 관련하여 데이터 특성에 따라 chunk size와 overlap 조정은 필수라는 것을 알게되었습니다. 제가 사용한 블로그 포스트 데이터는 운이 좋아서 처음 설정했던 chunk size 1000, overlap 150이 가장 성능이 좋았습니다만 사용하는 데이터의 어떤 정보를 품고 있고 전체 문서의 크기 문서에서 담고 있는 정보량이 얼마인지를 파악하는 실험은 필수라는 것을 알게 되었습니다. 그리고 이후에 진행했던 Text Splitter에 따른 성능 평가에서는 제가 사용하는 데이터는 블로그 포스트 데이터이고 마크다운 형식으로 작성했기 때문에 MarkdownHeaderTextSplitter를 이용한 청킹을 진행하게 되면 성능이 더 오를줄 알았습니다. 하지만 오히려 헤더 별로 잘리게 되어서 많은 정보가 포함되어 있지 않고 요약된 정보들이 포함되어 있는 상위 헤더들이 문서 검색의 상위 결과로 나오게 되어 오히려 성능이 떨어지는 것을 보면서 데이터 유형에 맞는 Text Splitter를 사용한다고 해서 무조건 성능이 오르는 것은 아니라는 것을 알게되었습니다. 다만 아쉬운 것이 있다면 의미 정보 기반으로 청킹을 해주는 SementicChunker 나 Parent-Child Chunking 등 좀 더 심화적인 Text Splitter를 이용한 실험은 진행해 보지 못한 것이 조금 아쉬웠습니다.

마지막으로 벡터 검색을 위해 의미 기반 벡터인 Dense 벡터만 사용하는 것보다는 키워드 기반인 BM25를 이용한 Sparse 벡터를 사용하는 것이 성능 향상에는 필수적이라는 것을 알게 되었습니다. 이전에 LangChain을 공부하면서 정리한 포스트에서도 나와있지만 입력 쿼리와의 의미 검색 수치가 높다는 이유만으로 사람이 봤을 때는 전혀 일치하지 않는 문서들이 검색되는 현상이 있었습니다. 이번에 직접 hybrid 문서 검색기를 구현해 보면서 실제로 성능 향상되는 것을 경험했고, 특히나 단순히 공백을 기준으로 Tokenizing 해주는 것보다 한국어 특성에 맞게 형태소 분석기를 이용한 Tokenizing을 했을 때 성능이 향상된 다는 것을 직접 체감할 수 있었습니다.

긴 글 읽어주셔서 감사드리며, 내용 중에 오타나 잘못된 내용 혹은 궁금한 사항이 있으실 경우에는 댓글 달아주시기 바랍니다.