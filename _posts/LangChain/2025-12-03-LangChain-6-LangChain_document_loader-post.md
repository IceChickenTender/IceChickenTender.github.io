---
title: "[LLM/RAG] LangChain - 6. LangChain의 Document Loader"
categories:
  - LangChain
tags:
  - LangChain
  
use_math: true  
toc: true
toc_sticky: true
toc_label: "LangChain의 Document Loader"
---

# 1. Document Loader 개요

LangChain 에서 Document Loader 는 다양한 소스에서 문서를 불러오고 처리하는 과정을 담당합니다. 특히 사전 지식이 필요한 지식 기반의 태스크, 정보 검색, 데이터 처리 작업 등을 처리할 때 반드시 필요합니다. Document Loader 의 주요 목적은 효율적으로 문서 데이터를 수집하고, 사용 가능한 형식으로 변환하는 것입니다.

	1. 다양한 소스 지원 : 웹 페이지, PDF 파일, 데이터베이스 등 다양한 소스에서 문서를 불러올 수 있습니다.
	2. 데이터 변환 및 정제 : 불러온 문서 데이터를 분석하고 처리하여, 랭체인의 다른 모듈이나 알고리즘이 처리하기 쉬운 형태로 변환합니다. 불필요한 데이터를 제거하거나, 구조를 변경할 수도 있습니다.
	3. 효율적인 데이터 관리 : 대량의 문서 데이터를 효율적으로 관리하고, 필요할 때 쉽게 접근할 수 있도록 합니다. 이를 통해 검색 속도를 향상시키고, 전체 시스템의 성능을 높일 수 있습니다.

# 2. 웹 문서(WebBaseLoader)

`WebBaseLoader` 는 특정 웹 페이지의 내용을 로드하고 파싱하기 위해 설계된 클래스로 태그를 제거하고 순수 텍스트만 추출하여 LangChain의 Document 형식으로 변환하는 클래스입니다. Python의 표준 urllib(또는 requests) 라이브러리로 요청을 보내고, bs4(BeautifulSoup)로 파싱합니다. 그리고 WebBaseLoader를 이용해 웹페이지에서 데이터를 크롤링하려면 필요한 데이터만 추출해야 하므로 어느정도 html에 대한 지식이 있어야 합니다.

예제 코드를 실행하기에 앞서 필요한 라이브러리부터 설치해 줍니다.

```bash
pip install langchain langchain-community
```

## 2.1 WebBaseLoader 파라미터

1. web_paths (Union[str, Sequence[str]])
  - 설명: 데이터를 로드할 대상 URL들의 리스트입니다.
  - 특징: 문자열 하나만 넣어도 되고, 리스트로 여러개를 넣어도 됩니다.

2. header_template(dict, Optional)
  - 설명: HTTP 요청을 보낼 때 사용할 헤더(Header) 정보입니다.
  - 역할: 웹사이트들이 봇(Bot) 접근을 막기위해 User-Agent를 검사할 때 브라우저인 척 속이는 역할을 합니다.

3. bs_kwargs(dict, Optional)
  - 설명: 내부적으로 사용되는 BeautifulSoup 객체에 전달할 파라미터입니다.
  - 활용(노이즈 제거): 웹 페이지 전체를 긁으면 네비게이션 바, 푸터(Footer), 광고 배너까지 다 들어옵니다. 이때 본문만 가져오라고 지시할 때 사용합니다.

4. verify_ssl(bool, Default:True)
  - 설명: HTTPS 요청 시 SSL 인증서를 검증할지 여부입니다.
  - 활용: 사내망(Private Network) 테스트 서버나, 인증서가 만료된 사이트를 긁어야 할 때 False로 둡니다.

5. requests_per_second(int, Default:2)
  - 설명: 여러 페이지를 긁을 때 서버에 부하를 주지 않도록 요청 속도를 제한합니다. 무작정 긁어오다간 차단당할 수 있으니 크롤링 해오는 페이지에서 제안한 제한값에 맞춰 조정해 주어야 합니다.

## 2.2 WebBaseLoader 이용하여 웹 페이지 데이터 가져오기

`web_paths` 매개변수는 로드할 웹 페이지의 URL 을 단일 문자열 또는 여러 개의 URL 시퀀스 배열로 지정할 수 있습니다. 여기서는 파이썬 튜플 형태로 2개의 URL 을 사용하고 있습니다.

`bs_kwargs` 매개변수는 BeautifulSoup 을 사용하여 HTML 을 파싱할 때 사용되는 인자들을 딕셔너리 형태로 제공합니다. 예제에서는 `bs4.SoupStrainer` 를 사용하여 특정 클래스 이름을 가진 HTML 요소만 파싱하도록 지정하고 있습니다. `article-header`, `article-title` 클래스를 가진 요소만 선택하여 파싱합니다.

`docs` 변수에는 로드된 문서들의 배열이 할당됩니다. 각 페이지별로 별도의 Document 객체로 변환되어 2개의 문서가 생성됩니다.

```python
# WebBaseLoader

import bs4
from langchain_community.document_loaders import WebBaseLoader

# 여러 개의 url 지정 가능

url1 = "https://blog.langchain.dev/customers-replit/"
url2 = "https://blog.langchain.dev/langgraph-v0-2/"

loader = WebBaseLoader(
    web_paths = (url1, url2),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("article-header", "article-content")
        )
    ),
)

docs = loader.load()

# 문서 개수 출력
print(f"docs num : {len(docs)}")

# 첫 번째 문서 출력
docs[0]
```

```
실행 결과

docs num : 2
Document(metadata={'source': 'https://blog.langchain.dev/customers-replit/'}, page_content='\nReplit is at the forefront of AI innovation with its platform that simplifies writing, running, and collaborating on code for over 30+ million developers. They recently released Replit Agent, which immediately went viral due to the incredible applications people could easily create with this tool.Behind the scenes, Replit Agent has a complex workflow which enables a highly custom agentic workflow with a high-degree of control and parallel execution. By using LangSmith, Replit gained deep visibility into their agent interactions to debug tricky issues.\xa0The level of complexity required for Replit Agent also pushed the boundaries of LangSmith. The LangChain and Replit teams worked closely together to add functionality to LangSmith that would satisfy their LLM observability needs. Specifically, there were three main areas that we innovated on:Improved performance and scale on large tracesAbility to search and filter within tracesThread view to enable human-in-the loop workflowsImproved performance and scale on large tracesMost other LLMOps solutions monitor individual API requests to LLM providers, offering a limited view of single LLM calls. In contrast, LangSmith from day one has focused on tracing the entire execution flow of an LLM application to provide a more holistic context.\xa0Tracing is important for agents due to their complex nature. It captures multiple LLM calls as well as other steps (retrieval, running code, etc). This gives you granular visibility into what’s happening, including at the inputs and outputs of each step, in order to understand the agent’s decision-making.\xa0Replit Agent was a ripe example for advanced tracing needs. Their agentic tool goes beyond simply reviewing and writing code, but also performs a wider range of functions – including planning, creating dev environments, installing dependencies, and deploying applications for users.\xa0As a result, Replit’s traces were very large - involving hundreds of steps. This posed significant challenges for ingesting data and displaying it in a visually meaningful way.To address this, the LangChain team improved their ingestion to efficiently process and store large volumes of trace data. They also improved LangSmith’s frontend rendering to display long-running agent traces seamlessly.Search and filter within traces to pinpoint issuesLangSmith has always supported search between traces, which allows users to find a single trace among hundreds of thousands based on events or full text search. But as Replit Agent’s traces got longer and longer, the Replit team needed to search within traces for specific events (oftentimes issues reported by alpha testers). This required augmenting existing search capabilities.In response, a new search pattern – searching within traces – was added to LangSmith. Instead of sifting and scrolling call-by-call within a large trace, users could now filter directly on a criteria they cared about (e.g. keywords in the inputs or outputs of a run). This greatly reduced Replit’s time needed to debug agent steps within a trace.Thread view to enable human-in-the-loop workflowsA key differentiator of Replit Agent was its emphasis on human-in-the-loop workflows. Replit Agent intends to be a tool where AI agents can collaborate effectively with human developers, who can come in and edit and correct agent trajectories as needed.With separate agents to perform roles like managing, editing, and verifying generated code,\xa0 Replit’s agents interacted with users continuously - often over long periods with multiple turns of conversation. However, monitoring these conversational flows was often difficult, as each user session would generate disjoint traces.\xa0To solve this, LangSmith’s thread view helped collate traces from multiple threads together that were related (i.e. from one conversation). This provided a logical view of all agent-user interactions across a multi-turn conversation, helping Replit better 1) find bottlenecks where users got stuck and 2) pinpoint areas where human intervention could be beneficial.\xa0ConclusionReplit is pushing the frontier of AI agent monitoring using LangSmith’s powerful observability features. By reducing the effort of loading long, heavy traces, the Replit team has greatly sped up the process of building and scaling complex agents. With faster debugging, improved trace visibility, and better handling of parallel tasks, Replit is setting the standard for AI-driven development.\t\n')
```

## 2.3 WebBaseLoader 이용하여 네이버 뉴스기사 데이터 가져오기

이번엔 네이버 뉴스기사를 크롤링해서 문서화 해보도록 하겠습니다.

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# 뉴스 기사 내용을 로드합니다.
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class":["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
    header_template = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
    },
)

docs = loader.load()
print(f"문서의 수: {len(docs)}")
print(docs)
```

```
Output:
문서의 수: 1
[Document(metadata={'source': 'https://n.news.naver.com/article/437/0000378416'}, page_content="\n출산 직원에게 '1억원' 쏜다…회사의 파격적 저출생 정책\n\n\n[앵커]올해 아이 낳을 계획이 있는 가족이라면 솔깃할 소식입니다. 정부가 저출생 대책으로 매달 주는 부모 급여, 0세 아이는 100만원으로 올렸습니다. 여기에 첫만남이용권, 아동수당까지 더하면 아이 돌까지 1년 동안 1520만원을 받습니다. 지자체도 경쟁하듯 지원에 나섰습니다. 인천시는 새로 태어난 아기, 18살될 때까지 1억원을 주겠다. 광주시도 17살될 때까지 7400만원 주겠다고 했습니다. 선거 때면 나타나서 아이 낳으면 현금 주겠다고 밝힌 사람이 있었죠. 과거에는 표만 노린 '황당 공약'이라는 비판이 따라다녔습니다. 그런데 지금은 출산율이 이보다 더 나쁠 수 없다보니, 이런 현금성 지원을 진지하게 정책화 하는 상황까지 온 겁니다. 게다가 기업들도 뛰어들고 있습니다. 이번에는 출산한 직원에게 단번에 1억원을 주겠다는 회사까지 나타났습니다.이상화 기자가 취재했습니다.[기자]한 그룹사가 오늘 파격적인 저출생 정책을 내놨습니다.2021년 이후 태어난 직원 자녀에 1억원씩, 총 70억원을 지원하고 앞으로도 이 정책을 이어가기로 했습니다.해당 기간에 연년생과 쌍둥이 자녀가 있으면 총 2억원을 받게 됩니다.[오현석/부영그룹 직원 : 아이 키우는 데 금전적으로 많이 힘든 세상이잖아요. 교육이나 생활하는 데 큰 도움이 될 거라 생각합니다.]만약 셋째까지 낳는 경우엔 국민주택을 제공하겠다는 뜻도 밝혔습니다.[이중근/부영그룹 회장 : 3년 이내에 세 아이를 갖는 분이 나올 것이고 따라서 주택을 제공할 수 있는 계기가 될 것으로 생각하고.][조용현/부영그룹 직원 : 와이프가 셋째도 갖고 싶어 했는데 경제적 부담 때문에 부정적이었거든요. (이제) 긍정적으로 생각할 수 있을 것 같습니다.]오늘 행사에서는, 회사가 제공하는 출산장려금은 받는 직원들의 세금 부담을 고려해 정부가 면세해달라는 제안도 나왔습니다.이같은 출산장려책은 점점 확산하는 분위기입니다.법정기간보다 육아휴직을 길게 주거나, 남성 직원의 육아휴직을 의무화한 곳도 있습니다.사내 어린이집을 밤 10시까지 운영하고 셋째를 낳으면 무조건 승진시켜 주기도 합니다.한 회사는 지난해 네쌍둥이를 낳은 직원에 의료비를 지원해 관심을 모았습니다.정부 대신 회사가 나서는 출산장려책이 사회적 분위기를 바꿀 거라는 기대가 커지는 가운데, 여력이 부족한 중소지원이 필요하다는 목소리도 나옵니다.[영상디자인 곽세미]\n\t\t\n")]
```

이번엔 여러 웹페이지를 한 번에 로드해보도록 하겠습니다. 

```python
loader = WebBaseLoader(
    web_paths=[
        "https://n.news.naver.com/article/437/0000378416",
        "https://n.news.naver.com/mnews/hotissue/article/092/0002340014?type=series&cid=2000063",
    ],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
    header_template={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
    },
)

# 데이터 로드
docs = loader.load()

# 문서 수 확인
print(len(docs))
print()


# 웹에서 가져온 결과를 출력
print(docs[0].page_content[:500])
print("==="*10)
print(docs[1].page_content[:500])
```

```
Output:
2


출산 직원에게 '1억원' 쏜다…회사의 파격적 저출생 정책


[앵커]올해 아이 낳을 계획이 있는 가족이라면 솔깃할 소식입니다. 정부가 저출생 대책으로 매달 주는 부모 급여, 0세 아이는 100만원으로 올렸습니다. 여기에 첫만남이용권, 아동수당까지 더하면 아이 돌까지 1년 동안 1520만원을 받습니다. 지자체도 경쟁하듯 지원에 나섰습니다. 인천시는 새로 태어난 아기, 18살될 때까지 1억원을 주겠다. 광주시도 17살될 때까지 7400만원 주겠다고 했습니다. 선거 때면 나타나서 아이 낳으면 현금 주겠다고 밝힌 사람이 있었죠. 과거에는 표만 노린 '황당 공약'이라는 비판이 따라다녔습니다. 그런데 지금은 출산율이 이보다 더 나쁠 수 없다보니, 이런 현금성 지원을 진지하게 정책화 하는 상황까지 온 겁니다. 게다가 기업들도 뛰어들고 있습니다. 이번에는 출산한 직원에게 단번에 1억원을 주겠다는 회사까지 나타났습니다.이상화 기자가 취재했습니다.[기자]한 그룹사가 오늘 파격적인 저출생 정책을 내놨
==============================

고속 성장하는 스타트업엔 레드팀이 필요하다


[이균성의 溫技] 초심, 본질을 잃을 때한 스타트업 창업자와 최근 점심을 같이 했다. 조언을 구할 게 있다고 했다. 당장 급한 현안이 있는 건 아니었다. 여러 번 창업한 경험이 있는데 지금 하고 있는 아이템은 대박 느낌이 든다고 헸다. 그런데 오히려 더 조심해야겠다는 생각이 들더란다. 조언을 구하고자 하는 바도 성장이 예상될 때 무엇을 경계해야 할지 알고 싶다는 거였다. 적잖은 스타트업 창업자를 만났지만 드문 사례였다.2년 가까이 스타트업 창업자를 릴레이 인터뷰 하면서 의미 있게 생각했던 것이 두 가지 있다. 첫째, 회사라는 단어보다 팀이라는 어휘를 주로 쓰고 있다는 점이었다. 그 표현의 유래나 의미 때문이라기보다는 팀이라는 말이 더 정겨워 뜻 깊게 생각된 듯하다. 이해관계보다 지향하는 뜻에 더 중점을 두고 하나의 마음으로 한 곳을 향해 달려가는 집단을 가리키는 표현이라는 생각에 더 정겨웠다.스타트업 대표들의 창업 동기는 대부분 ‘사회
```

IP 차단을 우회하기 위해 때때로 프록시를 사용할 필요가 있을 수 있습니다. 프록시를 사용하려면 로더에 프록시 딕셔너리를 전달해 사용할 수 있습니다.

```python
loader = WebBaseLoader(
    "https://www.google.com/search?q=parrots",
    proxies={
        "http": "http://{username}:{password}:@proxy.service.com:6666/",
        "https": "https://{username}:{password}:@proxy.service.com:6666/",
    },
    # 웹 기반 로더 초기화
    # 프록시 설정
)

# 문서 로드
docs = loader.load()
```

---

# 3. 텍스트 문서

## 3.1 TextLoader 파라미터

1. file_path(Union[str, Path])
  - 설명: 로드할 파일의 경로입니다.

2. encoding(str, Optional)
  - 설명: 로드할 파일의 인코딩입니다.

3. autodetect_encoder(bool)
  - 설명: 파일의 인코딩을 자동으로 탐지할지를 선택합니다.

## 3.2 TextLoader 이용하여 텍스트 파일 데이터 가져오기

`langchain_community` 라이브러리의 `document_loaders` 모듈에는 다양한 Document Loader 함수를 지원하고 있습니다. 이 중에서 `TextLoader` 를 사용하여 텍스트 파일을 불러올 수 있습니다. 그리고 텍스트 파일의 내용을 랭체인의 Document 객체로 변환하고 이를 리스트 형태로 반환합니다.

실습데이터 : [history.txt](https://github.com/tsdata/langchain-study/blob/main/data/history.txt)

현재 저는 구글 코랩에서 실습을 진행하고 있기 때문에 제 구글 드라이브와 연동해서 history.txt 파일을 loader 하도록 하였습니다.

```python
# Text 파일 Loader

from langchain_community.document_loaders import TextLoader

file_path = "/content/drive/MyDrive/LangChain/history.txt"

loader = TextLoader(file_path)
data = loader.load()

print(type(data))
print(len(data))

data
```

출력해보면 리스트 안에 Document 객체가 담겨 있는 것을 볼 수 있습니다.

```
실행 결과 

<class 'list'>
1
[Document(metadata={'source': '/content/drive/MyDrive/LangChain/history.txt'}, page_content='한국의 역사는 수천 년에 걸쳐 이어져 온 긴 여정 속에서 다양한 문화와 전통이 형성되고 발전해 왔습니다. 고조선에서 시작해 삼국 시대의 경쟁, 그리고 통일 신라와 고려를 거쳐 조선까지, 한반도는 많은 변화를 겪었습니다.\n\n고조선은 기원전 2333년 단군왕검에 의해 세워졌다고 전해집니다. 이는 한국 역사상 최초의 국가로, 한민족의 시원이라 할 수 있습니다. 이후 기원전 1세기경에는 한반도와 만주 일대에서 여러 소국이 성장하며 삼한 시대로 접어듭니다.\n\n4세기경, 고구려, 백제, 신라의 삼국이 한반도의 주요 세력으로 부상했습니다. 이 시기는 삼국이 각각 문화와 기술, 무력을 발전시키며 경쟁적으로 성장한 시기로, 한국 역사에서 중요한 전환점을 마련했습니다. 특히 고구려는 북방의 강대국으로 성장하여 중국과도 여러 차례 전쟁을 벌였습니다.\n\n7세기 말, 신라는 당나라와 연합하여 백제와 고구려를 차례로 정복하고, 한반도 최초의 통일 국가인 통일 신라를 건립합니다. 이 시기에 신라는 불교를 국교로 채택하며 문화와 예술이 크게 발전했습니다.\n\n그러나 10세기에 이르러 신라는 내부의 분열과 외부의 압력으로 쇠퇴하고, 이를 대체하여 고려가 성립됩니다. 고려 시대에는 과거제도의 도입과 더불어 청자 등 고려 고유의 문화가 꽃피었습니다.\n\n조선은 1392년 이성계에 의해 건국되어, 1910년까지 이어졌습니다. 조선 초기에는 세종대왕이 한글을 창제하여 백성들의 문해율을 높이는 등 문화적, 과학적 성취가 이루어졌습니다. 그러나 조선 후기에는 내부적으로 실학의 발전과 함께 사회적 변화가 모색되었으나, 외부로부터의 압력은 점차 커져만 갔습니다.\n\n19세기 말부터 20세기 초에 걸쳐 한국은 제국주의 열강의 침략을 받으며 많은 시련을 겪었습니다. 1910년, 한국은 일본에 의해 강제로 병합되어 35년간의 식민 지배를 받게 됩니다. 이 기간 동안 한국인들은 독립을 위한 다양한 운동을 전개했으며, 이는 1945년 일본의 패망으로 이어지는 독립으로 결실을 맺었습니다.\n\n해방 후 한반도는 남북으로 분단되어 각각 다른 정부가 수립되었고, 1950년에는 한국전쟁이 발발하여 큰 피해를 입었습니다. 전쟁 후 남한은 빠른 경제 발전을 이루며 오늘날에 이르렀습니다.\n\n한국의 역사는 오랜 시간 동안 수많은 시련과 도전을 겪으며 형성된 깊은 유산을 지니고 있습니다. 오늘날 한국은 그 역사적 배경 위에서 세계적으로 중요한 역할을 하고 있으며, 과거의 역사가 현재와 미래에 어떻게 영향을 미치는지를 이해하는 것은 매우 중요합니다.')]
```

Document 객체에는 `page_content` 필드와 `metadata` 필드가 들어 있습니다. `page_content` 는 텍스트로 변환된 문자열이 들어 있습니다.

---

# 4. 디렉토리 폴더(DirectoryLoader)

## 4.1 DirectoryLoader 이용하여 특정 폴더의 모든 파일을 가져오기

`DirectoryLoader` 를 사용하여 디렉토리 내의 모든 문서를 로드할 수 있습니다. `DirectoryLoader` 인스턴스를 생성할 때 문서가 있는 디렉토리의 경로와 해당 문서를 식별할 수 있는 glob 패턴을 지정합니다.

실습데이터1 : [history.txt](https://github.com/tsdata/langchain-study/blob/main/data/history.txt)
실습데이터2 : [places.txt](https://github.com/tsdata/langchain-study/blob/main/data/places.txt)

```python
# DirectoryLoader 이용

from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(path="/content/drive/MyDrive/LangChain", glob="*.txt", loader_cls=TextLoader)

data = loader.load()

print("첫 번째 데이터")
print(data[0].page_content)
print("\n\n")

print("두 번째 데이터")
print(data[1].page_content)

```

아래와 같이 두 개의 txt 파일을 읽어오는 것을 확인할 수 있습니다.

```
첫 번째 데이터
한국의 역사는 수천 년에 걸쳐 이어져 온 긴 여정 속에서 다양한 문화와 전통이 형성되고 발전해 왔습니다. 고조선에서 시작해 삼국 시대의 경쟁, 그리고 통일 신라와 고려를 거쳐 조선까지, 한반도는 많은 변화를 겪었습니다.

고조선은 기원전 2333년 단군왕검에 의해 세워졌다고 전해집니다. 이는 한국 역사상 최초의 국가로, 한민족의 시원이라 할 수 있습니다. 이후 기원전 1세기경에는 한반도와 만주 일대에서 여러 소국이 성장하며 삼한 시대로 접어듭니다.

4세기경, 고구려, 백제, 신라의 삼국이 한반도의 주요 세력으로 부상했습니다. 이 시기는 삼국이 각각 문화와 기술, 무력을 발전시키며 경쟁적으로 성장한 시기로, 한국 역사에서 중요한 전환점을 마련했습니다. 특히 고구려는 북방의 강대국으로 성장하여 중국과도 여러 차례 전쟁을 벌였습니다.

7세기 말, 신라는 당나라와 연합하여 백제와 고구려를 차례로 정복하고, 한반도 최초의 통일 국가인 통일 신라를 건립합니다. 이 시기에 신라는 불교를 국교로 채택하며 문화와 예술이 크게 발전했습니다.

그러나 10세기에 이르러 신라는 내부의 분열과 외부의 압력으로 쇠퇴하고, 이를 대체하여 고려가 성립됩니다. 고려 시대에는 과거제도의 도입과 더불어 청자 등 고려 고유의 문화가 꽃피었습니다.

조선은 1392년 이성계에 의해 건국되어, 1910년까지 이어졌습니다. 조선 초기에는 세종대왕이 한글을 창제하여 백성들의 문해율을 높이는 등 문화적, 과학적 성취가 이루어졌습니다. 그러나 조선 후기에는 내부적으로 실학의 발전과 함께 사회적 변화가 모색되었으나, 외부로부터의 압력은 점차 커져만 갔습니다.

19세기 말부터 20세기 초에 걸쳐 한국은 제국주의 열강의 침략을 받으며 많은 시련을 겪었습니다. 1910년, 한국은 일본에 의해 강제로 병합되어 35년간의 식민 지배를 받게 됩니다. 이 기간 동안 한국인들은 독립을 위한 다양한 운동을 전개했으며, 이는 1945년 일본의 패망으로 이어지는 독립으로 결실을 맺었습니다.

해방 후 한반도는 남북으로 분단되어 각각 다른 정부가 수립되었고, 1950년에는 한국전쟁이 발발하여 큰 피해를 입었습니다. 전쟁 후 남한은 빠른 경제 발전을 이루며 오늘날에 이르렀습니다.

한국의 역사는 오랜 시간 동안 수많은 시련과 도전을 겪으며 형성된 깊은 유산을 지니고 있습니다. 오늘날 한국은 그 역사적 배경 위에서 세계적으로 중요한 역할을 하고 있으며, 과거의 역사가 현재와 미래에 어떻게 영향을 미치는지를 이해하는 것은 매우 중요합니다.



두 번째 데이터
경복궁
서울의 중심에 위치한 경복궁은 조선 시대의 왕궁으로, 한국의 역사와 전통 문화를 체험할 수 있는 대표적인 명소입니다. 광활한 궁궐 안에는 경회루, 근정전 등 다양한 전통 건축물이 있으며, 정기적으로 궁궐 경비 교대식과 전통 공연이 열립니다.

남산 서울타워
서울의 스카이라인을 대표하는 남산 서울타워는 서울 시내를 한눈에 볼 수 있는 최고의 전망대입니다. 타워 주변의 남산 공원은 산책과 휴식을 즐기기에 적합하며, 연인들의 자물쇠 벽도 유명합니다.

부산 해운대 해수욕장
부산의 해운대 해수욕장은 국내외 관광객에게 사랑받는 한국 최대의 해수욕장 중 하나입니다. 넓은 백사장과 도심 속의 접근성이 좋은 위치로, 여름철에는 수많은 피서객으로 붐빕니다.

제주도
한국의 남쪽에 위치한 제주도는 화산섬으로, 아름다운 자연 풍경과 독특한 문화를 자랑합니다. 세계 자연 유산에 등재된 한라산, 청정 해변, 용두암 등 다양한 자연 명소와 함께 특색 있는 음식과 문화가 관광객을 맞이합니다.

경주
신라 천년의 고도 경주는 한국의 역사적인 도시 중 하나로, 불국사, 석굴암, 첨성대 등 수많은 유적지와 문화재가 있습니다. 신라의 역사와 문화를 체험할 수 있는 최적의 장소입니다.

인사동
서울의 인사동은 전통 찻집, 공예품 가게, 갤러리가 즐비한 문화 예술의 거리입니다. 한국의 전통 문화와 현대 예술이 공존하는 이곳에서는 다양한 기념품을 구입하고 전통 차를 맛볼 수 있습니다.

한강
서울을 가로지르는 한강은 도시의 휴식처로, 한강공원, 자전거 도로, 피크닉 장소 등을 제공합니다. 야경이 아름다운 한강에서는 다양한 레저 활동과 행사가 열리며, 여름에는 불꽃놀이 축제가 인기입니다.

순천만 국가정원
전라남도 순천에 위치한 순천만 국가정원은 다양한 식물과 아름다운 정원이 조화를 이루는 곳으로, 자연과 함께하는 힐링의 시간을 제공합니다. 인근의 순천만 습지는 천연 기념물로 지정된 생태 관광지입니다.
```

---

# 5. CSV 문서(CSVLoader)

## 5.1 CSVLoader 이용하여 CSV 파일 데이터 가져오기

`langchain_community` 라이브러리의 `document_loaders` 모듈의 `CSVLoader` 클래스를 사용하여 CSV 파일에서 데이터를 로드합니다. CSV 파일의 각 행을 추출하여 서로 다른 Document 객체로 변환합니다. 이들 문서 객체로 이루어진 리스트 형태로 반환합니다.

다음 코드는 `CSVLoader` 클래스의 인스턴스를 이용하여 주택금융관련 지수 데이터를 담고 있는 CSV 파일을 로드하고 있습니다. 인코딩 방식은 `cp949` 를 사용하였습니다.

실습 데이터 : [한국주택금융공사_주택금융관련_지수_20160101.csv](https://github.com/tsdata/langchain-study/blob/main/data/%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%AE%E1%86%A8%E1%84%8C%E1%85%AE%E1%84%90%E1%85%A2%E1%86%A8%E1%84%80%E1%85%B3%E1%86%B7%E1%84%8B%E1%85%B2%E1%86%BC%E1%84%80%E1%85%A9%E1%86%BC%E1%84%89%E1%85%A1_%E1%84%8C%E1%85%AE%E1%84%90%E1%85%A2%E1%86%A8%E1%84%80%E1%85%B3%E1%86%B7%E1%84%8B%E1%85%B2%E1%86%BC%E1%84%80%E1%85%AA%E1%86%AB%E1%84%85%E1%85%A7%E1%86%AB_%E1%84%8C%E1%85%B5%E1%84%89%E1%85%AE_20160101.csv)

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="/content/drive/MyDrive/LangChain/csv_sample.csv", encoding='cp949')
data = loader.load()

data[0]
```

```
실행 결과

Document(metadata={'source': '/content/drive/MyDrive/LangChain/csv_sample.csv', 'row': 0}, page_content='연도: 2004-01-01\n전국소득대비 주택가격 비율: 4.21\n서울소득대비 주택가격 비율: 4.89\n부산소득대비 주택가격 비율: 3.95\n대구소득대비 주택가격 비율: 3.73\n인천소득대비 주택가격 비율: 4.65\n광주소득대비 주택가격 비율: 2.81\n대전소득대비 주택가격 비율: 4.68\n울산소득대비 주택가격 비율: 2.66\n세종소득대비 주택가격 비율: 0\n경기소득대비 주택가격 비율: 4.17\n강원소득대비 주택가격 비율: 2.49\n충북소득대비 주택가격 비율: 2.62\n충남소득대비 주택가격 비율: 2.17\n전북소득대비 주택가격 비율: 3.12\n전남소득대비 주택가격 비율: 2.12\n경북소득대비 주택가격 비율: 2.12\n경남소득대비 주택가격 비율: 3.81\n제주소득대비 주택가격 비율: 2.99\n전국평균 대출금액  평균 연소득: 2.36\n서울평균 대출금액  평균 연소득: 2.61\n부산평균 대출금액  평균 연소득: 2.35\n대구평균 대출금액  평균 연소득: 2.24\n인천평균 대출금액  평균 연소득: 2.7\n광주평균 대출금액  평균 연소득: 1.6\n대전평균 대출금액  평균 연소득: 2.26\n울산평균 대출금액  평균 연소득: 1.67\n세종평균 대출금액  평균 연소득: 0\n경기평균 대출금액  평균 연소득: 2.42\n강원평균 대출금액  평균 연소득: 1.44\n충북평균 대출금액  평균 연소득: 1.53\n충남평균 대출금액  평균 연소득: 1.21\n전북평균 대출금액  평균 연소득: 1.9\n전남평균 대출금액  평균 연소득: 1.42\n경북평균 대출금액  평균 연소득: 1.31\n경남평균 대출금액  평균 연소득: 2.06\n제주평균 대출금액  평균 연소득: 1.28')
```

## 5.2 데이터 출처 정보를 특정 필드(열, column)로 지정

`CSVLoader` 를 사용하여 CSV 파일을 로드할 대, `source_column` 속성에 데이터의 출처 정보(`source`)로 사용될 열의 이름을 지정할 수 있습니다. 다음 예제에서는 "연도" 열이 각 행 데이터의 출처 정보로 사용됩니다. `source` 속성을 확인해 보면 `2004-01-01` 와 같이 해당 행의 "연도" 열에 있는 값이 적용된 것을 알 수 있습니다.

```python
loader = CSVLoader(file_path="/content/drive/MyDrive/LangChain/csv_sample.csv", encoding='cp949',
                   source_column='연도'
                   )
data = loader.load()

data[0]
```

```
실행 결과

Document(metadata={'source': '2004-01-01', 'row': 0}, page_content='연도: 2004-01-01\n전국소득대비 주택가격 비율: 4.21\n서울소득대비 주택가격 비율: 4.89\n부산소득대비 주택가격 비율: 3.95\n대구소득대비 주택가격 비율: 3.73\n인천소득대비 주택가격 비율: 4.65\n광주소득대비 주택가격 비율: 2.81\n대전소득대비 주택가격 비율: 4.68\n울산소득대비 주택가격 비율: 2.66\n세종소득대비 주택가격 비율: 0\n경기소득대비 주택가격 비율: 4.17\n강원소득대비 주택가격 비율: 2.49\n충북소득대비 주택가격 비율: 2.62\n충남소득대비 주택가격 비율: 2.17\n전북소득대비 주택가격 비율: 3.12\n전남소득대비 주택가격 비율: 2.12\n경북소득대비 주택가격 비율: 2.12\n경남소득대비 주택가격 비율: 3.81\n제주소득대비 주택가격 비율: 2.99\n전국평균 대출금액  평균 연소득: 2.36\n서울평균 대출금액  평균 연소득: 2.61\n부산평균 대출금액  평균 연소득: 2.35\n대구평균 대출금액  평균 연소득: 2.24\n인천평균 대출금액  평균 연소득: 2.7\n광주평균 대출금액  평균 연소득: 1.6\n대전평균 대출금액  평균 연소득: 2.26\n울산평균 대출금액  평균 연소득: 1.67\n세종평균 대출금액  평균 연소득: 0\n경기평균 대출금액  평균 연소득: 2.42\n강원평균 대출금액  평균 연소득: 1.44\n충북평균 대출금액  평균 연소득: 1.53\n충남평균 대출금액  평균 연소득: 1.21\n전북평균 대출금액  평균 연소득: 1.9\n전남평균 대출금액  평균 연소득: 1.42\n경북평균 대출금액  평균 연소득: 1.31\n경남평균 대출금액  평균 연소득: 2.06\n제주평균 대출금액  평균 연소득: 1.28')
```

---

## 5.3 CSV 파싱 옵션 지정

`CSVLoader` 클래스를 사용할 때 추가적인 CSV 관련 설정을 `csv_args` 매개변수를 통해 지정할 수 있습니다. `csv_args` 는 파이썬 표준 라이브러리인 `csv` 모듈에 전달될 추가 인자들을 담는 딕셔너리입니다. 다음 예제는 CSV 파일의 구분자(`delimiter`)로 줄바꿈 문자(`\n`)를 지정하고 있습니다. 줄바꿈 문자를 기준으로 각 필드를 구분하기 때문에, 기본 값인 콤마(`,`)를 적용했을 경우와 파싱된 결과에 차이가 있습니다.

```python
loader = CSVLoader(file_path="/content/drive/MyDrive/LangChain/csv_sample.csv", encoding='cp949',
                   csv_args={
                       'delimiter':'\n',})
data = loader.load()

data[0]
```

# 6. PDF 문서

`langchain_community` 라이브러리의 `document_loaders` 모듈에는 PDF 문서에서 텍스트를 추출하여 파일 내용을 Document 객체로 변환하는 다양한 유형의 Document Loader 를 제공합니다.

## 6.1 PDF 문서 페이지별 로드(PyPDFLoader)

### 6.1.1 PyPDFLoader 이용해 PDF 파일 데이터 가져오기

`langchain_community` 패키지에서 제공하는 `PyPDFLoader` 를 사용해 PDF 파일에서 텍스트를 추출합니다. 이 명령을 사용하려면 pypdf 라이브러리를 먼저 설치해야 합니다.

```python
!pip install -q pypdf
```

`PyPDFLoader` 인스턴스를 사용해 지정된 PDF 파일을 열고, Document 객체의 개수를 출력해봅니다.

실습 데이터 : [000660_SK_2023.pdf](https://github.com/tsdata/langchain-study/blob/main/data/000660_SK_2023.pdf)

```python
from langchain_community.document_loaders import PyPDFLoader

pdf_filepath = "/content/drive/MyDrive/LangChain/000660_SK_2023.pdf"
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()

len(pages)
```

```
실행 결과

21
```

10번째에 있는 Document 객체를 출력해 봅니다.

```
실행 결과

Document(metadata={'producer': 'Adobe PDF Library 15.0', 'creator': 'Adobe InDesign 16.2 (Macintosh)', 'creationdate': '2023-06-26T16:16:31+09:00', 'moddate': '2023-06-26T17:21:06+09:00', 'trapped': '/False', 'source': '/content/drive/MyDrive/LangChain/000660_SK_2023.pdf', 'total_pages': 21, 'page': 10, 'page_label': '11'}, page_content='2120\nESG Special Report 2023\n투자 검토 단계\nPre-Acquisition (인수 전)\n01\n포트폴리오 ESG 관리 체계\n장기적 관점에서 기업가치 제고를 실현하기 위해 핵심자산인 \n투자 포트폴리오의 경제적 가치와 함께 ESG 가치를 \n통합적으로 관리하기 위한 체계를 구축하고 있습니다.\n투자 검토 시점부터 인수 후, 회수 시점까지 투자\nLife Cycle에 걸쳐 적용되는 체계적인 ESG 관리를 \n기반으로 내부적으로는 ESG를 고려한 합리적인 투자의사 \n결정을 이행하고, 시장에서는 포트폴리오의 기업가치가 \n시장에서 제대로 평가받으며 나아가 사회·환경에 미치는 \n파급력을 높일 수 있도록 노력하겠습니다.\n포트폴리오 ESG 관리 원칙\nSK주식회사 투자회사\n기업가치 관점의\nESG 중점관리 \n항목 도출\n자사 ESG \n관리전략\nESG 성과 \n데이터 관리\n기업가치와\nESG 성과 \n연계성 분석\n포트폴리오 \nESG 관리전략 \nUpgrade\n성장단계\n산업특성\nESG Divestment 전략 검토\n       ESG Exit 리포트 발간\n ·    인수 이후 ESG Value-up 기반 Exit 전략 도출\n ·    중대 ESG 리스크/기회 현황 및   \nESG 관리·공시 수준 확인\n셀사이드(Sell-side) 점검사항 관리\n       중대 ESG 이슈 존재 여부 검토\n ·    매각 대상 시장 내 ESG 규제 준수 여부 확인\n ·    ESG 우수 영역에 대한 정보공개 및    \n기회 확대 방안 제시\n ·    국내외 책임투자 기준 부합 여부 확인\n ·    우수 관리 영역 정보공개 및   \n이해관계자 커뮤니케이션\n매각/투자 회수 단계\n03\nExit (투자 회수)\n정기 ESG 점검\n       투자회사 분류 \n ·   전체 포트폴리오를 16개 업종, 기업 규모에 따라 3개 그룹으로 구분\n       ESG 중점관리 항목 도출 \n  ·   ESG 외부평가 및 주가 상관관계 상위 영역 분석에 따라   \n산업 핵심관리 영역 도출\n       정기 평가 실시 \n   ·   연 2회 ESG 실적점검 실시,     \n중요 ESG 리스크/기회 관련 이슈 식별\n보유 단계\n02\nValue-up Period (보유 기간)\nESG 기반 주주 소통\n        주주 소통 대상 안건\n  ·   해당 산업 ESG 중점관리 항목과 연관된 리스크가 식별되었거나,  \n연중 중대한 ESG Controversy Issue(예상하지 못한 우려) 발생 시\n       투자회사 유형별 관여 방식\n ·    이사회 기타비상무이사를 통한 소통\n ·    ESG 점검 리포트 또는 주주서한 발송\n기후 리스크 관리\n        전환 리스크\n ·    탄소 규제/가격 변동에 따른 재무영향 점검  \n[수익성] 매출액/영업이익 대비 탄소 비용 추이 고려  \n[경제성] 투자 대비 감축 수단 효과 검토  \n[시장성] 경쟁사 대비 속도/수준 감안 대응전략 점검\n       물리적 리스크\n ·    자산 소재 지역의 취약한 이상기후 요인 식별\n ·    고위험 기업 대상 관리방안 마련 권고\nESG 실사\n       ESG 실사 수행\n ·   실사 대상 기업 산업 및 규모에 따른 \n체크리스트 생성\n ·   점검항목은 유형에 따라   \nESG 리스크와 관리체계로 구분\n ·   서면진단 및 현장실사 실시\n        시사점 도출\n ·    Valuation 반영을 통한   \n인수가액 조정\n ·   PMI(Post-Merger Integration, \n인수 합병 후 통합 과정)   \n개선 과제 제시\nESG는 의사결정과 행동에 반드시 \n반영되어야 하는 필수 요소이자 \n기회입니다. SK는 기업가치의 건강한 \n성장과 이해관계자의 지속가능한 행복을 \n위해 경영의사결정의 DNA로서 \nESG(환경/사회/지배구조)를 \n내재화하고 있습니다.\nESG 관리의 단계적 고도화\n포트폴리오 ESG 관리 역량 축적, \n글로벌 Top-tier 수준 ESG 관리체계 확보\n성장단계별 관리 차별화\n투자기업 성숙도(기업의 Life Cycle)에\n따라 적합한 ESG Value-up 실현\n산업별 중점관리항목 체계화\n산업별 기업가치에 큰 영향을 주는 \nESG 이슈 집중 관리\nWhere we are heading    |     How we get there    |     What we are preparing')
```

---

## 6.2 PDF 문서의 메타 데이터를 상세하게 추출하기(PyMuPDFLoader)

### 6.2.1 PyMuPDFLoader 이용하여 PDF 파일 데이터 가져오기

`langchain_community.document_loaders` 모듈의 PyMuPDFLoader 클래스는 PyMuPDF 를 사용하여 PDF 파일의 페이지를 로드하고, 각 페이지를 개별 `document` 객체로 추출합니다. 특히 PDF 문서의 자세한 메타데이터를 추출하는데 강점이 있습니다.

실습 데이터 : [000660_SK_2023_2023.pdf](https://github.com/tsdata/langchain-study/blob/main/data/000660_SK_2023.pdf)

`PyMuPDFLoader` 클래스를 사용하려면, 개발환경에 `PyMuPDF` 라이브러리를 설치해야합니다.

```python
!pip install pymupdf
```

`PyMuPDF` 인스턴스를 사용하여 지정된 PDF 파일을 로드하면, 각 페이지가 하나의 Document 객체로 일대일로 변환됩니다.

```python
from langchain_community.document_loaders import PyMuPDFLoader

pdf_filepath = "/content/drive/MyDrive/LangChain/000660_SK_2023.pdf"
loader = PyMuPDFLoader(pdf_filepath)

pages = loader.load()

print(len(pages))

```

```
실행 결과

21
```

첫 번째 Document 객체의 내용(`page_content`)을 출력해서 확인해 봅니다.

```python
from langchain_community.document_loaders import PyMuPDFLoader

pdf_filepath = "/content/drive/MyDrive/LangChain/000660_SK_2023.pdf"
loader = PyMuPDFLoader(pdf_filepath)

pages = loader.load()

pages[0].page_content
```

```
실행 결과

1
Where we are heading    |     How we get there    |     What we are preparing
ESG Special Report
2023 
NAVIGATING 
UNCERTAINTIES TO ENSURE  
SUSTAINABLE 
GROWTH
```

---

## 6.3 온라인 PDF 문서 로드(OnlinePDFLoader)

### 6.3.1 OnlinePDFLoader 이용하여 온라인 PDF 파일의 데이터를 가져오기

"Transformers" 논문 (`https://arxiv.org/pdf/1706.03762.pdf`)을 로드하여 페이지 내용을 추출하고, 로드된 페이지 수와 첫 페이지의 내용 일부를 출력하는 과정을 처리합니다. `langchain_community` 라이브러리의 `OnlinePDFLoader` 클래스를 사용합니다.

실습 데이터 : [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

구글 코랩 환경에서 실행하기 위해선 다음 라이브러리를 설치해 주어야 합니다.

```python
!pip install unstructured-pytesseract
```

```python
from langchain_community.document_loaders import OnlinePDFLoader

# Transformers 논문 로드
loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762")
pages = loader.load()

print(len(pages))
```

```
실행 결과

1
```

로드된 문서 객체의 내용을 출력하여 확인합니다. 첫 페이지의 텍스트 내용 중 처음 1000 자를 출력합니다.

```python
from langchain_community.document_loaders import OnlinePDFLoader

# Transformers 논문 로드
loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762")
pages = loader.load()

pages[0].page_content[:1000]
```

```
실행 결과

1706.03762v7 [cs.CL] 2 Aug 2023

arXiv

Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.

Attention Is All You Need

Ashish Vaswani* Noam Shazeer* Niki Parmar* Jakob Uszkoreit* Google Brain Google Brain Google Research Google Research avaswani@google.com noam@google.com nikip@google.com usz@google.com

Llion Jones* Aidan N. Gomez* ¢ Lukasz Kaiser* Google Research University of Toronto Google Brain llion@google.com aidan@cs.toronto.edu lukaszkaiser@google.com

Illia Polosukhin* + illia.polosukhin@gmail.com

Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing 
```

---

## 6.4 특정 폴더의 모든 PDF 문서 로드 (PyPDFDirectoryLoader)

### 6.4.1 PyPDFDirectoryLoader 이용하여 특정 폴더의 모든 PDF 파일 가져오기

`langchain_community.document_loaders` 모듈의 `PyPDFDirectoryLoader` 클래스는 지정된 디렉토리에서 모든 PDF 문서를 한 번에 가져옵니다.

실습 데이터1 : [00660_SK_2023.pdf](https://github.com/tsdata/langchain-study/blob/main/data/000660_SK_2023.pdf)
실습 데이터2 : [300720_한일시멘트_2023.pdf](https://github.com/tsdata/langchain-study/blob/main/data/300720_%E1%84%92%E1%85%A1%E1%86%AB%E1%84%8B%E1%85%B5%E1%86%AF%E1%84%89%E1%85%B5%E1%84%86%E1%85%A6%E1%86%AB%E1%84%90%E1%85%B3_2023.pdf)

`PyPDFDirectoryLoader` 의 인스턴스를 생성하고, load 메소드를 호출하여 해당 디렉토리의 모든 PDF 문서를 로드하고, data 변수에 할당합니다. `len(data)` 는 로드된 문서 객체의 총 개수를 반환합니다. `PyPDFDirectoryLoader` 가 디렉토리 내의 PDF 파일들을 가져와서 페이지별로 문서 객체로 변환하게 됩니다.

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("/content/drive/MyDrive/LangChain/")
data = loader.load()

len(data)
```

```
실행 결과

100
```

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("/content/drive/MyDrive/LangChain/")
data = loader.load()

print(data[0].metadata)
print("\n")
print(data[-1].metadata)
```

첫 번째 문서에는 SK 보고서가 두 번째 문서에는 한일시멘트 보고서가 들어간 것을 확인할 수 있습니다.

```
실행 결과

{'producer': 'Adobe PDF Library 15.0', 'creator': 'Adobe InDesign 16.2 (Macintosh)', 'creationdate': '2023-06-26T16:16:31+09:00', 'moddate': '2023-06-26T17:21:06+09:00', 'trapped': '/False', 'source': '/content/drive/MyDrive/LangChain/000660_SK_2023.pdf', 'total_pages': 21, 'page': 0, 'page_label': '1'}

{'producer': 'iLovePDF', 'creator': 'Adobe InDesign 16.4 (Macintosh)', 'creationdate': '2023-07-28T10:47:36+09:00', 'moddate': '2023-07-28T11:18:40+09:00', 'trapped': '/False', 'source': '/content/drive/MyDrive/LangChain/300720_한일시멘트_2023.pdf', 'total_pages': 79, 'page': 78, 'page_label': '79'}
```

# 7. LangChain Document Loader의 lazy_load()

LangChain에서 지원하는 모든 로더는 두 가지 메서드를 가지고 있습니다. load() 메서드와 lazy_load() 메서드입니다. load() 메서드는 모든 문서를 메모리에 리스트로 한꺼번에 올립니다. lazy_load()는 파이썬의 제너레이터(Generator)를 반환합니다. 문서를 하나씩 필요할 때만 메모리에 올립니다. 이것만 봤을 때는 굳이 lazy_load()를 쓸 필요 없이 그냥 load() 메서드를 사용하면 되는거 아닌가? 하는 생각이 드실 수 있습니다. 하지만 실무에서는 컴퓨터 혹은 서버가 처리할 수 있도록 조금 조금씩 문서 데이터가 들어오지 않습니다. 또한 대용량의 텍스트 데이터는 GB 단위로 올라가게 되며 이러한 대용량 데이터를 load() 메서드를 이용해 한 번에 읽으려고 한다면 OOM(OutOfMemory) 에러가 발생합니다. 즉 이러한 상황을 방지하기 위해 필요한게 lazy_load() 메서드이며, 이 메서드를 사용해 문서 데이터를 하나씩 읽고 -> DB에 저장하고 -> 메모리를 해제하는 파이프라인을 짜야 견고한 프로그램을 만들 수 있습니다. 아래 예제는 lazy_load()를 이용한 간단한 예제입니다.

```python
# 대용량 처리 시 권장 패턴
loader = TextLoader("huge_log_file.txt")

# 리스트로 받지 않고 반복문으로 순회
for doc in loader.lazy_load():
  # 여기서 벡터 Db 저장 로직 수행
  pass
```

# 마치며

LangChain에서 지원하는 다양한 문서 로더들 중에서 가장 많이 사용되는 로더들에 대해서 알아보았습니다. 다른 로더들은 차후에 필요하다고 생각이 되면 따로 다뤄 보도록 하겠습니다.

긴 글 읽어주셔서 감사드리며 본문 내용 중에 잘못된 내용이나 오타, 궁금하신 사항이 있으실 경우 댓글 남겨주시기 바랍니다.

# 참조

- 판다스 스튜디오 - 랭체인(LangChain) 입문부터 응용까지(https://wikidocs.net/book/14473)
- 테디노트 - LangChain 한국어 튜토리얼KR(https://wikidocs.net/book/14314)