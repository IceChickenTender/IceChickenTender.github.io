---
title: "[FastAPI] FastAPI 시작하기: 개발환경 구성부터 첫 API까지"
categories:
  - FastAPI
tags:
  - FastAPI
  
toc: true
toc_sticky: true
toc_label: "FastAPI 시작하기: 개발환경 구성부터 첫 API까지"
---

# 개요

AI/NLP 모델을 실제 서비스에 연결하기 위해서는 모델 개발뿐만 아니라 API와 BackEnd 기본 구조를 이해할 필요가 있습니다. 특히나 Python을 주로 사용해서 개발 되는 AI 모델들의 경우 Fast API를 주로 사용합니다. 특히나 토이 프로젝트를 진행하면서 간단하게 FastAPI를 이용하였는데, 이번에 마음 먹고 FastAPI 공식 문서를 이용해 기초부터 공부하면서 포스트로 정리해 보고자 하였습니다.

특히 이 포스트를 작성하는 목적은 단순히 공식 문서의 예제를 따라가는 것이 아니라, FastAPI의 기본 동작 구조와 API 설계의 핵심 개념을 다시 복습할 수 있도록 정리하는 것에 초점을 두었습니다.

# 1. PyCharm 프로젝트 구성

## 1.1 가상환경 구성

1. PyCharm의 "File" 탭에서 "New Project"를 클릭해서 새로운 프로젝트를 만들어 줍니다. 저는 FastAPI 학습용이기 때문에 "fastapi-study"로 만들어 주었습니다.

    <div align="center">
      <img src="/assets/images/fastapi/1/new_project.png" width="35%" height="40%"/>
    </div>

    <br>

2. 새로운 프로젝트를 만들 때 "Interpreter Type"은 "Project venv"를 선택해주면 자동으로 가상환경 세팅이 됩니다.

3. 가상환경이 세팅되어 있는지 확인하는 방법은 다음과 같습니다.

    1. "File > Settings > Project: 프로젝트명 > Python Interpreter"로 들어갑니다.

        <div align="center">
          <img src="/assets/images/fastapi/1/settings.png" width="35%" height="40%"/>
        </div>

        <br>

    2. 선택되어 있는 "Python Interpreter"의 경로가 "C:\...\프로젝트명\.venv\Scripts\python.exe" 인지 확인합니다.

        <div align="center">
          <img src="/assets/images/fastapi/1/python_interpreter.png" width="55%" height="40%"/>
        </div>

        <br>

    3. 만약 경로가 너무 길어 확인이 힘들다면 "Python Interperter" 제일 오른쪽에 있는 화살표를 누른 후 Show all을 누른 후에 확인해 줍니다.

        <div align="center">
          <img src="/assets/images/fastapi/1/show_all.png" width="55%" height="40%"/>
        </div>

        <br>

        <div align="center">
          <img src="/assets/images/fastapi/1/python_interpreter_path.png" width="55%" height="40%"/>
        </div>

        <br>

## 1.2 FastAPI 설치

이제 PyCharm에 구성한 가상환경에 FastAPI를 설치해 보도록 하겠습니다. 현재 FastAPI 공식 문서는 `uv` 사용을 기본 예시로 보여주고 있습니다만, 가상환경을 직접 구성한 경우에는 `pip install "afstapi[standard]"` 방식도 공식적으로 안내하고 있으며, 저는 가상환경을 구성해서 하므로 `pip install "afstapi[standard]"` 로 진행해 보도록 하겠습니다.

## 1.2.1 pip 업데이트

혹시 모르니 pip 업데이트를 먼저 진행해 줍니다. FastAPI 공식 가상환경 문서에서도 패키지 설치 전에 `pip`를 최신 버전으로 올리는 것을 권장하고 있습니다. 터미널을 키고 아래 명령어를 실행해 줍니다. 터미널은 PyCharm 아래쪽에 보시면 있습니다.

<div align="center">
  <img src="/assets/images/fastapi/1/terminal.png" width="55%" height="40%"/>
</div>

<br>

```bash
python -m pip install --upgrade pip
```

## 1.2.2 FastAPI 설치

공식 문서의 직접 설치 예시는 `pip install "fastapi[standard]"` 이고, 여기서는 현재 가상환경의 Python에 정확히 설치되도록 `python -m pip` 형태를 사용한 것입니다. 또한 `fastapi[standard]`로 설치하면 FastAPI 본체뿐 아니라 개발에 일반적으로 필요한 표준 의존성들도 함께 설치됩니다. 현재 공식 Tutorial도 이 설치 옵션을 기본으로 사용하고 있습니다.

```bash
python -m pip install "fastapi[standard]"
```

터미널에 아래 이미지와 같이 뜨면 정상적으로 설치가 된 것입니다.

<div align="center">
  <img src="/assets/images/fastapi/1/fastapi_install_success.png" width="75%" height="40%"/>
</div>

<br>

## 1.3 첫 FastAPI 애플리케이션 실행

FastAPI 공식 문서인 "First Steps(첫걸음)"에 있는 가장 간단한 다음 코드를 실행시켜 보도록 하겠습니다. 새로 만든 프로젝트에 "src" 디렉토리를 생성한 후에 "main.py"라는 python 파일을 만들었고 해당 파일안에 아래 코드를 넣어 주었습니다.

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
```

저는 PyCharm을 이용한 가상 환경에서 실행하므로 공식 문서에서 말하는 라이브 서버로 실행하려면 터미널에서 "fastapi dev main.py" 명령으로 실행을 했습니다.

그럼 <http://127.0.0.1:8000> 에 가서 JSON 응답을 확인해 보도록 하겠습니다. 아래 이미지는 보면 정상적으로 출력되는 것을 확인할 수 있습니다.

<div align="center">
  <img src="/assets/images/fastapi/1/result.png" width="35%" height="40%"/>
</div>

<br>

# 2. Path Operation

FastAPI에서 API를 작성할 때 가장 기본적으로 이해해야 하는 개념 중 하나는 **Path Operation**입니다. FastAPI에서는 특정 URL과 HTTP Method를 조합하여 클라이언트의 요청을 처리할 동작을 정의하며, 이를 Path Operation이라고 표현합니다.

## 2.1 FastAPI의 Path Operation

Path Operation을 이해하려면 먼저 `Path`와 `Operation`을 각각 구분해서 볼 필요가 있습니다.

### 2.1.1 Path

Path는 URL에서 도메인 이후 `/`부터 시작하는 경로를 의미합니다. 예를 들어 다음 URL이 있다고 하겠습니다. "https://exmaple.com/users/10" 여기에서 Path는 "/users/10"입니다.

FastAPI 공식 문서에서는 Path를 `Endpoint` 또는 `Route`라고 부르는 경우도 있다고 설명하고 있습니다. API에서는 이러한 Path를 이용하여 서로 다른 리소스나 기능을 구분합니다. 예를 들어 다음과 같이 각각 다른 리소스를 표현할 수 있습니다.

```
/users
/items
/posts
/models
```

### 2.1.2 Operation

Operation은 해당 Path에 대해 수행할 HTTP Method를 의미합니다. 대표적으로 다음과 같은 HTTP Method가 있습니다.

- GET
- POST
- PUT
- DELETE
- PATCH

일반적인 REST API에서는 다음과 같은 의미로 사용하는 경우가 많습니다.

- GET : 데이터 조회
- POST : 데이터 생성
- PUT : 데이터 수정
- DELETE : 데이터 삭제

FastAPI가 이러한 의미를 강제로 제한하는 것은 아니지만, 일반적인 API 설계에서는 이러한 관례를 따르는 경우가 많습니다. FastAPI와 OpenAPI에서는 각 HTTP Method를 하나의 `Operation`으로 표현합니다. 따라서 "@app.get("/")"는 "Path(/)"와 "Operation(GET)"을 표현합니다.

이를 정리하면 Path Operation은 특정 Path에 특정 HTTP Method로 요청이 들어왔을 때 수행할 API 동작을 정의하는 것이라고 이해할 수 있습니다.

## 2.2 Path Operation Decorator

FastAPI에서는 Python의 **Decorator**를 이용하여 Python 함수와 Path Operation을 연결합니다. 첫 예제의 "@app.get("/")" 부분이 Path Operation Decorator입니다.

```python
@app.get("/")
```

위 코드는 FastAPI에게 다음과 같은 의미를 전달합니다.

- Path: /
- Operation: GET
- 처리 함수: 바로 아래에 선언된 함수

즉

```python
@app.get("/")
async def root():
  ...
```

는 쉽게 표현하면 `/` Path로 `GET`요청이 들어오면 `root()` 함수를 이용하여 요청을 처리한다. 라는 의미입니다.

Python의 Decoratro 관점에서 이해하면 일반적인 Decorator

```python
@decorator
def func():
  ...
```

는 개념적으로 다음과 같은 형태로 이해할 수 있습니다.

```python
func = decorator(func)
```

FastAPI에서도 이 Decorator 기능을 이용하여 개발자가 정의한 Python 함수를 특정 API 요청과 연결할 수 있습니다.

FastAPI는 GET 이외에도 다른 HTTP Method에 대응하는 Decorator를 제공하고 있습니다.

```python
@app.get("/users") 
async def get_users(): 
  ... 

@app.post("/users") 
async def create_user(): 
  ... 

@app.put("/users/{user_id}") 
async def update_user(user_id: int): 
  ... 

@app.delete("/users/{user_id}") 
async def delete_user(user_id: int): 
  ...
```

각 Decorator는 다음과 같은 Path Operation을 나타냅니다.

- @app.get() : GET
- @app.post() : POST
- @app.put() : PUT
- @app.delete() : DELETE

FastAPI에는 `PATCH`, `HEAD`, `OPTIONS` 등 다른 HTTP Method에 대응하는 Decorator도 존재합니다. 또한 하나의 Python 파일에서 Path Operation Decorator를 여러번 사용할 수 있습니다.

## 2.3 Path Operation Function

Path Operation Decorator 바로 아래에 선언되는 Python 함수를 Path Operation Function이라고 합니다. 첫 예제를 다시 살펴보면

```python
@app.get("/") 
async def root(): 
  return {"message": "Hello World"}
```

위 코드에서 "async def root()" 가 Path Operation Function입니다. FastAPI는 요청을 받았을 때 Path Operation Decorator에 등록되어 있는 `root()` 함수를 호출합니다. 전체 흐름을 표현하면 다음과 같습니다.

```
Client 
↓ 
GET / 
↓ 
FastAPI 
↓ 
@app.get("/") 
↓
root() 
↓ 
return {"message": "Hello World"} 
↓ 
HTTP Response
```

즉, Path Operation Decorator가 HTTP 요청과 Python 함수를 연결하는 역할을 한다면, Path Operation Function은 실제로 해당 요청을 처리하는 Python 코드입니다.

# 3. Swagger UI와 OpenAPI

FastAPI는 작성된 API 정보를 바탕으로 OpenAPI 명세를 자동 생성하고, 이를 이용해 Swagger UI와 같은 API 문서 화면을 자동으로 제공합니다.

## 3.1 OpenAPI란?

API를 개발하면 해당 API를 사용하기 위해 다음과 같은 정보가 필요합니다.

- 어떤 URL로 요청해야 하는가?
- 어떤 HTTP Method를 사용하는가?
- 어떤 Parameter를 전달해야 하는가?
- 요청 데이터의 구조는 어떻게 되는가?
- 어떤 형태의 응답을 반환하는가?

이러한 정보를 정리한 것을 API 명세라고 합니다. 문제는 API마다 명세를 표현하는 방법이 제각각이면 사람이 아닌 프로그램이 이를 자동으로 이해하고 활용하기 어렵다는 문제가 발생합니다.

OpenAPI Specification(OAS)은 HTTP API를 설명하기 위한 표준화된 형식을 정의합니다. 즉, API 자체의 구조를 모두 동일하게 만드는 것이 아니라 서로 다른 API를 설명하는 방법을 표준화한 것입니다.

## 3.2 Swagger UI란?

Swagger UI는 OpenAPI 명세를 읽어 사람이 보기 쉬운 웹 문서 형태로 보여주는 도구입니다.

FastAPI 애플리케이션을 실행한 뒤 기본적으로 다음 주소에서 확인할 수 있습니다. "http://127.0.0.1:8000/docs"

Swagger UI에서는 작성된 API의 다음 정보를 확인할 수 있습니다.

- Path
- HTTP Method
- Parameter
- Request Body
- Response
- Status Code

또한 단순히 API 명세를 확인하는 것뿐만 아니라 `Try it out` 기능을 이용하여 브라우저에서 실제 API 요청을 보내고 응답을 확인할 수 있습니다. 따라서 개발 과정에서 Backend 개발자가 만든 API를 직접 확인하거나 Frontend 개발자 등이 API 사용 방법을 파악하고 테스트할 때 유용합니다.

중요한 것은 Swagger UI 자체가 API 명세의 원본은 아니라는 점입니다. Swagger UI는 OpenAPI 형식으로 작성된 API 정보를 읽어 사람이 사용하기 편한 화면으로 보여주는 역할을 합니다.

## 3.3 FastAPI와 OpenAPI / Swagger UI의 관계

FastAPI에서는 개발자가 OpenAPI 문서를 직접 작성하지 않아도 됩니다. 예를 들어 다음과 같이 API를 작성했다고 하겠습니다.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{users_id}")
async def get_user(user_id: int):
  return {"user_id": user_id}
```

FastAPI 코드에서 다음과 같은 정보를 파악할 수 있습니다.

- Path -> "/users/{user_id}"
- HTTP Method -> GET
- Parameter -> user_id
- Parameter Type -> integer

FastAPI는 이러한 정보를 이용하여 OpenAPI 명세를 자동으로 생성합니다. 그리고 Swagger UI는 이 OpenAPI 명세를 읽어 `/docs`에서 API 문서를 제공합니다.

# 4. Path Parameters

## 4.1 Path Parameter란?

Path Parameter는 URL Path 내부에 포함되는 동적인 값으로 Path Parameter는 URL 경로의 일부를 변수로 사용하여 특정 리소스를 식별하기 위한 값입니다. 예를 들어 "/users/10"과 같은 URL이 있다고 하면, 여기서 "/users"는 사용자 리소스를 의미하고, "10"은 그 중 특정 사용자를 식별하는 값입니다.

FastAPI에서는 다음과 같이 Path Parameter를 선언할 수 있습니다.

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):
  return {"user_id" : user_id}
```

위 코드에서 `{user_id}`가 Path Parameter입니다.

## 4.2 언제 사용하는가?

Path Parameter는 주로 특정 리소스 하나를 식별해야 할 때 사용합니다. 예를 들면 다음과 같은 Path들이 있다고 하겠습니다.

```
GET /users/10
GET /posts/35
GET /orders/A1024
```

각각은 다음과 같은 의미로 볼 수 있습니다.

```
/users/10 -> 10번 사용자
/posts/35 -> 35번 게시글
/orders/A1024 -> A1024주문
```

DB의 개념과 비교하면, Primary Key나 Unique한 값을 이용하여 특정 Row를 조회하는 것과 비슷하게 이해할 수 있습니다. 다만 Path Parameter가 반드시 DB의 Primary Key와 일치하는 것은 아닙니다.

## 4.3 타입 선언과 Validation

FastAPI에서는 Python 타입 힌트를 이용해 Path Parameter의 타입을 선언할 수 있습니다.

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
  return {"item_id": item_id}
```

`item_id: int`로 선언하면 FastAPI가 요청 값을 `int`로 변환하고, 변환할 수 없는 값이 들어오면 Validation Error를 반환합니다.

# 5. Query Parameters

## 5.1 Query Parameter란?

Query Parameter는 URL의 `?` 뒤에 `key=value` 형태로 전달되는 값입니다. Query Parameter는 특정 리소스를 식별하기 보다는 요청 결과에 조건이나 옵션을 추가하기 위해 사용하는 값입니다. URL 에서는 "?key=value"와 같은 형태로 표현합니다. 여러 값을 전달할 경우 `&`로 연결합니다. "?name=kim&limit=10"과 같은 형태로 쓰입니다. 

Path Parameter가 URL 경로의 일부라면 Query Parameter는 Path 뒤에 추가되는 부가적인 요청 정보라고 볼 수 있습니다.

FastAPI에서는 Path에 포함되지 않은 함수 매개변수를 기본적으로 Query Parameter로 인식합니다.

## 5.2 언제 사용하는가?

Query Parameter는 주로 리소스 목록에 대해 다음과 같은 조건을 지정할 때 사용합니다.

- 검색
- 필터링
- 정렬
- Pagination
- 조회 개수 제한
- 기타 선택 옵션

DB와 비교하면 `WHERE`, `ORDER BY`, `LIMIT` 등의 조건ㅇ르 지정하는 것과 비슷하게 이해할 수 있습니다.

따라서 Path Parameter와의 핵심 차이는 다음과 같습니다. Path Parameter는 "어떤 리소스를 대상으로 하는가"를 표현하고, Query Parameter는 "그 리소스를 어떤 조건이나 방식으로 조회할 것인가"를 표현합니다.

## 5.3 기본값과 Optional Parameter

FastAPI에서는 함수 매개변수의 기본값을 이용하여 Query Parameter의 기본값을 지정할 수 있습니다.

```python
@app.get("/users")
async def get_users(limit: int = 10):
  return {"limit": limit}
```

`limit` 값을 전달하지 않으면 기본값인 10이 사용됩니다.

Query Parameter는 선택적으로도 사용할 수 있습니다. 다음 예제 코드를 보도록 하겠습니다.

```python
@app.get("/users")
async def get_users(name: str | None = None):
  return {"name": name}
```

이 경우 `name`을 전달하지 않아도 요청이 가능하며, 값이 없으면 `None`이 됩니다. 반대로 기본값을 지정하지 않으면 필수 Query Parameter가 됩니다.

```python
@app.get("/users")
async def get_users(name: str):
  return {"name": name}
```

따라서 FastAPI에서는 함수의 타입 힌트와 기본값을 통해 Query Parameter의 타입과 필수 여부를 함께 표현할 수 있습니다.

# 6. Request Body

## 6.1 Request Body란?

`Request Body`는 Client가 Server에 보내는 HTTP Request의 본문(Body)에 담기는 데이터를 의미합니다.

지금까지 살펴본 Path Parameter와 Query Parameter 역시 Client가 Server로 데이터를 전달하는 방법이지만, 데이터가 전달되는 위치가 다릅니다.

- Path Parameter: URL 경로에 포함되는 데이터
- Query Parameter: URL의 Query String에 포함되는 데이터
- Request Body: HTTP Request의 Body에 포함되는 데이터

Request Body는 여러 필드로 구성된 구조화된 데이터를 서버로 전달할 때 주로 사용합니다. 반대로 Server가 Client에게 응답하면서 전달하는 본문의 데이터는 `Response Body`라고 합니다.

## 6.2 FastAPI에서 Request Body 사용하기

Fast API에서는 일반적으로 Pydantic의 `BaseModel`을 이용하여 Request Body의 데이터 구조를 정의합니다.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
  name: str
  description: str | None = None
  price: float
  tax: float | None = None

@app.post("/items/")
async def create_item(item: Item):
  return item
```

위 코드를 보면 Item은 Client가 전달할 Request Body의 데이터 구조를 나타냅니다. 그리고 Path Operation Function에서 Pydantic Model을 타입으로 선언합니다. FastAPI를 이를 보고 `item`을 Request Body에서 가져와야 하는 데이터로 판단합니다.

이후 Request Body로 전달된 데이터를 `Item` 모델에 맞게 처리하여 Python 객체로 함수에 전달합니다.

따라서 개발자는 Request Body의 JSON 데이터를 직접 파싱하지 않고, `item.name`, `item.price`와 같이 Python 객체로 사용할 수 있습니다.

## 6.3 Path / Query / Request Body 차이

세 가지 방식은 모두 Client가 Server로 데이터를 전달하는 방법이지만 용도가 다릅니다.

|구분|데이터 위치|주된 용도|
|-|-|-|
|Path Parameter|URL Path|특정 Resource 식별|
|Query Parameter|Query String|조회 조건, 필터링 등의 부가 정보 전달|
|Request Body|HTTP Request Body|구조화된 데이터 전달|

FastAPI는 Path Operation Function의 Parameter 선언을 바탕으로 각각 구분합니다.

- URL의 Path의 `{parameter}`와 이름이 일치하면 Path Parameter
- Path에 포함되지 않은 일반적인 단순 타입 Parameter는 Query Parameter
- Pydantic Model 타입의 Parameter는 Request Body

따라서 FastAPI에서는 별도의 Request Body 파싱 코드를 직접 작성하지 않아도 함수의 Parameter와 타입 선언을 통해 요청 데이터를 받을 수 있습니다.

## 6.4 Requst Body와 Pydantic

FastAPI에서 Request Body를 이해할 때 가장 중요한 부분은 Pydantic과의 관계입니다. Pydantic Model은 Request Body가 어떤 필드를 가지며 각 필드가 어떤 타입이어야 하는지를 정의합니다.

```python
class Item(BasemModel):
  name: str
  price: float
```

FastAPI는 이 모델을 기준으로 Client가 전달한 Request Body를 처리합니다.

```
HTTP Request Body 
        ↓ 
Pydantic Model 
        ↓ 
데이터 구조 및 타입 확인 
        ↓ 
Python Object 
        ↓ 
Path Operation Function
```

즉, Pydantic Model은 단순히 Request Body의 형태를 표현하는 역할만 하는 것이 아니라 전달받은 데이터가 정의된 구조와 타입에 맞는지 검증하는 역할도 합니다. 또한 FastAPI는 Pydantic Model의 정보를 활용하여 OpenAPI Schema와 API 문서를 자동으로 생성합니다. 따라서 Request Body의 핵심은 다음과 같이 정리할 수 있습니다.

> Request Body는 Client가 HTTP Request의 Body를 통해 Server로 전달하는 데이터이며, FastAPI에서는 Pydantic Model을 이용하여 그 데이터의 구조를 정의하고 검증한 뒤 Python 객체로 사용할 수 있도록 처리합니다.