---
title: "[MarkDown] Mark Down 문법 정리"
categories:
  - MarkDown
tags:
  - MarkDown

toc: true
toc_sticky: true
toc_label: "문법 정리"
---

요즘 대부분 회사에서 Git을 쓰고 있고 Github에 유용하고 다양한 오픈 소스들이 많아서 README.md가 무조건 쓰이고 있으며, 현재 다니는 회사의 이슈 게시글도 마크다운 문법으로 작성하도록 되어 있어 마크다운 문법을 공부해야지 하고 생각하고 있었습니다.   


하지만 혼자 공부한다고 여러 다른 기술 블로그를 깔짝 거리기만 하니 시간이 조금만 지나도 까먹고 해서 개인적으로 ppt 등과 같은 자료로 정리를 해두어야지 하는 필요성을 느꼈습니다.   


그러다 마침 블로그를 시작하게 되었고, 시작하는 김에 다른 분들도 보고 저도 제 블로그는 언제든 들어갈 수 있으니 여기다가 정리하자고 마음 먹게 되어 이렇게 마크다운 문법에 대해서 정리하게 되었습니다.

# 1. MarkDown에 대해
## 1.1 MarkDown 이란?
MarkDown은 텍스트 기반의 마크업 언어로 2004년 존 그루버에 의해 만들어 졌으며, 쉽게 쓰고 읽을 수 있으며, HTML로 변환이 가능합니다.
MarkDown은 특수 기호와 문자를 이용한 매우 간단한 구조의 문법을 사용하여 웹에서도 빠르게 컨텐츠를 작성하고, 직관적으로 인식이 가능합니다.
MarkDown이 최근 각광받기 시작한 이유는 깃허브(Github) 덕분이며, 특히 깃허브의 저장소(Repository)에 관한 정보를 기록하는 README.md 로 인해 자주 접하게 되면서 유명세를 타기 시작했습니다.
그리고 MarkDown을 통해 설치 방법, 소스코드 설명, 이슈 등을 간단하게 기록하고 가독성을 높일 수 있다는 강점이 부각되면서 점점 많은 곳에서 사용하게 되었습니다.

## 1.2 MarkDown의 장-단점
### 1.2.1 장점
1. 간결하다
2. 별도의 도구없이 작성 가능하다.
3. 다양한 형태로 변환이 가능하다.
4. 텍스트(Text)로 저장되기 때문에 용량이 적어 보관이 용이하다.
5. 텍스트 파일이기 때문에 버전 관리 시스템을 이용하여 변경 이력을 관리할 수 있다.
6. 지원하는 프로그램과 플랫폼이 다양하다.

### 1.2.2 단점
1. 표준이 없다
2. 표준이 없기 때문에 도구에 따라서 변환 방식이나 생성물이 다르다.
3. 모든 HTML 마크업을 대신하지 못한다.

# 2. 마크다운 문법
## 2.1 헤더(Headers)
- 글머리 1 ~ 6까지만 지원
- `#`다음에 띄어쓰기를 넣어줘야 적용이 됩니다.

```
# This is a Header1
## This is a Header
### This is a Header
#### This is a Header
##### This is a Header
###### This is a Header
```

## 2.2 BlockQuote
- 이메일에서 사용하는 `>` 블럭 인용 문자를 이용합니다.

```
> This is a first blockquote
>> This is a second blockquote
>>> This is a third blockquote
```

> This is a first blockquote
>> This is a second blockquote
>>> This is a third blockquote

- blockquote 안에는 다른 마크다운 요소를 포함할 수 있습니다.

```
> - List
> ``` code ```
```

> - List
> ``` code ```

## 2.3 목록(List)
- 순서 있는 목록(번호)
순서 있는 목록은 숫자와 점을 사용합니다.

```
1. 첫번째
2. 두번째
3. 세번째
```

1. 첫번째
2. 두번째
3. 세번째

현재까지는 어떤 번호를 입력해도 순서는 내림차순으로 적용됩니다.

```
1. 첫번째
3. 두번째
2. 세번째
```

1. 첫번째
3. 두번째
2. 세번째

- 순서 없는 목록(글머리 기호: `*`, `+`, `-`)

```
* 첫번째
  * 두번째
    * 세번째

+ 첫번째
  + 두번째
    + 세번째

- 첫번째
  - 두번째
    - 세번째
```

* 첫번째
  * 두번째
    * 세번째

+ 첫번째
  + 두번째
    + 세번째

- 첫번째
  - 두번째
    - 세번째

## 2.4 코드

### 2.4.1 들여쓰기

- 4개의 공백 또는 하나의 탭으로 들여쓰기를 만나면 변환되기 시작하며 들여쓰지 않은 행을 만날때까지 변환이 계속됩니다.

```
This is a normal paragraph:

	This is a code block.

end code block.
```

* * *
실제 적용 예시

This is a normal paragraph:

	This is a code block.

end code block.
* * *

- 개행을 넣어주지 않으면 제대로 인식이 되지 않는 문제가 존재하므로 사용 시 주의해야 합니다.

```
This is a normal paragraph:
	This is a code block.
end code block.
```

* * *
실제 적용 예시

This is a normal paragraph:
	This is a code block.
end code block.
* * *

### 2.4.2 코드 블럭
코드블럭은 다음과 같이 2가지 방식을 사용할 수 있습니다.

- `<pre><code>{code}</code></pre>` 이용 방식

```
<pre>
<code>
public class BootSpringBootApplication {
  public static void main(String[] args) {
    System.out.println("Hello, Honeymon");
  }
  
}
</code>
</pre>
```

<pre>
<code>
public class BootSpringBootApplication {
  public static void main(String[] args) {
    System.out.println("Hello, Honeymon");
  }

}
</code>
</pre>

***

- 코드 블럭 코드 ("```")을 이용하는 방법


````
```
public class BootSpringBootApplication {
	public static void main(String[] args) {
	    System.out.println("Hello, Honeymon");
  		}
	}   
```
````

```
public class BootSpringBootApplication {
	public static void main(String[] args) {
	    System.out.println("Hello, Honeymon");
  		}
	}
```

***

- 코드 블럭 시작점 ("```")에 사용하는 언어를 선언하여 해당 언어의 문법대로 강조가 가능합니다.

````
```java
public class BootSpringBootApplication {
  public static void main(String[] args) {
    System.out.println("Hello, Honeymon");
  }
}
```
````

```java
public class BootSpringBootApplication {
  public static void main(String[] args) {
    System.out.println("Hello, Honeymon");
  }
}
```

***

- 코드 블럭안에 백틱(`` ` ``) 넣는 방법   
여러 마크다운 문법을 찾아보는데 한국 블로그에는 그 어디에도 코드 블럭 안에 백틱(`` ` ``)을 넣는 방법이 없더라구요 그래서 제가 검색해서 찾은 내용 공유 드립니다. 역시 스택 오버 플로우 짱!

참조 링크 : <https://stackoverflow.com/questions/55586867/how-to-put-in-markdown-an-inline-code-block-that-only-contains-a-backtick-char>

백틱을 출력 하고 싶으면 출력하고 싶은 백틱보다 하나 더 많은 백틱을 양쪽에 감싸면 출력이 됩니다.

예시

백틱 하나만 출력 `` ` ``

```

`` ` ``

```

백틱 두 개 출력 ``` `` ```

```

``` `` ```

```

백틱 세 개 출력 ```` ``` ````

```

```` ``` ````

```

그렇다면 코드 블럭은 백틱 세 개를 사용하니까 코드 블럭 안에 3개 짜리 백틱을 출력하고 싶으면 백틱 4개로 감싸면 됩니다.

## 2.5 수평선

아래 줄은 모두 수평선을 만듭니다. 마크다운 문서를 미리보기로 출력할 때 페이지 나누기 용도로 많이 사용합니다.

```
* * *

***

*****

- - -

---------------------------------------
```

적용 예시

* * *

***

*****

- - -

---------------------------------------

## 2.6 링크
- 참조 링크

```
//문법
[link keyword][id]

[id] : URL "Optional Title here"

//실 적용 예시
Link: [Google][googlelink]

[googlelink]: https://google.com "Go google"
```

Link: [Google][googlelink]

[googlelink]: https://google.com "Go google"

- 외부 링크

```
//문법
[Ttile] (link)

//실 적용 예시
[Google](https://google.com, "Go google")
```

Link: [Google](https://google.com, "Go google")

- 자동 연결

```
일반적인 URL 혹은 이메일 주소인 경우 적절한 형식으로 링크를 형성합니다.

* 외부링크: <https://exmaple.com>
* 이메일링크: <address@example.com>
```

외부링크: <https://exmaple.com>   
이메일링크: <address@example.com>

## 2.7 강조

### 2.7.1 굵은 글자(bold)

```
**굵은 글자** 적용 방법
__굵은 글자__ 적용 방법
적용**굵은글자**방법
```

**굵은 글자** 적용 방법   
__굵은 글자__ 적용 방법   
적용**굵은글자**방법

### 2.7.2 이텔릭체

```
*이텔릭체* 적용 방법
_이텔릭체_ 적용 방법
적용*이텔릭체*방법
```

*이텔릭체* 적용 방법   
_이텔릭체_ 적용 방법   
적용*이텔릭체*방법

### 2.7.3 굵은 글자 및 이텔릭체
```
***굵은 글자 및 이텔릭체*** 적용 방법
___굵은 글자 및 이텔릭체___ 적용 방법
__*굵은 글자 및 이텔릭체*__ 적용 방법
**_굵은 글자 및 이텔릭체_** 적용 방법
적용***굵은 글자 및 이텔릭체***방법
```

***굵은 글자 및 이텔릭체*** 적용 방법   
___굵은 글자 및 이텔릭체___ 적용 방법   
__*굵은 글자 및 이텔릭체*__ 적용 방법   
**_굵은 글자 및 이텔릭체_** 적용 방법   
적용***굵은 글자 및 이텔릭체***방법

### 2.7.4 취소선 및 밑줄
```
~~취소선~~
<U>밑줄</U>
```

~~취소선~~   
<U>밑줄</U>

## 2.8 이미지

```
![Alt text](/path/to/img.jpg)
![Alt text](/path/to/img.jpg "Optional title")
```

![Alt text](/assets/images/profile_image.png)

***

![Alt text](/assets/images/profile_image.png "Optional title")

사이즈 조절 기능은 없기 때문에 html의 `<img width="" height=""></img>` 를 이용합니다.

```
<img src="/path/to/img.jpg" width="450px" height="300px" title="px(픽셀) 크기 설정" alt="RubberDuck"/><br/>
<img src="/path/to/img.jpg" width="40%" height="30%" title="px(픽셀) 크기 설정" alt="RubberDuck"/>
```

<img src="/assets/images/profile_image.png" width="450px" height="300px" title="px(픽셀) 크기 설정" alt="Quokka"/><br/>

***

<img src="/assets/images/profile_image.png" width="40%" height="30%" title="px(픽셀) 크기 설정" alt="Quokka"/>

이미지 정렬 또한 html의 <div align></div>를 이용합니다.

```
왼쪽 정렬
<div align="left">
  <img src="/assets/images/profile_image.png" width="40%" height="30%" title="px(픽셀) 크기 설정" alt="Quokka"/>
</div>

중앙 정렬
<div align="center">
  <img src="/assets/images/profile_image.png" width="40%" height="30%" title="px(픽셀) 크기 설정" alt="Quokka"/>
</div>

오른쪽 정렬
<div align="right">
  <img src="/assets/images/profile_image.png" width="40%" height="30%" title="px(픽셀) 크기 설정" alt="Quokka"/>
</div>

```

- 왼쪽 정렬
<div align="left">
  <img src="/assets/images/profile_image.png" width="40%" height="30%" title="px(픽셀) 크기 설정" alt="Quokka"/>
</div>

- 중앙 정렬
<div align="center">
  <img src="/assets/images/profile_image.png" width="40%" height="30%" title="px(픽셀) 크기 설정" alt="Quokka"/>
</div>

- 오른쪽 정렬
<div align="right">
  <img src="/assets/images/profile_image.png" width="40%" height="30%" title="px(픽셀) 크기 설정" alt="Quokka"/>
</div>

## 2.9 줄바꿈

3칸 이상 띄어쓰기(공백) (`   `)를 하고 개행을 하면 줄바꿈이 됩니다.

```
줄 바꿈을 하기 위해서는 문장 마지막에 3칸 이상을 띄어써야 한다.
이렇게 개행만 넣어주면 되지 않는다.

줄 바꿈을 하기 위해서는 문장 마지막에 3칸 이상을 띄어써야 한다.(공백3칸) 이렇게 하면 개행을 넣어 줬을 때 줄바꿈이 된다.
```

줄 바꿈을 하기 위해서는 문장 마지막에 3칸 이상을 띄어써야 한다.
이렇게 개행만 넣어주면 되지 않는다.

줄 바꿈을 하기 위해서는 문장 마지막에 3칸 이상을 띄어써야 한다.   
이렇게 하면 개행을 넣어 줬을 때 줄바꿈이 된다.

## 2.10 문자 정렬

MarkDown 에서는 따로 문자 정렬이 없어 이미지 정렬과 마찬가지로 html 문법을 사용합니다.

- 왼쪽 정렬

```
왼쪽 정렬 (Default)
{: .text-left}
정렬하고자 하는 문자열

중앙 정렬
{: .text-center}
정렬하고자 하는 문자열

오른쪽 정렬
{: .text-right}
정렬하고자 하는 문자열
```

왼쪽 정렬 (Default)
{: .text-left}
정렬하고자 하는 문자열

중앙 정렬
{: .text-center}
정렬하고자 하는 문자열

오른쪽 정렬
{: .text-right}
정렬하고자 하는 문자열

## 2.11 표(Table)

### 2.11.1 일반적인 표

```
|제목|내용|설명|
|------|---|---|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
```

|제목|내용|설명|
|------|---|---|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|

### 2.11.2 표 내부 셀 정렬

```
|제목|내용|설명|
|:---|---:|:---:|
|왼쪽정렬|오른쪽정렬|중앙정렬|
|왼쪽정렬|오른쪽정렬|중앙정렬|
|왼쪽정렬|오른쪽정렬|중앙정렬|
```

|제목|내용|설명|
|:---|---:|:---:|
|왼쪽정렬|오른쪽정렬|중앙정렬|
|왼쪽정렬|오른쪽정렬|중앙정렬|
|왼쪽정렬|오른쪽정렬|중앙정렬|

### 2.11.3 표 내부 셀 확장

```
|제목|내용|설명|
|:---|:---:|---:|
||중앙에서확장||
|||오른쪽에서 확장|
|왼쪽에서확장||
```

|제목|내용|설명|
|:---|:---:|---:|
||중앙에서확장||
|||오른쪽에서 확장|
|왼쪽에서확장||

## 2.12 접기

코드와 같이 내용이 긴 경우에는 접어 놓으면 포스트 글 내용을 불필요 하게 길지 않게 할 수 있습니다.

```
<details><summary>Click Me</summary>
Good!!
</details>    
```

<details><summary>Click Me</summary>
Good!!
</details>    

## 2.13 문자열 이스케이프 (Backslash Escapes)

MarkDown으로 글을 작성하다 보면 특수 문자(`*`, `+`) 등을 출력하고 싶을 경우가 있습니다. 이때는 백슬래쉬(`\`)를 이용하면 됩니다.

```
특수 문자 출력은 이렇게 \* \+
```

특수 문자 출력은 이렇게 \* \+



참조 링크   
[<https://gist.github.com/ihoneymon/652be052a0727ad59601>]   
[<https://www.markdownguide.org/basic-syntax/>]   
[<https://stackoverflow.com/questions/55586867/how-to-put-in-markdown-an-inline-code-block-that-only-contains-a-backtick-char>]   


블로그를 방문해 주셔서 감사합니다!   
오타나 수정 사항, 궁금한 것이 있다면 댓글로 작성 부탁 드립니다!