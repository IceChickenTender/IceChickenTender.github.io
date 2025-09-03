---
title: "[Python] Python 고급 문자열 기능"
categories:
  - Python
tags:
  - Python
  
toc: true
toc_sticky: true
toc_label: "Python 고급 문자열 기능"
---

Python 의 고급 문자열 기능에 대해서 알아보도록 하겠습니다. 저는 Python 이 `C` 나 `Java` 등에 비해 비교적 쉬운 언어라 Python 의 문법들이나 기법들에 대해서 주의 깊게 다루지 않아 자주 쓰던 것들도 까먹고 하는 문제가 있습니다. 거기다 최근에 다시 시작한 딥러닝 공부를 하면서 제 자신이 Python 에 대해서 많이 부족하다는 것을 깨달아 이를 보완하고자 블로그에 제가 공부한 것을 기록하면서 추후에 제가 또 까먹거나 필요할 때 바로 바로 볼 수 있고, 또 정리를 하면서도 공부가 된다고 하여 이렇게 블로그 정리를 해보고자 합니다. 이번엔 Python 문자열의 고급 기능들에 대해서 알아보고 정리해 보도록 하겠습니다. 이 포스트는 브라이언 오버랜드, 존 베냇 저자 `파이썬 스킬업` 이라는 책에 있는 내용을 토대로 작성하였으며, 책 내용 중 제가 필요하다고 생각되는 부분만 가져왔습니다.

# 1. 문자열 연산자(+, =, *, >, etc)

Python 의 문자열 타입 str에서 숫자 타입 연산자와 같은 연산자를 사용할 수 있습니다만, 다르게 동작합니다. 문자열에서 사용하는 연산자에 대해서 알아보고 어떻게 동작하는지 예시로 한 번 알아보도록 하겠습니다.

|연산자 문법|설명|
|----------|----|
|value = 문자열|문자열을 변수에 대입한다|
|문자열1 == 문자열2|문자열1과 문자열2가 같은 값을 가지고 있다면 True 를 반환한다 (다른 비교 연산자들과 같이 대소문자를 구분한다)
|문자열1 != 문자열2|문자열1과 문자열2가 서로 다른 값을 가지고 있다면 True 를 반환한다|
|문자열1 < 문자열2|문자열1이 문자열2보다 알파벳 순서가 앞이면 True 를 반환한다. 가령 'abc' < 'def' 는 True 를 반환하지만 'abc' < 'aaa' 는 False 를 반환한다|
|문자열1 > 문자열2|문자열1이 문자열2보다 알파벳 순서가 뒤면 True 를 반환한다.|
|문자열1 <= 문자열2|문자열1이 문자열2보다 알파벳 순서가 앞이거나 같다면 True 를 반환한다.|
|문자열1 >= 문자열2|문자열1이 문자열2보다 알파벳 순서가 뒤거나 같다면 True 를 반환한다.|
|문자열1 + 문자열2|두 문자열을 연결한다. 문자열1 끝에 문자열2를 붙인다.|
|문자열1 * n|문자열1을 정수 n만큼 반복하여 연결한다.|
|문자열1 in 문자열2|문자열1 전체가 문자열2에 포함되면 True 를 반환한다.|
|문자열1 not in 문자열2|문자열1 전체가 문자열2에 포함되지 않으면 True 를 반환한다.|
|문자열 is 객체|문자열과 객체가 메모리상에 동일 객체를 참조하고 있으면 True 를 반환한다. 간혹 None 이나 알지 못하는 객체 타입과 비교할 때 사용된다|
|문자열 is not 객체|문자열과 객체가 메모리상에 동일 객체를 참조하고 있지 않으면 True 를 반환한다.|

```python

# 1. 변수에 문자열 대입
value = '안녕하세요'
print(value) # 안녕하세요

# 2. 문자열 비교

# 동등 비교 문자열1 == 문자열2

a = "Python"
b = "Python"
c = "python"

print(a == b)  # True  : 값(내용)이 같으면 True (대소문자 구분)
print(a == c)  # False : 대소문자 다름
print("가나다" == "가나다")  # True : 한글도 유니코드 코드포인트 기준으로 비교

# 부등 비교: 문자열1 != 문자열2

print("abc" != "ABC")  # True  : 대소문자 다르면 True
print("123" != "0123") # True  : 길이나 문자 하나라도 다르면 True

# 사전식 비교(유니코드 순): >, >=, >, >=

# '<' 예시
print("abc" < "def")   # True  : 'a' < 'd'
print("abc" < "aaa")   # False : 'abc' vs 'aaa' -> 첫 글자 'a'는 같고, 두 번째 'b' > 'a' 이므로 False
print("Z"   < "a")     # True  : 'Z'(90) < 'a'(97)  ※ 아스키/유니코드 값 기준

# '>' 예시
print("홍길동" > "가나다")  # True  : '홍'이 '가'보다 코드포인트가 큼
print("Python" > "Java")    # True  : 'P' > 'J'

# '<=' '>=' 예시
print("abc" <= "abc")  # True  : 같으면 <=, >= 비교는 True
print("ab"  <= "abc")  # True  : 'ab'는 'abc'보다 사전식으로 앞섬

# 3. 문자열 연결: 문자열1 + 문자열2

first = '윌리엄'
last = '셰익스피어'
full_name = first + ' ' + last
print(full_name) # 윌리엄 셰익스피어

# 4. 반복: 문자열 * n

print("ha" * 3)  # "hahaha"
print("-"  * 10) # "----------"
print("abc" * 0) # "" (빈 문자열)

# 5. 포함 여부: 문자열1 in 문자열2

text = "파이썬은 재미있다!"
print("파이" in text)     # True  : 부분 문자열 포함
print("python" in text)  # False : 대소문자/언어 다르면 당연히 False

# 대소문자 무시하고 포함 확인하고 싶다면 둘 다 소문자/대문자로 통일
print("python" in "Python is fun".lower())  # True

# 6. 미포함 여부

lang = "Python"
print("JS"  not in lang)  # True
print("Py"  not in lang)  # False : "Py"는 "Python"에 포함됨

# 7. 객체 동일성 비교: is, is not

# 올바른 사용: None과 비교할 때
result = None
print(result is None)      # True
print(result is not None)  # False

# 잘못 쓰기 쉬운 패턴: 문자열 동등성에 is 사용 (지양!)
s1 = "abc"
s2 = "ab" + "c"            # 상수 폴딩으로 같은 객체가 될 수도 있음(파이썬 구현/상황에 따라 다름)
s3 = "".join(["ab", "c"])  # 런타임에 만들어져 보통은 다른 객체

print(s1 == s2)  # True  : 값이 같음
print(s1 is s2)  # (환경 따라 True/False) '우연히' True가 나올 수도 있어 의존하면 안 됨
print(s1 == s3)  # True  : 값이 같음
print(s1 is s3)  # 보통 False : 서로 다른 객체

# 'is'는 오직 "같은 객체인가?"를 물을 때만 사용
sentinel = object()
alias = sentinel
print(sentinel is alias)    # True  : 같은 객체를 가리킴
print(id(sentinel), id(alias))  # 객체 식별자(메모리 주소 유사값)도 동일

```

참고로 Python 에서 문자열을 비교할 때 알파벳 순서를 사용합니다. 좀 더 구체적으로 말하자면 문자 값의 아스키코드나 유니코드의 코드 순서를 따릅니다. 이렇게 하면 모든 대문자는 소문자보다 앞에 위치하며, 이 규칙 이외에는 문자들을 단순히 알파벳 순서로 비교할 수 있습니다. 그리고 문자열 비교를 할 때 광범위한 유니코드 문자로 이루어진 문자열을 사용하고 있다면 casefold 메서드를 사용해 대소문자를 구분하지 않고 값을 비교하는 것이 더 안전합니다.

```python
def compare_no_case(str1, str2):
	return str1.casefold() == str2.casefold()

print(compare_no_case('cat', 'CAT')) # True 값 반환
```

# 2. `join` 을 사용하여 문자열 만들기

우리는 `+` 를 이용해 문자열을 연결하는 방법에 대해서 알게 되었습니다. 하지만 이 경우 적은 숫자의 객체를 다루는 단순한 경우에는 효과가 좋지만 많이 반복할 경우 문자열 연산을 하기 위해 새로운 문자열 객체를 생성하게 되므로 메모리 누수가 일어나게 됩니다. 이러한 문제를 해결하기 위해서 `join` 을 사용합니다. 

```python
구분자_문자열.join(리스트)
```

`join` 은 리스트의 모든 문자열을 하나의 커다란 문자열로 만듭니다. 단순 문자열 연결하기 대신 join 을 사용하면서 얻는 이점은 수천 번 연산해야 하는 큰 사례에서 확인할 수 있습니다. 이런 사례에서 문자열 연결하기가 갖는 결점은 한 번만 사용하고 파이썬의 "가비지 컬렉션(garbage collection)"에 의해 버려질 수천 개의 문자열을 만들어야 한다는 것입니다. 그리고 빈번하게 실행된 가비지 컬렉션은 실행 시간의 비효율을 초래합니다. 아래 예시는 `join` 을 이용하면 아주 간단히 작업할 수 있는 것을 볼 수 있습니다.

```python

str_array = ['John', 'Paul', 'George', 'Ringo']

def print_worse(a_lst):
	s = ''
	for item in a_lst:
		s += item + ', '
	if len(s) > 0: # 문자열 끝의 콤바 + 빈칸 제거
		s = s[:-2]
	print(s)

def print_nice(a_lst):
	print(', '.join(a_lst))
 
print_worse(str_array) # John, Paul, George, Ringo
print_nice(str_array) # John, Paul, George, Ringo
```

# 3. 주요 문자열 메서드

Python 에서는 문자열 타입에 사용할 수 있는 기본 내장 함수를 제공합니다. 이 내장 함수들은 자주 사용되기 때문에 알아두고 있으면 유용합니다.

|메서드|설명|
|-----|----|
|input(프롬프트에 출력한 문자열)|문자열 입력을 위한 프롬프트 사용자|
|len(문자열)|문자열 내 문자 개수를 숫자로 반환|
|max(문자열)|가장 높은 코드(유니코드) 값을 가진 문자 반환, 빈 문자열에서 `ValueError` 발생|
|min(문자열)|가장 낮은 코드(유니코드) 값을 가진 문자 반환, `ValueError` 발생|
|reversed(문자열)|역순으로 항목을 내주는 이터레이터|
|sorted(문자열)|정렬된 문자들의 리스트를 반환|

```python

# 1. input()

# 콘솔에서 한 줄을 입력받아 str 로 반환합니다.
name = input("이름을 입력하세요: ")  # 예: 사용자가 '철수' 입력
print(type(name))          # <class 'str'>
print(f"안녕하세요, {name}님!")  # 안녕하세요, 철수님!

# 숫자를 받고 싶다면 형 변환을 해야 합니다.
age = int(input("나이를 입력하세요: "))  # '27' 입력 시 27(int)로 변환
print(f"내년에는 {age + 1}살!") # 내년에는 28살!


# 2. len()

s1 = "Python"
s2 = "파이썬"

print(len(s1))  # 6
print(len(s2))  # 3

# 3. max() / min()

t = "aZ3가"

print(max(t))  # '가'  : 유니코드 코드 포인트가 가장 큼
print(min(t))  # '3'   : 코드 포인트가 가장 작음

# 비교가 유니코드 코드 포인트 기반임을 확인
print(ord('a'), ord('Z'), ord('3'), ord('가'))  # 각 문자 -> 정수 코드값

# 빈 문자열은 에러: ValueError
empty = ""
# max(empty)  # ValueError
# 방어적으로 default 사용 (파이썬 3.4+)
print(max(empty, default=None))  # None


# 4. reversed()

s = "ABCDE"

rv = reversed(s)
print(rv)                   # <reversed object ...>  ← 역순 '문자'를 내주는 이터레이터

print(list(rv))             # ['E', 'D', 'C', 'B', 'A']  ← 한 번 소비하면 비워집니다

# 역순 문자열을 얻고 싶다면 join 또는 슬라이싱을 사용
s_rev = "".join(reversed(s))
print(s_rev)                # "EDCBA"

# 파이썬 관용구: 슬라이싱으로 뒤집기
print(s[::-1])              # "EDCBA"


# 5. sorted()

s = "baCA가1"

print(sorted(s))                   # ['1', 'A', 'C', 'a', 'b', '가'] (유니코드 코드값 오름차순)
print("".join(sorted(s)))          # "1ACab가"  ← 문자열로 합치기

# 내림차순
print("".join(sorted(s, reverse=True)))  # '가baCA1'

# 대소문자 무시 정렬
print("".join(sorted(s, key=str.lower))) # 'aAbC1가' (소문자로 변환한 값 기준)

```

# 4. 간단한 `is` 메서드

메서드의 이름이 `is`로 시작하는 모든 메서드는 `True` 나 `False` 를 반환합니다. 간혹 문자열의 어떤 특성을 이용하고자 할 때 `is` 가 들어간 메서드를 활용해야합니다. 이 메서드들을 표로 한 번 정리해 보았습니다.

|메서드 이름|테스트 통과 시 True 반환|
|----------|-----------------------|
|str.isalnum()|모든 문자가 글자와 숫자로 이루어졌으며, 최소한 문자가 하나 이상 있는 경우|
|str.isalpha()|모든 문자가 알파벳 글자로 이루어졌으며, 최소한 문자가 하나 이상 있는 경우|
|str.isdecimal()|모든 문자가 10진수 숫자로 이루어졌으며, 최소한 문자가 하나 이상 있는 경우, 유니코드를 사용함|
|str.isdigit()|모든 문자가 10진수 숫자로 이루어졌으며, 최소한 문자가 하나 이상 있는 경우|
|str.isidentifier()|문자열이 유효한 파이썬 식별자 이름 규칙을 지키고 있는 경우, 첫 문자는 반드시 문자가 언더스코어(_)여야하며, 각 문자는 글자, 숫자 혹은 언더스코이어야 한다.|
|str.islower()|모든 문자가 소문자로 이루어졌으며, 최소한 문자가 하나 이상 있는 경우(참고로 알파벳이 아닌 문자가 포함될 수도 있다)|
|str.isprintable()|모든 문자가 출력 가능한 문자인 경우, `\n` 과 `\t` 는 제외|
|str.isspace()|모든 문자가 공백 문자(whitespace)이며, 최소한 문자가 하나 이상 있는 경우|
|str.istitle()|모든 문자가 유효한 제목이며, 최소한 문자가 하나 이상 있는 경우, 첫 문자만 대문자고 나머지는 모두 소문자면 조건에 만족한다. 문자 사이에 공백 문자가 구분 문자가 있을 수 있다.|
|str.isupper()|모든 문자가 대문자로 이루어졌으며 최소한 문자가 하나 이상 있는 경우(참고로 알파벳이 아닌 문자가 포함될 수도 있다)|

```python
# 1. str.isalnum() - 영문자/한글/숫자만으로 구성

print("abc123".isalnum())     # => True
print("한글ABC123".isalnum())  # => True (한글/라틴/숫자 혼합 가능)
print("abc_123".isalnum())    # => False ('_'는 영숫자 아님)
print("".isalnum())           # => False (빈 문자열)

# 2. str.isalpha() - 알파벳만으로 구성

print("Python".isalpha())     # => True
print("한글ABC".isalpha())     # => True
print("abc123".isalpha())     # => False (숫자 포함)
print("abc!".isalpha())       # => False (! 포함)
print("".isalpha())           # => False

# 3. str.isdecimal() - 10진 숫자(유니코드) 만으로 구성

print("123456".isdecimal())   # => True
print("４５".isdecimal())      # => True (전각 숫자)
print("²".isdecimal())         # => False (제곱 표기: 2의 윗첨자)
print("一二三".isdecimal())     # => False (한자 숫자)
print("".isdecimal())          # => False

# 4. str.isdigit() - 숫자(윗첨자 등 일부 숫자 기호 포함)만으로 구성

print("123".isdigit())        # => True
print("²³".isdigit())         # => True (윗첨자 2,3 — isdecimal은 False)
print("①②".isdigit())         # => True 또는 False (플랫폼/버전에 따라 다를 수 있음)
print("一二三".isdigit())       # => False (한자 숫자는 isnumeric()에서 True)
print("12a".isdigit())        # => False
print("".isdigit())           # => False

# 5. str.isidentifier() - 파이썬 식별자로 유효한가?

print("variable".isidentifier())   # => True
print("_hidden".isidentifier())    # => True
print("변수".isidentifier())        # => True (유니코드 문자 가능)
print("3cats".isidentifier())      # => False (숫자로 시작 불가)
print("class".isidentifier())      # => True (키워드인지 여부는 별개)
# 키워드 여부는 keyword 모듈로 확인 가능:
#   import keyword; print(keyword.iskeyword("class"))  # True

# 6. str.islower() - 소문자만 포함

print("python".islower())     # => True
print("python!".islower())    # => True ('!'는 비문자)
print("Python".islower())     # => False
print("123".islower())        # => False (cased 문자가 없음)
print("ß".islower())          # => True (독일어 ß는 소문자)

# 7. str.isprintable() - 모든 문자가 출력 가능?

print("Hello, World!".isprintable())  # => True
print("공백 포함 ".isprintable())       # => True (스페이스는 출력 가능)
print("\n".isprintable())             # => False (제어 문자)
print("".isprintable())               # => True (빈 문자열은 True)

# 8. str.isspace() - 모두 공백 문자(스페이스/탭/개행 등)

print("   ".isspace())       # => True (스페이스만)
print("\t\n\r".isspace())    # => True (탭/개행/캐리지리턴)
print(" \u2003 ".isspace())  # => True (em-space 등 유니코드 공백)
print(" a ".isspace())       # => False (문자 포함)
print("".isspace())          # => False (빈 문자열)

# 9. str.istitle() - 각 단어가 제목 표기(첫 글자 대문자, 나머지 소문자)

print("Hello World".istitle())        # => True
print("Hello, World!".istitle())      # => True (구두점은 무시)
print("This Is A Title".istitle())    # => True
print("Hello 42Nd Street".istitle())  # => True (숫자 뒤 'Nd'는 제목 표기로 간주)
print("hello World".istitle())        # => False (첫 단어 소문자)


# 10. str.isupper() - 대문자만 포함

print("PYTHON".isupper())     # => True
print("PYTHON!".isupper())    # => True ('!'는 비문자)
print("PYTHON3".isupper())    # => True (숫자는 영향 없음)
print("PyTHON".isupper())     # => False (소문자 포함)
print("123".isupper())        # => False (cased 문자가 없음)

```

# 5. 대소문자 변환 메서드

문자열을 이용한 작업을 처리하다보면 대문자 혹은 소문자로 통일해서 처리해야 하는 경우가 잦습니다. 대소문자 변환 메서드와 함께 추가로 쓰이는 메서드도 함께 알아보도록 하겠습니다.

|메서드 이름|설명|
|----------|----|
|문자열.lower()|모두 소문자인 문자열을 생성한다.|
|문자열.upper()|모두 대문자인 문자열을 생성한다.|
|문자열.title()|문자열의 각 단어 첫 글자를 대문자로, 나머지를 소문자로 바꾼 문자열을 생성한다.|
|문자열.swapcase()|대소문자를 서로 변경한 문자열을 생성한다.|

Python 의 문자열은 변하지 않는 타입입니다. 그래서 위 메서드들은 변환된 새로운 문자열을 생성해줍니다. 다만 변환된 새로운 객체를 기존 객체에 대입해 주기 때문에 변경된 것으로 보입니다.

```python
# 1. 문자열.lower() 대문자를 소문자로 변환

s = "PyTHON 3.12! 한글 OK"
print(s.lower())     # => "python 3.12! 한글 ok" (한글은 케이스 개념이 없어 그대로)
print("ß".lower())   # => "ß" (더 강한 무케이스 비교가 필요하면 .casefold() 권장)

# 2. 문자열.upper() - 소문자를 대문자로 변환

s = "Hello, world! 123 한글"
print(s.upper())     # => "HELLO, WORLD! 123 한글"

# 3. 문자열.title() - 각 단어의 첫 글자를 대문자로, 나머지는 소문자로

print("hello world".title())                       # => "Hello World"
print("they're bill's friends from the UK".title())# => "They'Re Bill'S Friends From The Uk"
# ※ 아포스트로피를 단어 경계로 취급해 위처럼 보일 수 있음


# 4. 문자열.swapcase() - 대소문자 뒤집기

print("Python3 iS FUN!".swapcase())  # => "pYTHON3 Is fun!"
# 유니코드 전반에서는 swapcase를 두 번 적용해도 원문이 100% 보장되지는 않습니다.

```

# 6. 검색-교체 메서드

검색-교체 메서드는 문자열 메서드 중에서 유용해서 많이 사용됩니다. 그러므로 문자열의 검색-교체 메서드들을 알고 있는 것이 좋습니다. 아래는 자주 사용되는 검색-교체 메서드들을 표로 정리한 것입니다.

| 메서드 이름                               | 반환값    | 설명                                                                                                                           |
| ------------------------------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `startswith(prefix[, start[, end]])` | `bool` | 문자열이 `prefix`로 **시작**하면 `True`. `prefix`는 **튜플**도 가능. `start/end`는 슬라이스 규칙(`s[start:end]`)으로 해석. |
| `endswith(suffix[, start[, end]])`   | `bool` | 문자열이 `suffix`로 **끝나면** `True`. `suffix`는 **튜플**도 가능. `start/end` 지원(슬라이스와 동일).                    |
| `count(sub[, start[, end]])`         | `int`  | (겹치지 않는) `sub`의 출현 횟수. 빈 문자열이면 `len(s)+1`. 범위 지정은 슬라이스 규칙.                                        |
| `find(sub[, start[, end]])`          | `int`  | 가장 **왼쪽**(낮은 인덱스)의 위치를 반환, 없으면 **`-1`**. 범위 지정은 슬라이스 규칙.                                          |
| `index(sub[, start[, end]])`         | `int`  | `find()`와 동일하지만, 없으면 **`ValueError`** 발생.                                                         |
| `rfind(sub[, start[, end]])`         | `int`  | 가장 **오른쪽**(높은 인덱스)의 위치를 반환, 없으면 **`-1`**.                                                         |
| `rindex(sub[, start[, end]])`        | `int`  | `rfind()`와 동일하지만, 없으면 **`ValueError`** 발생.                                                        |
| `replace(old, new[, count])`         | `str`  | `old` → `new`로 **치환**한 **복사본** 반환. `count`를 주면 그 횟수만큼만 치환. (미지정 또는 `-1`이면 모두)                     |

```python
s = "abc_abc-XYZ.txt"

# 1. startswith / endswith
print(s.startswith("abc"))                 # True
print(s.endswith(".txt"))                  # True
print(s.startswith(("def", "abc")))        # True (튜플 허용)
print(s.endswith((".csv", ".txt"), 0, 11)) # False: s[0:11] 범위에서는 ".txt"가 없음

# 2. count (겹치지 않는 출현만 카운트)
print("ababa".count("aba"))   # 1
print("ababa".count("ba"))    # 2
print("abc".count(""))        # 4 (len(s)+1)

# 3. find / index
print(s.find("abc"))          # 0
print(s.find("XYZ"))          # 8
print(s.find("nope"))         # -1
# print(s.index("nope"))      # ValueError (없으면 예외)

# 4. rfind / rindex
print(s.rfind("abc"))         # 4 (오른쪽에서 가장 가까운 위치)
# print(s.rindex("nope"))     # ValueError

# 5. replace
print(s.replace("abc", "###"))        # "###_###-XYZ.txt"
print(s.replace("abc", "###", 1))     # "###_abc-XYZ.txt" (한 번만)
```

# 7. split 을 활용한 문자열 쪼개기

입력받은 문자열을 다루는 가장 흔한 프로그래밍 작업은 토큰화(tokenizing)입니다. 파이썬의 split 메서드는 토큰화 작업을 쉽고 편리하게 할 수 있도록 도와줍니다. `split` 메소드의 구성은 다음과 같습니다.

```
문자열.split(구분_문자열=None)
```

알아볼 예제는 split 메서드의 다양한 활용법에 대해서 알아보도록 하겠습니다.

```python

# 1. 구분 문자열이 생략되거나 None 이면 공백 문자(빈칸, 탭, 개행 문자)를 기준으로 분리한다.

stooge_list = 'Moe Larry Curly Shemp'.split()
print(stooge_list) # ['Moe', 'Larry', 'Curly', 'Shemp']

# 2. None 혹은 기본 인자로 split 수행 시 공백 문자의 개수는 상관 없다.

stooge_list = 'Moe    Larry    Curly    Shemp'.split()
print(stooge_list) # ['Moe', 'Larry', 'Curly', 'Shemp']

# 3. 구분 문자열이 명시 되면 구분자에 의해서 정확하게 구분되어야 한다.

stooge_list = 'Moe    Larry    Curly    Shemp'.split(' ')
print(stooge_list) # ['Moe', '', '', '', 'Larry', '', '', '', 'Curly', '', '', '', 'Shemp']

```

# 8. 앞뒤 문자 제거하기

사용자나 텍스트 파일로부터 문자열을 입력받으면 앞뒤에 붙은 빈칸 혹은 개행을 제거하여 원하는 형태로 변경해야 할 경우가 있습니다. 혹은 문자열을 앞뒤로 감싸고 있는 숫자나 다른 문자를 제거해야 하는 경우도 있습니다. Python 의 str 은 이렇게 앞뒤 문자를 제거하는 몇 가지 메서드를 제공합니다.

|메서드 이름|설명|
|----------|----|
|문자열.strip(제거문자열=' ')|앞뒤 문자 지우기|
|문자열.lstrip(제거문자열=' ')|앞 문자 지우기|
|문자열.rstrip(제거문자열=' ')|뒤 문자 지우기|

특히 `rstrip` 메서드의 경우 코딩 테스트를 Python 으로 진행할 경우 입력의 개행을 지우기 위해 굉장히 많이 사용됩니다.

```python

# 1. 문자열.strip() - 앞/뒤 모두에서 제거

# 1) 기본: 공백류(스페이스, 탭, 개행 등) 제거
s = " \t  Hello World \n"
print(s.strip())     # => "Hello World"

# 2) 지정한 문자들의 '집합'을 양끝에서 제거 (가운데는 그대로)
s = "***--Hello--***"
print(s.strip("*-")) # => "Hello"  (* 또는 - 를 끝에서 계속 깎아냄)

# 3) 경로 꼬리 슬래시 제거 예
path = "///usr/local///"
print(path.strip("/"))  # => "usr/local"

# 4) '집합'이라는 점 주의: "abc" 전체가 아니라 a/b/c 각각을 의미
s = "abcabcXYZcab"
print(s.strip("abc"))   # => "XYZ"  (양끝에서 a/b/c 문자들을 제거한 뒤 멈춤)

# 5) 유니코드 공백도 제거 (NBSP, EM SPACE 등)
u = "\u00A0Hello\u2003"
print(u.strip())        # => "Hello"


# 2. 문자열.lstrip() - 앞(왼쪽)만 제거

# 앞쪽 공백 제거
s = "   [DATA]"
print(s.lstrip())       # => "[DATA]"

# 앞쪽의 '0'들만 제거 (숫자 포맷 정리 등에 유용)
num = "000123.4500"
print(num.lstrip("0"))  # => "123.4500"  (앞의 0만)

# 지정 문자 집합 예시
token = ">>>INFO>>> message"
print(token.lstrip("><I NFO"))  # => "> message"
# (왼쪽에서 >, <, I, 공백, N, F, O 중에 있는 문자를 계속 제거)

# 3. 문자열.rstrip() - 뒤(오른쪽)만 제거

# 뒤쪽 공백/개행 제거
line = "result=42   \n"
print(line.rstrip())         # => "result=42"

# 뒤쪽의 '0'들만 제거 (소수점 0 꼬리 자르기)
num = "000123.4500"
print(num.rstrip("0"))       # => "000123.45"

# 로그 꼬리 마커 제거
log = "status:OK;;;;"
print(log.rstrip(";:"))      # => "status:OK"

```

# 마치며

Python 의 문자열의 고급 기능들에 대해서 알아보았습니다. 평소 많이 쓰던 것들도 많았지만, 자주 접해보지 못했던 메서드들에 대해서 알아보고, 예제를 통해서 구체적으로 어떻게 사용되는지도 알아보았습니다. 긴 글 읽어주셔서 감사드리며, 내용 중 잘못된 내용, 오타 혹은 궁금한 내용이 있으시다면 댓글 달아주시기 바랍니다.