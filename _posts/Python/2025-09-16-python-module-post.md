---
title: "[Python] Python 모듈 기초"
categories:
  - Python
tags:
  - Python
  
toc: true
toc_sticky: true
toc_label: "Python 모듈 기초"
---

이번엔 Python 의 모듈 기초에 대해서 공부하고 알아보도록 하겠습니다. 이번 Python 의 모듈 기초에서는 모듈 import 방법과 import 하는 여러 방법들 그리고 일반적으로 많이 사용되는 Python 의 모듈들에 대해서 알아보고 기초적인 사용 방법에 대해서 정리해 보도록 하겠습니다. 이번 블로그 작성에는 "혼자 공부하는 파이썬" 이라는 책을 참고하였습니다.

# 1. 표준 모듈

Python 은 모듈이라는 기능을 활용해 코드를 분리하고 공유합니다. 모듈을 가져올 때는 다음과 같은 구문을 사용합니다. 일반적으로 모듈을 가져오는 `import` 구문은 코드의 가장 위에 작성합니다.

```python
import 모듈_이름
```

## 1.1 import 방식들

### from 구문

모듈에는 정말 많은 변수와 함수가 들어 있습니다. 하지만 그중에서 우리가 활용하고 싶은 기능은 극히 일부일 수 있으며, math.cos(), math.sin() 처럼 앞에 무언가를 계속 입력하는 것이 귀찮다고 느껴질 수 있습니다. 이럴 때 `from` 구문을 사용하면 됩니다 형태는 다음과 같습니다.

```python
from 모듈_이름 import 가져오고 싶은 변수 또는 함수
```

이 때 가져오고 싶은 변수 또는 함수에 여러 개의 변수 또는 함수를 입력할 수도 있습니다. 이런 구문을 사용하면 가져온 기능은 모듈이름을 앞에 붙이지 않고도 사용할 수 있습니다.

```python
from math import sin, cos, tan, floor, ceil

print(sin(1)) # 0.8414709848078965
print(cos(1)) # 0.5403023058681398
print(tan(1)) # 1.5574077246549023

```

만약 앞에 모듈이름을 붙이는 것이 싫고, 모든 기능을 가져오는 것이 목적이라면 `\*` 기호를 사용합니다. `\*` 기호는 컴퓨터에서 "모든 것"을 의미합니다. 따라서 다음과 같이 코드를 입력하면 모듈 내부의 모든 것으 가져오는 것을 의미합니다.

```python
from math import *
```

다만 모든 것을 가져오면 식별자 이름에서 충돌이 발생할 수 있습니다. 따라서 `from`구문을 사용할 때는 최대한 필요한 것들만 가져와서 사용하는 것이 좋습니다.

### as 구문

모듈을 가져올 때 이름 충돌이 발생하는 경우가 있을 수 있습니다. 추가로 모듈의 이름이 너무 길어서 짧게 줄여 사용하고 싶은 경우도 있을 수 있습니다. 이럴 때는 다음과 같이 `as` 구문을 사용합니다.

```python
import 모듈_이름 as 사용하고_싶은_식별자
```

이를 활용하면 이전의 코드에서 `math`로 사용하던 `math` 모듈을 `m` 이라는 이름 등으로 사용할 수 있습니다.

```python
import math as m

print(m.sin(1)) # 0.8414709848078965
print(m.cos(1)) # 0.5403023058681398
print(m.tan(1)) # 1.5574077246549023
```

## 1.2 random 모듈

모듈을 불러오는 방법을 알았으니, 이제 다양한 모듈을 살펴보도록 하겠습니다. 일단 가장 간단한 `random`모듈부터 살펴봅도록 하겠습니다. `random` 모듈은 랜덤한 값을 생성할 때 사용하는 모듈입니다.

다음과 같은 방법으로 가져올 수 있습니다. 물론 from 구문 또는 as 구문과도 조합해서 사용할 수 있습니다.

```python
import random
```

공식 문서에 올라와 있는 `random` 모듈의 예시 중에 몇 가지를 예제를 통해 확인해 보도록 하겠습니다.

```python
import random

# random() : 0.0 <= x < 1.0 사이의 float 값을 리턴합니다.
print("- random():", random.random())

# uniform(min, max): 지정한 ㅓㅂㅁ위 사이의 float 값을 리턴합니다.
print("- uniform(10, 20):", random.uniform(10, 20))

# randrange(): 지정한 범위의 int 를 리턴합니다.
# - randrange(max): 0부터 max 사이의 값을 리턴합니다.
# - randragne(min, max): min부터 max 사이의 값을 리턴합니다.
print("- randrange(10)", random.randrange(10))

# choice(list): 리스트 내부에 있는 요소를 랜덤하게 선택합니다.
print("- choice([1, 2, 3, 4, 5]):", random.choice([1, 2, 3, 4, 5]))

# shuffle(list): 리스트 요소들을 랜덤하게 섞습니다.
sample_list = [1, 2, 3, 4, 5]
random.shuffle(sample_list)
print("- shuffle([1, 2, 3, 4, 5]):", sample_list)

# sample(list, k=<숫자>): 리스트의 요소 중에 k개를 뽑습니다.
print("- sample([1, 2, 3, 4, 5], k=2):", random.sample([1, 2, 3, 4, 5], k=2))
```

```
실행 결과

- random(): 0.7423878144392989
- uniform(10, 20): 13.860169730723673
- randrange(10) 5
- choice([1, 2, 3, 4, 5]): 2
- shuffle([1, 2, 3, 4, 5]): [2, 4, 1, 5, 3]
- sample([1, 2, 3, 4, 5], k=2): [4, 3]
```

## 1.3 sys 모듈

sys 모듈은 시스템과 관련된 정보를 가지고 있는 모듈입니다. 명령 매개변수를 받을 때 많이 사용되므로 간단하게 살펴보도록 하겠습니다.

```python
import sys

# 며령 매개변수를 출력합니다.
print(sys.argv)
print("---")

# 컴퓨터 환경과 관련된 정보를  출력합니다.
print("copyright:", sys.copyright)
print("---")
print("version:", sys.version)

# 프로그램을 강제로 종료합니다.
sys.exit()
```

```
실행 결과

['/usr/local/lib/python3.12/dist-packages/colab_kernel_launcher.py', '-f', '/root/.local/share/jupyter/runtime/kernel-69910077-a07f-4488-9898-698cbb9e8395.json']
---
copyright: Copyright (c) 2001-2023 Python Software Foundation.
All Rights Reserved.

Copyright (c) 2000 BeOpen.com.
All Rights Reserved.

Copyright (c) 1995-2001 Corporation for National Research Initiatives.
All Rights Reserved.

Copyright (c) 1991-1995 Stichting Mathematisch Centrum, Amsterdam.
All Rights Reserved.
---
version: 3.12.11 (main, Jun  4 2025, 08:56:18) [GCC 11.4.0]
An exception has occurred, use %tb to see the full traceback.

SystemExit
/usr/local/lib/python3.12/dist-packages/IPython/core/interactiveshell.py:3561: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
  warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)
```

5번 째 라인의 `sys.argv`라고 되어 있는 부분이 바로 명령 매개변수입니다. 프로그램을 실행할 대 추가로 입력하는 값들을 의미합니다.

## 1.4 os 모듈

`os` 모듈은 운영체제와 관련된 기능을 가진 모듈입니다. 새로운 폴더를 만들거나 폴더 내부의 파일 목록을 보는 일도 모두 `os` 모듈을 활용해서 처리합니다. 그럼 간단하게 `os` 모듈의 몇 가지 변수와 함수를 사용해 보겠습니다.

```python
import os

# 기본 정보를 몇 개 출력해봅시다.
print("현재 운영체제:", os.name)
print("현재 폴더:", os.getcwd())
print("현재 폴더 내부의 요소", os.listdir())

# 폴더를 만들고 제거합니다(폴더 제거는 폴더가 비어있을 때만 제거 가능)
os.mkdir("hello")
print(os.path.exists("hello")) # 생성된 폴더가 있는지 확인해 봅니다.

os.rmdir("hello")
print(os.path.exists("hello")) # 생성된 폴더가 있는지 확인해 봅니다.

# 파일을 생성하고 + 파일 이름을 변경합니다.
with open("original.txt", "w") as file:
	file.write("hello")

print(os.path.exists("original.txt"))
os.rename("original.txt", "new.txt")
print(os.path.exists("new.txt"))

# 파일을 제거합니다.
os.remove("new.txt")
print(os.path.exists("new.txt"))

# 시스템 명령어 실행
os.system("dir")

```

```
실행 결과

현재 운영체제: posix
현재 폴더: /content
현재 폴더 내부의 요소 ['.config', 'sample_data']
True
False
True
True
False
0
```
## 1.5 datetime 모듈

`datetime` 모듈은 date(날짜), time(시간)과 관련된 모듈로, 날짜 형식을 만들 때 자주 사용되는 코드들로 구성되어 있습니다. 그럼 예제를 통해 `datetime` 모듈의 다양한 사용 방법에 대해서 알아보도록 하겠습니다.

```python
import datetime

# 현재 시작을 구하고 출력하기
print("# 현재 시각 출력하기")
now = datetime.datetime.now()
print(now.year, "년")
print(now.month, "월")
print(now.day, "일")
print(now.hour, "시")
print(now.minute, "분")
print(now.second, "초")
print()

# 시간 출력 방법
print("# 시간을 포맷에 맞춰 출력하기")
output_a = now.strftime("%Y.%m.%d %H:%M:%S")
output_b = "{}년 {}월 {}일 {}시 {}분 {}초".format(now.year,
	now.month, now.day, now.hour, now.minute, now.second)

output_c = now.strftime("%Y{} %m{} %d{} %H{} %M{} %S{}").format(*"년월일시분초")
print(output_a)
print(output_b)
print(output_c)
print()
```

```
실행 결과

# 현재 시각 출력하기
2025 년
9 월
16 일
8 시
30 분
31 초

# 시간을 포맷에 맞춰 출력하기
2025.09.16 08:30:31
2025년 9월 16일 8시 30분 31초
2025년 09월 16일 08시 30분 31초
```

output_a 처럼 `strftime()` 함수를 사용하면 시간을 형식에 맞춰 출력할 수 있습니다. 다만 한국어 등의 문자는 매개변수에 넣을 수 없습니다. 그래서 이를 보완하고자 output_b 와 output_c 같은 형식을 사용합니다.

이 외에도 datetime 모듈은 다양한 시간 처리 기능을 가지고 있습니다.

```python
import datetime

now = datetime.datetime.now()

# 특정 시간 이후의 시간 구하기
print("# datetime.timedelta로 시간 더하기")
after = now + datetime.timedelta(
	weeks=1, days=1, hours=1, minutes=1, seconds=1)

print(after.strftime("%Y{} %m{} %d{} %H{} %M{} %S{}").format(*"년월일시분초"))
print()

# 특정 시간 요소 교체하기
print("# now.replace()로 1년 더하기")
output = now.replace(year=(now.year + 1))
print(output.strftime("%Y{} %m{} %d{} %H{} %M{} %S{}").format(*"년월일시분초"))
```

```
실행 결과

# datetime.timedelta로 시간 더하기
2025년 09월 24일 09시 42분 04초

# now.replace()로 1년 더하기
2026년 09월 16일 08시 41분 03초
```

'timedelta()' 함수를 사용하면 특정한 시간의 이전 또는 이후를 구할 수 있습니다. 다만 `timedelta()` 함수는 1년 후, 2년 후 등의 몇 년 후를 구하는 기능이 없습니다. 그래서 1년 후를 구할 때는 `replace()` 함수를 사용해 아예 날짜 값을 교체하는 것이 일반적입니다.

## 1.6 time 모듈

시간과 관련된 기능을 다룰 때는 `time` 모듈을 사용합니다. `time` 모듈로도 날짜와 관련된 처리를 할 수 있지만, 그런 처리는 `datetime` 모듈을 사용하는 경우가 더 많습니다.

`time` 모듈은 4장에서 살펴보았던 것처럼 유닉스 타임(1970년 1월 1일 0시 0분 0초를 기준으로 계산한 시간 단위)을 구할 때, 특정 시간 동안 코드 진행을 정지할 때 많이 사용합니다. 그럼 예제를 통해 구체적으로 알아보도록 하겠습니다.

그럼 굉장히 자주 사용되는 `time.sleep()` 함수를 알아보겠습니다. `time.sleep()` 함수는 특정 시간 동안 코드 진행을 정지할 때 사용되는 함수입니다. 매개변수에는 정지하고 싶은 시간을 초 단위로 입력합니다.

```python
import time
import datetime

print("지금부터 5초 동안 정지합니다.")
now = datetime.datetime.now()
print(now.strftime("%Y{} %m{} %d{} %H{} %M{} %S{}").format(*"년월일시분초"))
time.sleep(5)
now = datetime.datetime.now()
print(now.strftime("%Y{} %m{} %d{} %H{} %M{} %S{}").format(*"년월일시분초"))
print("프로그램을 종료합니다.")
```

코드를 실행하면 "지금부터 5초 동안 정지합니다!"를 출력하고 5초 동안 정지합니다. 그리고 5초 후에 "프로그램을 종료합니다"를 출력합니다. 매우 자주 사용하는 기능인데다가 어렵지 않으므로 외워두면 좋습니다.

## 1.7 urllib 모듈

이번에는 `urllib` 모듈에 대해서 살펴보겠습니다. `urllib` 모듈은 "URL 을 다루는 라이브러리"라는 의미입니다. 이때 URL 이란 "Unifor Resources Locator"를 의미하는 말로, 어렵게 표현하면 네트워크의 자원이 어디에 위치하는지 확인할 때 사용하는 것입니다.

`urllib` 모듈은 인터넷 주소를 활용할 때 사용하는 라이브러리입니다. 그러면 예제를 통해 살펴보도록 하겠습니다.

```python
from urllib import request

# urlopen() 함수로 구글의 메인 페이지를 읽습니다.
target = request.urlopen("https://google.com")
output = target.read()

# 출력합니다
print(output)
```

일단 `from urllib import request` 를 사용해 `urllib` 모듈에 있는 `request` 를 가져왔습니다. 이 때 `request` 도 모듈이라 이후의 코드에서 request 모듈 내부에 있는 `urlopen()` 함수를 `request.urlopen()` 형태로 사용했습니다.

`urlopen()` 함수는 URL 주소의 페이지를 열어주는 함수입니다. 구글의 메인 페이지 주소를 넣어 보았는데, 이렇게 입력하면 웹 브라우저에 "https://google.com"를 입력해서 접속하는 것처럼 Python 이 "https://google.com"에 들어가 줍니다.

이어서 `read()` 함수를 호출하면 해당 웹 페이지에 있는 내용을 읽어서 가져옵니다. 코드를 실행하면 다음과 같은 결과를 출력합니다.

```
실행 결과

b'<!doctype html><html itemscope="" itemtype="http://schema.org/WebPage" lang="en"><head><meta content="Search the world\'s information, including webpages, images, videos and more. Google has many special features to help you find exactly what you\'re looking for." name="description"><meta content="noodp, " name="robots"><meta content="text/html; charset=UTF-8" http-equiv="Content-Type"><meta content="/images/branding/googleg/1x/googleg_standard_color_128dp.png" itemprop="image"><title>Google</title><script nonce="Bv3EBtcfGwq5bWUpU84Vew">(function(){var _g={kEI:\'IijJaLikLrms5NoPva-toQ0\',kEXPI:\'0,202854,2,82,3946975,104243,425603,226411,1,63632,5230292,11847,199,36811986,25228681,123988,14280,14115,62507,2663,3431,3319,23878,7042,2097,4600,328,6225,64164,15049,8211,3286,4134,30380,28333,10904,12054,25322,5932,353,10700,31,7740,409,5870,7707,7,5774,27611,4719,837,10968,6251,28,7,1,1293,4989,1,5970,4650,2360,459,9288,5683,3604,1083,6245,10437,7,1,2729,632,15919,15304,566,6026,1,9631,1032,6436,9743,2646,98,5,4,1,321,843,3276,983,667,2388,1722,8177,730,5031,1472,2,1,8117,4426,1279,408,183,1,2,1380,31,104,3044,311,1057,2839,2114,2,1347,5,2,681,127,4930,328,453,2,7,803,864,1500,1949,1719,1205,154,4,4,4212,1,2,882,2,2230,569,3342,1,1,418,1,162,458,9,143,3012,14,781,3347,318,1003,942,1,228,2,3296,648,120,112,5,2516,1835,969,317,3359,515,21,89,7,645,6,5,495,5,786,1420,240,2,548,746,613,5,1774,2,1460,5,7,1138,2,540,30,66,1,169,22,3372,859,507,139,274,59,195,40,796,223,

생략...
```

# 2. 모듈 만들기

Python 은 모듈 만드는 방법이 간단합니다. 단순하게 파이썬 파일을 만들고, 이를 외부에서 읽어 들이면 모듈이 됩니다. 너무 간단하고 유연해서 모듈을 코드로 인식하고 실행해 버리는 문제 등이 발생할 수 있습니다. 그러나 파이썬은 이를 막기 위한 다양한 대처 방법도 제공해 줍니다. 또한 모듈을 구조화해서 패키지를 만드는 기능도 제공해 줍니다.

이번에는 원의 반지름과 넓이를 구하는 간단한 모듈을 만들어 보면서 모듈을 만드는 방법, 모듈 실행과 관련된 안전 장치를 설치하는 방법, 패키지를 만드는 방법에 대해서 알아보도록 하겠습니다.

먼저 module_basic 디렉터리를 만들어 `main.py`, `test_module.py` 파일을 넣어주시기 바랍니다. `main.py` 가 메인 코드로 활용할 부분입니다.

## 2.1 모듈 만들기

Python 에서는 모듈 만들기가 매우 쉽습니다. 모듈 내부에 변수와 함수 등을 잔뜩 넣어주면 되는데, 간단하게 이전에 만들어 봤던 함수들을 넣어 보겠습니다.

```python
# test_module.py 파일
PI = 3.141592

def number_input():
	output = input("숫자 입력> ")
	return float(output)

def get_circumference(radius):
	return 2 * PI * radius

def get_circle_area(radius):
	return PI * radius * radius
```

```python
# main.py
import test_module as test

radius = test.number_input()
print(test.get_circumference(radius))
print(test.get_circle_area(radius))
```

```
실행 결과

숫자 입력> 10
62.83184
314.1592
```

## 2.2 __name__=="__main__"

다른 사람들이 만든 Python 코드들을 보다 보면 `__name__="__main__"` 이라는 코드를 많이 볼 수 있스빈다. 많은 Python 개발자들도 이게 뭔지도 모르고 그냥 사용하는 경우가 많은데 이 의미가 무엇인지 한 번 짚고 넘어가도록 하겠습니다.

### __name__

파이썬 코드 내부에서는 `__name__` 이라는 변수를 사용할 수 있습니다. `__name__` 이라는 변수에 어떤 값이 들어 있는지 확인해 보겠습니다.

```python
print(__name__) # __main__
```

프로그래밍 언어에서는 프로그램의 진입점을 엔트리 포인트(entry point) 또는 메인(main)이라고 부릅니다. 그리고 이러한 엔트리 포인트 또는 메인 내부에서의 `__name__` 은 `__main__` 입니다.

### 모듈의 __name__

엔트리 포인트가 아니지만 엔트리 포인트 파일 내에서 import 되었기 때문에 모듈 내 코드가 실행됩니다. 모듈 내부에서 `__name__`을 출력하면 모듈의 이름을 나타냅니다. 간단하게 코드를 구성해 보겠습니다.

```python
# main.py 파일
import test_module

print("# 메인의 __name__ 출력하기")
print(__name__)
print()
```

```
실행 결과

# 모듈의 __name__ 출력하기
test_module

# 메인의 __name__ 출력하기
__main__
```

코드를 실행하면 엔트리 포인트 파일에서는 "__main__" 을 출력하지만, 모듈 파일에서는 모듈 이름을 출력하는 것을 볼 수 있습니다.

### __name__ 활용하기

엔트리 포인트 파일 내부에서는 `__name__`이 "__main__"이라는 값을 갖습니다. 이를 활용하면 현재 파일이 모듈로 실행되는지, 엔트리 포인트로 실행되는지 확인할 수 있습니다.

예를 들어 다음 코드를 살펴보겠습니다. test_module.py 라는 이름으로 프로그램을 만들었습니다. 그리고 "이러한 형태로 활용한다"라는 것을 보여주기 위해 간단한 출력도 넣었습니다.

```python
# test_module.py 파일
PI = 3.141592

def number_input():
	output = input("숫자 입력> ")
	return float(output)

def get_circumference(radius):
	return 2 * PI * radius

def get_circle_area(radius):
	return PI * radius * radius

# 활용 예
print("get_circumference(10):", get_circumference(10))
print("get_circle_area(10):", get_circle_area(10))
```

```python
# main.py
import test_module as test

radius = test.number_input()
print(test.get_circumference(radius))
print(test.get_circle_area(radius))
```

```
실행 결과

get_circumference(10): 62.83184
get_circle_area(10): 314.1592
숫자 입력> 10
62.83184
314.1592
```

그런데 현재 `test_module.py` 라는 파일에는 "이런 식으로 동작해요!"라는 설명을 위해 추가한 활용 예시 부분이 있습니다. 모듈로 사용하고 있는데, 내부에서 출력이 발생하니 문제가 됩니다.

이 때 현재 파일이 엔트리 포인트인지 구분하는 코드를 활용합니다. 조건문으로 `__name__` 이 "__main__" 인지 확인만 하면 됩니다.

```python
# test_module.py 파일
PI = 3.141592

def number_input():
	output = input("숫자 입력> ")
	return float(output)

def get_circumference(radius):
	return 2 * PI * radius

def get_circle_area(radius):
	return PI * radius * radius

# 활용 예
if __name__ == "__main__":
	print("get_circumference(10):", get_circumference(10))
	print("get_circle_area(10):", get_circle_area(10))
```

```python
# main.py
import test_module as test

radius = test.number_input()
print(test.get_circumference(radius))
print(test.get_circle_area(radius))
```

```
실행 결과

숫자 입력> 10
62.83184
314.1592
```

자주 사용되는 형태의 코드입니다. 인터넷에서 다른 사람들이 만든 코드를 보다 보면 100% 확률로 만날 수 있을텐데, 그럴 때 당황하지 않도록 꼭 기억하시기 바랍니다.

## 2.3 패키지

패키지는 모듈이 모여서 구조를 이루면 패키지가 됩니다. 

### 패키지 만들기

패키지를 만들어 보겠습니다. 일단 폴더를 다음과 같이 구성합니다. `main.py` 파일은 엔트리 포인트로 사용할 파이썬 파일이며, `test_package` 폴더는 패키지로 사용할 폴더입니다.

`test_package` 폴더 내부에 모듈을 하나 이상 넣어 패키지를 만들어 보겠습니다. 예제로 `module_a.py` 파일과 `module_b.py` 파일을 생성해 보겠습니다. 이어서 파일들에 다음과 같이 입력합니다.

```python
# ./test_package/module_a.py 의 내용
variable_a = "a 모듈의 변수"
```

```python
# ./test_package/module_b.py 의 내용
variable_b = "b 모듈의 변수"
```

```python
# main.py

# 패키지 내부의 모듈을 읽어 들입니다.
import test_package.module_a as a
import test_package.module_b as b

# 모듈 내부의 변수를 출력합니다.
print(a.variable_a)
print(b.variable_b)
```

```
실행 결과

a 모듈의 변수
b 모듈의 변수
```

### __init__.py 파일

패키지를 읽을 때 어떤 처리를 수행해야 하거나 패키지 내부의 모듈들을 한꺼번에 가져오고 싶을 때가 있습니다. 이럴 때는 패키지 폴더 내부에 `__init__.py` 파일을 만들어 사용합니다.

`test_package` 폴더 내부에 다음과 같이 `__init__.py` 파일을 만들어 보겠습니다. 패키지를 읽어 들일 때 `__ini__.py` 를 가장 먼저 실행합니다.

따라서 패키지와 관련된 초기화 처리 등을 할 수 있습니다. `__init__py` 에서는 `__all__` 이라는 이름의 리스트를 만드는데, 이 리스트에 지정한 모듈들이 `from <패키지_이름> import *`을 할 때 전부 읽어 들여집니다.

```python
# __init__.py
# "from test_package import *"
# 모듈을 읽어 들일 때 가져올 모듈

__all__ = ["module_a", "module_b"]

# 패키지를 읽어 들일 때 처리를 작성할 수도 있습니다.
print("test_package 를 읽어 들였습니다.")
```

```python
# 패키지 내부의 모듈을 모두 읽어 들입니다.
from test_package import *

# 모듈 내부의 변수를 출력합니다.
print(module_a.variable_a)
print(module_b.variable_b)
```

```
실행 결과

test_package 를 읽어 들였습니다.
a 모듈의 변수
b 모듈의 변수
```

# 마치며

Python 의 모듈의 기초에 대해서 알아보았습니다. 내용 중 잘못된 내용이나 오타, 궁금하신 내용이 있으시다면 댓글 달아주시기 바랍니다. 긴 글 읽어주셔서 감사합니다.