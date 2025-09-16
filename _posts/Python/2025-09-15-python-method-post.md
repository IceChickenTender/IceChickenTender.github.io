---
title: "[Python] Python 함수 기초"
categories:
  - Python
tags:
  - Python
  
toc: true
toc_sticky: true
toc_label: "Python 함수 기초"
---

이전에도 느꼈지만 최근 다시 LLM 을 위해 딥러닝을 공부하면서 저 스스로가 Python 기초가 조금 부족하다고 생각하여 여태까지 미뤄왔던 Python 의 기초들에 대해서 짚고 가려고 합니다. 대학원생 시절에 딥러닝 공부를 하면서도 Python 기초가 부족했었지만 그래도 Python 문법 자체가 다른 언어들에 비해서 쉽고, 또 당장 프로젝트를 위한 모듈을 만들었어야 했어서 그 당시에는 Python 기초 공부할 시간적 여유가 없었는데 그걸 이제야 하게 되었습니다. 그럼 "혼자 공부하는 파이썬(윤인서 저)"이라는 책을 참고하여 Python 의 함수에 대해서 공부해보도록 하겠습니다. 너무 기초적인 것들은 건너뛰고 제가 몰랐던 내용이나 중요하다고 생각되는 것들만 다루도록 하겠습니다.

# 1. 함수 기초

## 1.1 Python 함수의 구성

Python 의 함수에서 사용되는 용어부터 정리해 보도록 하겠습니다.

- 매개변수 : 함수 호출 시 함수 내부로 전달되는 값(인수)를 받아서 함수가 처리할 데이터를 나타내는 변수를 말합니다.
- 리턴값 : 함수 실행 결과를 호출한 코드로 돌려주는 값을 말합니다.

함수의 기본 형태는 다음과 같습니다.

```python
def 함수_이름():
	dosomething
```

그렇다면 실제로 함수를 구현해 보도록 하겠습니다.

```python
def print_3_times():
	print("안녕하세요")
	print("안녕하세요")
	print("안녕하세요")

print_3_times()
```

```
실행 결과

안녕하세요
안녕하세요
안녕하세요
```

이번엔 매개변수를 이용한 함수 예제를 보도록 하겠습니다.

```python
def print_n_times(value, n):
	for i in range(n):
		print(value)

print_n_times("안녕하세요", 5)
```

```
실행 결과

안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
```

함수의 괄호 내부에 value 와 n 이라는 식별자를 입력했습니다. 이렇게 입력하면 매개변수가 됩니다. 이렇게 매개변수를 만들면 함수를 호출할 때 값을 입력해서 함수쪽으로 전달할 수 있습니다.

## 1.2 가변 매개변수

앞서 살펴본 함수에서는 함수를 선언할 때의 매개변수와 함수를 호출할 때의 매개변수가 같아야 했습니다. 적어도 안 되고, 많아도 안됩니다. 그러나 우리가 많이 사용하는 `print()` 함수는 매개변수를 원하는 만큼 입력할 수 있습니다. `print()` 함수와 같이 매개변수를 원하는 만큼 받을 수 있는 함수를 가변 매개변수라고 부릅니다. 가변 매개변수를 사용한 함수의 구조는 다음과 같습니다.

```python
def 함수_이름(매개변수, 매개변수 ..., *가변 매개변수):
	dosomething
```

가변 매개변수를 사용할 때는 다음과 같은 제약이 있습니다.

- 가변 배개변수 뒤에는 일반 매개변수가 올 수 없습니다.
- 가변 매개변수는 하나만 사용할 수 있습니다.

그럼 예제를 보도록 하겠습니다.

```python
def print_n_times(n, *values):
	# n번 반복합니다.
	for i in range(n):
		# values 는 리스트처럼 활용합니다.
		for value in values:
			print(value)
		print()

print_n_times(3, "안녕하세요", "즐거운", "파이썬 프로그래밍")
```

```
실행 결과

안녕하세요
즐거운
파이썬 프로그래밍

안녕하세요
즐거운
파이썬 프로그래밍

안녕하세요
즐거운
파이썬 프로그래밍
```

가변 매개변수 뒤에는 일반 매개변수가 올 수 없다고 했습니다. 만약 가변 매개변수 뒤에 일반 매개변수가 올 수 있다고 한다면 어디까지가 가변 매개변수이고, 어디가 일반 매개변수인지 구분하기 힘듭니다. 그래서 Python 은 내부적으로 가변 매개변수 뒤에 일반 매개변수가 오지 못하게 막은 것입니다.

## 1.3 기본 매개변수

`print()` 함수의 자동 완성 기능으로 나오는 설명을 적어보면 다음과 같습니다.

```python
print(vlaue, ..., sep=' ', end='\n', file=sys.stdout, flush=False)
```

가장 앞에 있는 value 가 바로 '가변 매개변수'입니다. 가변 매개변수 뒤에는 일반 매개변수가 올 수 없다고 했는데 매개변수가 왔습니다. 그런데 뭔가 특이하게 '매개변수=값' 형태로 되어 있습니다. 이는 `기본 매개변수`라고 부르며, 매개변수를 입력하지 않았을 경우 매개변수에 들어가는 기본값입니다. 기본 매개변수도 다음과 같은 제약이 있습니다.

- 기본 매개변수 뒤에는 일반 매개변수가 올 수 없습니다.

그럼 예제를 통해 보도록 하겠습니다.

```python
def print_n_times(value, n=2):
	# n번 반복합니다.
	for i in range(n):
		print(value)

print_n_times("안녕하세요")
```

```
실행 결과

안녕하세요
안녕하세요
```

## 1.4 키워드 매개변수

키워드 매개변수를 알아보기 전에 가변 매개변수와 기본 매개변수 둘을 같이 써도 되는지 한 번 알아보도록 하겠습니다.

### 기본 매개변수가 가변 매개변수보다 앞에 올 때

기본 매개변수가 가변 매개변수보다 앞에 올 때는 기본 매개변수의 의미가 사라집니다. 다음 코도의 실행 결과를 예측해 봅시다. `n` 값에는 무엇이 들어갈까요?

```python
def print_n_times(n=2, *values):
	for i in range(n):
		for value in values:
			print(value)
		print()

print_n_times("안녕하세요", "즐거운", "파이썬 프로그래밍")
```

매개변수가 순서대로 입력되므로 `n` 에는 "안녕하세요"가 들어가고, values 에는 ["즐거운", "파이썬 프로그래밍"]이 들어가게됩니다. 그런데 `range()` 함수의 매개변수에는 숫자만 들어올 수 있으므로 다음과 같은 오류가 발생합니다.

```

TypeError                                 Traceback (most recent call last)
/tmp/ipython-input-119174662.py in <cell line: 0>()
      5                 print()
      6 
----> 7 print_n_times("안녕하세요", "즐거운", "파이썬 프로그래밍")

/tmp/ipython-input-119174662.py in print_n_times(n, *values)
      1 def print_n_times(n=2, *values):
----> 2         for i in range(n):
      3                 for value in values:
      4                         print(value)
      5                 print()

TypeError: 'str' object cannot be interpreted as an integer

```

따라서 기본 매개변수는 가변 매개변수 앞에 쓰면 안된다는 것을 기억하시면 됩니다.

### 가변 매개변수가 기본 매개변수보다 앞에 올 때

그러면 반대로 가변 매개변수가 기본 매개변수보다 앞에 올 때는 어떻게 되는지 보도록 하겠습니다.

```python
def print_n_times(*values, n=2):
	for i in range(n):
		for value in values:
			print(value)
		print()

print_n_times("안녕하세요", "즐거운", "파이썬 프로그래밍", 3)
```

코드를 실행하면 입력 받은 4개의 값들을 2번 씩 출력하도록 실행이 됩니다.

```
실행 결과

안녕하세요
즐거운
파이썬 프로그래밍
3

안녕하세요
즐거운
파이썬 프로그래밍
3
```

그래서 두 가지를 함께 사용하기 위해 Python 은 키워드 매개변수라는 기능을 만들었습니다. 그럼 이제 키워드 매개변수에 대해서 알아보도록 하겠습니다.

키워드 매개변수는 매개변수 이름을 직접적으로 지정해서 값을 입력합니다. 그럼 직전에 봤던 예제를 조금 수정해 보도록 하겠습니다.

```python

def print_n_times(*values, n=2):
	for i in range(n):
		for value in values:
			print(value)
		print()

#print_n_times("안녕하세요", "즐거운", "파이썬 프로그래밍", 3) -> 직전 예시
print_n_times("안녕하세요", "즐거운", "파이썬 프로그래밍", n=3)
```

이전 예제와는 달리 우리가 의도한 대로 세 개의 문자열이 세 번 출력되도록 하는 것을 확인할 수 있습니다.

```
실행 결과

안녕하세요
즐거운
파이썬 프로그래밍

안녕하세요
즐거운
파이썬 프로그래밍

안녕하세요
즐거운
파이썬 프로그래밍
```

## 1.5 리턴

`input()` 함수를 보도록 하겠습니다. `input()` 함수는 함수를 실행하고 나면 다음과 같은 형태로 함수의 결과를 받아서 사용합니다. 이와 같은 함수의 결과를 리턴값이라고 부릅니다.

```python
#input() 함수의 리턴값을 변수에 저장합니다.

value = input(">" )

# 출력합니다.
print(value)
```

# 2. 함수 고급 활용

Python 에서는 함수를 조금 더 편리하게 사용할 수 있는 여러 방법들이 있습니다. 그 대표적인 것이 튜플(tupe)과 람다(Lambda)를 활용하는 것입니다.

- 튜플 : 함수와 함께 사용되는 리스트와 비슷한 자료형으로, 리스트와 다른 점은 한 번 결정된 요소는 바꿀 수 없다는 것
- 람다 : 매개변수로 함수를 전달하기 위해 함수 구문을 작성하는 것이 번거롭고, 코드 공간 낭비라 생각이 들 때 간단하고 쉽게 선언하는 방법

## 2.1 튜플을 활용한 함수

튜플은 함수의 리턴에 많이 사용합니다. 함수의 리턴에 튜플을 사용하면 여러 개의 값을 리턴하고 할당할 수 있기 때문입니다. 다음 예제를 보도록 하겠습니다.

```python
def test():
	return (10, 20)

a, b = test()

print("a: ", a)
print("b: ", b)
```

```
실행 결과 

a:  10
b:  20
```

## 2.2 람다를 활용한 함수

Python 은 함수를 매개변수로 전달할 수 있습니다. 그리고 이런 Python 은 이를 더 효율적으로 활용할 수 있게 람다라는 기능을 제공합니다. 

### 함수의 매개변수로 함수 전달하기

함수를 매개변수로 전달하는 것부터 람다까지 알아보도록 하겠습니다.

다음은 함수의 매개변수로 함수를 전달하는 코드입니다.

```python
# 매개변수로 받은 함수를 10번 호출하는 함수
def call_10_times(func):
	for i in range(10):
		func()

# 간단히 출력하는 함수
def print_hello():
	print("안녕하세요")

call_10_times(print_hello)
```

프로그램을 실행하면 print_hello() 함수를 10번 실행합니다. 따라서 "안녕하세요"라는 문자열을 10번 출력합니다.

```
실행 결과

안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
```

### filter() 함수와 map() 함수

함수를 매개변수로 전달하는 대표적인 표준 함수로 `map()` 함수와 `filter()` 함수가 있습니다. 

- map() : 리스트, 튜플 같은 여러 값(반복 가능한 객체)의 각 요소에 지정한 함수를 하나씩 적용해, 그 결과값들을 순서대로 모아둔 이터레이터를 만들어주는 함수입니다.
- filter() : 리스트, 튜플 등의 각 요소를 지정한 조건 함수에 넣어 True 가 되는 값들만 골라, 그 선택된 요소들을 순서대로 담은 이터레이터를 만들어주는 함수입니다.

예제를 통해 구체적으로 알아보도록 하겠습니다.

```python
# 함수를 선언합니다.
def power(item):
	return item * item

def under_3(item):
	return item < 3

# 변수를 선언합니다

list_input_a = [1, 2, 3, 4, 5]

# map() 함수를 사용합니다.
output_a = map(power, list_input_a)
print("# map() 함수의 실행결과")
print("map(power, list_input_a): ", output_a)
print("map(power, list_input_a): ", list(output_a))
print()

# filter() 함수를 사용합니다.
output_b = filter(under_3, list_input_a)
print("# filter() 함수의 실행결과")
print("filter(under_3, output_b):", output_b)
print("filter(under_3, output_b):", list(output_b))


```

`map()` 함수와 `filter()` 함수는 모두 첫 번째 매개변수에 함수, 두 번째 매개변수에 리스트를 넣습니다. 일단 `map()` 함수를 살펴보도록 하겠습니다. 첫 번째 매개변수에는 값을 제곱해 주는 `power()` 함수를 넣었습니다.

두 번째 매개변수에는 `[1, 2, 3, 4, 5]` 라는 리스트를 넣었습니다. 그리고 결과로 `[1, 2, 3, 4, 5]` 내부의 요소에 `power()` 함수가 적용된 `[1, 4, 9, 16, 25]`를 얻었습니다.

이제 `filter()` 함수를 살펴보도록 하겠습니다. 첫 번째 매개변수에는 `item < 3` 을 판정하는 `under_3()`함수를 넣었습니다. 두 번째 매개변수에는 `[1, 2, 3, 4, 5]` 라는 리스트를 넣었습니다. 그리고 결과로 `[1, 2, 3, 4, 5]` 내부의 요소 중에 3 보다 작은 값을 가지는 `[1, 2]`를 얻었습니다.

### 람다의 개념

매개변수로 함수를 전달하가 위해 함수 구문을 작성하는 것도 번거롭고, 코드 공간 낭비라는 생각이 들 수 있습ㄴ디ㅏ. 많은 개발자들이 이러한 생각을 했고, 그래서 람다(lambda)라는 개념을 생각했습니다.

람다는 "간단한 함수를 쉽게 선언하는 방법"입니다. 다음과 같은 형태로 만듭니다.

```python
lambda 매개변수 : 리턴값
```

이전 `power()` 함수와 `under_3()` 함수를 람다로 변환해 보도록 하겠습니다. `def` 키워드로 선언했던 함수를 `lambda`로 바꾸고, `return` 키워드를 따로 쓰지 않았다는 정도의 차이가 생겼습니다.

```python
# 함수를 선언합니다.
power = lambda x: x*x
under_3 = lambda x: x<3

# 변수를 선언합니다

list_input_a = [1, 2, 3, 4, 5]

# map() 함수를 사용합니다.
output_a = map(power, list_input_a)
print("# map() 함수의 실행결과")
print("map(power, list_input_a): ", output_a)
print("map(power, list_input_a): ", list(output_a))
print()

# filter() 함수를 사용합니다.
output_b = filter(under_3, list_input_a)
print("# filter() 함수의 실행결과")
print("filter(under_3, output_b):", output_b)
print("filter(under_3, output_b):", list(output_b))
```

```
실행 결과

# map() 함수의 실행결과
map(power, list_input_a):  <map object at 0x7cde960886d0>
map(power, list_input_a):  [1, 4, 9, 16, 25]

# filter() 함수의 실행결과
filter(under_3, output_b): <filter object at 0x7cde96088700>
filter(under_3, output_b): [1, 2]
```

람다는 간단한 함수를 쉽게 선언하는 방법이라고 했는데, 왜 사용하는지가 의심스러울 정도로 복잡합니다. 람다는 다음과 같이 함수의 매개변수에 곧바로 넣을 수 있습니다.

```python
list_input_a = [1, 2, 3, 4, 5]

# map() 함수를 사용합니다.
output_a = map(lambda x: x*x, list_input_a)
print("# map() 함수의 실행결과")
print("map(power, list_input_a): ", output_a)
print("map(power, list_input_a): ", list(output_a))
print()

# filter() 함수를 사용합니다.
output_b = filter(lambda x:x<3, list_input_a)
print("# filter() 함수의 실행결과")
print("filter(under_3, output_b):", output_b)
print("filter(under_3, output_b):", list(output_b))
```

실행 결과는 이전과 같습니다. 람다를 사용하면 코드를 더 깔끔하게 작성할 수 있고, 함수가 매개변수로 넣어졌다고 확인하고 어떤 함수인지를 알기 위해 다시 찾아 올라가는 수고를 하지 않아도 됩니다.

지금은 매개변수가 하나인 람다만을 살펴봤는데, 다음과 같이 매개변수가 여러 개인 람다도 만들 수 있습니다.

```python
lambda x, y : x*y
```

# 마치며

Python 함수의 기초에 대해서 포스트로 정리해보았습니다. 내용 중에 잘못된 내용이나 오타, 궁금하신 것이 있으시다면 댓글 달아주시기 바랍니다. 긴 글 읽어주셔서 감사합니다.