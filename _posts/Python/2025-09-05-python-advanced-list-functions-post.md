---
title: "[Python] Python 고급 리스트 기능에 대해"
categories:
  - Python
tags:
  - Python
  
use_math: true
toc: true
toc_sticky: true
toc_label: "Python 고급 리스트 기능"
---

이번엔 Python 에서 가장 많이 사용되는 리스트의 고급 기능들에 대해서 알아보도록 하겠습니다. "파이썬 스킬업!" 이라는 책을 참고하였습니다.

# 1. 리스트 변수 복사

Python 으로 코딩을 하다보면 리스트를 복사해야 할 때가 많습니다. 하지만 Python 은 객체 참조이기 때문에 잘못 복사했다간 얕은 복사가 되버리는 경우가 많습니다. 그래서 보통은 깊은 복사를 많이 하지만 특정 상황에서는 얕은 복사를 이용한 리스트 변수 복사를 활용하는게 더 효율적일 때가 있습니다. 그래서 리스트 변수 복사 방법에 대해서 알아보도록 하겠습니다.

다음과 같이 리스트의 얕은 복사를 진행할 경우 `a_list` 와 `b_list` 에 변경이 생기면 양쪽 모두에 영향을 주게 되어 버그를 발생시키는 주 원인이 됩니다.

```python
a_list = [2, 5, 10]
b_list = a_list

b_list.append(100)
a_list.append(200)
b_list.append(1)
print(a_list) # [2, 5, 10, 100, 200, 1]
print(b_list) # [2, 5, 10, 100, 200, 1]

```

그래서 만약 리스트 전체 항목을 별도로 복사하고 싶다면 항목 간 복사를 수행해야 합니다. 가장 간단한 방법은 슬라이싱을 사용하는 것입니다.

```python
my_list = [1, 10, 5]
yr_list = my_list[:]

my_list.append(100)
yr_list.append(200)

print(my_list) # [1, 10, 5, 100]
print(yr_list) # [1, 10, 5, 200]
```

이 같은 방법은 리스트 객체 자체를 복사하는 것이 아닌 리스트 안에 있는 요소 객체들을 복사하는 것이기 때문에 원본 리스트나 복사한 리스트의 요소가 변경이 되어도 영향을 주지 않습니다. 하지만 리스트의 요소가 변경 가능한 것들(리스트, 딕셔너리 등)이라면 얕은 복사로 인한 문제가 똑같이 발생할 수 있기 때문에 이러한 경우에는 깊은 복사 방법을 사용해 주어야 합니다.

# 2. 슬라이싱을 이용한 값 대입하기

리스트는 슬라이싱을 이용해 값을 대입할 수 있습니다. 예제로 바로 알아보도록 하겠습니다. 아래 예제를 보면 `my_list` 의 1부터 3까지 있는 요소를 삭제하고 그 자리에 `707` 과 `777` 을 삽입하게 됩니다.

```python
my_list = [10, 20, 30, 40, 50, 60]
my_list = [1:4] = [707, 777]

print(my_list) # [10, 707, 777, 50, 60]

```

슬라이싱 범위의 길이가 0인 인덱스를 인덱스를 넣을 수도 있습니다. 이렇게 되면 기존 값을 삭제하지 않고, 해당 위치에 새로운 리스트 항목을 삽입합니다. 예제를 통해 알아보도록 하겠습니다.

```python
my_list = [1, 2, 3, 4]
my_list[0:0] = [-50, -40]

print(my_list) # [-50, -40, 1, 2, 3, 4]

my_list[1:1] = [100, 200]

print(my_list) # [-50, 100, 200, -40, 1, 2, 3, 4]
```

다만 이러한 슬라이싱을 이용한 값 대입을 사용할 땐 다음과 같은 제약사항이 있습니다.

- 슬라이싱을 이용해 값을 넣을 때에 넣고자 하는 값은 항상 리스트나 다른 컬렉션에 담겨있어야 하며, 그 리스트나 컬렉션을 이용해야 합니다.
- 슬라이싱을 이용해 값을 넣을 때에 스텝이 명시된다면, 슬라이싱 문법을 통해 추출되는 요소의 개수와 삽입할 데이터의 길이가 반드시 같아야 합니다. 이 경우는 예제를 통해서 구체적으로 알아보도록 하겠습니다.

```python
my_list = [10, 20, 30, 40, 50, 60]

print(my_list[1:4:2]) # [20, 40] 로 2개의 요소가 추출됨

my_list[1:4:2] = [1, 2, 3] # 2개가 아닌 다른 길이의 리스트를 대입하게 되면 에러가 발생함
```

```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipython-input-3996935856.py in <cell line: 0>()
      3 print(my_list[1:4:2])
      4 
----> 5 my_list[1:4:2] = [1, 2, 3]
      6 
      7 print(my_list) # [10, 707, 777, 50, 60]

ValueError: attempt to assign sequence of size 3 to extended slice of size 2

```

```python
my_list = [10, 20, 30, 40, 50, 60]

print(my_list[1:4:2])

my_list[1:4:2] = [1, 2]

print(my_list) # [10, 1, 30, 2, 50, 60] 스텝이 적용되어 `20`, `40` 이 있던 곳에만 값이 들어가게 됨 
```

# 3. 리스트 연산자

리스트에는 여러 연산자들이 있습니다. 이를 표로 정리하면 다음과 같습니다.

|연산자/문법|상세 설명|
|----------|---------|
|리스트1 + 리스트2|이어 붙이기가 수행되면서 리스트1과 리스트2의 모든 항목이 포함된 새로운 리스트를 생성한다.|
|리스트1 * n 혹은 n * 리스트1|리스트1의 항목을 n 번 반복한 리스트를 생성한다.|
|리스트1=리스트2|리스트1에 리스트2가 참조하고 있는 객체를 대입한다. 결과적으로 리스트1은 리스트2의 별칭이 된다.|
|리스트1==리스트2|각 항목을 비교하여 리스트1과 리스트2의 내용이 같으면 True 를 반환한다.|
|리스트1!=리스트2|리스트1과 리스트2의 내용이 같으면 False 를 반환한다. 그렇지 않으면 True 를 반환한다.|
|항목 in 리스트|리스트 내에 '항목'이 존재하면 True 를 반환한다.|
|항목 not in 리스트|리스트 내에 '항목'이 존재하지 않으면 True 를 반환한다.|
|리스트1 < 리스트2|항목 간 미만 비교를 수행한다.|
|리스트1 <= 리스트2|항목 간 이하 비교를 수행한다.|
|리스트1 > 리스트2|항목 간 초과 비교를 수행한다.|
|리스트1 >= 리스트2|항목 간 이상 비교를 수행한다.|

아래는 위 표에 있는 리스트 연산자들의 예제입니다.

```python
# 준비
a = [1, 2, 3]
b = [4, 5]

# 1) 리스트1 + 리스트2  : 이어 붙여 새 리스트 생성
c = a + b
print(c)          # → [1, 2, 3, 4, 5]
print(c is a)     # → False (새 리스트)

# 2) 리스트1 * n 혹은 n * 리스트1 : n번 반복한 새 리스트
d = a * 3
print(d)          # → [1, 2, 3, 1, 2, 3, 1, 2, 3]

# 가변 객체를 *로 복제하면 내부가 공유됨
x = [[0]] * 3
x[0][0] = 99
print(x)          # → [[99], [99], [99]] (같은 내부 리스트를 참조)
# 안전하게 만들려면: x = [[0] for _ in range(3)]

# 3) 리스트1 = 리스트2 : 같은 객체(별칭)로 만듦
p = [10, 20]
q = p             # 대입 → 별칭
q[0] = 999
print(p)          # → [999, 20] (서로 영향)
print(p is q)     # → True

# 4) 리스트1 == 리스트2 : 내용(원소들)이 같으면 True
u = [1, 2, 3]
v = [1, 2, 3]
print(u == v)     # → True
print(u is v)     # → False (다른 객체)

# 5) 리스트1 is 리스트2 : 동일 객체(아이덴티티)일 때만 True
w = u
print(u is w)     # → True
print(u is v)     # → False (내용 같아도 객체가 다르면 False)

# 6) 항목 in 리스트
print(2 in a)     # → True
print(9 in a)     # → False

# 7) 항목 not in 리스트
print(9 not in a) # → True
print(2 not in a) # → False

# 8) 리스트1 < 리스트2 : 사전식(lexicographical) 비교
print([1, 2, 9] < [1, 3])      # → True   (처음 다른 원소: 2 < 3)
print([1, 2] < [1, 2, 0])      # → True   (앞부분 같으면 더 짧은 쪽이 작다)
print([3, 0] > [2, 100])       # → True   (첫 원소 3 > 2)

# 혼합 타입 비교는 TypeError 가능 (예: [1] < ['1'])
try:
    print([1] < ['1'])
except TypeError as e:
    print(type(e).__name__)    # → TypeError

```

# 4. 리스트 함수

파이썬의 기본 함수 중 리스트와 함께 사용할 수 있는 len, max, min, sorted, versed, sum 과 같은 유용한 함수가 많습니다.

|메서드|설명|
|-----|----|
|len(컬렉션)|컬렉션 길이 반환|
|max(컬렉션)|컬렉션의 항목 중 최대인 값 반환|
|min(컬렉션)|컬렉션의 항목 중 최소인 값 반환|
|reversed(컬렉션)|역순으로 정렬된 이터레이터 반환|
|sorted(컬렉션)|정렬된 리스트 생성|
|sum(컬렉션)|모든 항목을 더한 값 반환, 항목들은 무조건 숫자여야 함|

아래는 예제 코드입니다.

```python
# 샘플 데이터
nums   = [3, 10, -2, 7]
words  = ["pear", "apple", "banana"]
pairs  = [("kim", 82), ("lee", 95), ("park", 82)]

# 1) len(컬렉션): 길이
print(len(nums))          # → 4
print(len({"a", "b", "b"}))  # → 2 (집합은 중복 자동 제거)
print(len("hello"))       # → 5

# 2) max(컬렉션): 최대값 (key/default 지원)
print(max(nums))                  # → 10
print(max(words, key=len))        # → "banana"  (가장 긴 문자열)
print(max([], default=None))      # → None      (빈 컬렉션 처리)

# 3) min(컬렉션): 최소값 (key/default 지원)
print(min(nums))                  # → -2
print(min(words, key=len))        # → "pear"    (가장 짧은 문자열)
print(min([], default=0))         # → 0

# 4) reversed(컬렉션): 역순 이터레이터 반환 (필요시 list(...)로 소비)
rev_it = reversed(nums)
print(rev_it)                     # → <list_reverseiterator object ...>
print(list(rev_it))               # → [7, -2, 10, 3]

# 5) sorted(컬렉션): 정렬된 "새 리스트" 생성 (원본 불변, key/reverse 지원)
print(sorted(nums))                       # → [-2, 3, 7, 10]
print(sorted(nums, reverse=True))         # → [10, 7, 3, -2]
print(sorted(words, key=len))             # → ['pear', 'apple', 'banana'] (길이 기준)
# 정렬 안정성(Stable sort): 같은 key면 원래 순서 유지
print(sorted(pairs, key=lambda x: x[1]))  # → [('kim', 82), ('park', 82), ('lee', 95)]

# 6) sum(컬렉션): 숫자 합 (start 인자로 초기값 가능)
print(sum(nums))                 # → 18
print(sum(nums, 100))            # → 118  (초기값 100부터 더함)
# 주의: 비숫자 포함 시 TypeError
# sum(["a", "b"])  # ← TypeError

```

# 5. 리스트에서의 람다 함수 사용

리스트에서 연산 처리를 할 때 단순한 1회용 함수를 만들고 싶을 수 있습니다. 이럴 때 사용하는 것이 람다 함수입니다. 람다 함수는 변수에 대입하지 않는 이상 이름이 존재하지 않는 함수이며, 일반적으로 한 번만 사용하기 위해 만들어집니다. 

```
lambda 인수들: 반환값
```

아래는 리스트에서 람다를 사용한 예제입니다. 여러 예제들이 있지만 저는 정렬에서 많이 사용해서 정렬 예제만 가져와봤습니다.

```python
words = ["pear", "apple", "banana", "kiwi"]

# 길이 기준 오름차순
print(sorted(words, key=lambda s: len(s)))
# → ['kiwi', 'pear', 'apple', 'banana']

# 다중 키(길이 → 사전식)
print(sorted(words, key=lambda s: (len(s), s)))
# → ['kiwi', 'pear', 'apple', 'banana']

# 튜플 리스트: 두 번째 값 기준 내림차순
pairs = [("kim", 82), ("lee", 95), ("park", 82)]
pairs.sort(key=lambda x: x[1], reverse=True)
print(pairs)  # → [('lee', 95), ('kim', 82), ('park', 82)]

```

# 마치며

Python 리스트의 고급 기능들에 대해서 알아보았습니다. 추후에 다른 내용들도 알게 되면 추가하도록 하겠습니다. 긴 글 읽어주셔서 감사드리며, 잘못된 내용이 있거나 오타, 궁금하신 것이 있으시다면 댓글 달아주시기 바랍니다.