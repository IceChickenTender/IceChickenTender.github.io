---
title: "[Python] Python Dictionary 에 대해"
categories:
  - Python
tags:
  - Python

use_math: true  
toc: true
toc_sticky: true
toc_label: "Python Dictionary"
---

제가 암기력이 부족해서 Python 코딩을 하는 중에 다른걸 하다가 다시 Python 을 다루게 되면 Python 의 문법들을 자주 까먹습니다. 아마도 Python 의 문법이 쉬워서 중요하게 생각하지 않아서 그런 듯 합니다. 그래서 이번에 안까먹도록 하기 위해 공부도 하고, 까먹더라도 쉽게 제 블로그에서 바로 참조할 수 있도록 블로그에 정리를 해놓고자 해서 이렇게 포스팅하게 되었습니다.

# 1. Dictionary 란?

Python 의 Dictionary 는 `key-value` 쌍으로 데이터를 저장하는 자료형입니다. 리스트와 달리 순서(index) 가 아닌 고유한 키(key)를 사용하여 값을 빠르게 조회할 수 있습니다.

- key 는 변경 불가능(immutable) 타입만 사용 가능(예 : 문자열, 숫자, 튜플)
- 값은 모든 타입 사용 가능
- 중괄호 `{}` 또는 `dict()` 를 사용하여 생성

```python
# 빈 딕셔너리 생성
my_dict = {}

# key-value 형태로 값 추가
my_dict = {"name": "Alice", "age": 25, "job": "Engineer"}
```

---

# 2. Dictionary 특징

## 1. Dictinoary 의 본질 : 해시 테이블

Python 의 Dictionary 는 해시 테이블(Hash Table) 기반으로 구현되어 있습니다.
즉, 키(key)를 해시 함수로 변환한 값을 인덱스로 사용하여 빠른 접근 속도를 보장합니다.

- Dictionary 의 기능별 시간 복잡도
	- 검색 : $O(1)$
	- 삽입 : $O(1)$
	- 삭제 : $O(1)$
- 최악의 경우 : $O(n)$ (해시 충돌 발생 시)

## 2. 해시 충돌과 오픈 어드레싱

Dictionary 에서 키 충돌이 발생하면 Python 은 오픈 어드레싱(Open Addressing) 방식을 사용합니다. 충돌이 발생하면 다른 슬롯을 찾아 값을 저장합니다.

- 해시 충돌 관리 방식은 Probing 기법을 사용합니다.
- 키 비교 시 `__hash__()` 와 `__eq__()` 모두 활용합니다.

## 3. Dictionary 의 메모리 특성

Dictionary 는 성능 최적화를 위해 여유 공간(over-allocation)을 가집니다. 즉 요소를 추가할 때마다 크기를 딱 맞게 조정하지 않고, 일정 비율로 확장합니다.

- Load Factor 가 특정 임계치를 넘으면 리사이징 발생
- 따라서 잦은 삽입/삭제 작업에서 메모리와 성능 트레이드 오프가 존재

## 4. Dictionary 의 불변 키 조건

Dictionary 의 키는 반드시 해시로써 기능해야 하므로, 불변(immutable) 해야 합니다. 따라서 허용되는 키 타입이 정해져 있습니다.

- 허용되는 키 타입
	- 문자열, 숫자, 튜플(내부 요소가 불변일 경우)
- 허용되지 않는 키 타입
	- 리스트, 집합 등 가변 객체

```python
d = {}
d[[1,2,3]] = "mutable"  # TypeError 발생

```

```
실행 결과

TypeError                                 Traceback (most recent call last)
/tmp/ipython-input-784898883.py in <cell line: 0>()
      1 d = {}
----> 2 d[[1,2,3]] = "mutable"  # TypeError 발생

TypeError: unhashable type: 'list'

```


---

# 3. Dictionary 기본 활용

## 1. Dictionary 생성 방법

```python


# 1. 중괄호 사용
person1 = {"name": "Bob", "age":30}
print(person1)

# 2. dict() 생성자 사용
person2 = dict(name="Bob", age=30)
print(person2)

# 3. 리스트/튜플 쌍으로 변환
person3 = dict([("name", "Bob"), ("age", 30)])
print(person3)

# 4. 딕셔너리 컴프리헨션
squares = {x: x**2 for x in range(5)}
print(squares)

```

```
실행 결과

{'name': 'Bob', 'age': 30}
{'name': 'Bob', 'age': 30}
{'name': 'Bob', 'age': 30}
{0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

---

## 2. 값 접근하기

존재하지 않는 키를 `[]` 로 접근하면 `KeyError` 발생, `get()`은 안전하게 사용 가능.

```python 
person = {"name": "Charlie", "age": 28}

# 1. 직접 접근
print(person["name"])  # Charlie

# 2. get() 메서드 (존재하지 않으면 None 반환)
print(person.get("job"))        # None
print(person.get("job", "N/A")) # N/A

```

---

## 3. 값 수정 및 추가

```python

person = {"name": "Dana", "age": 22}

# 값 수정
person["age"] = 23

# 새로운 key-value 추가
person["job"] = "Designer"

print(person) # {'name': 'Dana', 'age': 23, 'job': 'Designer'}

```

---

## 4. 값 삭제하기

```python

person = {"name": "Eve", "age": 35, "job": "Doctor"}

# 1. del 키워드
del person["job"] # {'name': 'Eve', 'age': 35}

# 2. pop() 메서드 (값 반환 후 삭제)
age = person.pop("age") # {'name': 'Eve'}

# 3. popitem() (마지막 요소 삭제)
last_item = person.popitem() # {}

# 4. clear() (모든 요소 삭제)
person = {"name": "Eve", "age": 35, "job": "Doctor"} # {'name': 'Eve', 'age': 35, 'job': 'Doctor'}
person.clear() # {}
```

## 5. Dictionary 반복문 활용

```python
person = {"name": "Grace", "age": 26, "job": "Analyst"}

# key 반복
for key in person.keys():
	print(key)


# value 반복
for value in person.values():
	print(value)

# key-value 반복
for key, value in person.items():
	print(f"{key} => {value}")

```

## 6. Dictionary 컴프리헨션 활용

```python
# 1. 제곱수 딕셔너리
squares = {x: x**2 for x in range(1, 6)}
# # {1:1, 2:4, 3:9, 4:16, 5:25}

#2. 조건 추가
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
# {0:0, 2:4, 4:16, 6:36, 8:64}
```

# 4. Dictionary 고급 활용 

## 1. defaultdict

값이 존재하지 않는 경우 기본값을 자동으로 생성하는 dict 변형입니다.

```python
from collections import defaultdict

counts = defaultdict(int)
words = ["apple", "banana", "apple"]

for w in words:
	counts[w] += 1

print(counts)  # {'apple': 2, 'banana': 1}
```

## 2. OrderedDict

Dictionary 의 요소들에 순서가 필요할 경우 사용합니다. Python 3.7 이후 기본 dict 도 순서를 유지하지만, `OrderedDict` 는 순서 기반 비교와 같은 추가 기능을 제공합니다.

```python
from collections import OrderedDict

od1 = OrderedDict(a=1, b=2)
od2 = OrderedDict(b=2, a=1)

print(od1 == od2)  # False (순서 비교)

```
## 3. Dictionary 정렬

기본적으로 Dictionary 는 자동 정렬이 불가능합니다. 따라서 키, 값, (키, 값) 쌍을 꺼내어 sorted로 정렬이 가능하며, 반한되는 결과는 리스트 또는 정렬된 `OrderedDict` 입니다.

### 3.1 키(key) 기준 정렬

```python
d = {"banana": 3, "apple": 5, "pear": 2}

# 키 기준 오름차순
sorted_by_key = dict(sorted(d.items()))
print(sorted_by_key)
# {'apple': 5, 'banana': 3, 'pear': 2}

# 키 기준 내림차순
sorted_by_key_desc = dict(sorted(d.items(), reverse=True))
print(sorted_by_key_desc)
# {'pear': 2, 'banana': 3, 'apple': 5}

```

### 3.2 값(value) 기준 정렬

```python
d = {"banana": 3, "apple": 5, "pear": 2}

# 값 기준 오름차순
sorted_by_value = dict(sorted(d.items(), key=lambda x: x[1]))
print(sorted_by_value)
# {'pear': 2, 'banana': 3, 'apple': 5}

# 값 기준 내림차순
sorted_by_value_desc = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
print(sorted_by_value_desc)
# {'apple': 5, 'banana': 3, 'pear': 2}

```

### 3.3 복잡한 조건으로 정렬

예시로, 값 우선 정렬 -> 값이 같으면 키 기준으로 정렬할 수도 있습니다.

```python
d = {"banana": 3, "apple": 5, "pear": 3, "orange": 2}

# 값 기준 정렬, 값 같으면 키 알파벳순
sorted_complex = dict(sorted(d.items(), key=lambda x: (x[1], x[0])))
print(sorted_complex)
# {'orange': 2, 'banana': 3, 'pear': 3, 'apple': 5}

```

### 3.4 OrderedDict 사용

Python 3.7+ 에서는 일반 dict 도 입력 순서를 유지하지만, 정렬된 상태 유지를 원한다면 `OrderedDict` 를 명시적으로 사용하는 방법도 있습니다.

```python
from collections import OrderedDict

d = {"banana": 3, "apple": 5, "pear": 2}
ordered = OrderedDict(sorted(d.items(), key=lambda x: x[1]))
print(ordered)
# OrderedDict([('pear', 2), ('banana', 3), ('apple', 5)])

```

### 3.5 Dictionary 정렬 시 성능 고려

- `sorted()` 는 $O(n \log(n))$ 복잡도를 가집니다.
- 정렬된 결과는 새로운 리스트/딕셔너리이므로 원본에 영향을 주지 않습니다.
- 대규모 데이터에서 반복 정렬이 필요하다면
	- `heapq` (우선순위 큐)
	- `bisect` (이진 탐색 삽입)
	- `pandas.DataFrame` 같은 전용 라이브러리 활용이 효율적입니다.

# 5. Dictionary 메소드 : 실무에서 꼭 알아야 할 것들

## 0. 한눈에 보는 요약 표

| 메소드                                     | 핵심 목적     | 반환값            | 주요 부작용                        | 평균 복잡도 |
| --------------------------------------- | --------- | -------------- | ----------------------------- | ------ |
| `get(key, default=None)`                | 안전 조회     | 값 또는 `default` | 없음                            | O(1)   |
| `setdefault(key, default=None)`         | 존재 확인+초기화 | 최종 값           | 키 없으면 삽입                      | O(1)   |
| `update(mapping_or_iterable, **kwargs)` | 병합/갱신     | `None`         | 대상 dict 변경                    | O(m)   |
| `pop(key, default=__noarg__)`           | 꺼내기+삭제    | 값              | 키 없고 `default` 없으면 `KeyError` | O(1)   |
| `popitem()`                             | 마지막 쌍 꺼내기 | `(key, value)`, 튜플 | LIFO 삭제                       | O(1)   |
| `clear()`                               | 전부 삭제     | `None`         | 비움                            | O(n)   |
| `keys()`                                | 키 뷰       | `dict_keys`    | 뷰는 동적 반영                      | O(1)   |
| `values()`                              | 값 뷰       | `dict_values`  | 뷰는 동적 반영                      | O(1)   |
| `items()`                               | (키,값) 뷰   | `dict_items`   | 뷰는 동적 반영                      | O(1)   |
| `copy()`                                | 얕은 복사     | 새 dict         | 중첩 객체는 공유                     | O(n)   |
| `fromkeys(iterable, value=None)`        | 동일 값 초기화  | 새 dict         | 클래스메소드                        | O(n)   |

## 1. get(key, default=None)

사용 목적 : 키가 없어도 예외 없이 값을 읽고 싶을 때 사용

```python
user = {"name": "Alice", "age": 30}
user.get("name")            # 'Alice'
user.get("job", "N/A")      # 'N/A'

```

- 반환 : 값이 존재하면 값을, 없으면 `default` 파라매터에 지정한 값을(기본은 `None`을 반환)
- 복잡도 : 평균 $O(1)$
- 메소드 사용시 주의할 점
	- 값이 실제로 `None`인 경우와 키가 없는 경우를 구분해야 한다면 `in`으로 먼저 확인해야 함
	- `default` 는 평가가 즉시 이루어집니다. 계산 비용이 큰 기본값은 `setdefault` 또는 지연 계산 패턴 사용

## 2. setdefault(key, default=None)

사용 목적 : 카운팅/그룹핑에서 초기 리스트/셋 등을 만들 때 유용함

```python
groups = {}

for name, score in [("A", 10), ("B", 12), ("A", 8)]:
    groups.setdefault(name, []).append(score)
# {'A': [10, 8], 'B': [12]}

```

- 반환 : 최종적으로 해당 키에 저장된 값
- 부작용 : 키가 없으면 `default` 에 지정된 값을 삽입
- 복잡도 : 평균 $O(1)$
- 메소드 사용 시 주의할 점
	- `default` 는 키가 없고 삽입할 때만 평가되므로, 비싼 초기값 생성에 적합함
	- `default` 로 가변 객체(예: `[]`, `{}`) 를 직접 리터럴로 넣을 때 공유 버그 주의 보통 위 예시처럼 매 호출마다 새로운 `[]` 가 만들어지므로 안전하지만, 함수 바깥에 한 번 만든 리스트를 재사용하면 모든 키가 같은 객체를 참조할 수 있음
	- 빈도수 카운트에는 `collections.Counter` 가 더 단순하고 빠른 경우가 많음

## 3. update(mapping_or_iterable, **kwargs)

사용 목적 : 다른 매핑(또는 `(k v)` iterable)으로 키를 덮어쓰며 병합하기 위해

```python
cfg = {"host": "localhost", "port": 8000}
cfg.update({"port": 9000, "debug": True})
# {'host': 'localhost', 'port': 9000, 'debug': True}

```

- 반환 : 아무것도 반환하지 않음 (`None` 을 반환)
- 복잡도 : $O(m)$ (병합할 항목의 수 m)
- 우선순위 : 같은 키면 파라매터에 있는 값으로 덮어씀

## 4. pop(key, default=__noarg__)

사용 목적 : 일반적으로 값을 반환받으면서 딕셔너리에 있는 데이터를 제거하기 위해 주로 사용

```python
env = {"MODE": "prod", "REGION": "ap-northeast-2"}
mode = env.pop("MODE")             # 'prod', 키 제거됨
fallback = env.pop("DEBUG", False) # 키 없으면 False 반환

```

- 반환 : 삭제한 값 (또는 `default` 로 지정한 값)
- 예외 : 키가 없고, `default` 값 미제공 시 `KeyError` 발생
- 복잡도 : 평균 $O(1)$

## 5. popitem()

사용 목적 : 마지막에 삽입된 `(key, value)` 을 꺼내면서 해당 데이터를 삭제함, 스택처럼 사용 가능

```python
d = {"a": 1, "b": 2, "c": 3}
k, v = d.popitem()  # ('c', 3)

```

- 반환 : `(key, value)` 튜플
- 예외 : 딕셔너리가 비어있다면 `KeyError` 발생
- 복잡도 : $O(1)$

## 6. clear()

사용 목적 : 딕셔너리에 있는 데이터를 모두 비울 때 사용

```python
cache.clear()

```

- 반환 : 아무것도 반환하지 않음
- 복잡도 : $O(n)$
- 부작용 : 공유 참조된 뷰(keys(), itmes()) 도 즉시 비워진 상태를 반영하므로 비운 딕셔너리의 뷰를 사용할 경우 주의해야 함

## 7. keys(), values(), items()

사용 목적 : 동적 뷰 객체(`dict_keys`, `dict_values`, `dict_items`)를 제공

- 반환 : `dict_keys`, `dict_values`, `dict_items` (이터러블한 뷰) 반환
- 복잡도 : 뷰 생성에는 $O(1)$, 순회는 $O(n)$

## 8. copy() (얕은 복사)

사용 목적 : 동일한 1레벨 구조만 복제

```python
orig = {"a": [1, 2], "b": {"x": 1}}
shallow = orig.copy()
shallow["a"].append(3)
# orig['a']도 [1,2,3]로 변함 (내부 리스트는 공유)

```

- 반환 : 새로운 dict 반환
- 복잡도 : $O(n)$
- 주의점 : 중첩 구조가 있다면 `copy.deepcopy` 필요

## 9. fromkeys(iterable, value=None) (클래스 메소드)

사용 목적 : 동일한 기본값으로 빠르게 초기화할 때 사용

```python
fields = ("id", "name", "email")
row = dict.fromkeys(fields, None)
# {'id': None, 'name': None, 'email': None}

```

- 반환 : 새로운 dict 반환
- 복잡도 : $O(n)$
- 주의점 : `value` 가 가변 객체일 경우 모든 키가 같은 객체를 참조하므로 주의

```python
bad = dict.fromkeys(["a", "b"], [])
bad["a"].append(1)
bad["b"]  # [1] 이 출력됨  ← 같은 리스트 공유

```

# 6. 마치며

Python Dictionary 에 대해서 제가 추후에 참조하기 위해서 한 번 정리해 보았습니다. 혹시나 다른 분들에게도 도움이 되었으면 좋겠습니다. 긴 글 읽어주셔서 감사드리며, 내용 중에 잘못된 내용이나 오타가 있거나 궁금하신 것이 있으시다면 댓글 달아주시기 바랍니다.