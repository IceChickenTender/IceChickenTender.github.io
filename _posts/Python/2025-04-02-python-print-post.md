---
title: "[Python] 파이썬 출력 함수 print 사용법 정리"
categories:
  - Python
tags:
  - Python
  
toc: true
toc_sticky: true
toc_label: "파이썬 출력 함수 print 사용법 정리"
---

이번엔 Python 출력 함수인 print 에 대해서 알아보고자 합니다. 특히 단순히 출력하는 방법에 대해서는 다루지 않고 추후에 제가 다시 보면서 사용하기 위헤 잘 까먹거나 유용한 것들 위주로 정리를 해보았습니다. 또한 직접 타이핑을 하면서 배울 수 있게 코드 블록 복사 기능은 제거하였습니다.

# 구분자를 이용해 출력하기

## sep

print 함수에 여러 개의 값이 있을 경우 `sep` 구분자를 이용해 각 값 사이 사이에 넣어 줄 수 있습니다.

```python
print(1, 2, sep = " ")
print(1, 2, sep = ",")
print(1, 2, sep = "/")
```

```
콘솔 출력
1 2
1,2
1/2
```

## end

print 함수는 end 에 기본적으로 `\n` 이 들어가 있어, end 의 값을 바꾸지 않으면 다음줄로 넘어가게 됩니다. 이 때 여러줄에 걸쳐 출력되는 것들을 한 줄로 출력하고 싶다면 이 end 값을 바꿔 주면 됩니다.

```python
for i in range(5):
    print(i)

for i in range(5):
    print(i, end=" ")
```

```
콘솔 출력
0
1
2
3
4
0 1 2 3 4 
```

# 타입 포맷을 이용해 출력하기

C 언어에서 처럼 특정 타입 포맷을 이용해 출력할 수 있습니다. 문자열 사이에 특정 변수의 값을 출력하고자 할 때 주로 사용됩니다

```python
x = 10
print("x is %d" % x)

y = "code"
print("y is %s" % y)

print("x is %d and y is %s" % (x, y))
```

```
콘솔 출력
x is 10
y is code
x is 10 and y is code
```

## 타입 포맷 종류

1. %s : 문자열
2. %c : 문자
3. %d : 정수
4. %f : 실수

# format 함수 이용

문자열 사이에 특정 변수의 값을 출력하고자 할 때 주로 사용됩니다. 하지만 타입 변수를 사용할 때 처럼 타입을 명시해 주지 않아도 됩니다. 그리고 변수의 개수가 여러개일 경우에는 `{}` 안에 인덱스 숫자를 넣어주어야 합니다.

```python
x, y = 10, "code"

print("x is {0}" .format(x))
print("x is {new_x}" .format(new_x=x))

print("\n")

print("x is {0} and y is {1}" .format(x, y))
print("x is {new_x} and y is {new_y}" .format(new_x=x, new_y=y))
print("y is {1} and x is {0}" .format(x, y))
print("y is {new_y} and x is {new_x}" .format(new_x=x, new_y=y))
```

```
콘솔 출력
x is 10
x is 10


x is 10 and y is code
x is 10 and y is code
y is code and x is 10
y is code and x is 10
```

# f 문자열 포멧 이용

문자열 앞에 f를 붙이면, 중괄호와 변수 이름만으로 문자열에 원하는 변수를 삽입할 수 있습니다.

```python
x, y = 10, "code"

print(f"x is {x}")
print(f"y is {y}")
print(f"x is {x} and y is {y}")
```

```
콘솔 출력
x is 10
y is code
x is 10 and y is code
```

# 실수형 변수를 소수점에 맞춰 출력하기

## 타입 포맷(%f) 이용

%f 를 이욥하며 `%`와 `f` 사이에 `.`과 출력하고자 하는 소수점 자리수를 넣어줍니다.

```python
x = 3.141592653
print("%.4f" % x)
```

```
콘솔 출력
3.1416
```

## format 이용

`{}`에 `:.자릿수` 를 넣어줍니다.

```python
x = 3.141592653
print("{0:.4f}" .format(x))
```

```
콘솔 출력
3.1416
```

## f 문자열 포맷 이용

`{}` 안에 변수 옆에 `:.자릿수f` 를 넣어줍니다.

```python
x = 3.141592653
print(f"{x:.4f}")
```

```
콘솔 출력
3.1416
```