---
title: "[Algorithm] 소수 판별 방법"
categories:
  - Algorithm
tags:
  - Algorithm

use_math: true  
toc: true
toc_sticky: true
toc_label: "소수 판별 방법"
---

이번 포스트는 baekjoon 에서 알고리즘 문제를 풀다가 소수와 관련된 문제를 풀면서 시간 초과가 떠서 빠르게 소수를 찾는 방법이 무엇이 있는지 찾아보다가 차후 코테를 위한 알고리즘 공부를 위한 재학습을 위해 소수를 판별하는 방법들을 정리해 놓으면 좋겠다고 생각하여 본 포스트를 작성하게 되었습니다.

# 1. 가장 일반적인 소수 판별 방법

소수(prime number)란 1보다 큰 자연수 중 1과 자기 자신 두 개만을 약수로 갖는 수를 말합니다. 합성수(composite number)란 1보다 큰 자연수 중 소수가 아닌 수를 말하며 3개 이상의 약수를 갖습니다. "1"은 소수도 합성수도 아닙니다.

소수와 합성수 그리고 "1"에 대한 성질입니다. 이를 참고하여 우리는 일반적으로 다음과 같이 소수를 구하게 됩니다. 예를 들어 127이 소수인지 판별해 보도록 하겠습니다.

우선 1은 소수도 합성수도 아니라고 하였습니다. 그렇다면 시작은 2부터 시작합니다. 그리고 소수는 1과 자기 자신만을 약수로 갖는다고 하였기 때문에 2부터 126까지의 수로 127을 나누었을 때 나머지가 0이 되는 것이 있는지 세어 봅니다. 나머지가 0이 되는 수가 하나도 없다면 이 수는 소수입니다. 이를 Python 코드로 나타내면 다음과 같습니다.

```python
def is_prime(num):
    if num < 2:
        return False
    
    for i in range(2, num):
        if num % i == 0:
            return False
    return True

num = 127

if is_prime(num):
    print(f"{num} is prime number!")
else:
    print(f"{num} is not prime number!")
```

```
127 is prime number!
```

하지만 이 방법은 문제에서 처리해야할 입력값의 범위가 100,000,000을 넘어가게 되면 시간초과가 발생하게 됩니다. 그렇다면 좀 더 빨리 찾는 방법들은 무엇이 있는지 알아보도록 하겠습니다.

<br>

# 2. 제곱근을 이용한 소수 판별 방법

소수를 구하는데 제곱근을 이용할 수 있는 이유는 소수인지 판별해야 되는 변수 n에 대해서 a*b = n (a>1, b>1)이라 할 때 a와 b는 n의 약수이다. 라는 성질 때문입니다. 즉 a와 b둘 다 확인할 필요 없이 a와 b 중 작은 수만 확인해봐도 특정 숫자가 소수인지 아닌지 판별이 가능합니다. 이 때 작은 수의 범위는 n의 제곱근 이하가 됩니다. 그러므로 2부터 n-1까지가 아닌 2부터 n의 제곱근의 +1만큼까지만 확인을 해주면 됩니다. Python 코드는 다음과 같습니다.

```python
import math

def is_prime(num):

    if num < 2:
        return False

    for i in range(2, int(math.sqrt(num))+1):
        if num%i == 0:
            return False
    return True

num = 127

if is_prime(num):
    print(f"{num} is prime number!")
else:
    print(f"{num} is not prime number!")
```

```
127 is prime number!
```

그럼 제곱근을 이용했을 때 얼마나 시간이 감소하는지를 확인하기 위해 100,000 보다 작은 수 중에서 가장 큰 소수인 99991을 구하는 시간을 한 번 재보도록 하겠습니다. 1,000,000 을 기준으로 하고자 했지만 너무 오래 걸려 100,000으로 줄였습니다. 

```python
def is_prime1(num):
    if num < 2:
        return False

    for i in range(2, num):
        if num % i == 0:
            return False
    return True

import math
import time

def is_prime2(num):

    if num < 2:
        return False

    for i in range(2, int(math.sqrt(num))+1):
        if num%i == 0:
            return False
    return True

start_time = time.time()

max_value = -1

for i in range(100001):
    if is_prime1(i) and i > max_value:
        max_value = i

end_time = time.time()

print(f"가장 큰 소수는 {max_value} 입니다.")
print(f"is_prime1을 이용한 소요 시간은 {(end_time-start_time)} 입니다.")

start_time = time.time()

max_value = -1

for i in range(100001):
    if is_prime2(i) and i > max_value:
        max_value = i

end_time = time.time()

print(f"가장 큰 소수는 {max_value} 입니다.")
print(f"is_prime2을 이용한 소요 시간은 {(end_time-start_time)} 입니다.")
```

```
가장 큰 소수는 99991 입니다.
is_prime1을 이용한 소요 시간은 24.80061364173889 입니다.
가장 큰 소수는 99991 입니다.
is_prime2을 이용한 소요 시간은 0.17161989212036133 입니다.
```

결과를 보면 일반적으로 처음부터 모두 검사할 경우에는 24초가 걸리고 제곱근을 사용할 경우에는 0.17초가 걸리는 것을 확인할 수 있습니다.

# 3. Miller-Rabin 소수 판별 법

# 3.1 이론적 배경

## 3.1.1 페르마의 소정리

페르마의 소정리는 소수 판별의 기초가 되는 정리입니다.

> 만약 \( n \)가 소수이고, \( 1 < a < n \)인 정수 \( a \)에 대해 다음이 성립합니다:

$$
a^{n-1} \equiv 1 \pmod{n}
$$

하지만 이 조건만으로는 충분하지 않으며, 소수가 아닌 합성수도 위 조건을 만족하는 경우가 있습니다. 이러한 합성수를 `카마이클 수` 라고 하며, 대표적으로 `561` 이라는 수가 있습니다.

## 3.1.2 Miller-Rabin 소수판별법

Miller-Rabin 소수판별법은 큰 수에 대해 매우 빠르게 소수 여부를 판단할 수 있는 확률적 소수 판별법입니다. 페르마의 소정리를 기반으로 하며, 테스트 반복 횟수를 늘리면 오류 확률을 무시할 수 있을 정도로 낮출 수 있습니다.

Miller-Rabin 소수판별법은 소수의 특별한 성질을 이용합니다. n 이 홀수인 소수라고 할 때 n-1 은 $ 2^sd $ 라고 할 수 있으며 이 때, s 는 정수이고 d는 홀수가 되며 소수인 n 은 다음 식을 만족하게 됩니다.

$$
a^{d} \equiv 1 \pmod{n}
$$

하지만 a 는 1 < a < n 이기 때문에 아주 운이 나쁜 경우 a 를 잘못 고르게 되어 위 식을 만족하지 못할 수 있습니다. 그에 대한 예시로 n 은 97로 소수이고, a 가 5일 경우를 보도록 하겠습니다.

- a = 97 (소수)
- n-1 = 96 = $ 2^{5} \cdot 3 $ s = 32, d = 3
- a = 5

$$
\begin{array}{l}
a^{d} \equiv 1 \pmod{n}
5^3 = 125, 125 \mod 97 = 28 \\
\Rightarrow a^d \equiv 28 \mod 97 \ne 1
\end{array}
$$

위와 같이 소수인 경우라도 식을 만족하지 못할 때가 있습니다. 이를 위해 $a^{n-1}$ 의 제곱근들을 모두 검사하는 조건을 추가합니다.

위에서 언급한 s 를 $2^r$ 으로 나타내도록 하겠습니다. 그러면 수식은 다음과 같습니다.

$$

a^{2^r \cdot d} \equiv -1 \pmod{n} \ for \ some \ 0 \le r \le s-1

$$

소수인 경우 $a^{n-1}$ 제곱근들 중에서 하나는 무조건 $\equiv -1$ 을 만족하게 됩니다.

즉 정리를 하자면 특정 자연수 n 이 소수라면 1번과 2번식 중에서 하나는 만족하게 됩니다. 

### mod 연산에서 -1의 의미

mod 연산에서 -1 의 의미는 mod 연산을 당하는 수가 mod 연산을 하는 수보다 1 작을 때 표현하는 것으로 이렇게 표현하는 이유는 수론에서 수학적 대칭성을 강조하기 위해서입니다.   

# 4. 알고리즘으로써 Miller-Rabin 소수판별법

## 4.1 알고리즘 절차

### 4.1.1 전처리 : $n-1 = 2^r \cdot d$ 꼴로 분해

### 4.1.2 조건 검사

임의의 정수 $a \in [2, n-2]$를 선택하고 이론적 배경에서 알아본 두 가지 조건 중 하나라도 만족하면 소수일 가능성이 있습니다.

### 4.1.3 여러 번 반복

특정 합성수의 경우 고르는 a 값에 따라 위 두 가지 조건 중 하나를 만족할 수 있습니다. 이를 위해 여러번 반복하여 신뢰도를 높입니다.
Miller-Rabin 은 n 이 홀수인 합성수 인데 여러 수 중 강한 거짓증거(strong liar) 인 a가 선택되면 합성수임에도 불구하고 소수로 오인할 수 있으며, 이러한 오류율은 $ \le \frac{1}{4}$ 이며, 이는 이미 "Michael O. Rabin, “Probabilistic algorithm for testing primality,” Journal of Number Theory, 1980." 에 의해 증명이 되었습니다. 즉 오류가 존재할 수 있기 때문에 오류율을 극한으로 낮추기 위해 여러 번 반복을 진행합니다. 그래서 k 번 진행을 하게 된다면 오류율은 $\le \frac{1}{4^k}$ 가 됩니다.

## 4.2 예시

### 4.2.1 예시1: 합성수에서 조건 실패

#### 입력

- $ n = 561$ (합성수)
- $n-1 = 560 = 2^4 \cdot 35$
- $a = 2$

#### 계산

- $x_0 = 2^{35} \mod 561 = 263$
- $x_1 = 263^2 \mod 561 = 166$
- $x_2 = 166^2 \mod 561 = 67$
- $x_3 = 67^2 \mod 561 = 1$

어느 순간에도 $x_i \equiv -1 \mod 561$ 즉, 560이 나타자니 않으므로 합성수로 판별됨

### 4.2.2 예시 2: 두 번째 조건 만족

#### 입력

- $n=97$
- $n-1 = 96 = 2^5 \cdot 3$
- $a = 5$

#### 계산

- $x_0 = 5^3 \mod 97 = 28$
- $x_1 = 28^2 \mod 97 = 8$
- $x_2 = 8^2 \mod 97 = 64$
- $x_3 = 64^2 \mod 97 = 22$
- $x_4 = 22^2 \mod 97 = \mathbf{96} \equiv -1 \mod 97$

$x_4 = 22^2 \mod 97 = \mathbf{96} \equiv -1 \mod 97$ 로 조건에 부합

## 4.3 알고리즘으로써 시간복잡도

시간복잡도는 한 번 테스트할 때마다 $O(\log{n})$ 의 시간복잡도를 가지며, k 번 반복을 한다면 $O(k\log{n})$ 의 시간복잡도를 가집니다.

# 5. python 코드

```python
import random

def is_prime(n, k=5) :

    # n 이 1보다 작거나 같으면 소수가 아니므로 False return
    if n <= 1:
        return False
    if n <= 3: # n 이 3보다 작거나 같으면 2와 3 밖에 없으니 소수이므로 True return
        return True
    if n % 2 == 0: # 3보다 큰 n 이 2로 나누어 떨어지면 소수가 아니므로 False return
        return False


    r, d = 0, n-1

    # 조건 검사를 위한 r와 d 를 구함
    while d % 2 == 0:
        d //= 2
        r += 1

    # 높은 신뢰도를 위해 k 번 실행 k의 값이 커질 수록 오류율이 1/4^k 만큼 줄어듦
    for _ in range(k):
        a = random.randrange(2, n-1) # 2~n-1 값 사이에 있는 랜덤 숫자를 선택
        x= pow(a, d, n) # a^d 를 구하고 그 값을 mod n 을 진행

        # x 가 1 이거나 n-1 이면 소수라고 보고 continue
        if x == 1 or x == n-1:
            continue

        # 위 조건에서 소수 판별이 되지 않으면 두 번째 조건 판별을 진행
        for _ in range(r-1):
            x = pow(x, 2, n)
            if x == n-1:
                break

        else: # 위 두 조건 모두 만족하지 못하면 False return
            return False
    
    # k 번 반복하는 동안 return 하지 않으면 소수라고 판단하고 True return
    return True
```

# 마치며

소수판별법 중 하나인 Miller-Rabin 소수판별법에 대해서 알아보았습니다. 부족한 제가 알기론 소수는 컴퓨터 공학에서 암호학이 아니면 그렇게 많이 쓰이진 않는 것으로 알고 있습니다만 그래도 알고리즘 문제에서 간혹 소수와 관련된 문제들이 출제되다 보니 이번 기회에 한 번 정리를 해보았습니다.   
긴글 읽어주셔서 감사드리며 잘못된 내용이나 오타가 있을 경우 댓글 달아주시기 바랍니다.

# 참고문헌

[위키피디아-Miller-Rabin 소수판별법](https://ko.wikipedia.org/wiki/%EB%B0%80%EB%9F%AC-%EB%9D%BC%EB%B9%88_%EC%86%8C%EC%88%98%ED%8C%90%EB%B3%84%EB%B2%95)