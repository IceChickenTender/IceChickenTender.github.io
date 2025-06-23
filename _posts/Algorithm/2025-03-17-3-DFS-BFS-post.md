---
title: "[Algorithm] 탐색 알고리즘 DFS/BFS"
categories:
  - Algorithm
  - Python
tags:
  - Algorithm
  - Python

toc: true
toc_sticky: true
toc_label: "탐색 알고리즘 DFS/BFS"
---

이번엔 그래프가 주어졌을 때의 탐색 알고리즘인 DFS와 BFS 에 대해서 알아보도록 하겠습니다. 저도 여러번 코딩 테스트를 진행해 보았는데 저는 알고리즘이 매우 취약해서 이런 그래프 문제가 나올 때마다 매번 문제 풀이를 모두 다 하지 못했었습니다. 이제는 이런 그래프 문제가 나오더라도 풀 수 있게 이번에 꼼꼼히 한 번 정리해 보도록 하겠습니다.

# 그래프 기본 개념

그래프는 **노드(Node)** 와 **간선(Edge)** 로 표현되며 이 때 노드를 **정점** 이라고도 말합니다. 그래프 탐색이란 하나의 노드를 시작으로 다수의 노드를 방문하는 것을 말합니다. 또한 두 노드가 간선으로 연결되어 있다면 두 노드는 인접하다라고 표현합니다.

<div align="center">
<img src="/assets/images/algorithm/3/graph.png" width="50%" hegiht="40%">
</div>

<br>

프로그래밍에서 그래프는 크게 2가지 방식으로 표현할 수 있습니다. 코딩 테스트에서는 이 두 방식 모두 필요하니 두 개념에 대해 알고 있어야 합니다.

## 인접 행렬

인접 행렬 방식은 2차원 배열로 그래프의 연결관계를 표현 하는 방식입니다. 연결이 되어 있지 않은 노드끼리는 무한의 비용으로 작성합니다. 실제 코드에서는 논리적으로 정답이 될 수 없는 큰 값 중에서 999999999, 987654321 등의 값으로 초기화하는 경우가 많습니다. 아래 예시를 이용해 코드로 인접행렬을 코드로 구현해 보겠습니다.

<div align="center">
<img src="/assets/images/algorithm/3/graph_ex_1.png" width="50%" hegiht="40%">
</div>

<br>

```python
INF = 99999999 # 무한 비용 선언

# 2 차원 리스트를 이용해 인접 행렬 표현

graph = [
  [0, 7, 5],
  [7, 0, INF]
  [5, INF, 0]

]
```

<br>

## 인접 리스트

인접 리스트 방식에서는 다음 그림처럼 모든 노드에 연결된 노드에 대한 정보를 차례대로 연결하여 저장합니다.

<div align="center">
<img src="/assets/images/algorithm/3/adjacency_list.png" width="50%" hegiht="40%">
</div>

<br>

인접 리스트는 연결 리스트라는 자료구조를 이용해 구현하는데, C++ 나 자바와 같은 프로그래밍 언어에서는 별도록 연결 리스트 기능을 위한 표준 라이브러리를 제공합니다. 반면에 파이썬은 기본 자료형인 리스트 자료형이 append()와 메소드를 제공하므로, 전통적인 프로그래밍 언어에서의 배열과 연결 리스트의 기능을 모두 기본으로 제공합니다. 파이썬으로 인접 리스트를 이용해 그래프를 표현하고자 할 때에도 단순히 2차원 리스트를 이용하면 된다는 점만 기억하시면 됩니다. 다음 예제는 그래프를 인접 리스트 방식으로 처리할 때 데이터를 초기화한 코드입니다. 

```python
# 행이 3개인 2차원 리스트로 인접 리스트 표현
graph = [[] for _ in range(3)]

# 노드 0에 연결된 노드 정보 저장(노드, 거리)
graph[0].append((1, 7))
graph[0].append((2, 5))

# 노드 1에 연결된 노드 정보 저장(노드, 거리)
graph[1].append((0, 7))

# 노드 2에 연결된 노드 정보 저장(노드, 거리)
graph[2].append((0, 5))

print(graph)
```

```
콘솔 출력
[[(1, 7), (2, 5)], [(0, 7)], [(0, 5)]]
```

인접 행렬과 인접 리스트에는 어떤 차이가 있을까요? 코딩 테스트를 위해 공부하는 것이므로 메모리와 속도 측면에서 살펴보도록 하겠습니다. 메모리 측면에서 보자면 인접 행렬 방식은 모든 관계를 저장하므로 노드 개수가 많을수록 메모리가 불필요하게 낭비됩니다. 반면에 인접 리스트 방식은 연결된 정보만을 저장하기 때문에 메모리를 효율적으로 사용합니다. 하지만 인접 리스트 방식은 인접 행렬 방식에 비해 특정한 두 노드가 연결되어 있는지에 대한 정보를 얻는 속도가 느립니다. 인접 리스트 방식에서는 연결된 데이터를 하나씩 확인해야 하기 때문입니다.   
예를 들어 노드 0과 노드 1이 연결되어 있는지 확인해보도록 하겠습니다. 인접 행렬 방식에서는 graph[0][1]만 확인하면 됩니다. 반면 인접 리스트 방식에서는 노드 0에 대한 인접 리스트를 앞에서부터 차례대로 확인해야 합니다. 그러므로 특정한 노드의 연결된 모든 인접 노드를 순회해야 하는 경우, 인접 리스트 바익이 인접 행렬 방식에 비해 메모리 공간의 낭비가 적습니다.

# DFS

DFS는 Depth-Frist Search 의 약자로 깊이 우선 탐색이라고도 부르며, 그래프에서 깊은 부분을 우선적으로 탐색하는 알고리즘입니다. DFS 탐색 알고리즘은 특정한 경로로 탐색하다가 특정한 상황에서 최대한 깊숙이 들어가서 노드를 방문한 후, 다시 돌아가 다른 경로로 탐색하는 알고리즘입니다. DFS는 스택을 이용하며 구체적인 동작과정은 다음과 같습니다.

1. 탐색 시작 노드를 스택에 삽입하고 방문처리를 한다.
2. 스택의 최상단 노드에 방문하지 않은 인접 노드가 있으면 그 인접 노드를 스택에 넣고 방문 처리를 한다. 방문하지 않은 인접 노드가 없으면 스택에서 최상단 노드를 꺼낸다.
3. 2 번의 과정을 더 이상 수행할 수 없을 때까지 반복한다.

직관적인 이해를 위해 그림을 이용해 설명을 진행하도록 하겠습니다. 그리고 일반적으로 인접한 노드 중에서 방문하지 않은 노드가 여러 개 있으면 번호가 낮은 순서부터 처리합니다.

<div align="center">
<img src="/assets/images/algorithm/3/graph_ex_2.png" width="50%" hegiht="40%">
</div>

<br>

방문 처리된 노드는 회색으로, 현재 처리하는 스택의 최상단 노드는 하늘색으로 표현했습니다.

step 1. 시작 노드인 `1` 을 스택에 삽입하고 방문 처리를 한다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs1.png" width="50%" hegiht="40%">
</div>

<br>

step 2. 스택의 최상단 노드인 `1`에 방문하지 않은 인접 노드 `2`, `3`, `8`이 있다. 이 중에서 가장 작은 노드인 `2`를 스택에 넣고 방문 처리를 한다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs2.png" width="50%" hegiht="40%">
</div>

<br>

step 3. 스택의 최상단 노드인 `2`에 방문하지 않은 인접 노드 `7`이 있다. 따라서 `7`번 노드를 스택에 넣고 방문처리를 한다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs3.png" width="50%" hegiht="40%">
</div>

<br>

step 4. 스택의 최상단 노드인 `7`에 방문하지 않은 인접 노드 `6`과 `8`이 있다. 이 중에서 가장 작은 노드인 `6`을 스택에 넣고 방문 처리를 한다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs4.png" width="50%" hegiht="40%">
</div>

<br>

step 5. 스택의 최상단 노드인 `6`에 방문하지 않은 인접 노드가 없다. 따라서 스택에서 `6`번 노드를 꺼낸다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs5.png" width="50%" hegiht="40%">
</div>

<br>

step 6. 스택의 최상단 노드인 `7`에 방문하지 않은 인접 노드 `8`이 있다. 따라서 `8`번 노드를 스택에 넣고 방문 처리를 한다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs6.png" width="50%" hegiht="40%">
</div>

<br>

step 7. 스택의 최상단 노드인 `8`에 방문하지 않은 인접 노드가 없다. 따라서 스택에서 8번 노드를 꺼낸다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs7.png" width="50%" hegiht="40%">
</div>

<br>

step 8. 스택의 최상단 노드인 `7`에 방문하지 않은 인접 노드가 없다. 따라서 스택에서 `7`번 노드를 꺼낸다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs8.png" width="50%" hegiht="40%">
</div>

<br>

step 9. 스택의 최상단 노드인 `2`에 방문하지 않은 인접 노드가 없다. 따라서 스택에서 `2`번 노드를 꺼낸다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs9.png" width="50%" hegiht="40%">
</div>

<br>

step 10. 스택의 최상단 노드인 `1`에 방문하지 않은 인접 노드 `3`을 스택에 넣고 방문 처리한다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs10.png" width="50%" hegiht="40%">
</div>

<br>

step 11. 스택의 최상단 노드인 `3`에 방문하지 않은 인접 노드 `4`와 `5`가 있다. 이 중에서 가장 작은 노드인 `4`를 스택에 넣고 방문 처리를 한다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs11.png" width="50%" hegiht="40%">
</div>

<br>

step 12. 스택의 최상단 노드인 `4`에 방문하지 않은 인접 노드 `5`가 있다. 따라서 `5`번 노드를 스태에 넣고 방문 처리를 한다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs12.png" width="50%" hegiht="40%">
</div>

<br>

step 13. 남아 있는 노드에 방문하지 않은 인접 노드가 없다. 따라서 모든 노드를 차례대로 꺼내면 다음과 같다.

<div align="center">
<img src="/assets/images/algorithm/3/dfs13.png" width="50%" hegiht="40%">
</div>

<br>

결과적으로 노드의 탐색 순서(스택에 들어간 순서)는 다음과 같습니다.   
1->2->7->6->8->3->4->5   
깊이 우선 탐색 알고리즘인 DFS는 스택의 자료구조에 기초한다는 점에서 구현이 간단합니다. 실제로는 스택을 쓰지 않아도 되며 탐색을 수행함에 있어서 데이터의 개수가 N 개인 경우 O(N)의 시간이 소요된다는 특징이 있습니다. 스택을 이용한 DFS 예제 코드는 다음과 같습니다.

```python
def dfs_stack(graph, start, visited):

    stack = [] # 스택으로 사용할 리스트
    print(start, end=" ") # 첫 시작은 출력
    stack.append(start) # 스택에 삽입
    visited[start] = True # visited 리스트에서 해당 인덱스의 값을 True 로 변경

    while stack: # stack 이 완전히 빌 때까지 반복

        current = stack[-1] # 현재 방문 중인 노드
        visited_count = 0 # 현재 방문 중인 노드의 인접 노드들이 모두 방문 되었는지 체크하기 위한 변수

        for i in graph[current]: # 현재 위치의 노드의 인접 노드들을 체크
            if not visited[i]: # 만약 방문하지 않은 노드라면 출력하고, 스택에 삽입하고, 방문 체크 그리고 루프 탈출
                print(i, end =" ")
                stack.append(i)
                visited[i] = True
                break
            else: # 방문을 했다면 visited_count 하나 증가
                visited_count += 1

        if visited_count == len(graph[current]): # visited_count 와 현재 방문 중인 노드의 인접 노드들 중 방문한 노드들의 개수와 같다면 stack 에서 pop 을 진행
            stack.pop()

graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7],
]

visited = [False] * 9
dfs_stack(graph, 1, visited)
```

다만 스택을 이용한 DFS 알고리즘은 인접 노드들의 숫자가 정렬 되지 않을 경우 출력이 달라질 수 있습니다. 그리고 DFS는 스택을 이용하는 알고리즘이기 때문에 실제 구현은 재귀 함수를 이용했을 때 매우 간결하게 구현할 수 있으며 인접 노드들의 정렬에 상관없이 출력이 일정합니다. 재귀 함수를 이용한 예제 소스코드는 다음과 같습니다.

```python
"""
DFS 예제

1->2->7->6->8->3->4->5 로 출력되는 그래프를 예제로 함
본 코드에서는 각 노드를 배열의 인덱스에 맞추기 위해 1을 빼준 것이 아닌 노드 번호 그대로를 사용
그렇기 때문에 노드가 8개 임에도 불구하고 사용된 배열들의 크기가 9로 사용
"""

# DFS 메서드 정의
def dfs(graph, v, visited):
    # 현재 노드를 방문 처리하고 출력
    visited[v] = True
    print(v, end = ' ')

    #현재 노드와 연결된 다른 노드를 재귀적으로 방문
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)

# 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)
graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7],
]

# 각 노드가 방문된 정보를 리스트 자료형으로 표현 (1차원 리스트)
visited = [False] * 9

# 정의된 DFS 함수 호출
dfs(graph, 1, visited)
```

```
콘솔 출력
1 2 7 6 8 3 4 5
```

위 예제에서는 인접 리스트를 이용했습니다. 앞으로 DFS 와 관련된 문제는 위 예제코드를 기반으로 문제 풀이를 할 것이기 때문에 왠만하면 암기를 해두는 것이 좋습니다.

<br>

# BFS 

BFS 알고리즘은 너비 우선 탐색이라는 의미를 가집니다. 쉽게 말해 가까운 근처에 있는 노드부터 탐색하는 알고리즘입니다. DFS 는 최대한 멀리 있는 노드를 우선으로 탐색하는 방식으로 동작한다고 했는데 BFS 는 그 반대로 동작합니다. BFS 구현에는 선입선출 방식인 큐 자료구조를 이용합니다. 인접한 노드를 반복적으로 큐에 넣도록 알고리즘을 작성하면 자연스럽게 먼저 들어온 것이 먼저 나가게 되어 가까운 노드부터 탐색을 진행하게 됩니다. BFS 의 동작 방식은 다음과 같습니다.

1. 탐색 시작 노드를 큐에 삽입하고 방문 처리를 한다
2. 큐에서 노드를 꺼내 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문 처리를 한다.
3. 2 번의 과정을 더 이상 수행할 수 없을 때까지 반복한다.

DFS 알고리즘을 설명할 때 사용했던 그래프를 이용해 BFS 도 DFS 와 같이 그림 예제를 통해 설명을 하도록 하겠습니다.

step 1 시작 노드인 `1`을 큐에 삽입하고 방문 처리를 한다. 방문 처리된 노드는 회색으로, 큐에서 꺼내 현재 처리하는 노드는 하늘색으로 표현했다.

<div align="center">
<img src="/assets/images/algorithm/3/bfs1.png" width="50%" hegiht="40%">
</div>

<br>

step 2 큐에서 노드 `1`을 꺼내고 방문하지 않은 인접 노드 `2`, `3`, `8`을 모두 큐에 삽입하고 방문 처리를 한다.

<div align="center">
<img src="/assets/images/algorithm/3/bfs2.png" width="50%" hegiht="40%">
</div>

<br>

step 3 큐에서 노드 `2`를 꺼내고 방문하지 않은 인접 노드 `7`을 큐에 삽입하고 방문 처리를 합니다.

<div align="center">
<img src="/assets/images/algorithm/3/bfs3.png" width="50%" hegiht="40%">
</div>

<br>

step 4 큐에서 노드 `3`을 꺼내고 방문하지 않은 인접 노드 `4`, `5`를 모두 큐에 삽입하고 방문 처리를 합니다.

<div align="center">
<img src="/assets/images/algorithm/3/bfs4.png" width="50%" hegiht="40%">
</div>

<br>

step 5 큐에서 노드 `8`을 꺼내고 방문하지 않은 인접 노드가 없으므로 무시합니다.

<div align="center">
<img src="/assets/images/algorithm/3/bfs5.png" width="50%" hegiht="40%">
</div>

<br>

step 6 큐에서 노드 `7`을 꺼내고 방문하지 않은 인접 노드 `6`을 큐에 삽입하고 방문 처리를 합니다.

<div align="center">
<img src="/assets/images/algorithm/3/bfs6.png" width="50%" hegiht="40%">
</div>

<br>

step 7 남아 있는 노드에 방문하지 않은 인접 노드가 없습니다. 따라서 모든 노드를 차례대로 꺼내면 최종적으로 다음과 같습니다.

<div align="center">
<img src="/assets/images/algorithm/3/bfs7.png" width="50%" hegiht="40%">
</div>

<br>

결과적으로 노드의 탐색 순서는 다음과 같습니다.   
1->2->3->8->7->4->5->6   
BFS 는 큐 자료구조에 기초한다는 점에서 구현이 간단합니다. 실제로 구현함에 있어 앞서 언급한 대로 dqeue 라이브러리를 사용하는 것이 좋으며 탐색을 수행함에 있어 O(N)의 시간이 소요됩니다. 일반적인 경우 실제 수행 시간은 DFS 보다 좋은 편이라는 점까지만 기억해 주시면 될 것 같습니다.

```python
"""
BFS 예제 코드
"""

from collections import deque

# BFS 메서드 정의

def bfs(graph, start, visitied):
    # 큐 구현을 위해 deque 라이브러리 사용
    queue = deque([start])

    # 현재 노드를 방문 처리
    visited[start] = True

    #큐가 빌 때까지 반복
    while queue:
        #큐에서 하나의 원소를 뽑아 출력
        v = queue.popleft()
        print(v, end = ' ')

        # 해당 원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
        for i in graph[v]:
            if not visitied[i]:
                queue.append(i)
                visitied[i]=True

# 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)

graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]

# 각 노드가 방문된 정보를 리스트 자료형으로 표현 (1차원 리스트)
visited = [False] * 9

# 정의된 BFS 함수 호출
bfs(graph, 1, visited)
```

```
콘솔 출력
1 2 3 8 7 4 5 6
```

<br>

# 마치며

이번엔 코딩 테스트에서 그래프 문제로 자주 출제되는 DFS와 BFS에 대해서 알아 보았습니다. 사실 제 생각에 코딩 테스트는 개념도 중요하지만 수학 처럼 개념을 익힌 뒤에 실제 문제 풀이를 많이 해보면서 감을 잡아야 한다고 생각합니다. 이번엔 개념만 알아보았고 이후에 코딩 테스트 관련 알고리즘들의 개념을 모두 정리한 뒤 각 알고리즘 별 문제 풀이도 같이 진행하고자 합니다.   
긴 긁 읽어주셔서 감사드리며 잘못된 내용 혹은 궁금하신 내용이 있으시다면 댓글 달아주시기 바랍니다!

