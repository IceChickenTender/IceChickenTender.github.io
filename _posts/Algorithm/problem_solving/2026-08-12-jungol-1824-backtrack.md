---
title: "[AlgorithmSolving][Backtracking] JUNGOL_1824: 스도쿠"
categories:
  - AlgorithmSolving
tags:
  - AlgorithmSolving
  
use_math: true
toc: true
toc_sticky: true
toc_label: "JUNGOL_1824: 스도쿠"
---

# 1. 문제 내용

- 문제:

  스도쿠는 18세기 스위스 수학자가 만든 '라틴 사각형'이랑 퍼즐에서 유래한 것으로 현재 많은 인기를 누리고 있다. 

  이 게임은 아래 그림과 같이 가로, 세로 각각 9개씩 총 81개의 작은 칸으로 이루어진 정사각형 판 위에서 이뤄지는데, 

  게임 시작 전 몇 몇 칸에는 1부터 9까지의 숫자 중 하나가 쓰여 있다.

  <div align="center">
    <img src="/assets/images/algorithm/problem_solving/jungol_1824/sudoku.png" width="45%" height="40%"/>
  </div>

  나머지 빈 칸을 채우는 방식은 다음과 같다. (1) 각각의 가로줄과 세로줄에는 1부터 9까지의 숫자가 한 번씩만 나타나야 한다. (2) 굵은 선으로 구분되어 있는 3x3 정사각형 안에도 1부터 9까지의 숫자가 한 번씩만 나타나야 한다.

 

  위의 예의 경우, 첫째 줄에는 1을 제외한 나머지 2부터 9까지의 숫자들이 이미 나타나 있으므로 첫째 줄 빈칸에는 1이 들어가야 한다.
  
  또한 위쪽 가운데 위치한 3x3 정사각형의 경우에는 3을 제외한 나머지 숫자들이 이미 쓰여있으므로 가운데 빈 칸에는 3이 들어가야 한다.

  이와 같이 빈칸을 차례로 채워 가면 다음과 같은 최종 결과를 얻을 수 있다. 

  <div align="center">
    <img src="/assets/images/algorithm/problem_solving/jungol_1824/sudoku_result.png" width="45%" height="40%"/>
  </div>

  (위 사각형에 있는 음영은 문제의 편의상 표시한 것이다.)

  게임 시작 전 스도쿠 판에 쓰여 있는 숫자들의 정보가 주어질 때 모든 빈 칸이 채워진 최종 모습을 출력하는 프로그램을 작성하시오.

- 입력:

  아홉 줄에 걸쳐 한 줄에 9개씩 게임 시작 전 스도쿠판 각 줄에 쓰여 있는 숫자가 한 칸씩 띄워서 차례로 주어진다.

  스도쿠 판의 빈 칸의 경우에는 0이 주어진다. 

  스도쿠 판을 규칙대로 채울 수 없는 경우의 입력은 주어지지 않는다.

- 출력:

  모든 빈 칸이 채워진 스도쿠 판의 최종 모습을 아홉줄에 걸쳐 한 줄에 9개씩 한 칸씩 띄워서 출력한다.

  스도쿠 판을 채우는 방법이 여럿인 경우는 그 중 하나만을 출력한다.

- 예제 입력:

  ```
  0 3 5 4 6 9 2 7 8
  7 8 2 1 0 5 6 0 9
  0 6 0 2 7 8 1 3 5
  3 2 1 0 4 6 8 9 7
  8 0 4 9 1 3 5 0 6
  5 9 6 8 2 0 4 1 3
  9 1 7 6 5 2 0 8 0
  6 0 3 7 0 1 9 5 2
  2 5 8 3 9 4 7 6 0
  ```

- 예제 출력:

  ```
  1 3 5 4 6 9 2 7 8
  7 8 2 1 3 5 6 4 9
  4 6 9 2 7 8 1 3 5
  3 2 1 5 4 6 8 9 7
  8 7 4 9 1 3 5 2 6
  5 9 6 8 2 7 4 1 3
  9 1 7 6 5 2 3 8 4
  6 4 3 7 8 1 9 5 2
  2 5 8 3 9 4 7 6 1
  ```

# 2. 문제 유형 파악

이 문제는 비어있는 81개의 칸에 1부터 9까지의 숫자를 채워 넣는 전형적인 백트래킹 문제입니다.

- 백트래킹이 필요한 이유: 
  
  단순히 가로, 세로, 3X3 조건에 맞는 숫자를 순서대로 채워 넣는다고해서 스도쿠가 완성되지 않습니다. 당장 조건에 맞아 보여서 숫자를 채웠지만, 나중에 다른 칸을 채우다 보니 어떤 숫자도 들어갈 수 없는 모순에 빠질 수 있기 때문입니다. 따라서 막다른 길에 다다랐을 때, 이전에 내렸던 선택을 취소하고 다른 숫자를 넣어보는 탐색 과정이 필수적이므로 완전 탐색과 가지치기를 결합한 백트래킹으로 접근해야 합니다.

# 3. 문제를 풀기 위한 핵심 아이디어

이 문제를 시간 초과 없이 효율적으로 해결하기 위한 핵심은 "상태 공간 트리의 깊이를 줄이고, 유망성 검사 시간을 $O(1)$로 단축하는 것입니다.

1. 탐색의 대상을 '빈칸'으로만 한정하기

  81개의 전체 맵을 돌면서 재귀를 호출하면 불필요한 연산이 기하급수적으로 늘어납니다. 탐색을 시작하기 전, 스도쿠 판을 한 번 순회하며 숫자가 0인 빈칸들의 좌표(행, 열)만 리스트(`blanks`)에 수집합니다. 이제 재귀 함수의 깊이(Depth)는 이 빈칸 리스트의 인덱스가 되며, 오직 숫자를 채워야 할 곳만 핀포인트로 탐색할 수 있습니다.

2. 빠른 유망성 검사를 위한 3개의 상태 배열

  빈칸에 1부터 9까지의 숫자를 넣어볼 때, 매번 `for`문을 돌며 가로, 세로, 3x3 박스를 검사하면 막대한 시간 지연이 발생합니다. 이를 해결하기 위해 각 구역의 숫자 점유 여부를 기록하는 3개의 2차원 배열(Boolean 배열)을 도입합니다.

    - row_check[r][num] : r번째 행에 숫자 num이 존재하는가?
    - col_check[r][col] : r번째 열에 숫자 num이 존재하는가?
    - square_check[sq_idx][num]: sq_idx 번째 3x3 박스에 숫자 num이 존재하는가?
  
  이 배열들을 활용하면 단일 `if`문 하나로 해당 자리에 숫자를 넣을 수 있는지 단 $O(1)$만에 판별할 수 있습니다.

3. 3x3 박스 인덱스 (sq_idx) 구하기

9x9 격자에는 총 9개의 3x3 박스가 존재합니다. 현재 좌표 `(r, c)`가 0번부터 8번 중 몇 번째 박스에 속하는지 구하는 공식은 다음과 같습니다.

  - 공식: (r//3)*3 + (c//3)
  - 행을 3으로 나눈 몫에 3을 곱하여 '박스가 시작되는 덩어리 줄'을 찾고, 열을 3으로 나눈 몫을 더해 최종 박스 번호를 도출합니다.

4. 상태의 기록과 복구

  숫자를 내려놓으며 상태 배열을 `True`로 만들었다면, 재귀 탐색이 실패하고 뒤로 돌아올 때는 반드시 다시 배열을 False로 되돌려놓아야 합니다. 그래야 해당 자리에 다른 후보 숫자를 넣어보거나, 더 윗단계를 수정할 수 있습니다.

5. 언어의 실행 환경 한계 극복

  스도쿠의 백트래킹은 경우에 따라 수십만 번 이상의 단순 반복과 재귀 호출이 일어납니다. 파이썬의 실시간 통역(Interpreter) 방식으로는 시간 초과를 피하기 매우 까다롭습니다. 따라서 자주 반복되는 코드를 기계어로 캐싱해버리는 JIT 컴파일러 기반의 PyPy3로 제출하여 언어 레벨의 실행 속도 한꼐를 돌파해야 합니다.

# 4. 코드

```python
import sys

def solve():
    input = sys.stdin.readline
    # 2차원 리스트 생성
    board = [list(map(int, input().split())) for _ in range(9)]
    blanks = []

    # O(1) 탐색을 위한 상태 배열
    row_check = [[False] * 10 for _ in range(9)]
    col_check = [[False] * 10 for _ in range(9)]
    square_check = [[False] * 10 for _ in range(9)]

    # 초기 보드 상태 세팅
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                num = board[i][j]
                row_check[i][num] = True
                col_check[j][num] = True
                square_check[(i // 3) * 3 + (j // 3)][num] = True
            else:
                blanks.append((i, j)) # 빈칸 좌표 수집
                
    def get_candidates_count(r, c):
        count = 0
        sq_idx = (r // 3) * 3 + (c // 3)
        for num in range(1, 10):
            if not row_check[r][num] and not col_check[c][num] and not square_check[sq_idx][num]:
                count += 1
        return count

    # blanks 리스트를 후보 숫자가 적은 빈칸부터 오름차순 정렬
    blanks.sort(key=lambda x: get_candidates_count(x[0], x[1]))

    def backtrack(depth):
        # Base Case: 모든 빈칸을 다 채웠다면 출력 후 강제 종료
        if depth == len(blanks):
            for row in board:
                print(*row) # 언패킹을 사용하여 한 줄씩 빠르게 출력
            sys.exit(0)

        # 현재 채워야 할 빈칸 좌표와 3x3 박스 인덱스를 '반복문 밖에서 한 번만' 계산
        r, c = blanks[depth]
        sq_idx = (r // 3) * 3 + (c // 3)

        for num in range(1, 10):
            # 함수 호출 오버헤드를 없애고 직접 조건문으로 O(1) 검사
            if not row_check[r][num] and not col_check[c][num] and not square_check[sq_idx][num]:

                # 상태 업데이트 (Do)
                board[r][c] = num
                row_check[r][num] = True
                col_check[c][num] = True
                square_check[sq_idx][num] = True

                # 다음 빈칸 탐색
                backtrack(depth + 1)

                # 상태 복구 (Undo)
                board[r][c] = 0
                row_check[r][num] = False
                col_check[c][num] = False
                square_check[sq_idx][num] = False

    backtrack(0)

solve()
```