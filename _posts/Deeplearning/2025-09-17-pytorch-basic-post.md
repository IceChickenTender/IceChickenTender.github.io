---
title: "[Deeplearning] Pytorch 기초"
categories:
  - Deeplearning
  - Pytorch
tags:
  - Deeplearning
  - Pytorch
  
use_math: true 
toc: true
toc_sticky: true
toc_label: "Pytorch 기초"
---

# 1. Tensor

Pytorch 모델 연산을 위해서는 Pytorch 의 기본 단위인 Tensor 를 이용해야 합니다. 따라서 텐서의 종류와 연산을 잘 숙지해야만 효유렂ㄱ인 코드를 작성할 수 있습니다.

## 1.1 여러 가지 Tensor

Tensor 는 Pytorch 의 기본 단위이며, GPU 연산을 가능하게 합니다. 또한 Numpy 의 배열과 유사하여 손쉽게 다룰 수 있습니다. 

`torch.empty` 를 통해 크기가 5x4 인 빈 텐서를 생성합니다. 이 때 초기화되지 않은 행렬인 경우 해당 시점에 할당된 메모리에 존재하던 값들이 초깃값으로 나타납니다.

```python
import torch
import numpy as np

x = torch.empty(5,4)
print(x)
```

```
Output:

tensor([[8.4798e+04, 4.3065e-41, 8.4800e+04, 4.3065e-41],
        [8.4802e+04, 4.3065e-41, 8.4800e+04, 4.3065e-41],
        [8.4803e+04, 4.3065e-41, 8.4803e+04, 4.3065e-41],
        [8.4804e+04, 4.3065e-41, 8.4801e+04, 4.3065e-41],
        [8.4805e+04, 4.3065e-41, 8.4802e+04, 4.3065e-41]])
```

`torch` 는 `ones`, `zeros`, `empty` 등 넘파이에서 사용되는 동일한 형태의 함수들을 많이 제공하고 있습니다.

```python
import torch

torch.ones(3, 3)
```

```
Output:

tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
```

```python
import torch

torch.zeros(2) # 2행 영 벡터
```

```
Output:

tensor([0., 0.])
```

```python
import torch

torch.rand(5, 6)
```

```
Output:

tensor([[0.2059, 0.9352, 0.7966, 0.5072, 0.5273, 0.7979],
        [0.0672, 0.8546, 0.3399, 0.5666, 0.7043, 0.8849],
        [0.9940, 0.8452, 0.3362, 0.5954, 0.7204, 0.3742],
        [0.4486, 0.6853, 0.0660, 0.5973, 0.2877, 0.4527],
        [0.0630, 0.0097, 0.3998, 0.9447, 0.8573, 0.0923]])
```

# 1.2 리스트, 넘파이 배열을 Tensor 로 만들기

`torch.tensor()` 를 통해 텐서로 변환이 가능합니다. 또한 `torch.FloatTensor()`, `torch.LongTensor()` 와 같이 구체적인 텐서 타입을 정의할 수도 있습니다.

```python
import torch

l = [13, 4]
torch.tensor(l)
```

```
Output:

tensor([13,  4])
```

```python
import torch
import numpy as np

r = np.array([4, 56, 7])
torch.tensor(r)
```

```
Output:

tensor([ 4, 56,  7])
```

## 1.3 텐서의 크기, 타입 연산

`.size()` 는 텐서의 크기를 확인할 수 있으며 매우 자주 사용됩니다. `x.size()` 는 x 텐서 (5x4)의 크기이므로 `torch.Size([5, 4])` 로 출력됩니다. 따라서 `x.size()[1]` 는 4를 출력합니다. 혹은 `x.size(1)` 로도 표현이 가능합니다.

```python
import torch

x = torch.empty(5,4)
x.size()[1]
```

```
Output:

4
```

```python
type(x)
```

```
Output:

torch.Tensor
```

텐서의 사칙연산, 제곱, 몫 계산 등의 기본 연산은 넘파이와 동일합니다.

```python
import torch

x = torch.rand(2,2) # 2x2 랜덤 행렬
y = torch.rand(2,2) # 2x2 랜덤 행렬

print(x)
print()

print(y)
print()

print(x+y)
```

```
Output:

tensor([[0.6952, 0.3446],
        [0.9267, 0.8361]])

tensor([[0.1341, 0.9824],
        [0.5800, 0.9629]])

tensor([[0.8293, 1.3269],
        [1.5066, 1.7990]])
```

`torch,add(x, y)` 는 `x+y` 와 같은 의미입니다. `y.add(x)` 는 y 에 x 를 더한다는 의미입니다.

```python
print(torch.add(x, y))
print()

print(y.add(x))
```

```
Output:

tensor([[0.7732, 1.3242],
        [0.7352, 0.3525]])

tensor([[0.7732, 1.3242],
        [0.7352, 0.3525]])
```

`y.add_(x)` 는 y 에 x 를 더하여 y 를 갱신합니다. 즉, y 는 기존 y 와 x 가 더해진 값으로 바뀌어있습니다. 이와 같이 기존의 값을 덮어 씌우는 방식을 인플레이스(in-place) 방식이라고 합니다.

```python
print(y)
print()

y.add_(x)
print(y)
print()
```

```
Output:

tensor([[0.3894, 0.1670],
        [0.4131, 0.1102]])

tensor([[1.2173, 0.2576],
        [1.2094, 0.2270]])
```

## 1.4 텐서의 크기 변환

`view` 는 텐서 크기를 바꿔주는 함수입니다. 즉, `x.view(64)` 는 8x8을 일렬로 만든 텐서입니다. 

```python
import torch
x = torch.rand(8, 8) # 8x8 랜덤 행렬
print(x.size())
```

```
Output:

torch.Size([8, 8])
```

```python
a = x.view(64)
print(s.size())
```

```
Output:

torch.Size([64])
```

-1은 원래 크기가 되게 하는 값으로 전체 성분이 64개가 되게 하는 4x4x4 배열을 만들어야 합니다. 따라서 -1은 4도 자동 할당됩니다.

```python
import torch

b = x.view(-1, 4, 4)
print(b.size())
```

```
Output:

torch.Size([4, 4, 4])
```

## 1.5 텐서에서 넘파이 배열로 변환

텐서 뒤에 `.numpy()` 를 붙여주면 넘파이 배열로 변환됩니다.

```python
import torch

x = torch.rand(8,8)
y = x.numpy()
type(y)
```

```
Output: numpy.ndarray
```

## 1.6 단일 텐서에서 값으로 반환하기

`.item()` 은 손실 함숫값과 같이 숫자가 하나인 텐서를 텐서가 아닌 값으로 만들어 줍니다. 

```python
import torch

x = torch.ones(1)
print(x.item())
```

```
Output: 1.0
```

# 2. 역전파

모델 파라미터의 최적화는 미분의 성질과 연쇄 법칙을 기반으로 한 역전파를 통해 진행됩니다. 역전파는 모델이 복잡할수록 계산 과정이 복잡해져 코드를 직접 구현하기에는 어려움이 있습니다. 따라서 파이토치는 다양하게 사용할 수 있는 최적화 방법을 제공하고 있습니다.

## 2.1 그래디언트 텐서

일반적으로 인공 신경망의 최적화라는 것은 손실 함수의 최솟값이 나오게 하는 신경망의 최적 가중치를 찾는 과정입니다. 따라서 최적화를 위해 변화량을 나타내는 미분은 필수적인 요소입니다. 이 때 깊은 인공 신경망의 구조는 입력값이 들어와 다중의 층을 지나 출력값을 산출하는 합성 함수 형태임을 알 수 있습니다. 따라서 미분의 성질과 연쇄 법칙(chain rule)을 통해 원하는 변수에 대한 미분값을 계산할 수 있습니다. 다만 층이 깊어지거나 구조가 복잡할 수록 계산이 복잡해지기 때문에 사람이 직접 계산하기는 매우 힘듭니다. 파이토치는 앞서 언급한 일련에 계산 과정을 자동으로 해주는 자동 미분 계산 함수를 제공하고 있습니다. 따라서 최적화 과정인 역전파를 쉽게 작성할 수 있습니다.

```python
import torch

x = torch.ones(2,2,requires_grad=True)

y = x+1
z = 2*y**2
r = z.mean()
print("Result:", r)
```

```
Output: Result: tensor(8., grad_fn=<MeanBackward0>)
```

`requires_grad = True`는 해당 텐서를 기준으로 모든 연산들을 추적하여 그래디언트(Gradient)라고 하는 미분값의 모임(배열)을 계산할 수 있게 합니다. 즉, x 에 대해서 영ㄴ쇄 법칙을 이용한 미분이 가능하다는 것입니다. 예시를 보면 y 는 x 에 대한 식, z 는 y 에 대한 식, r 은 z 에 대한 식입니다. 따라서 이는 합성 함수의 개념으로써 최종 함수 r 은 x 에 대해서 표현 및 미분이 가능합니다.

여기서 수학적인 이해가 필요한 부분이 있는데, 미분을 한다는 것은 미분이 가능한 함수라는 것이고, 함수라는 것은 미분하려는 변수가 함수의 조건을 만족해야 한다는 의미입니다. 함수의 조건은 저으이역에 속하는 주어진 값 x 는 오직 하나에 대한 r 값이 치역에 존재해야 합니다. 따라서 y 와 z 는 함수의 조건에 만족하지 않고 일련의 계산 과정이기 때문에 y 와 z 를 x 에 대해서 미분을 하려고 했을 때 에러가 납니다. 따라서 모델의 최적화를 위해 단일값이 나오는 손실 함수를 정의하는 것입니다.

```python
r.backward()
print(x.grad)
```

```
Output:

tensor([[2., 2.],
        [2., 2.]])
```

r 을 기준으로 역전파를 진행하겠다는 의미이므로, $\frac{dr}{dx}$ $r = \frac{z_1+z_2+z_3+z_4}{4}$ 이고 $z_1 = 2{y_i}^2=2(x_i+1)^2$ 이므로 $\frac{dr}{dx} = x_i + 1$ 입니다.

모든 x 의 성분이 1이므로 그래디언트 `x.grad` 를 통해 나온 미분값은 모두 2가 됩니다.

## 2.2 자동 미분 - 선형회귀식

자동 미분을 위해 과거에는 Tensor 를 덮어씌워 사용하는 Variable 을 사용했습니다. 하지만 현재 텐서는 자동 미분을 위한 기능을 직접 제공하기 때문에 Variable 을 사용하지 않고 Tensor 를 바로 이용합니다.

아래는 데이터 생성을 위한 코드로 x 는 0~4까지의 수를 세로 벡터(5,1)로 만든 입력 특성 행렬입니다. `unsqueeze(1)` 을 통해 배치 차원 N=5, 특성 차원 D=1 꼴로 맞춰 줍니다.
`y = 2*x + noise` 는 정답(타깃)으로, 기울기 2, 절편 0인 선형관계에 랜덤 잡음을 더해 현실적인 회귀 문제는 만듭니다.

```python
import torch
from matplotlib import pyplot as plt

# 0,1,2,3,4 를 float32 텐서로 만들고 (5,) → (5,1) 형태로 차원을 하나 늘림
# 선형회귀에서 특성행렬 X를 (N, D) 꼴로 사용하기 위해 unsqueeze(1)
x = torch.FloatTensor(range(5)).unsqueeze(1)

# 타깃 y = 2*x + ε (ε ~ U[0,1))  : 실제 관계(기울기=2)에 약간의 잡음을 더해 학습 문제를 구성
y = 2*x + torch.rand(5,1)

# 입력 특성 수 D (=열 개수). 여기서는 1차원 특성이라 1
num_features = x.shape[1]
```

`w` 와 `b` 는 각각 가중치(기울기) 와 편향(절편)입니다. `requires_grad=True` 를 주면 Pytorch 가 이 텐서로부터 파생되는 연산 그래프를 추적하여, 나중에 `loss.backward()` 때 연쇄법칙으로 `∂loss/∂w`, `∂loss/∂b`를 계산해 `w.grad`, `b.grad`에 저장합니다.

선형식은 `y=wb+b` 로 표현됩니다. 따라서 w 는 5x1 데이터와 곱할 수 이썽야 하며, 예측값이 하나로 나와야 하므로 크기가 1(피쳐수)x1(출력값 크기)인 배열로 정의합니다. 따라서 `wx` 는 5x1이 됩니다.

편향 `b` 는 모든 인스턴스에 동일한숫자를 더해주는 것이므로 크기가 1인 텐서로 정의합니다.

우리의 목표는 `wx+b` 가 잘 예측을 할 수 있는 `w` 와 `b`를 찾는 것입니다. 초깃값에 대한 좋은 정보가 있을 경우에는 좋은 값으로 초깃값을 설정한다면 수렴이 빠르고 정확도도 높아질 수 있지만, 모르는 경우에는 초깃값을 무작위로 줍니다. 이 예시에서는 `torch.randn` 을 이용합니다.

`x`, `y` 와 `w`, `b` 텐서의 가장 큰 차이는 `requires_grad=True` 유무입니다. 데이터는 변하지 않는 값으로서 업데이트가 필요 하지 않는 반면, `w`, `b` 값은 역전파를 통해 최적값을 찾는 것이므로, `w`, `b` 에 `requires_grad` 를 `True` 로 활성화시킵니다.

```python
# 가중치 w (D×1)와 편향 b (스칼라)를 난수로 초기화
# requires_grad=True: 역전파 시 이 변수들에 대한 미분(그래디언트)을 추적/계산하도록 설정
w = torch.randn(num_features, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
```

가중치를 업데이트하는 최적화 방법은 매우 다양합니다. 그중 가장 널리 사용되는 방법이 경사하강법입니다. 경사하강법(Gradient descent)은 목적 함수인 손실 함수를 기준으로 그래디언트를 계산하여 변수를 최적화하는 기법입니다. 이 예시에서는 가장 기본적인 최적화 방법인 확률적 경사하강법(SGD)을 사용합니다.

`torch.optim.SGD` 내부에 변수를 리스트로 묶어 넣어주고 적절한 학습률(learning rate)을 정하여 자동으로 가중치와 편향을 업데이트합니다.

```python
# 학습률(스텝 크기) 설정
learning_rate = 1e-3

# SGD(확률적 경사하강법) 옵티마이저에 업데이트 대상 파라미터 목록과 학습률 전달
optimizer = torch.optim.SGD([w,b], lr=learning_rate)
```

선형 모델 $\hat{\mathbf{y}} \;=\; X\,\mathbf{w} \;+\; b\,\mathbf{1}$ 를 그대로 텐서 연산으로 구현했습니다. 여기서 `torch.matmul(x,w)` 는 행렬곱으로 $(5,1)@(1,1) \rightarrow (5,1)$ rㅕㄹ과가 나오고, `b` 가 브로드캐스팅되어 모든 샘플에 더해집니다.

최적화는 계산을 누석시키지 때문에 매 에폭마다 누적된 값을 `optimizer.zero_grad()` 을 통해 초기화합니다.

예측값과 실제값을 이용해 손실 함수를 계산합니다. 여기서 사용된 함수는 MSE(Mean Square Error)로 회귀에서 가장 흔히 쓰는 평균제곱오차입니다. 손실이 작아질수록 예측이 정답에 가까워진다는 뜻입니다.

매 에폭(epoch) 마다 손실 함수값을 저장하기 위해 빈 리스트 `loss_stack` 을 생성합니다. 

```python
# 총 1001회(0~1000 epoch) 반복 학습
for epoch in range(1001):
  # 이전 step에서 누적된 gradient를 0으로 초기화 (PyTorch는 기본적으로 누적)
  optimizer.zero_grad()

  # 예측값 y_hat = Xw + b
  # x: (5,1), w: (1,1) → matmul 결과 (5,1)
  # b: (1,) 이므로 브로드캐스팅으로 (5,1)에 더해짐
  y_hat = torch.matmul(x,w) + b

  # 평균제곱오차(MSE) = mean((y_hat - y)^2)
  loss = torch.mean((y_hat - y)**2)

  # 역전파: 현재 loss를 w, b에 대해 미분하여 w.grad, b.grad에 그래디언트 저장
  loss.backward()

  # 경사하강 스텝: w ← w - lr * w.grad,  b ← b - lr * b.grad
  optimizer.step()

  # 시각화/기록용으로 scalar 값만 추출해 리스트에 저장(.item()은 Python float로 변환)
  loss_stack.append(loss.item())

  # 100 epoch마다 진행상황(손실값) 로그 출력
  if epoch % 100 ==0:
    print(f"Epoch: {epoch}:{loss.item()}")
```

최종 학습된 `w`, `b` 로 예측값을 산출합니다. 이 때 최적화를 사용하지 않으므로 `requires_grad` 를 비활성화합니다. 이 때 `with torch.no_grad():` 를 이용하여 구문 내부에 있는 `requires_grad` 가 작동하지 않도록 할 수 있습니다. 그리고 matplotlib 을 이용해 loss 와 관련된 그래프를 그려보았습니다.

```python
# 추론/시각화 구간에서는 그래디언트 추적이 불필요하므로 비활성화
with torch.no_grad():
  # 학습된 파라미터(w, b)로 예측값 계산: y_hat = Xw + b
  y_hat = torch.matmul(x, w) + b

  # 전체 그림(figure) 생성: 가로 10인치, 세로 5인치 크기
  plt.figure(figsize=(10,5))

  # 1행 2열(subplot 1x2) 중 첫 번째(왼쪽) 서브플롯 활성화
  plt.subplot(121)

  # 학습 동안 기록한 손실값(loss_stack)을 선 그래프로 표시
  plt.plot(loss_stack)

  # 현재(왼쪽) 서브플롯의 제목 설정
  plt.title("Loss")

  # 1행 2열(subplot 1x2) 중 두 번째(오른쪽) 서브플롯 활성화
  plt.subplot(122)

  # 원본 데이터 점 산포도: 파란 점('.b')
  # (x, y)가 텐서여도 matplotlib가 내부적으로 넘파이로 변환해 그릴 수 있음
  plt.plot(x, y, '.b')

  # 모델 예측 곡선: 빨간 실선('r-')
  plt.plot(x, y_hat, 'r-')

  # 범례 추가 (왼쪽이 원본, 오른쪽이 예측) — 'groud'는 'ground' 오타일 가능
  plt.legend(['groud truth', 'prediction'])

  # 현재(오른쪽) 서브플롯의 제목 설정
  plt.title("Prediction")

  # 위에서 정의한 두 개의 서브플롯을 화면에 렌더링
  plt.show()

```

<div align="center">
  <img src="/assets/images/deeplearning/pytorch/loss_graph.png" width="60%" height="50%"/>
</div>

# 3. 데이터 불러오기

메모리와 같은 하드웨어 성능의 한계 등의 이유로 한 번에 전체 데이터를 모델에 주어 학습하기 힘들기 때문에 일반적으로 배치 형태의 묶음으로 데이터를 나누어 모델 학습에 이용합니다. 또한 모델을 학습할 때 데이터의 특징과 사용 방법에 따라 학습 성능의 차이가 날 수 있습니다. 따라서 데이터를 배치 형태로 만드는 법과 데이터를 전처리하는 방법에 대해서 알아보도록 하겠습니다.

## 3.1 파이토치 제공 데이터 사용

`torchvision.transforms.Compose` 는 여러 개의 변환(transform)을 순서대로 묶어 한 번에 적용할 수 있도록 해주는 래퍼(Wrapper) 입니다. 즉, 이미지를 `Compose` 객체에 넣으면 등록된 변환들이 정해진 순서대로 차례차례 실행됩니다. 예시에서는 16x16 으로 이미지 크기 변환 후 텐서 타입으로 변환합니다. 만약 원본 이미지의 너비, 높이가 다를 경우 너비, 높이를 각각 지정을 해야 하기 때문에 `tr.Resize((16, 16))`이라고 입력해야 합니다.

`torchvision.datasets`에서 제공하는 `CIFAR10` 데이터를 불러옵니다(CIFAR10 은 카테고리가 10개인 이미지 데이터 세트입니다). `root`에는 다운로드 받을 경로를 입력합니다. `train=True`이면 학습 데이터를 불러오고, `train=False`이면 테스트 데이터를 불러옵니다. 마지막으로 미리 선언한 전처리를 사용하기 위해 `transform=transf`을 입력합니다.

```python
import torch # 파이토치 기본 라이브러리
import torchvision # 이미지와 관련된 파이토치 라이브러리
import torchvision.transforms as tr # 이미지 전처리 기능들을 제공하는 라이브러리
from torch.utils.data import DataLoader, Dataset # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리
import numpy as np
import matplotlib.pyplot as plt

# 여러 변환을 순차적으로 적용하기 위한 Compose
# - Resize(16): 원본 CIFAR-10(32×32)을 16×16으로 축소
# - ToTensor(): [0,255] 범위의 PIL 이미지를 [0.0,1.0] 범위의 torch.Tensor(C,H,W)로 변환
trasf = tr.Compose([tr.Resize(16), tr.ToTensor()])

# 학습용 CIFAR-10 데이터셋 로드
# - root="./data": 데이터가 저장될 경로
# - train=True: 학습 세트
# - download=True: 없으면 다운로드
# - transform=trasf: 위에서 정의한 변환 적용
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=trasf
)

# 테스트용 CIFAR-10 데이터셋 로드 (동일 변환 적용)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=trasf
)

# 첫 번째 샘플의 (이미지 텐서, 레이블) 중 이미지 텐서의 크기 출력
# CIFAR-10은 RGB 3채널이므로 (C,H,W) = (3,16,16)이 기대값
print(trainset[0][0].size())
```

```
Output: torch.Size([3, 16, 16])
```

일반적으로 데이터셋은 이미지와 라벨이 동시에 들어있는 튜플(tuple) 형태입니다. 따라서 `trainset[0]`은 학습 데이터의 첫 번째 데이터로 이미지 한 장과 라벨 숫자 하나가 저장되어 있습니다. 즉, `trainset[0][0]`은 이미지이며, `trainset[0][1]`은 라벨입니다. 현재 이미지 사이즈는 3x16x16입니다. 여기서 3은 채널 수를 말하고, 16x16은 이미지의 너비와 높이를 의미합니다. 일반적인 컬러 사전은 RGB 이미지이기 때문에 채널이 3개이고, (너비)x(높이)x(채널 수)로 크기가 표현됩니다. 하지만 파이토치에서는 이미지 한 장이 (채널 수)x(너비)x(높이)로 표현되니 유의해야합니다.

`DataLoader`는 데이터를 미니 배치 형태로 만들어 줍니다. 따라서 배치 데이터에 관한 배치 사이즈 및 셔플 여부 등을 선택할 수 있습니다. 즉 `batch_size=50, shuffle=True`은 무작위로 데이터를 섞어 한 번에 50개의 이미지를 묶은 배치로 제공하겠다는 의미입니다.

`CIFAR10`의 학습 이미지는 50,000장이고, 배치 사이즈가 50장이므로 1,000은 배치의 개수가 됩니다.

```python
trainloader = DataLoader(trainset, batch_size=50, shuffle=True)
testloader = DataLoader(testset, batch_size=50, shuffle=False)

print(len(trainloader))
```

```
Output: 1000
```

배치 이미지를 간단히 확인하기 위해 파이썬에서 제공하는 `iter`와 `next` 함수를 이용합니다. 이를 통해 `trainloader`의 첫 번째 배치를 불러 올 수 있습니다.

배치 사이즈는 (배치 크기)x(채널 수)x(너비)x(높이)를 의미합니다. 즉, 배치 하나에 이미지 50개가 들어가 있음을 알 수 있습니다.

```python
images, labels = next(iter(trainloader))
print(images.size())
```

```
Output: torch.Size([50, 3, 16, 16])
```

`image[1]` 의 크기는 (3,16,16)입니다. 이 때 그림을 그려주기 위해서 채널 수가 가장 뒤로 가는 형태인 (16,16,3)을 만들어야 합니다. permutate(1,2,0)은 기존 차원의 위치인 0,1,2를 1,2,0으로 바꾸는 함수입니다. 따라서 0번째의 크기가 3인 텐서를 마지막으로 보냅니다. 마지막으로 `numpy()`를 이용해 넘파이 배열로 변환합니다.

```python
oneshot = images[1].permute(1,2,0).numpy()
plt.figure(figsize=(2,2))
plt.imshow(oneshot)
plt.axis("off")
plt.show()
```

<div align="center">
  <img src="/assets/images/deeplearning/pytorch/CIFAR10_sample.png" width="40%" height="30%"/>
</div>

# 마치며

파이토치의 기본에 대해서 알아보았습니다. 파이토치의 텐서와 파이토치를 이용해 학습을 위한 역전파 방법에 대해서 알아보았고, 마지막으로 파이토치에서 제공하는 데이터셋을 불러오는 방법에 대해서도 알아보았습니다. 이후에는 여러 인공 신경망에 대해서 다뤄보도록 하겠습니다. 긴 글 읽어주셔서 감사드리며, 잘못된 내용, 오타 혹은 궁금하신 사항이 있으실 경우 댓글 달아주시기 바랍니다.
