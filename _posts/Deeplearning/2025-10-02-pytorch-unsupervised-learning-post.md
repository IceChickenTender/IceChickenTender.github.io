---
title: "[Deeplearning] 파이토치를 이용한 비지도 학습 알아보기 "
categories:
  - Deeplearning
  - Pytorch
tags:
  - Deeplearning
  - Pytorch
  
use_math: true
toc: true
toc_sticky: true
toc_label: "파이토치를 이용한 비지도 학습"
---

이번엔 파이토치를 이용해 비지도 학습에 대해서 알아보도록 하겠습니다. 우리가 실제 데이터를 이용하여 모델을 구축하고 데이터를 분석한다고 가정할 때 가장 어려운 부분은 데이터 수집 및 가공입니다. 학습용으로 다루는 데이터들은 일반적으로 양도 충분하고 정답도 있어 지도 학습을 진행할 수 있습니다. 하지만 정답이 없는 경우에는 지도 학습을 위해 데이터 전체에 정답을 달아줘야 합니다. 이 때 데이터가 많다면 시간과 비용이 많이 들고 상황에 따라서는 정답을 어떻게 줘야 할지 애매한 경우가 있습니다. 또한 실수로 정답을 잘못 먹는 경우도 생길 수 있습니다.
비지도 학습은 정답 없이 데이터의 특성을 파악하는 학습 방법입니다. 즉, 모델에게 정답을 알려주지 않기 때문에 라벨링이 없는 데이터를 사용할 수 있다는 장점이 있습니다. 하지만 정답을 모르기 때문에 상대적으로 지도 학습보다는 성능이 낮을 수 있습니다. 대표적인 비지도 학습의 종류로는 입력 데이터로부터 특성을 뽑아 유사 성질들을 군집화 하는 클러스터링(Clustering)과 새로운 데이터를 생성해 내는 오토인코더(Autoencoder)와 생성적 적대 신경망(GAN)이 있습니다.

# 1. K-평균 알고리즘

클러스터리은 입력 데이터로부터 특성을 뽑아 유사 성질들을 군집화하는 비지도 학습이며 종류가 매우 다양합니다. 정답이 없어 클러스터링 종류에 따라 군집을 다르게 할 수 있기 때문에 종류 선택과 각 알고리즘의 하이퍼 파라미터 값을 잘 따져 봐야 합니다. K-평균(K-means) 알고리즘은 각 클러스터의 평균을 기준으로 점들을 배치시키는 알고리즘입니다. 이 때, 특정 거리 함수를 통해 각 중심(Centroid)과 입력값의 거리를 측정하고 그 중 가장 가까운 그룹으로 할당을 합니다. 예를 들어 그룹이 3개라면 각 그룹의 중심이 3개가 있고 어떤 한 점과 중심과의 거리를 계산합니다. 이 때 첫 번째 중심과의 거리가 가장 가까웠다면 이 점은 첫 번째 그룹으로 귀속됩니다. 모든 점이 할당되면 각 그룹의 평균을 구해 중심을 업데이트하고 위 과정을 반복하여 클러스터링을 시행합니다.

## 1.1 데이터 만들기

skleanr.datasets 에는 다양한 데이터를 제공합니다. 이번 K-means 예제에서는 make_circles 를 이용합니다. 총 500개의 점을 생성합니다.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

x, y = make_circles(n_samples=500, noise=0.1)
plt.figure(figsize=(9,6))
plt.scatter(x[:,0], x[:,1], c=y)
plt.show()
```

<div align="center">
  <img src="/assets/images/deeplearning/pytorch/unsupervised_learning/k_means_data_image.png" width="50%" height="40%"/>
</div>

## 1.2 텐서 데이터 변환하기

```python
x = torch.FloatTensor(x)
```

## 1.3 K-평균 알고리즘

거리 함수를 정의합니다. 여기서는 L2 거리 함수로 두 점(중심과 각 점)들의 거리를 측정합니다.
각 점과 중심과의 거리를 계산하여 가장 거리가 가까운 점의 인덱스를 반환합니다.

```python
def l2distance(a, b):
  return torch.argmin(torch.sum((a-b)**2, dim=1), dim=0)
```

K-평균 알고리즘에서는 클러스터의 수를 정해줘야만 합니다. 따라서 군집의 수를 알 수 없을 때 적절한 숫자를 찾아야 합니다. 여기서는 기본값을 num_clusters=2 로 했고, max_iteration 은 중심이 업데이트 횟수를 의미하며, 기본값을 5로 설정했습니다.

`centroids = torch.rand(num_clusters, x.size(1)).to(device)` 초기 중심을 랜덤으로 할당합니다. 하나의 벡터 크기는 입력값의 피쳐 개수 x.size(1)와 같아야 합니다.

`h = x[m].expand(num_clusters, -1)` 은 입력값 하나를 각 중심까지의 거리를 구해야 하므로 중심점의 개수 만큼 확장하기 위해 expand 를 이용해 입력값을 중심점의 개수 즉 클러스터 개수 만큼 복사하여 확장합니다.

모든 입력값과 모든 중심과의 거리를 계산하여 가장 가까운 그룹으로 할당합니다. 만약 클러스터의 개수가 2개라면 어떤 입력값과 2개의 중심점과 비교한 후 첫 번째 중심점과 가깝다면 첫 번째 그룹으로 할당합니다.   

지정한 max_iteration 만큼 업데이트를 진행하며, 할당된 그룹에 있는 점들의 평균값으로 각 중심점을 업데이트 합니다.

```python
def kmeans(x, num_clusters=2, max_iteration=5):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  x = x.to(device)
  centroids = torch.rand(num_clusters, x.size(1)).to(device)
  for update in range(max_iteration):
    y_assign = []
    for m in range(x.size(0)):
      h = x[m].expand(num_clusters, -1)
      assign = l2distance(h, centroids)
      y_assign.append(assign.item())
    
    y_assign = np.array(y_assign)
    if update != max_iteration-1:
      for i in range(num_clusters):
        idx = np.where(y_assign==i)[0]
        centroids[i] = torch.mean(x[idx], dim=0)
  return y_assign, centroids
```

## 1.4 알고리즘 실행 및 그래프 그리기

K-평균 알고리즘은 sklearn 에서 제공하는 from sklearn.cluster import KMeans 가 많이 사용됩니다. 하지만 파이토치를 통해 코드를 작성한다면 추후에 GPU 연산도 할 수 있으며 requires_grad 를 사용하여 다른 모델과 조합할 경우 역전파도 이용할 수 있습니다. 아래 이미지에서 별로 찍힌 것이 군집된 점들의 중심점입니다.

```python
y_pred, centroids = kmeans(x,2)
plt.figure(figsize=(9,6))
plt.scatter(x[:,0], x[:,1], c=y_pred)
plt.plot(centroids[:,0].cpu(), centroids[:,1].cpu(), '*', markersize=30)
plt.show()
```

<div align="center">
  <img src="/assets/images/deeplearning/pytorch/unsupervised_learning/kmeans_result_graph.png" width="50%" height="40%"/>
</div>

# 2. 오토인코더

오토인코더는 정답 없이 모델을 학습시키는 비지도 학습 모델입니다. 일반적으로 대칭형 구조를 지니고 있으며, 입력 데이터를 압축하는 인코더(Encoder) 부분과 압축을 푸는 디코더(Decoder) 부분으로 구성되어 있습니다. 따라서 인코더를 통해 차원 축소가 된 잠재 변수(Latent variable)를 가지고 별도로 계산을 할 수도 있고 디코더를 통해 입력값과 유사한 값을 생성할 수도 있습니다. 기본적으로 입력값 x 와 출력값 $\acute{x}$ 를 이용하여 MSE 를 정의하고 이를 기준으로 학습을 진행합니다.

<div align="center">
  <img src="/assets/images/deeplearning/pytorch/unsupervised_learning/autoencoder_image.png" width="70%" height="60%"/>
</div>

## 2.1 스택 오토인코더

### 라이브러리 불러오기

```python
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
```

### GPU 연산 정의 및 MNIST 데이터 불러오기

`torchvision.datasets.MNIST`를 이용해 데이터를 불러오고 transforms 를 이용하여 텐서 데이터로 변환합니다.

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = torchvision.datasets.MNIST(root="./data/", download=True, train=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
```

### 모델 구축하기

기본 오토인코더 모델은 층을 여러개 쌓았다고 해서 스택 오토인코더(Stakced Autoencoder)라고도 불립니다.

인코더와 디코더를 각각 nn.Sequential 로 묶어 보기 좋게 합니다.

MNIST 이미지의 크기는 1x28x28 입니다. 따라서 nn.Linear 에 넣어주기 위해 사진을 일렬러 편 후 인코더 부분에 크기가 784(28x28)인 벡터 하나가 들어오게 됩니다.

인코더에서는 층을 자유롭게 쌓아 노드를 10개까지 줄입니다. 즉, 잠재 변수의 크기가 10으로 정의됩니다.

디코더에서는 크기가 줄어든 잠재 변수 벡터를 다시 크기를 늘려줍니다.

마지막은 같은 크기의 이미지가 나와야 하므로 28\*28로 입력합니다.

MNIST 이미지의 픽셀값은 0이상 1이하입니다. 따라서 nn.Sigmoid() 를 이용해 범위를 정해서 수렴을 빨리 하게 할 수 있습니다.

forward 함수에서는 encoder 와 decoder 를 차례대로 연ㅅ나할 수 있도록 코드를 작성합니다.

```python
class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 10),
                nn.ReLU())
    
    self.decoder = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 28*28),
                nn.Sigmoid())
  
  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
```

### 모델, 손실 함수, 최적화 기법 정의하기

우리의 목적은 입력 이미지와 유사한 출력 이미지를 얻는 것입니다. 따라서 입력 이미지와 출력 이미지의 L2 거리를 계산하는 MSE 손실 함수를 사용하고, 최적화 방법은 Adam 을 사용합니다.

```python
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

### 학습하기

현재 오토인코더의 층은 합성곱 층이 아니고 일렬 노드로 구성된 nn.Linear 입니다. 따라서 이미지를 일렬로 펴서 넣어주기 위해 `outputs = model(inputs.view(-1, 28*28))` 에서 `inputs.view(-1,28*28)`  을 이용해 차원을 변경해 줍니다.

벡터 형태로 나온 출력값을 다시 정사각형 이미지로 변환하기 위해 `outputs = outputs.view(-1, 1, 28, 28)` 로 원래 이미지 차원으로 변경해 줍니다.

```python
for epoch in range(51):
  running_loss = 0.0
  for data in trainloader:
    inputs = data[0].to(device)
    optimizer.zero_grad()
    outputs = model(inputs.view(-1, 28*28))
    outputs = outputs.view(-1, 1, 28, 28)
    loss = criterion(inputs, outputs)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  cost = running_loss / len(trainloader)
  print("[%d] loss: %.3f"%(epoch+1, cost))
```

```
Output: 

[1] loss: 0.083
[2] loss: 0.060
[3] loss: 0.051
[4] loss: 0.044
... 이하 생략 ...
```

### 모델 평가

실제로 학습한 모델이 출력해주는 이미지가 실제 이미지와 비슷한지 확인하기 위해 평가 데이터를 이용해 실제 이미지와 모델이 만든 이미지와의 비교를 진행해 보았습니다.

```python
testset = torchvision.datasets.MNIST(root="./data/", download=True, train=False, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True) # 10개 이미지를 시각화

model.eval()

dataiter = iter(testloader)
images, labels = next(dataiter)
images = images.to(device)


with torch.no_grad():
    flattened_images = images.view(-1, 28*28)
    outputs = model(flattened_images)

images_to_show = images.cpu().numpy()
outputs_to_show = outputs.view(-1, 1, 28, 28).cpu().numpy()

n_images = 10  # 표시할 이미지 개수

fig, axes = plt.subplots(nrows=2, ncols=n_images, figsize=(15, 3))
plt.gray() # 흑백으로 설정

print("Original Images")
print("Reconstructed Images")

for i in range(n_images):
    # 원본 이미지 표시 (첫 번째 행)
    ax = axes[0, i]
    ax.imshow(np.squeeze(images_to_show[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # 복원된 이미지 표시 (두 번째 행)
    ax = axes[1, i]
    ax.imshow(np.squeeze(outputs_to_show[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
```

아래 이미지 에서 위는 실제 평가 데이터 이미지이고 아래는 모델이 만든 이미지입니다. 모델이 만든 이미지가 원본 보다는 조금 흐릿하긴 하지만 그래도 완전히 못 알아볼 정도는 아니어서 어느 정도 오토인코더가 동작한다고 볼 수 있습니다.

<div align="center">
  <img src="/assets/images/deeplearning/pytorch/unsupervised_learning/autoencoder_result_image.png" width="70%" height="60%"/>
</div>

# 3. 디노이징 오토인코더

오토인코더의 기본적인 목적은 새로운 데이터를 만드는 것입니다. 따라서 출력 데이터를 입력 데이터에 가까워지도록 학습한다면 기존의 데이터와 매우 유사하여 새로운 데이터를 만드는 의미가 무색해질 수 있습니다. 따라서 입력갑셍 과적합되지 않도록 입력값에 노이즈를 주입시키거나 신경망에 드롭 아웃을 적용하여 출력 데이터와 노이즈가 없는 원래 입력 데이터를 가지고 손실 함수를 계산합니다. 따라서 노이즈가 있는 이미지를 가지고 노이즈가 없는 이미지와 유사한 데이터를 만드는 구조이기 때문에 이를 디노이징 오토인코더(Denoising autoencoder)라고 합니다. 실제로 이미지 복원이나 노이즈 제거 등에 사용됩니다. 이번 예제에서는 가우시안 노이즈를 주입하여 학습 하는 방법을 구현해보겠습니다.

<div align="center">
  <img src="/assets/images/deeplearning/pytorch/unsupervised_learning/denoising_autoencoder_image.png" width="60%" height="50%"/>
</div>

### 노이즈 데이터 만들기

임포트와 데이터, 모델 코드는 이전 오토인코더 코드와 동일합니다. 우선 학습 데이터에서 일정 부분을 추출해 노이즈를 삽입해 보고, 원본 데이터와 비교를 위해 이미지 출력을 진행해 보도록 하겠습니다.

```python
dataiter = iter(trainloader)
images, labels = next(dataiter)
images = images.to(device)
dirty_images = images + torch.normal(0, 0.5, size=images.size()).to(device)

images_to_show = images.cpu().numpy()
dirty_images_to_show = dirty_images.cpu().numpy()

n_images = 10

fig, axes = plt.subplots(nrows=2, ncols=n_images, figsize=(15, 3))
plt.gray() # 흑백으로 설정

for i in range(n_images):
    ax = axes[0, i]
    ax.imshow(np.squeeze(images_to_show[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = axes[1, i]
    ax.imshow(np.squeeze(dirty_images_to_show[i]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
```

가우시안 노이즈를 주입한 결과 원본 이미지에 예전 TV 송출이 중단된 지지직 거리던 흑백이 추가된 것을 확인할 수 있습니다. 그럼 이렇게 노이즈를 추가한 데이터로 학습을 진해앻 보도록 하겠습니다.

<div align="center">
  <img src="/assets/images/deeplearning/pytorch/unsupervised_learning/noise_input_image.png" width="70%" height="60%"/>
</div>

### 학습하기

이전 오토인코더의 학습 부분에 가우시안 노이즈를 주입해주면 됩니다. 노이즈 텐서의 사이즈는 이미지 사이즈와 같아야 하기 때문에 size=input.size() 를 이볅하고 평균과 표준펴낯는 임의로 0, 0.5를 넣어주었습니다.

```python
for epoch in range(51):
  running_loss = 0.0
  for data in trainloader:
    inputs = data[0].to(device)
    optimizer.zero_grad()
    dirty_inputs = inputs + torch.normal(0, 0.5, size=inputs.size()).to(device) # 가우시안 노이즈 주입
    outputs = model(inputs.view(-1, 28*28))
    outputs = outputs.view(-1, 1, 28, 28)
    loss = criterion(inputs, outputs)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  cost = running_loss / len(trainloader)
  print("[%d] loss: %.3f"%(epoch+1, cost))
```

이미지 출력은 이전 오토인코더의 이미지 결과를 출력 코드를 그대로 사용하였습니다.

아래 이미지는 디노이징 오토인코더로 학습한 모델을 이용해 이미지를 생성한 결과입니다. 위는 원본 이미지이고, 아래는 모델이 생성한 이미지입니다.

저의 주관으로는 오토인코더 모델보다는 입력에 노이즈를 섞은 디노이징 오토인코더의 결과가 조금 더 좋아보입니다.

<div align="center">
  <img src="/assets/images/deeplearning/pytorch/unsupervised_learning/denoise_autoencoer_result_image.png" width="70%" height="60%"/>
</div>

# 4. 합성곱 오토인코더

합성곱 오토인코더(Convolutional autoencoer)는 nn.Linear 대신 합성곱 층 nn.Conv2d 를 사용하는 구조입니다. 따라서 이미지 데이터가 일렬로 펴지지 않고 그대로 들어와 연산이 진행됩니다. 기본적으로 잠재 변수 h 는 일렬 형태인 벡터이기 때문에, 인코더에서 나온 피쳐맵을 일렬로 펴서 h 를 추출하고 다시 h 를 은닉층을 거친 뒤 사각형 모양의 피쳐맵으로 만들어 디코더에 넣어주어야 합니다. 실제 h 를 벡터화하는 바법은 응용 범위가 다양합니다. 그럼 코드로 한 번 알아보도록 하겠습니다.

### 피쳐맵을 벡터화하기

`Flatten` 클래스는 인코더를 거친 후, 정 가운데에 있는 $h$(hidden layer 혹은 Latent Variables) 를 위한 클래스로 피쳐맵 형태에서 일렬로 변환하는 기능을 가진 클래스입니다. 인코더를 거친 피쳐맵의 크기는 (배치 사이즈, 채널 수, 이미지 너비, 이미지 높이)입니다. 따라서 배치 사이즈가 현재 이미지의 개수이므로 벡터가 배치 사이즈 만큼 존재해야 합니다. 즉, x.view(batch_size, -1)를 이용해 각 피쳐 데이터를 일렬로 변환합니다.

```python
class Flatten(torch.nn.Module):
  def forward(self, x):
    batch_size = x.shape[0]
    return x.view(batch_size, -1)
```

### 벡터를 사각형 피쳐맵으로 변환하기

`Deflatten` 클래스는 잠재 변수 h(혹은 hidden_layer) 에서 출력층으로 갈 때 일렬인 벡터를 CNN 에 맞게 피쳐맵으로 바꿔주는 클래스입니다.

`feature_size`는 잠재 변수 h 를 이용해 구합니다. 잠재 변수 h 의 크기는 (배치사이즈, 채널 수\*이미지 너비\*이미지 높이)입니다. 따라서 벡터 사이즈는 채널 수\*이미지 너비*이미지 높이이고 너비와 높이가 같다면 채널 수\*이미지 너비\*\*2가 됩니다. 그래서 이미지 한 변의 길이는 (벡터 사이즈//채널 수)\*\*.5 가 됩니다.

잠재 변수를 이용해 구한 `feature_size` 를 이용해 피쳐맵의 크기를 (배치 사이즈, 채널 수, 이미지 너비, 이미지 높이) = (s[0], self.k, feature_size, feature_size) 로 변환합니다.

```python
class Deflatten(nn.Module):
  def __init__(self, k):
    super(Deflatten, self).__init__()
    self.k = k

  def forward(self, x):
    s = x.size()
    feature_size = int((s[1]//self.k)**5)
    return x.view(s[0], self.k, feature_size, feature_size)
```

### 모델 구축하기

모델의 decoder 를 보면 nn.Conv2d 대신 nn.ConvTranspose2d 를 사용합니다. 그 이유는 크기가 작은 입력값을 크기가 큰 입력값으로 만들기 위함입니다. nn.ConvTranspose2d 는 입력 성분(Conv의 결과)을 출력 성분(Conv의 입력)으로 미분하여 그 값을 입력 벡터와 곱해 출력 벡터를 산출하고 그 결과 벡터를 행렬 형태로 변환하는 연산입니다. 이렇게 말하면 잘 이해가 되질 않습니다. 

쉽게 설명하자면 Conv2d 는 큰 이미지를 작은 특징 맵으로 요약하는 과정입니다. 마치 넓은 사진에서 돋보기(커널)를 움직이며 중요한 특징(윤곽선, 질감 등)만 찾아 작은 스케치북에 옮겨 그리는 것과 같으며, 이러한 과정을 진행하면 이미지는 작은 피쳐맵으로 압축되게 됩니다. 하지만 현재 우리가 알아보고 있는 합성곱 오토인코더의 디코더는 인코더에서 Conv2d 를 통해 압축된 피쳐맵을 다시 원본 이미지로 되돌려야 합니다. 그래서 Conv2d 반대로 동작하는 ConvTranspose2d 를 사용하는 것입니다. ConvTranspose2d 는 작은 특징 맵(요약본)을 가지고 원래의 큰 이미지로 확장/복원 할 수 있습니다. 업 샘플링(Upsampling) 컨볼루션이라고도 불립니다. ConvTranspose2d 를 이용해 도출되는 확장/복원된 피쳐매의 크기는 다음 식에 의해 크기를 산출할 수 있습니다.

(출력값의 크기) = (입력값의 크기-1)x(보폭)-2*(패딩)+(필터의 크기) + (출력값 패딩)
{: .text-center}


```python
class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    k = 16
    self.encoder = nn.Sequential(
        nn.Conv2d(1, k, 3, stride=2), nn.ReLU(),
        nn.Conv2d(k, 2*k, 3, stride=2), nn.ReLU(),
        nn.Conv2d(2*k, 4*k, 3, stride=1), nn.ReLU(),
        Flatten(), nn.Linear(1024, 10), nn.ReLU())

    self.decoder = nn.Sequential(
        nn.Linear(10, 1024), nn.ReLU(),
        Deflatten(4*k),
        nn.ConvTranspose2d(4*k, 2*k, 3, stride=1), nn.ReLU(),
        nn.ConvTranspose2d(2*k, k, 3, stride=2), nn.ReLU(),
        nn.ConvTranspose2d(k, 1, 3, stride=2, output_padding=1), nn.Sigmoid())

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
```

### 모델, 손실 함수, 최적화 기법 정의하기

```python
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

### 학습하기

```python
for epoch in range(51):
  running_loss = 0.0
  for data in trainloader:
    inputs = data[0].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  cost = running_loss /len(trainloader)
  if epoch % 10 == 0:
    print("[%d] loss: %.3f"%(epoch+1, cost))
```

### 모델을 이용한 이미지 출력하기

```python

test_dataset = torchvision.datasets.MNIST(root="./data/", download=True, train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True) # 10개의 이미지를 시각화합니다.

model.eval()

dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)

with torch.no_grad():
    outputs = model(images)

original_images = images.cpu().numpy()
reconstructed_images = outputs.cpu().numpy()

n = 10  # 표시할 이미지 개수
plt.figure(figsize=(20, 4))
print("Top row: Original Images, Bottom row: Reconstructed Images")

for i in range(n):
    # 원본 이미지 표시
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.squeeze(original_images[i]), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n // 2:
        ax.set_title('Original Images')

    # 복원된 이미지 표시
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.squeeze(reconstructed_images[i]), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n // 2:
        ax.set_title('Reconstructed Images')
plt.show()
```

<div align="center">
  <img src="/assets/images/deeplearning/pytorch/unsupervised_learning/conv_autoencoder_result_image.png" width="70%" height="60%"/>
</div>

# 5. 생성적 적대 신경망(GAN)

생성적 적대 신경망(Generative adverarial network, GAN)은 아이디어 자체만으로 매우 가치있는 모델로 평가 받고 있습니다. 실제로 얼굴 변환, 생성, 음성 변조, 그림 스타일 변환, 사진 복원 등 다양한 기술로 응용되고 있습니다. 기본적으로 생성적 적대 신경망은 진짜 같은 가짜 데이터를 만들어 내는 기술입니다.

