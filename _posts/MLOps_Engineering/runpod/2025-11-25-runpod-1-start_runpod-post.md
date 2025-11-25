---
title: "[MLOps/Engineering] RunPod - 1. RunPod 시작하기"
categories:
  - MLOps/Engineering
  - RunPod
tags:
  - MLOps/Engineering
  - RunPod
  
use_math: true
toc: true
toc_sticky: true
toc_label: "RunPod 시작하기"
---

# 개요

최근에 RunPod라는 온라인 상에서 Docker 환경으로 시간당 돈을 지불하고 GPU를 사용할 수 있도록 하는 사이트를 알게 되었습니다. RunPod를 알게된 계기는 구글 코랩을 이용해 딥러닝 모델들을 공부하던 중에 코랩보다 좀 더 저렴하게 사용할 수 있는 곳은 없을까 하고 찾아보다가 알게 되었습니다. 코랩은 무료 버전에서도 T4 GPU를 사용할 수 있지만 하루에 1\~2시간이 한계입니다. 그래서 저는 저에게 투자한다고 생각하고 월 구독료 9.99 달러인 코랩 프로를 사용했는데 코랩 프로도 한 달에 100 컴퓨팅 유닛을 주고 100 컴퓨팅 유닛을 다 사용하면 무료로 사용하는 것과 똑같이 T4 GPU만 사용할 수 있고 시간도 1\~2시간 밖에 사용하지 못한다는 것을 알게 되어 이 돈이면 차라리 좀 더 저렴한 것을 사용하는게 낫겠다는 생각에 RunPod 사용법에 대해서 알아보았고, 이후에 저를 위해서라도 블로그의 포스트로 정리해 놓자고 생각해서 이 포스트를 쓰게 되었습니다.

# 1. Runpod 란?

RunPod 란 딥러닝 모델 학습, 배포, 그래픽 렌더링을 위해 고성능 GPU를 합리적인 가격에 빌려 쓸 수 있는 클라우드 GPU 임대 플랫폼입니다. AWS나 Google Cloud 같은 거대 클라우드 서비스(Hyperscaler)가 범용적인 서비스를 제공한다면, RunPod는 AI와 그래픽 작업에 특화된 GPU 인프라를 쉽고 저렴하게 제공하는데 집중합니다.

RunPod는 우선 압도적인 가성비를 자랑합니다. RunPod는 두 가지 형태로 서비스를 제공하는데 "Community Cloud"는 개인이나 소규모 데이터센터가 유휴 GPU 자원을 공유하는 형태입니다. 마치 에어비앤비(Airbnb)처럼 남는 GPU를 빌려주기 때문에 가격이 매우 저렴합니다. 그 다음으로는 "Secure Cloud"로 RunPod의  파트너 데이터센터에서 제공하는 검증된 서버입니다. 보안과 안정성이 보장되어 기업이나 중요한 프로젝트에 적합합니다.

두 번째로 RunPod는 컨테이너 기반의 편리한 환경을 제공합니다. RunPod는 기본적으로 Docker 컨테이너 기술을 사용합니다. 이후에 RunPod 사용방법에 대해서 살펴보면서 알게 되겠지만 사용자가 복잡한 환경 설정을 처음부터 할 필요없이, PyTorch, TensorFlow, Stable Diffusion 등이 이미 설치된 '템플릿'을 선택하면 몇 번의 클릭만으로 바로 작업 환경이 열립니다. 또한 사용자가 직접 커스텀 Docker 이미지를 올릴 수도 있어 자유도가 높습니다.

마지막으로 "Serverless GPU"(추론 및 배포 용도)로 단순히 GPU 서버를 임대하는 것을 넘어, AI 모델을 서비스(API)로 배포할 때 사용하는 Serverless 기능도 제공합니다. 서버를 계속 켜두는 것이 아니라, AI가 요청을 처리하는 시간(초 단위)만큼만 비용을 지불하는 방식이라 효율적입니다.

# 2. RunPod를 사용하는 이유

1. 저렴한 비용
위에서 설명한 것과 같이 RunPod는 아마존 AWS나 구글 클라우드와 같이 비용적인 측면에서 굉장히 저렴합니다. 사실 A100 이상의 GPU의 경우 RunPod나 다른 대기업의 클라우드 서비스와 비교해도 큰 차이는 없습니다. 하지만 RunPod만의 특징 중 하나는 개인이나 소규모 데이터센터에서 제공하는 "Community Cloud"의 가격이 굉장히 저렴하고, 무엇보다 가성비가 좋은 RTX 4090과 같은 RTX GPU를 시간당 대략 0.6 달러에 제공하고 있습니다. 다른 클라우드 서비스의 경우 일반적으로 기업에서 사용하는 것을 염두에 두고 서비스를 제공하고 있어 정해둔 시간 단위 만큼의 비용을 청구하는 온 디맨드가 잘 없고, 있다고 해도 기본 T4 GPU 부터 시작하기 때문에 생각보다 많은 비용이 듭니다.

2. 사용 편의성이 굉장히 좋다
저도 개인적으로 공부를 하면서 모델을 실제로 돌려봐야 하기 때문에 여러 클라우드 서비스를 알아보았습니다. 이전 회사에서 아마존 AWS를 사용했었기 때문에 AWS를 살펴보았는데 개인이 사용하기에는 일단 운영체제부터 CPU, RAM 등 설정해야 할 것이 너무 많고, 비용 계산이나 결제 방법 그리고 개인이 고성능의 GPU를 사용하기 위해선 특별히 또 무언갈 신청해야 하는 등 굉장히 불편합니다. 하지만 RunPod는 회원가입을 하고, 사용을 원하는 GPU를 선택하고, 운영체제와 필요한 라이브러리들이 설치된 template을 선택만하면 간단하게 사용하고자 하는 서버가 하나 생성이 됩니다.

사실 본인이 금액적으로 여유가 충분하다면 기업들이 많이 사용하는 아마존 AWS를 사용하는 것이 가장 큰 도움이 될 것입니다. 하지만 저는 그럴 정도로 여유가 많지 않고, 또 공부 목적으로 간단하게만 실행하면 되기 때문에 RunPod를 선택한 것 같습니다. 또한 본인이 IT 회사를 운영하거나 서버를 관리하는 직책이 아닌 이상 회사에서 직접 클라우드 서비스를 이용해 서버를 신청하는 일도 거의 하지 않을거고, 또 회사에 들어가서 일을 하다보면 어쩔 수 없이 서버의 설정을 한다든지 해야 하기 때문에 서버 관리나 서버에 대한 지식은 회사에서도 충분히 쌓을 수 있다고 생각합니다 그래서 굳이 개인적으로 공부를 한다든지 토이 프로젝트를 하는데 꼭 AWS를 사용할 필요는 없다고 생각합니다.

# 3. RunPod로 GPU 서버 pod 구축하기

그럼 이제 RunPod에서 GPU 서버를 사용하기 위한 pod 구축에 대해서 알아보도록 하겠습니다. 이번엔 RunPod에서 pod를 직접 구축해 보고 구축한 Pod와 VsCode를 연결하는 것까지 알아보도록 하겠습니다.

## 3.1 준비 작업

RunPod에 접속하기 위해서는 비밀번호 방식보다는 SSH 키를 사용하는 것이 보안상 안전하고 접속도 편리합니다.

1. 로컬 컴퓨터(내 PC)에서 SSH키 생성

	- Windows Powershell을 열고 아래 명령어를 입력합니다.

```bash
ssh-keygen -t ed25519 -C "runpod-key"
```

	- 위 명령어를 치고나면 다른 설정과 관련된 질문들이 나오는데 일단 기본값을 사용해도 되니 key 값이 나올 때까지 엔터를 눌러주도록 하겠습니다.
	- 완료되면 아래 이미지와 같이 "C:\Users\사용자명\.ssh\" 폴더가 생성되고 "C:\Users\사용자명\.ssh\id_ed25519" 파일이 생성되었다고 뜨면서 key 값이 출력됩니다.

<div align="center">
  <img src="/assets/images/mlops_engineering/runpod/1/key_generator_image.png" width="50%" height="40%"/>
</div>

	- 만약 key 값이 기억이 나지 않으시면 `type $env:USERPROFILE\.ssh\id_ed25519.pub`을 입력하면 key 값이 출력됩니다 "env" 뒤에 주소값은 해당 폴더로 가셔서 주소값을 복사해서 사용하시는게 정신건강에 이로우실 겁니다. 명령어를 출력하면 아래 화면과 같이 key 값이 출력됩니다.

<div align="center">
  <img src="/assets/images/mlops_engineering/runpod/1/key_print_image.png" width="50%" height="40%"/>
</div>

2. RunPod에 키 등록

RunPod 회원가입과 요금 충전등은 기본적으로 할 줄 아실 듯 하니 생략하도록 하겠습니다. 저는 구글 계정을 연동해서 회원가입을 했고, 해외 결제가 가능한 신용카드로 충전을 진행했습니다.

	- RunPod 웹사이트 접속 -> 왼쪽 사이드바에서 Setting을 찾아 클릭 -> SSH Public Keys 항목으로 이동합니다.
	- 복사한 키를 붙여넣고 Add Key를 클릭합니다.

<div align="center">
  <img src="/assets/images/mlops_engineering/runpod/1/runpod_ssh_key_setting.png" width="65%" height="40%"/>
</div>

## 3.2 RTX 4090 인스턴스(Pod) 대여

이제 실제로 서버를 빌려보도록 하겠습니다.

1. Community Cloud 혹은 Secure Cloud 선택

	- 옆의 사이드바 메뉴에서 Manage의 Pods 항목을 클릭합니다.
	- 위의 Select an Instance 의 두 번째 항목에서 Community Cloud 혹은 Secure Cloud 중 하나를 선택합니다.

<div align="center">
  <img src="/assets/images/mlops_engineering/runpod/1/runpod_gpu_select.png" width="65%" height="40%"/>
</div>

2. GPU 선택
	- RTX 4090을 찾아서 클릭합니다.

3. 템플릿 및 디스크 설정
	- Template: RunPod PyTorch 2.x (최신 버전)을 선택하는 것이 가장 무난합니다.
	- Container Disk: 기본값(20GB)은 작을 수 있으므로 40GB 정도로 늘려줍니다.
	- Volume Disk: 이 부분이 실제 파일이 저장되는 곳입니다. LLM 모델을 다운로드 받으려면 넉넉해야 합니다. 최소 100GB 이상으로 설정해 주세요
	- Container 와 Volume Disk 설정은 template 설정에서 edit를 누르면 확인할 수 있습니다. 아래 이미지를 참조해 주세요

<div align="center">
	<img src="/assets/images/mlops_engineering/runpod/1/setting_template.png" width="65%" height="40%"/>
</div>

<br>

<div align="center">
	<img src="/assets/images/mlops_engineering/runpod/1/runpod_pod_setting.png" width="65%" height="40%"/>
</div>

4. 실행: 설정을 마치고 Deploy를 클릭합니다.

## 3.3 VS Code와 연결하기

Pod가 생성되고 `Running` 상태가 되면 연결할 수 있습니다.

1. 접속 정보 확인

	- 생성된 Pod의 Connect 버튼을 누릅니다.
	- `SSH over exposed TCP` 탭을 보면 아래와 같은 명령어가 보입니다.
		```bash
		ssh root@123.456.78.9 -p 12345 -i ~/.ssh/id_ed25519
		```

2. VS Code에 접속해서 연결을 진행합니다.

	- VS Code를 켜고 좌측 메뉴의 Extension에서 `Remote - SSH`를 검색해서 설치합니다.

		<div align="center">
			<img src="/assets/images/mlops_engineering/runpod/1/vscode_extension.png" width="65%" height="40%"/>
		</div>
	- 설치 후 VS Code를 재실행한 후 `F1`을 눌러 `Remote-SSH:Connect to Host...`를 선택합니다.
	- `Add New SSH Host`를 선택하고, RunPod에서 복사한 SSH 명령어를 붙여넣습니다.
	- 설정이 저장되면, 다시 `Connect to Host`를 눌러 방금 추가한 IP를 선택합니다.
		- 간혹 설정 저장이 되지 않는 경우도 있다고 합니다. 제가 그런 케이스였습니다. 이런 경우에는 다음과 같이 진행해 줍니다.
		- `F1`을 눌러 `Remote-SSH: Open SSH Configuration File`을 선택해 줍니다.
		- `C:USERPROFILE\.ssh\config` 파일을 선택해 줍니다.
		- 그리고 다음과 같이 등록을 해줍니다. (HostName, Port 는 본인 것을 사용하셔야 합니다.)
			```config
			Host RunPod-4090            # VS Code 목록에 표시될 별명 (아무거나 가능)
    		HostName 123.456.7.89   # RunPod 명령어의 @ 뒤에 있는 IP 주소
    		User root               # RunPod 명령어의 ssh 바로 뒤 계정 (보통 root)
    		Port 1000              # RunPod 명령어의 -p 뒤에 있는 숫자
    		IdentityFile ~/.ssh/id_ed25519  # RunPod 명령어의 -i 뒤에 있는 경로
			```
	- 새 창이 뜨면서 연결이 완료됩니다. (OS 선택 창이 뜨면 Linux 선택)	

## 3.4 기본 개발 환경 세팅

RunPod에 접속했다면 VS Code에서 터미널(단축키 Ctrl + \`)을 열고 아래 사항을 꼭 확인해야 합니다.

1. 작업 경로 이동(`/workspace`)

	- RunPod는 `/workspace` 폴더 안에 있는 파일만 영구 보존 됩니다.
	- 나머지 폴더에 설치한 라이브러리나 파일은 Pod를 껐다 켜면(Terminated가 아닌 Stop 후 Start) 초기화 될 수 있습니다.
	- 항상 터미널에서 아래 명령어로 이동 후 작업을 시작하세요.
		```bash
		cd /workspace
		```

2. 가상 환경 설정 (선택사항 이지만 추천)

	- 기본 시스템 Python을 써도 되지만, 꼬이는 것을 방지하기 위해 `conda`나 `venv`를 사용하는 것이 좋습니다.
	- RunPod의 PyTorch 템플릿은 보통 Conda가 설치되어 있습니다.

3. GPU 확인

	- 터미널에 `nvidia-smi`를 입력하여 RTX 4090이 정상적으로 잡히는지 확인합니다.
		<div align="center">
			<img src="/assets/images/mlops_engineering/runpod/1/nvidia_smi.png" width="65%" height="40%"/>
		</div>

## 3.5 비용 관리

RunPod는 클라우드 시스템이므로 시간당 관리가 핵심입니다.

- Stop(일시 정지)
	- GPU 사용 요금은 멈추지만, 디스크(Volume) 보관 요금은 계속 나갑니다. (매우 저렴하긴 합니다.)
	- 나중에 다시 켜서 작업을 이어서 할 수 있습니다.
- Terminate(삭제)
	- Pod와 저장된 데이터가 모두 삭제됩니다. 더 이상 과금되지 않습니다.
	- 작업이 완전히 끝났거나, 중요 데이터를 로컬로 백업했다면 반드시 Terminate 해야 합니다.

공부나 토이 프로젝트를 하다보면 중간에 끊길 수 있습니다. 학습된 모델이나 구현하고자 한 모델을 구현한 후 테스트를 한 이후 모델을 따로 저장한 이후에 꼭 Terminate 를 해야 하며, 아직 해야할 작업이 남아 있다면 디스크 보관 요금이 매우 저렴하기 때문에 일주일 정도는 괜찮으니 Stop과 Start 로 생성한 Pod를 유지하는게 좋습니다.

마지막 팁으로 모델 학습은 언제 끝날지 알 수 없습니다. 하지만 대략적으로 알고 있는 경우 혹시나 깜빡하고 Pod를 켜놓을 수도 있으니 터미널에서 아래 리눅스 명령어로 자동 종료를 걸 수 있습니다.

```bash
# 3시간(180분) 뒤에 시스템을 끄도록 예약 (Pod가 Stop 상태로 전환됨)
sudo shutdown -h +180
```

# 마치며

RunPod 사용방법에 대해서 알아보았습니다. 이번엔 단순히 Pod를 생성하고 VS Code와 연결하는 것까지만 알아보았습니다. 다음엔 Pod와 VSCode에 github를 연동하고, VSCode에서 Jupyter Notebook을 이용해 실제 모델을 구현해보고 학습까지 진행해 보도록 하겠습니다.

긴 글 읽어주셔서 감사드리며, 본문 내용 중에 잘못된 내용이나 오타, 궁금하신 사항이 있으실 경우 댓글 달아주시기 바랍니다.