---
title: "[MLOps/Engineering] RunPod - 2. Github 와 연동하기"
categories:
  - MLOps/Engineering
  - RunPod
tags:
  - MLOps/Engineering
  - RunPod
  
use_math: true
toc: true
toc_sticky: true
toc_label: "Github와 연동하기"
---

# 개요

저번 포스트에서는 RunPod에서 Pod를 생성해 PC에 있는 VSCode와 연동하는 것까지 알아보았습니다. 이번엔 github와 연동해서 github에 있는 코드를 불러와 구글 코랩과 같이 VSCode에서 Jupyter Notebook 으로 코드를 실행해 보고, huggingface의 klue/roberta-base 모델 학습까지 한 번 진행해 보도록 하겠습니다. 이번 포스트는 윈도우 환경에서 진행했으며 추후에 Mac이나 Linux에서 진행할 때는 따로 다루도록 하겠습니다.

# 1. github 연동하기

## 1.1 github 연동에 사용할 master key 생성

먼저 RunPod 전용으로 쓸 SSH 키를 내 컴퓨터에서 만듭니다.

1. PowerShell을 키고, key 파일을 생성할 폴더에 가서 아래 명령어를 실행해 github 연동에 사용할 master key 생성하기

	```bash
	# 경로를 현재 폴더로 지정하고(-f), 알고리즘은 ed25519(-t) 사용
	ssh-keygen -t ed25519 -C "runpod-master" -f ./runpod_key
	```

2. 파일 확인 : 폴더에 아래 두 개의 파일이 생겼는지 확인
	- runpod_key : 비밀키
	- runpod_key.pub : 공개키(github에 등록할 key 값이 들어 있는 파일)

## 1.2 github에 공개키 등록

만들어진 "자물쇠"를 github에 달아주는 작업입니다.

1. 공개키 내용 복사
	- 메모장이나 VSCode로 방금 만든 runpod_key.pub 파일을 엽니다.
	- 안에 있는 `ssh-ed25519 ...`로 시작하는 문자열 전체를 복사합니다.

2. github에 접속
	- github 로그인 -> 우측 상단 프로필 클릭 -> Settings로 들어갑니다.
	- 좌측 메뉴에서 SSH and GPG keys를 클릭합니다.
	- New SSH key 초록색 버튼을 클릭합니다.

3. 공개키 등록
	- Title : `RunPod Master Key`(알기 쉬운 이름 아무거나)
	- Key: 복사한 내용을 붙여 넣기
	- Add SSH key 클릭

등록을 마치면 아래 이미지와 같이 됩니다.

<div align="center">
  <img src="/assets/images/mlops_engineering/runpod/2/github_key_register.png" width="65%" height="40%"/>
</div>

## 1.3 Pod에 '비밀키' 넣기 (새로운 Pod를 만들 때마다 수행)

1. VSCode로 Pod 연결

2. 비밀키 Pod에 넣기
	- Pod에 연결한 이후 왼쪽의 EXPLORER을 `/workspace`와 연결해 줍니다.
	- 연결한 workspace에 비밀키 파일인 runpod_key 파일을 넣어줍니다.
	- `/root/.ssh/` 폴더가 있는지 확인해보고 폴더가 없다면 생성해 줍니다.
	- runpod_key 파일을 /root/.ssh/ 폴더로 옮겨주고 파일 이름을 `id_ed25519`로 바꿔줍니다.
		- id_ed25519로 바꿔주는 이유는 기본적으로 github가 id_ed25519라는 이름의 키를 찾기 때문에 이렇게 하면 설정을 더 건드릴 필요가 없습니다.

3. id_ed25519 파일의 권한 설정을 바꿔주기

리눅스는 보안상 키 파일의 권한이 너무 열려 있으면(누구나 읽을 수 있으면) 사용을 거부하며 에러가 발생합니다. 따라서 권한을 바꿔주어야 합니다.

4. VS Code에서 터미널에 열어 아래 명령어를 실행합니다.
	- 아래 명령은 root 계정만 해당 파일을 읽고 쓸 수 있게 하라는 명령입니다.

	```bash
	chmod 600 /root/.ssh/id_ed25519
	```

5. 확인 및 사용

- 터미널에 아래 명령어를 쳐서 `Hi [사용자명]! You've successfully authenticated...` 라는 메시지가 뜨는지 확인해 봅니다. 메시지가 뜬다면 성공입니다.

	```bash
	ssh -T git@github.com
	```

# 2. VS Code와 Github 연동하기

내 github에 있는 프로젝트를 RunPod의 `/workspace`로 가져오는 과정입니다.

1. VS Code에서 F1 또는 Ctrl + Shift + P 키를 누릅니다.

2. 입력창에 Git: Clone 이라고 치고 엔터를 누릅니다.

3. 주소 입력:
	- Clone from GitHub를 선택합니다. 그러면 아래 이미지와 같이 자신의 github repo 목록이 뜹니다. 여기서 사용하고자 하는 저장소를 선택해 줍니다.

		<div align="center">
  			<img src="/assets/images/mlops_engineering/runpod/2/github_repo_list.png" width="65%" height="40%"/>
		</div>

	- Github에서 복사한 주소를 직접 붙여넣어도 됩니다.

4. 저장 위치 선택
	- 폴더 선택 창이 뜨면 반드시 `/workspace` 폴더를 선택하고 OK를 누릅니다.

		<div align="center">
  			<img src="/assets/images/mlops_engineering/runpod/2/select_workspace.png" width="65%" height="40%"/>
		</div>

5. "Would you like to open the cloned repository?" 라고 물으면 Open을 누릅니다.

# 3. VS Code에서의 Git 사용 팁

1. VS Code에서의 브랜치 변경
	- VS Code의 왼쪽 아래를 보면 아래 이미지와 같이 현재 선택된 브랜치가 있습니다. 이 부분을 클릭해 줍니다.

		<div align="center">
  			<img src="/assets/images/mlops_engineering/runpod/2/vscode_git_branch_change.png" width="65%" height="40%"/>
		</div>

	- 그러면 아래 이미지와 같은 창이 뜨는데 여기서 사용하고자 하는 브랜치를 선택해 줍니다.

		<div align="center">
  			<img src="/assets/images/mlops_engineering/runpod/2/select_branch.png" width="65%" height="40%"/>
		</div>

2. VS Code에서의 git 사용

왼쪽 사이드바 메뉴를 보시면 "Source Control"이라는 메뉴가 있습니다. 이걸 클릭하시면 git에서 지원하는 기능들이 있어 그대로 사용하시면 됩니다.

<div align="center">
	<img src="/assets/images/mlops_engineering/runpod/2/git_source_control.png" width="65%" height="40%"/>
</div>

# 4. VS Code와 RunPod 의 Pod 에서 실제 모델 실행해 보기

일단 현재 저의 저장소에는 이전에 작업하던 ipynb 파일이 있고, 해당 파일에는 klue/roberta-base 모델과 KLUE의 STS 데이터셋(두 문장이 얼마나 유사한지에 대한 데이터셋)을 이용해 입력의 문장 임베딩을 잘 생성하도록 학습하는 예제 코드가 있습니다.

예제 코드 자체가 코랩 환경으로 세팅되어 있어 실행하다 보면 에러가 발생하지만 결국 잘 실행되는 것을 확인하였습니다.

아래 그림은 학습 데이터를 이용해 모델을 학습 하는 과정입니다.

<div align="center">
	<img src="/assets/images/mlops_engineering/runpod/2/embedding_model_train_check.png" width="65%" height="40%"/>
</div>

<br>

학습 전 성능은 아래와 같습니다.

<div align="center">
	<img src="/assets/images/mlops_engineering/runpod/2/prev_train_performance.png" width="65%" height="40%"/>
</div>

<br>

학습 후 성능은 아래와 같습니다.

<div align="center">
	<img src="/assets/images/mlops_engineering/runpod/2/after_train_performance.png" width="65%" height="40%"/>
</div>


# 마치며

RunPod에서 생성한 Pod와 VS Code에 Github를 연동하는 것에 대해서 알아보았습니다. 이제 그나마 저렴하게 딥러닝 모델 학습 혹은 토이 프로젝트를 진행할 준비가 되었습니다. 일단 여기까지 진행하면서 느낀 것은 RunPod가 비용이 저렴하긴 하지만 생각보다 여러 가지로 불편한 부분도 있는 듯 합니다. 아마도 제가 RunPod를 사용한지 얼마 안돼서 그런 듯 합니다. 이후에는 RunPod를 더 사용해 보면서 직접 template을 수정해 본다든지, 아니면 사용하면서 좀 더 쉽게 사용할 수 있는 여러 팁들이 생기게 된다면 그 내용들을 정리해서 포스트로 작성해보고자 합니다. 

긴 글 읽어주셔서 감사드리며, 본문 내용 중에 잘못된 내용이나 오타, 궁금하신 사항이 있으신 경우 댓글 달아주시기 바랍니다.
