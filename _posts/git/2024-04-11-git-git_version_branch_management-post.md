---
title: "[Git] Git branch 관리 "
categories:
  - Git
tags:
  - Git
toc: true
toc_sticky: true
toc_label: ""
---

이번 포스트는 git branch 관리에 대한 내용입니다. 이러한 내용에 대해서 포스트를 쓰게 된 이유는 회사에서도 형상 관리 툴로 git을 사용하고 있고, 요즘엔 git을 사용하는게 선택이 아닌 필수이기 때문입니다. 또한 회사에서 git branch 관련해서 참고할만한 사이트가 있었고, 해당 사이트를 참고해서 개인적인 블로그 등에 정리를 하고자 마음을 먹고 있었기도 해서 이번 기회에 개인적으로 정리를 하고자 포스트를 작성하게 되었습니다.

### 0. 개요

Git 은 merging/branching 기능이 있어, 분산 작업과 합치는 작업이 용이하다. 하지만 무분별하게 branch를 늘리게 되면 하나의 프로젝트에서 가지가 뻗어 나온 것이 아닌 각 각의 브랜치가 하나의 프로젝트로써 작용하게 되므로 Git을 사용하는 의미가 퇴색되어 버립니다.

따라서 Git 을 사용할 때 Git의 장점을 최대한으로 살릴 수 있도록 사용해야 하며, 이러한 구조로 Git 을 사용할 수 있는 모델을 사용해야 합니다.

현재 널리 사용되고 있는 Git 관리 모델로는 `master`, `develop`, `hotfix`, `feature`, `release` 5개 브랜치만을 사용하는 모델이 널리 사용되고 있으며, 전체적인 모델 개요도는 다음과 같습니다.

<img src="/assets/images/git/1/1.png" width="60%" hegiht="40%">

### 1. 주요 Branch

주요 branch 로는 `master`와 `develop` 두 개의 branch 를 항상 유지 합니다.

제가 회사를 다니면서 `master`, `develop` branch 에서 발생한 특이 사항에 대한 것에 말씀을 드리자면, `Major.Middle.Miner.Bug` 와 같은 구조로 version 관리를 한다고 가정했을 때 `Major`, `Middle` 은 호환이 되지 않는 버전인데, 이러한 구조에서 4.4 버전 제품을 고객사에 팔고, 4.5 버전이 릴리즈가 되었는데 4.4에서는 있던 기능을 4.5에서는 제거했지만, 4.5에서 제거했던 기능이 4.4에서 버그가 발생하는 경우에는 어떻게 branch 관리를 해야하는지에 대한 논의가 있었습니다.

저의 선임은 주요 branch에 이전 버전 기능과 최신 버전 기능이 모두 존재해야 한다고 주장을 했고, 최대한 그렇게 되도록 관리를 했습니다. 하지만 이럴 경우에 `master`, `develop` branch 는 오류 투성이가 되어 정상적으로 사용할 수 없는 branch 가 되어 버렸고, 단순히 코드를 저장하기만 하고 사용할 수 없었습니다. 거기다 주요 branch 들이 제 기능을 하지 못하니 `master` branch 에서 뻗어져 나오는 `hotfix` branch 나 `develop` branch 에서 뻗어져 나오는 `release` branch 도 특정 commit 에서 branch 를 만드는 등 정상적이지 않게 작업을 진행했었습니다.

저는 이러한 경험을 토대로 추후에 제가 git branch 를 관리하게 되면서 저는 `master`, `develop` branch 는 최신 버전만을 유지하며, 이전 버전의 내용들은 과감히 버리는 방법을 택했습니다. 제가 선택한 방법이 적용되면서 제가 관리하던 프로젝트의 `master`, `develop` branch 는 제 기능대로 동작을 하였습니다. 그리고 여기서 한 가지 주의할 점으로는 이전 버전과 최신 버전에 모두 적용되는 수정 사항의 경우에는 주요 branch 에도 모두 반영을 해주어야 하는 꼼꼼함이 필요하다는 것이었습니다.

아마도 저와 같은 경험을 하시는 분들은 잘 없을 것으로 생각됩니다. 왜냐하면 최신 버전이 나오면 구버전에서 최신 버전으로 업그레이드를 진행하면 되지 왜 구버전을 관리를 해주는 거지 하고 생각을 하실테니까요 하지만 제가 다니는 회사의 제품 구조상 업그레이드가 쉽지 않고, 버전 별로 기능들의 차이가 심하게 나는 제품들이기 때문에 이러한 문제가 발생했었습니다. 그래서 저는 저와 같은 경험을 하시는 분들에게 도움을 드리고자 주요 branch 에서 겪은 저의 경험에 대한 내용도 적게 되었습니다.

<img src="/assets/images/git/1/2.png" width="30%" hegiht="35%">

#### master branch

`master` branch 는 배포할 코드를 보관하는 브랜치로 `develop` 에서부터 파생되는 `release`, `hotfix` branch 가 merge 되는 branch 입니다. `master` branch 에서 코드를 직접 수정하는 것은 최대한 피해야 하며, `release`, `hotfix` branch 가 merge 되고 `tag` 가 부착되는 branch 이기도 합니다.

#### develop branch

`develop` branch 는 현재 개발 중인 코드를 보관하는 branch 입니다. `develop` branch 에는 항상 최신 코드가 포함되어 있어야 하며, `develop` branch 가 안정되고, 새로운 버전 릴리즈를 진행해야 하면 `release` branch 를 `develop` branch 에서 생성하여 릴리즈를 진행 합니다.

### 2. 보조 Branch

보조 branch 로는 `feature`, `release`, `hotfix` branch 가 있습니다. 보조 branch 들을 생성할 때 branch 이름은 사용하는 git 툴에 따라 다른 듯 한데 저는 sublime merge 혹은 sourcetree 를 사용해봤을 때는 git flow finish 기능을 사용하기 위해서는 `보조브랜치/브랜치명` 으로 branch 명을 지정하시는 것을 추천 드립니다.

#### feature branch

`featrue` branch 는 곧 배포할 (다음 혹은 언젠가) 기능을 개발하는 branch 입니다. `feature` branch 는 보통 신규 기능 개발, 신규 개발 급의 리팩토링과 같은 작업을 진행할 때 생성하는 branch 입니다. `feature` branch 는 따로 분리된 branch 에서 개발을 진행하며 기능이 완료되는 시점에 `develop` branch 로 merge 합니다. 일반적으로 origin 혹은 remote 에 push 하지 않고 local 에만 생성하는 branch 입니다. 만약에 공동으로 작업을 해야 하는 경우라면 origin 혹은 remote 에 push 해서 작업 해도 됩니다. 이러한 feature 단위의 branch 를 통해서 releaes 대상 feature 를 선택하거나 버리기 용이하기 때문에 `develop` branch 에서 작업하기 보다는 `feature` branch 를 생성해서 작업하는 것을 권장합니다.

`feature` branch 작업 과정은 다음과 같습니다.

1. 새로운 기능 개발을 위해 `feature` branch 를 추가 branch 이름은 `feature/new` 
2. develop branch 에서 `feature/new`branch 를 생성
3. 개발을 진행하고, 개발이 끝나서 다음 release 에 기능을 추가하기로 했다면 `develop` branch 에 merge 합니다.

#### release

`release` branch 는 배포를 준비하는 branch 입니다. `dvelop` branch 에서 작업하던 코드를 release 하기에 앞서 버전 넘버 부여, 버그 수정, 검증 등 제품 release 전에 해야할 활동들을 하는 branch 입니다.

`release` branch 를 만드는 시점은 `develop` branch 가 배포할 수 있는 상태가 되었다고 판단이 되었을 때로, 이 때 배포하고자 하는 기능이 `develop` branch 에 merge 되어 있어야 하고, 이번에 배포되지 않을 기능의 경우에는 `release` branch를 만들 때까지 기다렸다가 `develop` branch 에 포함시켜야 합니다.

`release` branch 는 프로젝트의 버전 정책에 맞는 `release/버전넘버` 로 생성합니다. 

`release` branch 작업 과정은 다음과 같습니다.

1. `develop` branch 로부터 `release` branch 생성
2. 코드 내의 버전 정보와 같은 메타 데이터를 release 하는 것에 맞게 수정
3. release 를 위한 검증을 진행, 개발하면서 발견하지 못한 버그를 수정하기 위한 것으로 QA 를 진행해야 합니다.
4. release 버전 검증 중에 버그가 발견되면 수정하고, `release` branch 에서 수정합니다.
5. 3~4 를 반복하며 QA 를 진행합니다.
6. 더 이상 수정 사항이 없고, release 준비가 끝났다고 판단이 되면 release 를 진행합니다.
7. `release` branch 를 `master` branch 로 merge 합니다.
8. `release` branch 가 반영된 `master` branch 를 `develop` branch 로 merge 합니다
    - 참고로 `release` branch 가 반영된 `master` branch 대신 `release` branch 를 `develop` branch 로 merge 하는 경우도 있습니다만 git 툴에서 제공해 주는 git flow finish 에서는 `release` branch 가 반영된 `master` branch 를 `develop` branch 로 merge 하도록 되어 있습니다.
9. `master` branch 에 버전 정보를 포함한 `tag` 를 부착합니다.
10. `release` branch 를 삭제합니다.

#### hotfix branch

배포 이후에 버그가 발생하지 않는다면 정말 좋겠지만, 사소하거나 심각하거나 버그는 항상 존재하기 마련입니다. 또한 긴급하게 특정 기능을 포함시켜 배포를 진행해야 하는 경우도 있습니다. 그럴 때 `hotfix` branch 는 그러한 버그 수정 혹은 긴급 배포를 위한 목적으로 만들어지는 branch 입니다. 그리고 `hotfix` branch 는 별도의 branch 로 만들어 수정 작업을 진행하므로 한 개발자가 `hotfix` branch 에서 버그를 수정하는 동안 다른 개발자들은 `develop` branch 에서 개발을 계속 진행할 수 있습니다. hotfix 의 경우에도 버전 정보를 포함한 `tag` 를 부착해 주어야 하는데 `jenkins` 등과 같은 빌드 툴을 이용할 때 어떤 버전에서 버그가 수정되었는지를 알아야 하기 때문입니다.

<img src="/assets/images/git/1/3.png" width="30%" hegiht="35%">

`hotfix` branch 작업 과정은 다음과 같습니다.

1. 배포한 버전에서 버그가 발생하면 `master` branch 에서 `hotfix` branch 를 생성합니다.
2. 버그를 수정합니다.
3. `hotfix` branch 를 `master` branch 에 merge 합니다.
4. `develop` branch 에는 `hotfix` branch 가 반영된 `master` branch 를 merge 합니다.
5. `master` branch 에 `tag` 를 부착합니다.
6. `hotfix` branch 를 삭제합니다.

### 추가로

Commit 시에 다음과 같은 룰을 지키면 git 을 사용함에 있어 효율성이 극대화 됩니다.

1. 하나의 commit은 하나의 comment로 설명이 가능해야 한다. 조금 귀찮더라도 commit을 나누어 하는 것이 좋다.
    - 추후 history 를 확인해야 할 일이 생겼을 때 history 파악이 용이
    - 버그 fix 한 commit 을 공유할 때 다른 사람들이 이해하기가 쉬움
2. 팀 내의 공통으로 사용하는 formatter 가 있어야 합니다.
    - 코드 규칙 (Coding Convention) 이나 Version History 등을 관리할 때 공통으로 사용하는 format 이 없다면 conflict 가 자주 발생합니다.
3. `tag` 를 부착할 때에는 version 명은 `v1.0.0.0` 과 같은 형식은 사용합니다.
    - version 명에는 강제성이 없습니다만 다른 사람들이 자기가 빌드한 라이브러리 사용하는 것을 고려한다면 관례적으로 사용하는 방법을 사용하는 것이 좋고, 관례적으로 `tag` 에 version 명을 달 때에는 `v` 를 달아서 사용한다고 합니다.

### 마치며

이번엔 Git branch 관리 Model 에 대해서 알아보았습니다. 최근엔 제가 소개한 Model 이 Git branch 관리의 정성적인 모델이 되었습니다. 제가 정리한 내용이 다른 분들에게도 도움이 되었으면 좋겠고, 포스트 내용 중에 잘못된 내용이나 오타 등이 있다면 댓글 남겨주시고, 포스트 읽어주셔서 감사합니다!

### 참조

<https://amazingguni.github.io/blog/2016/03/git-branch-%EA%B7%9C%EC%B9%99>   
<https://nvie.com/posts/a-successful-git-branching-model/>

