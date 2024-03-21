---
title: "[jenkins] jenkins 빌드 시 발생하는 Maven Cannot Find Symbol Error 처리 방법"
categories:
  - jenkins
tags:
  - jenkins
toc: true
toc_sticky: true
toc_label: "Cannot Find Symbol"
---

이번엔 회사에서 수정한 어떤 모듈의 라이브러리가 잘 동작하는지 테스트를 위해 jenkins 빌드를 진행했지만 Maven의 `Cannot Find Symbol` 에러가 발생했습니다. 
다음에도 비슷한 에러가 발생했을 때 즉각적으로 대처를 하기위해 이 에러가 왜 발생했고 어떻게 처리했는지에 대해서 블로그에 이와 관련된 포스트를 작성해 보려고 합니다.

해결방법을 바로 보고자 한다면 오른쪽 상단에서 `3. 에러 해결 방법`을 클릭하시면 해당 내용으로 바로 넘어갑니다.

## 1. 에러가 발생했을 때의 상황과 환경
회사에서 일을 하다가 발생한 거라 자세한 모듈 이름 대신 모듈명은 `A 모듈`, `B 모듈` 과 같이 사용하도록 하겠습니다.

1. 환경
- A모듈에서 DB property 옵션을 읽어서 처리하기 위해서 B모듈이 필요
- A모듈에 DB property 읽는 것에 수정할게 있어 B모듈에 새로운 클래스를 추가

2. 상황
- jenkins는 nexus와 연동되어 있으며, jenkins 빌드 시에는 maven 빌드를 사용
- B모듈의 1.0.2 버전이 이미 올라가 있었으나 B모듈의 버전업이 힘든 상황이라 nexus에 있던 기존 1.0.2 버전의 B모듈을 지우고 재업로드를 진행
- jenkins 빌드 시 A모듈에서 B모듈에 추가한 클래스를 찾지 못해 `Cannot Find Symbol` 에러가 발생   
   <img src="/assets/images/jenkins/1/1.png" width="50%" hegiht="40%">

## 2. Cannot Find Symbol
`Cannot Find Symbol`에러가 발생하는 이유로 제가 아는 것은 두 가지가 있습니다 추가로 더 발생하는 이유가 있다면 댓글에 써주시길 바랍니다.

1. 참조하는 라이브러리들 중에 같은 이름의 클래스가 있을 경우
2. 참조하는 라이브러리들 중에 참조하고자 하는 클래스, 메소드, 변수가 없을 경우

## 3. 에러 해결 방법

1. 같은 에러가 로컬 pc의 A모듈 빌드할 때도 발생을 하였고, 구글링 하는 중에 메이븐 저장소 문제일 것 같다고 판단
2. 로컬 pc의 메이븐 저장소에서 문제가 되는 B모듈 라이브러리를 지운 후에 A모듈 빌드를 진행하니 정상적으로 빌드가 됨

3. jenkins도 로컬 pc에서와 같이 진행을 하면 될 것 같다고 판단 하였지만 회사 jenkins는 물리적 서버가 아닌 클라우스 환경이고 권한이 없어 접근이 불가하여 메이븐 저장소에 있는 B모듈 라이브러리 수정이 불가능한 상황

4. jenkins project의 환경의 build step에서 execute shell로 삭제가 가능할 것으로 판단

5. build steps에 execute shell을 추가하기 전에 메이븐 저장소 경로 파악
    - 저장소 경로 파악 방법
      - Dashboard의 Jenkins관리의 시스템 설정으로 이동   
      <img src="/assets/images/jenkins/1/2.png" width="80%" hegiht="80%">
      - `Local Maven Repository`에 있는 경로 확인   
      Default면 `~/.m2/repository` 입니다.   
      <img src="/assets/images/jenkins/1/3.png" width="80%" hegiht="80%">

6. build steps에 execute shell 추가

    - execute shell 추가   
    `Add build step`에서 Execute Shell 클릭하면 추가 됩니다.   
    <img src="/assets/images/jenkins/1/4.png" width="50%" hegiht="40%">
    - 메이븐 저장소에서 라이브러리 삭제를 위한 명령어 추가   
      저는 execute shell을 오류가 발생하던 메이븐 빌드 실행 전에 실행하도록 추가해 주었습니다.   
      <img src="/assets/images/jenkins/1/5.png" width="50%" hegiht="40%">   
        -  execute shell로 rm -rf 명령이 실행된 모습   
        <img src="/assets/images/jenkins/1/6.png" width="50%" hegiht="40%">
    - 오류가 해결되고 빌드가 성공한 모습   
    <img src="/assets/images/jenkins/1/7.png" width="50%" hegiht="40%">

jenkins 빌드 시 maven 빌드에서 발생하는 `cannot find symbol` 에러에 대해서 알아보았습니다. 로컬 pc의 ide에서 발생하는 `cannot find symbol`은 해결이 쉽지만 jenkins와 같이 솔루션 배포할 때 말고는 잘 사용하지 않아 jenkins에 대해서 잘 알지도 못하고, 간단한 에러가 발생해도 로컬 pc의 ide와는 환경이 달라 해결에 애를 많이 먹긴했습니다.   

웹에서 검색을 해봐도 해당 내용에 대해 원하는 내용이 잘 없어 이번 기회에 이렇게 정리를 해보았습니다. 다른 분들에게도 도움이 되었으며 하고, 포스트에 수정해야 할 점이나 궁금하신 사항은 댓글 남겨주시면 감사하겠습니다.

긴 글 읽어주셔서 감사합니다.