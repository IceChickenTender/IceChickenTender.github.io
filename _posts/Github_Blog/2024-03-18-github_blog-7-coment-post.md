---
title: "[Github_Blog] 7. 댓글 기능 추가하기"
categories:
  - Github_Blog
tags:
  - Github_Blog
toc: true
toc_sticky: true
toc_label: "댓글 기능 추가하기"
---

# 댓글 기능 추가하기

## 1. DISQUS 가입하기
- 댓글로 사용할 수 있는 것들 중에는 disqus를 포함해 discourse, facebook, staticman, staticman_v2, utternaces, custem 등이 있습니다만 저는 한 번에 댓글을 모아볼 수 있고, 세팅과 예제가 많은 disqus로 시작했습니다.
- 우선은 https://disqus.com/에 접속해서 가입을 진행해 주시길 바랍니다.

## 2. shortname 찾기
- DISQUS 메인화면에서 HOME으로 이동
- `Add Disqus To Site`로 이동
  - Get Started 클릭
  - `I want to install Disqus on my site` 진행
  - 자신의 github blog 사이트 등록 진행
- shortname 가져오기
  - 등록한 자신의 github blog를 선택
  - Settings 항목으로 이동
    - 왼쪽의 SITE에서 General로 이동
  - 아래 이미지와 같이 Shortname에 있는 값을 사용하면 됨

<img src="/assets/images/github_blog/7/1.png" width="50%" hegiht="40%">

## 3. _config.yml 수정하기
- 수정 파일 : ./_config.yml
  - Comments 항목에서 수정
    - provider를 disqus로 수정 이 때 ""를 붙여야 함
    - shortname 수정
      - shortname은 이전에 알려주었던 방법으로 disqus에서 가져오며, ""는 붙이지 않음

## 4. We were unable to load Disqus 오류 해결
- disqus 설정에서 신뢰된 사이트에 자신의 블로그가 등록되지 않아서 발생한 문제
  - disqus 설정 -> advanced -> trusted Domains에 자신의 블로그 주소를 입력해주면 해결
    - https를 제거한 주소를 입력해 주어야 함
  - shortname 확인
    - shortname이 잘못되어 발생한 오류일 수 있으므로 shortname 확인

## 5. 댓글 기능 추가된 것 확인 
- 로컬에서는 댓글 기능 테스트가 힘드므로 git에 push한 후에 아래 이미지와 같이 되는지 확인
- 
<img src="/assets/images/github_blog/7/2.png" width="50%" hegiht="40%">