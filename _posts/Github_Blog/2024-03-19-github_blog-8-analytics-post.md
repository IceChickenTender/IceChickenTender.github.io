---
title: "[Github_Blog] 8. 방문자 통계 내기"
categories:
  - Github_Blog
tags:
  - Github_Blog
toc: true
toc_sticky: true
toc_label: "방문자 통계 내기"
---

# 방문자 통계 내기

## 1. Gogle Analytics 등록하기
- Google Analytics에 접속하면 무료로 시작할 수 있습니다.
  - Google 로그인 후 계정 이름 설정
    - 계정 이름은 크게 영향을 끼치지 않으므로 자유롭게 작성
  - 블로그 주소를 속성 이름으로 등록하고, 시간대를 설정

## 2. tracking ID 찾기
- Analystics의 가장 하단에 있는 관리 버튼을 클릭
- 관리자-사용자 화면에서 데이터 스트림 버튼 클릭
- 자신의 github blog 주소를 찾아서 클릭
- 측정 ID에 있는 태그 값 가져오기
- _config.yml 수정하기 (아래 이미지 처럼 수정)
  - 수정 후 git push

<img src="/assets/images/github_blog/8/1.png" width="50%" hegiht="40%">

## 3. Analytics Test
- 연결 후 블로그에 접속하면 아래 이미지와 같이 업데이트 됨

<img src="/assets/images/github_blog/8/2.png" width="50%" hegiht="40%">