---
title: "[Github_Blog] 5. minimal mistake 본문 영역 및 글자 크기"
categories:
  - Github_Blog
tags:
  - Github_Blog
toc: true
toc_sticky: true
toc_label: "본문 영역 및 글자 크기"
---

# minimal mistake 본문 영역 및 글자 크기 설정하기

## 1. WIDTH 조절하기
아래 이미지와 같이 본문 영역을 조절하고 싶을 경우 진행

<img src="/assets/images/github_blog/5/1.png" width="50%" hegiht="40%">

수정 파일 : ./_sass/_minimal-mistakes/_variables.scss

<img src="/assets/images/github_blog/5/2.png" width="50%" hegiht="40%">

이미지에 있는 변수들을 설정하여 WIDTH 조절
local 사이트를 띄운 다음 자신에게 맞는 크기를 찾아서 적용하면 됨

## 2. BLOG FONT 조절하기
수정 파일 : ./_sass/_minimal-mistakes/_reset.scss
아래 이미지 부분에서 수정하여 적용

<img src="/assets/images/github_blog/5/3.png" width="50%" hegiht="40%">

## 3. 포스트 밑줄 제거
처음 블로그를 세팅하면 각 포스트가 하이퍼링크로 처리되어 밑줄이 있는 것을 확인할 수 있으며, 이것이 거슬릴 경우 제거 가능

<img src="/assets/images/github_blog/5/4.png" width="50%" hegiht="40%">

수정 파일 : ./_sass/_minimal-mistakes/_base.scss
수정 내용은 아래 이미지를 참고

<img src="/assets/images/github_blog/5/5.png" width="50%" hegiht="40%">

