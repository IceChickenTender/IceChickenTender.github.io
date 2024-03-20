---
title: "[Github_Blog] 6. 파비콘(Favicon) 설정하기"
categories:
  - Github_Blog
tags:
  - Github_Blog
toc: true
toc_sticky: true
toc_label: ""
---

# 파비콘(Favicon) 설정하기

## 1. 파비콘(Favicon) 이란?
- 파비콘(Favicon)이란, 인터넷 웹 브라우저의 주소창에 표시되는 대표 아이콘입니다.
- 세팅을 해주지 않으면 크롬 기준으로 지구본 모양이 나오게 됩니다.
- 꾸미는 것에 크게 관심이 없으신 분이라면 굳이 설정을 해주시지 않아도 되지만 각자의 개성을 나타내면 좋으니 꾸미는 것을 추천 드립니다.

## 2. 원하는 이미지 찾기
- 금손이시라면 원하는 이미지를 직접 만들어도 되고, 그게 아니라면 찾아서 사용하셔도 됩니다.
- 유용한 이미지 공유 사이트들은 다음과 같습니다.
  - flaticon : https://www.flaticon.com/
  - lcoon-mono : https://icoon-mono.com/
  - Thenounproject : https://thenounproject.com/
  - lconfinder : https://www.iconfinder.com/
  - Freepik : https://www.freepik.com/
  - FreeVectors : https://www.freevectors.net/

## 3. 파비콘 아이콘 만들기
- https://realfavicongenerator.net/에 접속해서 `Select your Favicon image`를 클릭해서 원하는 이미지를 넣어주세요.
- `Favicon Generator. For real.` 창이 뜬다면 맨 아래의 Generate your Favicons and HTML code를 눌러주세요
- Dwonload your package 옆의 Favicon package 버튼을 클릭해서 압축 파일을 받아주시고 압축파일을 해제해서 ./assets/logo.ico 라는 폴더에 정리해 주세요
- 정리 후의 이미지는 아래와 같습니다.

<img src="/assets/images/github_blog/6/1.png" width="50%" hegiht="40%">

## 4. 파비콘 적용하기
- Generate your Favicons and HTML code를 통해 생성된 HTML 코드를 적용해 주세요
- 수정 파일 : /_includes/_head/custom.html
- href를 아래 이미지와 같이 수정해서 적용해 줍니다.
