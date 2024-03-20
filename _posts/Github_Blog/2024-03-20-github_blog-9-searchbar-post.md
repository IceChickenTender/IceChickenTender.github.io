---
title: "[Github_Blog] 9. 검색창 노출시키기"
categories:
  - Github_Blog
tags:
  - Github_Blog
toc: true
toc_sticky: true
toc_label: "검색창 노출시키기"
---

# 검색창 노출시키기

## 1. Google search Console
-  Google search Console은 google에서 검색을 했을 때 자신의 사이트가 보여질 수 있도록 하는 google의 서비스입니다.
-  https://search.google/search-console/about 에 접속해서 시작하기 버튼 클릭
-  도메인을 구매하지 않았기 때문에 url 접두어로 선택
   -  자신의 블로그 url을 입력
   -  웹사이트에 HTML 파일 업로드를 진행하여 소유권 확인
   -  다음 화면에서 HTML 파일 다운로드

<img src="/assets/images/github_blog/9/1.png" width="50%" hegiht="40%">

## 2. HTML 파일 세팅
- _config.yml 파일이 있는 곳에 HTML 파일을 위치 시키기
  - HTML 파일을 git에 푸쉬하면 Google search Console에 다음과 같은 창이 뜸
  
<img src="/assets/images/github_blog/9/2.png" width="50%" hegiht="40%">

## 3. sitemap.xml 만들기
- gooel 크롤러가 github blog url을 체크할 수 있도록 하는 sitemap.xml 만들어 주기
  - 위치는 _config.xml 파일이 있는 곳에 위치
  - 만들어 준 후에 git push

<img src="/assets/images/github_blog/9/3.png" width="50%" hegiht="40%">

## 4. robot.txt 만들기
- robots.txt는 검색 로봇에게 사이트 및 웹 페이지를 수집할 수 있도록 허용하거나 제한하는 국제 권고안입니다.
- sitemap.xml 파일이 있는 곳에 robts.txt 파일 만들어준 후 git push

<img src="/assets/images/github_blog/9/4.png" width="50%" hegiht="40%">

## 5. Google search Console에 최종 sitemap.xml 등록
- SiteMaps에서 자신의 블로그 주소에 sitemap.xml을 넣고 제출로 sitemap.xml 파일 등록
  
<img src="/assets/images/github_blog/9/5.png" width="50%" hegiht="40%">

- 구글에서 색인 생성하는데 3~5일 정도 소요되므로 이후에 `site:자신의 블로그 주소`로 검색해서 검색창에 노출되는지 확인
  
<img src="/assets/images/github_blog/9/6.png" width="50%" hegiht="40%">