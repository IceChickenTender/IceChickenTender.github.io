---
title: "[Github_Blog] 2. _config.yml 파일 수정하기"
categories:
  - Github_Blog
tags:
  - Github_Blog
toc: true
toc_sticky: true
toc_label: " _config.yml 수정 "
---

반갑습니다 minimal mistakes를 이용해 github blog를 구축해보기 두 번째 시간 입니다.
이번에는 자신만의 블로그로 꾸미기 위해 프로필 이미지, 블로그 이름 등을 꾸미기 위해 \_config.yml 파일에 대해서 알아보고 수정까지 진행해 보도록 하겠습니다.

# \_config.yml 수정하기

## 1. \_config.yml 이란?
\_config.yml 파일은 화면에 내 정보를 나타내기 위해 사용하는 파일로 블로그의 기본적인 정보들을 나타내 줄 수 있습니다.

&nbsp;

## 2. 기본 정보 설정
- 수정파일 : github_blog_path/\_config.yml
- 아래 이미지는 블로그 전반의 기본 사항들 입니다. 각 변수들을 직접 변경하면서 로컬서버에서 확인해 보세요
- 저 같은 경우에는 까먹을까 싶어서 주석을 달아 놓았습니다.

<script src="https://gist.github.com/IceChickenTender/05d37181e63c874201760f5c61295d52.js"></script>

&nbsp;

## 3. 프로필 영역 설정
- 수정파일 : github_blog_path/\_config.yml
- 블로그 좌측에 보여지는 프로필 영역을 설정할 수 있습니다.
- 저는 프로필 사전 보관을 github_blog_path/assets/images 폴더를 따로 만들어서 진행하였습니다. 저 처럼 하지 않으셔도 되고 assets 폴더를 바로 사용하셔도 됩니다.
- E-mail 기재 시 확인할 수 있도록 mailto를 붙여주셔야 합니다.

<script src="https://gist.github.com/IceChickenTender/7af19d869a35a4885d5521222a170029.js"></script>

&nbsp;

## 4. 하단 프로필 영역 설정
- 수정 파일 : github_blog_path/\_config.yml
- 프로필의 맨 하단에 있는 footer의 기재 사항 입니다.
- 좌측 프로필 영역과 동일한 데이터라 복사, 붙여넣기만 하면 금방 만들어지는 영역입니다.
- 현재 저는 이 부분은 사용하지 않습니다만 참고용으로 알려드립니다.

<script src="https://gist.github.com/IceChickenTender/8857ed19ba67349482ef89869cc33e90.js"></script>

&nbsp;

## 5. 첫 화면 게시물 개수 설정
- 수정 파일 : github_blog_path/\_config.yml
- 두 번째 줄에 위치한 `paginate`의 속성 값이 보여줄 개수 값 입니다.

<script src="https://gist.github.com/IceChickenTender/d22991551bf56455eaa810cdc8298dd5.js"></script>

&nbsp;

## 6. 유의 사항

프로필 정보를 수정하기 위해 \_config.yml 파일을 수정할 때 빠른 테스트를 위해 로컬 서버에서 하는 경우에는 바로 \_config.yml 파일 수정을 한다고 해서 바로 수정되지는 않습니다.
명령 프롬프트(cmd)로 실행 중인 서버를 종료하고 재기동해야 적용됩니다.

블로그를 방문해 주셔서 감사드리며   
오타나, 수정 사항 등이 있으시면 댓글 달아 주시면 감사하겠습니다!