---
title: "[Github_Blog] _config.yml 파일 수정하기"
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

<div align="center">
  <img src="/assets/images/github_blog/2/config_yml_basic_info.png" width="90%" height="90%" title="config_yml_1" alt="config_yml_basic_info"/>
</div>

&nbsp;

## 3. 프로필 영역 설정
- 수정파일 : github_blog_path/\_config.yml
- 블로그 좌측에 보여지는 프로필 영역을 설정할 수 있습니다.
- 저는 프로필 사전 보관을 github_blog_path/assets/images 폴더를 따로 만들어서 진행하였습니다. 저 처럼 하지 않으셔도 되고 assets 폴더를 바로 사용하셔도 됩니다.
- E-mail 기재 시 확인할 수 있도록 mailto를 붙여주셔야 합니다.

<div align="center">
  <img src="/assets/images/github_blog/2/config_yml_profile.png" width="90%" height="90%" title="config_yml_profile" alt="config_yml_profile"/>
</div>

&nbsp;

## 4. 하단 프로필 영역 설정