---
title: "[Github_Blog] Category 세팅하기 "
categories:
  - Github_Blog
tags:
  - Github_Blog
toc: true
toc_sticky: true
toc_label: "Category 세팅"
---

안녕하세요 오늘은 포스트를 카테고리, 태그 등의 특정 항목으로 분류하도록 하는 세팅 방법에 대해서 알아 보고자 합니다.
세팅을 하게 되면 화면 오른쪽 상단에 분류 항목들이 뜨게되며, 분류 항목들을 클릭하게 되면 세팅한 분류 항목대로 포스트가 분류가 됩니다.
본인의 취향에 따라 여러가지 분류로 진행할 수 있는데 저는 Category, Tag, Year로 진행하도록 하겠습니다.

# Category 세팅하기

## 1.1 navigation.yml 수정
- 수정 파일 : github_blog_path/\_data/navigation.yml
- 클릭하면 게시글들을 분류 내용대로 정리한 url로 이동하게 하는 것으로 오른쪽 상단에 위치 합니다.

<script src="https://gist.github.com/IceChickenTender/41976dc17e73ac379d49979f006610e3.js"></script>

&nbsp;

## 1.2 원하는 page 수정
- 수정 파일 : github_blog_path/\_pages/원하는 md 파일 (아래 예시는 introduce.md 입니다)
- navigation.yml에서 설정한 분류 항목들에 대한 설정을 진행합니다.

<div align="center">
  <img src="/assets/images/intoruce.png" width="90%" height="90%" title="introduce" alt="introduce"/>
</div>