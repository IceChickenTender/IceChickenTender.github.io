---
title: "[Github_Blog] 4. 포스팅 글 써보기"
categories:
  - Github_Blog
tags:
  - Github_Blog
toc: true
toc_sticky: true
toc_label: "포스팅 글 써보기"
---

# 포스팅 글 써보기

이제 블로그 포스트를 쓰기 위한 준비는 얼추 갖추어 졌습니다.   
블로그 포스트 쓰는 방법에 대해서 알아보도록 하겠습니다.

## 1. toc 다루기
toc은 목차와 같은 기능으로 MarkDown 문법의 헤더(Header)와 연동됩니다.   
toc에 연동된 헤더(Header)가 등록이 되고 등록된 헤더(Header)를 클릭하면 클릭한 헤더(Header)로 이동합니다.

- toc 구성요소
	1. toc : 포스트에서 toc을 사용할 것인지 결정하는 변수로 `true` 혹은 `false` 값을 이용합니다.
		- true : toc을 사용함
		- false : toc을 사용하지 않음
	2. toc_sticky : toc을 포스트의 오른쪽 상단에 고정할 것인지 결정하는 변수로 `toc`과 같이 `true` 혹은 `false` 값을 이용합니다.
		- true : toc_sticky 기능을 사용함
		- false : toc_sticky 기능을 사용하지 않음
	3. toc_label : 사용할 toc에 레이블을 붙여주는 것으로 적용하고자 하는 문자열 값을 넣어주면 됩니다.
		- 적용할 문자열은 `""` 로 감싸주어야 합니다.
- 예시는 다음 이미지와 같습니다.

<figure class="half">
	<img src="/assets/images/github_blog/4/toc_example_1.png" style="width:50%">
	<img src="/assets/images/github_blog/4/toc_example_2.png" style="width:50%">
</figure>

## 2. \_posts 폴더 추가

\_posts 폴더를 만들고 이 폴더에 md 파일을 추가하여 포스트를 작성할 수 있습니다.   
저는 \_posts 폴더에 카테고리별로 폴더를 나누어 관리를 하고 있습니다.

<div align="center">
	<img src="/assets/images/github_blog/4/posts_folder.png" width="50%" height="50%" title="1234" alt="1234"/>
</div>

## 3. 포스터 양식 확인

categories, tags 형식이 맞지 않으면 포스트가 제대로 업로드 되지 않음 그래서 처음 포스트를 작성한다면 ./docs/posts 에 있는 아무 포스트 내용을 복사해서 필요없는 부분을 지우고 사용하는 것을 추천

<figure class="half">
	<img src="/assets/images/github_blog/4/1.png" style="width:50%">
	<img src="/assets/images/github_blog/4/2.png" style="width:50%">
</figure>


