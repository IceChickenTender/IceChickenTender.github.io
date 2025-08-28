---
title: "[Github_Blog] Minimal Mistakes 코드 블럭 자동 줄바꿈"
categories:
  - Github_Blog
tags:
  - Github_Blog
toc: true
toc_sticky: true
toc_label: "Minimal Mistakes 코드 블럭 자동 줄바꿈"
---

포스트를 작성하면서 코드 블럭에 길이가 긴 내용을 쓸 때 줄바꿈이 되지 않고, 옆으로 스크롤바가 생기면서 코드 블럭 안의 내용을 한 번에 확인할 수가 없는 문제가 발생했고, 해당 문제를 해결한 과정을 포스트로 기록하기 위해 이 포스트를 작성하게 되었습니다.

# 1. 개요

포스트를 작성하다가 코드 블럭 안에 긴 내용을 작성하게 되었는데 아래 이미지와 같이 코드 블럭 안의 내용을 한 번에 확인할 수 없는 문제가 발생하였습니다.

<div align="center">
<img src="/assets/images/github_blog/13/code_block1.png" width="70%" hegiht="60%">
</div>

---

# 2. 문제 원인

문제의 원인은 Minimal Mistakes 테마에서는 기본적으로 코드 블럭에 `overflow-x: auto;` 스타일이 적용되어 있어서, 코드가 길어지면 자동으로 가로 스크롤바가 생기도록 되어 있었습니다. 그래서 문제를 해결하기 위해선 CSS 를 커스터마이징해서 스크롤바 없이 코드 블럭 안의 내용 전체가 블럭 크기 안에서 줄바꿈 되도록 해야합니다.

---

# 3. 해결 방법

github blog 를 관리하고 있는 디렉토리로 가서 `_sass/minimal-mistkaes/_custom.scss` 또는 최종적으로 scss import 를 진행하는 `assets/css/main.scss` 파일에 아래와 같은 내용을 추가해 주세요


```scss
// 코드 블럭 자동 줄바꿈 적용
pre {
  white-space: pre-wrap;   // 기본 pre 대신 줄바꿈 허용
  word-wrap: break-word;   // 긴 단어도 줄바꿈
  overflow-x: visible;     // 가로 스크롤 제거
}
```

저는 `assets/css/main.scss` 파일에 수정 내용을 적용하였습니다만 만약 `_sass/minimal-mistakes/_custom.scss` 에 수정을 하였는데 적용이 되지 않는다면 `assets/css/main.scss` 를 확인해 보시고 만약 아래와 같이 `import`가 되어 있지 않다면 `import` 시켜주시기 바랍니다.

```scss
@import "minimal-mistakes";
```

적용 후에는 아래 이미지와 같이 코드 블럭 안의 전체 내용을 한 번에 볼 수 있고, 좌우 스크롤바도 사라진 것을 확인할 수 있습니다.

<div align="center">
<img src="/assets/images/github_blog/13/code_block2.png" width="70%" hegiht="60%">
</div>


