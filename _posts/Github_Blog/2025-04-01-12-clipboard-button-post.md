---
title: "[Github_Blog] Minimal Mistakes 코드 블록 복사 기능 추가"
categories:
  - Github_Blog
tags:
  - Github_Blog
toc: true
toc_sticky: true
toc_label: "Minimal Mistakes 코드 블록 복사 기능 추가"
---

이번엔 제가 사용하고 있는 github blog 테마인 Minimal Mistakes 에서 코드 블록을 복사할 수 있는 기능 구현에 대해 알아보고자 합니다. 최근에 코딩 테스트를 위한 알고리즘 공부도 하고, 공부한 알고리즘에 대한 문제를 풀면서 해당 문제 리뷰를 하고자 하다 예제 샘플을 올렸는데 이걸 매번 드래그해서 복사하자니 저도 그렇고 제 블로그를 보시는 분들도 귀찮겠다는 생각이 들어 구현을 해보게 되었습니다. 그럼 시작해 보겠습니다.

# 구현 과정

Minimal Mistakes 의 rouge highlighter 기능을 사용한다는 전제로 구현하였습니다. rouge highlighter 는 `%` 를 이용해 하이라이팅 등의 기능을 하는 것을 말합니다.

## 복사 버튼 HTML 파일 만들기

`/_includes/code-header.html` 파일 추가 후에 아래 내용을 붙여 넣습니다.

{% include code-header.html %}
```html
<div class="code-header">
  <button class="copy-code-button" title="Copy code to clipboard">
    <img class="copy-code-image" src="/assets/images/copy.png" />
  </button>
</div>
```

`src` 에 있는 이미지의 경우 직접 넣어주셔야 합니다. 저는 Flaticon 같은 곳에서 찾아서 대충 스크린샷 찍어서 `/assets/images/copy.png` 로 직접 저장해 주었습니다.

## CSS 를추가하여 위치와 스타일 적용하기

`/_sass/minimal-mistakes/_page.scss` 파일의 맨 밑 부분에 아래 코드를 넣어 주세요

{% include code-header.html %}
```css
.code-header {
  position: relative;
  top: 50px;
  right: 10px;
  z-index: 1;
}

.copy-code-button {
  float: right;
  cursor: pointer;
  background-color: $code-background-color;
  padding: 5px 5px;
  border-radius: 5px;
  width: 30px;
  height: 35px;
  margin-top: -40px;
  border: none;
}

.copy-code-button:focus {
  outline: none;
}

.copy-code-button:hover {
  background-color: #0f1214;
}

.copy-code-image {
  filter: invert(1);
}

```

## 클립보드에 코드를 복사하도록 해주는 Javascript 코드 추가하기

`/assets/scripts` 디렉토리에 copyCode.js 라는 이름으로 파일을 생성해 주고 아래 내용을 넣어 줍니다.

{% include code-header.html %}
```js
const codeBlocks = document.querySelectorAll(
  ".code-header + .highlighter-rouge"
);
const copyCodeButtons = document.querySelectorAll(".copy-code-button");

copyCodeButtons.forEach((copyCodeButton, index) => {
  const code = codeBlocks[index].innerText;
  let id;

  copyCodeButton.addEventListener("click", () => {
    window.navigator.clipboard.writeText(code);

    const img = copyCodeButton.querySelector("img");
    img.src = "/assets/images/check.png";

    if (id) {
      clearTimeout(id);
    }

    id = setTimeout(() => {
      img.src = "/assets/images/copy.png";
    }, 2000);
  });
});
```

그리고 `/_includes/footer.html` 에 copyCode.js 에 대한 내용을 아래와 같이 추가해 줍니다. 파일 최하단에 추가해 주시면 됩니다.

```html
<script src="/assets/scripts/copyCode.js"></script>
```

## Jekyll Liquid 태그를 이용해 코드 블록에 복사 버튼 추가 하기

아래 이미지와 같이 코드 블록에 Jekyll Liquid 태그를 이용해 코드 블록에 복사 버튼을 추가하면 됩니다.

<div align="center">
<img src="/assets/images/github_blog/12/1.png" width="50%" hegiht="40%">
</div>

<br>

# 최종 결과

아래와 같이 코드 블록 오른쪽 상단에 작은 버튼이 생긴 것을 확인할 수 있고, 이 버튼을 클릭하면 코드 블록의 내용이 복사가 되는 것을 확인할 수 있습니다.

<div align="center">
<img src="/assets/images/github_blog/12/2.png" width="50%" hegiht="40%">
</div>