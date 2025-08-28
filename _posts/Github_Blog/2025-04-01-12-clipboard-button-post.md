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

이번엔 제가 사용하고 있는 github blog 테마인 Minimal Mistakes 에서 코드 블록을 복사할 수 있는 기능 구현에 대해 알아보고자 합니다. 특히 ChatGPT 를 쓰면서 GPT 가 보여주는 코드 블록처럼 구현을 진행해 보고자 합니다. 오른쪽 상단에 코드 블럭 복사 버튼을 만들고, 왼쪽 상단에는 코드 블럭에 사용된 언어 이름을 표시하도록 구현하고자 합니다. 그리고 구현 과정에 대해 포스트를 작성하게 된 계기는 코드 블럭 복사 버튼과 언어이름 표시와 관련해서 아무리 찾아봐도 자료가 없었고, 혹시나 해서 ChatGPT 에게 물어보아 GPT 의 도움을 토대로 구현을 하게 되었고, 이 과정을 다른 사람들에게도 공유를 해서 저 처럼 답답한 심정을 느꼈을 사람들에게 조금이나마 도움이 되었으면 해서 작성하게 되었습니다.

# 구현 과정

## 코드 블럭 복사 버튼 구현 과정

### 1. 코드 블럭에 적용할 복사 버튼을 위한 JavaScript 파일 생성 및 적용

포스트의 모든 코드 블럭에 복사 버튼을 자동으로 삽입하고, 클릭 시 해당 코드 내용을 클립보드로 복사하는 JavaScript 파일을 `/assets/scripts/` 디렉토리에 추가해 줍니다. 저는 파일 이름을 `copy-code-auto.js` 로 하였습니다. 코드 내용은 다음과 같습니다.

```js
/* /assets/scripts/copy-code-auto.js */
(function () {
  const blocks = document.querySelectorAll('div.highlighter-rouge, figure.highlight, div.highlight');

  blocks.forEach((block) => {
    // 컨테이너 결정
    const container = block.classList.contains('highlight') ? block : (block.querySelector('.highlight') || block);
    if (!container) return;

    // 버튼 중복 삽입 방지
    if (container.querySelector('.copy-code-button')) return;

    // 코드 엘리먼트
    const codeEl = container.querySelector('pre code') || container.querySelector('code');
    if (!codeEl) return;

    // 헤더(버튼) 생성
    const header = document.createElement('div');
    header.className = 'code-header';
    const btn = document.createElement('button');
    btn.className = 'copy-code-button';
    btn.type = 'button';
    btn.title = 'Copy code to clipboard';
    // 이미지 아이콘 쓰시려면 아래 한 줄로 교체
    // btn.innerHTML = '<img class="copy-code-image" src="{{ "/assets/images/copy.png" | relative_url }}" />';
    btn.textContent = '복사하기';

    header.appendChild(btn);

    // 컨테이너에 삽입 (상단)
    container.style.position = getComputedStyle(container).position === 'static' ? 'relative' : getComputedStyle(container).position;
    container.insertBefore(header, container.firstChild);

    // 클릭 복사
    btn.addEventListener('click', async () => {
      const text = codeEl.innerText;
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          const ta = document.createElement('textarea');
          ta.value = text;
          ta.style.position = 'fixed';
          ta.style.top = '-9999px';
          document.body.appendChild(ta);
          ta.focus(); ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
        }
        btn.textContent = '복사됨!';
        setTimeout(() => (btn.textContent = '복사하기'), 1200);
      } catch {
        btn.textContent = '복사실패';
        setTimeout(() => (btn.textContent = '복사하기'), 1200);
      }
    });
  });
})();
```

### 2. footer 에 스크립트 추가

`/includes/footer.html` 의 마지막 줄에 아래와 같이 추가해 줍니다. `footer.html` 은 사이트 공통 하단을 담당하는 파일로, 저작권/링크 같은 푸터 디자인뿐 아니라, 자신이 만든 JavaScript 파일으 불러와 기능을 확장하는데 사용됩니다. 그리고 js 파일을 `footer.html` 에 넣어주는 이유는 HTML 페이지가 위에서부터 아래로 순서대로 렌더링이 되는데, 스크립트를 `<head>`에 넣으면 DOM 요소들이 생성되기 전에 실행될 수 있어 에러가 납니다. `footer.html` 은 페이지 하단에 위치하기 때문에 모든 코드 블럭이 생성된 후 실행됩니다. 그러므로 `footer.html` 은 DOM 조작 스크립트를 넣기 적합하기 때문에 `footer.html` 에 넣어주는 것입니다.

```html
<script src="{{ '/assets/scripts/copy-code-auto.js' | relative_url }}" defer></script>
```

### 3. 버튼 추가를 위한 scss 파일 수정

`/sass/minimal-mistakes/_page.scss` 파일의 가장 마지막 부분에 아래와 같이 추가해 줍니다.

```scss
.code-header {
  position: relative;
  top: 0;  // 자동 주입 시 컨테이너 첫 줄에 오므로 0부터 시작
  z-index: 1;
  display: flex;
  justify-content: flex-end;
}

.copy-code-button {
  cursor: pointer;
  background-color: $code-background-color;
  color: #ffffff;
  padding: 4px 8px;
  border-radius: 6px;
  border: none;
  margin: .4rem;
  font-size: .8rem;
}

.copy-code-button:hover { background-color: #0f1214; }
```

## 코드 블럭 왼쪽 상단에 언어 이름 표시 구현 과정

### 1. highlighter 적용을 위한 `_config.yml` 파일 수정

highlighter 가 적용되지 않아 최종적으로 언어이름 표시가 되지 않을 수 있습니다. 자신의 github blog 디렉토리에 있는 `_config.yml` 파일에서 `Conversion` 과 `Markdown Processing` 을 찾고 아래와 같이 추가해 줍니다.

```yml
markdown: kramdown
highlighter: rouge

kramdown:
  input: GFM
  syntax_highlighter: rouge
  syntax_highlighter_opts:
    # 라인번호 쓰면 <table> 구조가 되기도 하므로 당장은 끔
    block: { line_numbers: false }
  # (선택) 미지정 언어 추측 방지
  # rouge:
  #   guess_lang: false

```

### 2. 언어 이름 표시를 위한 JavaScript 추가

코드 블럭 왼쪽 상단에 언어 이름을 자동으로 표시해주는 역할을 하는 JavaScript 가 필요합니다. 이 JavaScript 는 코드 블럭에 언어를 명시하면, 실제 HTML 출력 시 `class="language-xxx"` 형태가 붙는데, 그 정보를 읽어와서 배지를 달아주는 역할을 합니다. 저는 `/assets/js/` 에 `code-lang-label.js` 파일로 추가를 해주었습니다.

```js
/* assets/js/code-lang-label.js */
(function () {
  function onReady(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn);
    } else {
      fn();
    }
  }

  onReady(function () {
    const containers = document.querySelectorAll('div.highlighter-rouge, figure.highlight, div.highlight');
    if (!containers.length) return;

    const DISPLAY = {
      js: 'JavaScript', javascript: 'JavaScript',
      ts: 'TypeScript', typescript: 'TypeScript',
      py: 'Python', python: 'Python',
      bash: 'Bash', shell: 'Shell', sh: 'Shell', zsh: 'Zsh',
      html: 'HTML', css: 'CSS',
      json: 'JSON', yaml: 'YAML', yml: 'YAML',
      md: 'Markdown', markdown: 'Markdown',
      java: 'Java', c: 'C', cpp: 'C++', cxx: 'C++',
      csharp: 'C#', 'c#': 'C#',
      go: 'Go', rust: 'Rust', kotlin: 'Kotlin',
      php: 'PHP', ruby: 'Ruby', r: 'R', swift: 'Swift',
      sql: 'SQL', scala: 'Scala', perl: 'Perl', dart: 'Dart',
      scss: 'SCSS', sass: 'Sass'
    };

    function pickLang(container) {
      // 1) 컨테이너 클래스에서 language-xxx
      for (const cls of container.classList) {
        if (cls.startsWith('language-')) return cls.slice(9);
      }
      // 2) 내부 code/pre, (라인번호 구조까지 포함)
      const code =
        container.querySelector('pre code') ||
        container.querySelector('code') ||
        container.querySelector('td.code pre code');
      if (code) {
        for (const cls of code.classList) {
          if (cls.startsWith('language-')) return cls.slice(9);
        }
      }
      return null;
    }

    containers.forEach((outer) => {
      const container = outer.classList.contains('highlight')
        ? outer
        : (outer.querySelector('.highlight') || outer);

      if (!container) return;

      // 중복 방지
      if (container.dataset.langBadgeApplied === '1') return;

      const langRaw = pickLang(outer) || pickLang(container);
      if (!langRaw) return;

      const label = DISPLAY[langRaw.toLowerCase()] || langRaw.toUpperCase();

      // 코드 헤더가 있으면 헤더 왼쪽에 배지 엘리먼트 삽입
      const header = container.querySelector('.code-header');
      if (header) {
        header.style.display = 'flex';
        header.style.alignItems = 'center';
        header.style.justifyContent = 'space-between';

        // 이미 배지가 있으면 스킵
        if (!header.querySelector('.code-lang-badge')) {
          const badge = document.createElement('span');
          badge.className = 'code-lang-badge';
          badge.textContent = label;

          // 버튼이 오른쪽에 가도록, 배지는 맨 앞에 삽입
          header.insertBefore(badge, header.firstChild);
        }
        // === pre 패딩을 읽어 header 좌우 패딩을 '강제로' 동일하게 맞춤 ===
        try {
          const pre = container.querySelector('pre');
          const csPre = pre ? getComputedStyle(pre) : null;
          const csCont = getComputedStyle(container);

          // pre의 padding-left가 0이면 컨테이너 padding-left를 사용
          const padL = csPre && csPre.paddingLeft !== '0px'
            ? csPre.paddingLeft
            : (csCont.paddingLeft !== '0px' ? csCont.paddingLeft : '1rem');

          const padR = csPre && csPre.paddingRight !== '0px'
            ? csPre.paddingRight
            : (csCont.paddingRight !== '0px' ? csCont.paddingRight : '1rem');

          header.style.paddingLeft = padL;
          header.style.paddingRight = padR;
        } catch (e) {
          // 문제가 있어도 페이지 죽지 않게 무시
        }

      } else {
        // 헤더가 없으면 부모에 data-lang 달고 ::before 로 표시
        outer.classList.add('has-lang-badge');
        outer.setAttribute('data-lang', label);
        const cs = getComputedStyle(outer).position;
        if (cs === 'static') outer.style.position = 'relative';
      }

      container.dataset.langBadgeApplied = '1';
    });
  });
})();

```

### 3. footer.hmlt 에 추가한 JavaScript 적용

`/includes/footer.html` 에 추가한 JavaScript 파일을 적용해 줍니다. `footer.html` 파일의 가장 아래 부분에 아래와 같이 추가해 줍니다.

```html
<!-- 코드 블럭 코드 언어 배치 삽입에 필요 -->
<script src="{{ '/assets/js/code-lang-label.js' | relative_url }}" defer></script>
```

### 4. 버튼 추가를 위한 scss 파일 추가

버튼 추가를 위한 scss 파일을 추가합니다. `/assets/css/main.scss` 에 추가해줘도 되지만 그렇게 되면 중복되는 내용들 때문에 나중에 적용하고자 하는 내용이 제대로 적용되지 않을 수 있어서 `/_sass/minimal-mistakes/_custom.scss` 라는 새로운 파일에 추가를 해주고자 합니다. 내용은 아래와 같습니다.

```scss
/* 헤더와 코드 본문의 좌우 패딩을 '같게' 맞춰 정렬 */
.highlighter-rouge > .highlight > .code-header,
figure.highlight > .code-header,
.highlight > .code-header {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  width: 100% !important;
  box-sizing: border-box !important;

  /* 패딩은 JS에서 동기화해 주지만, 기본값을 넣어 둡니다 */
  padding-left: 1rem !important;
  padding-right: 1rem !important;

  /* (선택) 배경을 코드 블록과 동일하게 하고 싶으면 유지 */
  background: $code-background-color;
  position: relative;
  z-index: 3;
}

/* 배지는 header 패딩에 맞춰 정렬 → margin 제거 */
.code-header .code-lang-badge {
  margin: 0 !important;
  padding: .15rem .5rem;
  font-size: .75rem;
  line-height: 1;
  border-radius: .4rem;
  background: rgba(0,0,0,.35);
  color: #fff;
  border: 1px solid rgba(255,255,255,.25);
  user-select: none;
}

/* 헤더가 없을 때를 위한 보강(그대로 유지) */
.highlight, .highlighter-rouge, figure.highlight { position: relative; }
.has-lang-badge::before {
  content: attr(data-lang);
  position: absolute;
  top: .6rem;
  left: 1rem;         /* 헤더 기본값과 맞춤 */
  z-index: 2;
  display: inline-block;
  padding: .15rem .5rem;
  font-size: .75rem;
  line-height: 1;
  border-radius: .4rem;
  background: rgba(0,0,0,.35);
  color: #fff;
  border: 1px solid rgba(255,255,255,.25);
  user-select: none;
}

/* 코드 본문 패딩(테마가 다르면 0일 수 있으니 기본값 보강) */
.highlight pre {
  padding-left: 1rem !important;
  padding-right: 1rem !important;
  box-sizing: border-box;
}

```

# 최종 결과

최종 결과는 아래 이미지와 같습니다. 아래 이미지와 같이 오른쪽 상단에는 복사하기 버튼이, 왼쪽 상단에는 코드 블럭에 적용된 언어이름이 표시 되는 것을 확인할 수 있습니다.

<div align="center">
<img src="/assets/images/github_blog/12/result.png" width="70%" hegiht="60%">
</div>