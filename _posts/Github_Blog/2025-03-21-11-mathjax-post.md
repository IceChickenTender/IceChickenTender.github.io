---
title: "[Github_Blog] GitHub 블로그에서 마크다운 수학 수식 사용하기"
categories:
  - Github_Blog
tags:
  - Github_Blog
use_math: true

toc: true
toc_sticky: true
toc_label: "GitHub 블로그에서 마크다운 수학 수식 사용하기"
---

이번엔 Jekyll Github 블로그에서 마크다운을 사용할 때 수학 수식을 적용하는 방법에 대해 알아보고자 합니다. 

# 계기

저는 포스트를 작성하면서 좀 더 깔끔하게 보이게 하기 위해서 마크다운에 수학 수식도 사용 가능한지 알아보았고, 여러 블로그의 글들을 보면서 마크다운의 수학 수식 적용 법을 적용해 보았지만 다른 사람들 처럼 저는 마크다운에 수학 수식이 적용되지 않았습니다. 그러다 어떤 티스토리 포스트에서 티스토리에서는 마크다운 수학 수식이 적용되지 않았고, html, css 기반인 티스토리에서 마크다운 수식을 적용하기 위해 Mathjax 를 적용했다는 글을 보게 되었고, Jekyll Github 도 이와 비슷하게 적용할 수 있지 않을까 해서 찾아 보았더니 그와 관련된 포스트가 있었고 그 포스트의 내용대로 적용하니 저도 마크다운의 수학 수식을 사용할 수 있게 되었습니다. 그렇다면 이제 어떻게 적용하면 되는지 알아보도록 하겠습니다.

# Mathjax 적용 방법

## 1. 마크다운의 엔진 변경

_config.yml 파일의 내용 중에 `# Conversion` 부분의 내용을 아래와 같이 수정합니다.

```yml
# Conversion
markdown: kramdown
highlighter: rouge
lsi: false
excerpt_separator: "\n\n"
incremental: false
```

제 _config.yml 파일은 위와 같이 되어 있어 변경하지는 않았습니다.

## 2. _includes 디렉토리에 mathjax_support.html 파일을 새로 생성하고 아래와 같이 작성합니다.

```html
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    TeX: {
      equationNumbers: {
        autoNumber: "AMS"
      }
    },
    tex2jax: {
    inlineMath: [ ['$', '$'] ],
    displayMath: [ ['$$', '$$'] ],
    processEscapes: true,
  }
});
MathJax.Hub.Register.MessageHook("Math Processing Error",function (message) {
	  alert("Math Processing Error: "+message[1]);
	});
MathJax.Hub.Register.MessageHook("TeX Jax - parse error",function (message) {
	  alert("Math Processing Error: "+message[1]);
	});
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
```

## 3. _layout 디렉토리의 default.html 파일의 `<head>` 부분에 다음 코드를 삽입합니다.


```
{`%` if page.use_math `%`}
	{`%` include mathjax_support.html `%`}
{`%` endif `%`}
```

아마 기존에 `<head>` 부분에 다른 두 줄이 있어서 그 두 줄 밑에 넣어 주면 됩니다. 그리고 위 코드에서 `%` 옆에 작은 따옴표를 붙였는데 현재 코드 블록에 `%`를 넣게 되면 강조 구문으로 인식해 출력이 되지 않아 임시 방편으로 넣은 것이므로 저 작은 따욤표는 제거해 주시기 바랍니다.

## 4. 수학식을 표시할 포스트의 front-matter 에 use_math: true 로 적용합니다. 

front-matter 는 jekyll 에서 포스트의 카테고리와 태그 등을 지정하는 곳입니다. 

```
categories:
  - Github_Blog
tags:
  - Github_Blog
use_math: true
```

## 5. 제대로 수식이 적용 되는지 테스트 해보기

`$N^2$` 을 넣었을 때 `$N^2$` 이 나오지 않고 $N^2$ 이 나오게 된다면 성공입니다.

# 참조

<https://chaelin0722.github.io/blog/mathjex/>