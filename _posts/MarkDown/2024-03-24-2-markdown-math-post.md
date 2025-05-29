---
title: "[MarkDown] 마크다운에서의 수학 수식 문법 정리"
categories:
  - MarkDown
tags:
  - MarkDown

use_math: true
toc: true
toc_sticky: true
toc_label: "마크다운에서의 수학 수식 문법 정리"
---

이번엔 마크다운에서 수학 수식 문법 정리에 대한 포스트를 작성해 보고자 합니다. 매번 다른 포스트들을 찾고 하자니 귀찮고, 저의 블로그에 정리를 해놓으면 굳이 검색할 필요 없이 제 블로그의 포스트를 참고하기 위해서 포스트를 작성하게 되었습니다.   
참고로 모든 마크다운에서 제공하는 것은 아니며, 저는 사용하는 것이 Github 블로그기 때문에 Github 블로그의 jekyll 에서 Mathjax 를 적용하였습니다. 이에 대한 방법으로는 다음 링크를 참조해 주시기 바랍니다.   
<https://icechickentender.github.io/github_blog/11-mathjax-post/>

# Mathjax 란?

MathML, LaTeX 및 ASCIIMathML 마크 업을 사용하여 웹 브라우저에 수학 표기법을 표시하는 크로스 브라우저 JavaScript 라이브러리이며 웹 사이트에서 수학식 표시를 가능하게 해줍니다.

# 수식 사용을 위한 방법

`$$`로 묶어줌으로써 mathjax를 사용한다는 것을 컴퓨터에게 전달합니다. 예를 들어 $N^2$ 을 표현하고 싶다고 가정한다면 `$N^2$` 을 사용하면 $N^2$ 과 같이 출력됩니다.

# 그리스 문자

다양한 그리스 문자들을 사용할 수 있습니다. 그리스 문자들은 수학 수식에서 알파($\alpha$), 베타($\beta$) 등과 같은 기호입니다. 그리고 `$$` 안에서 쓰고자 할 때 꼭 표현에 백슬래쉬(`\`)를 붙어주어야 합니다. 관련해서 테이블로 정리해 보았습니다.

|그리스 문자|표현|
|:------:|:-:|
|$\alpha$(alpha)|$\alpha$|
|$\beta$(beta)|$\beta$|
|$\gamma$(gamma)|$\gamma$|
|$\delta$(delta)|$\delta$|
|$\epsilon$(epsilon)|$\epsilon$|
|$\varepsilon$(varepsilon)|$\varepsilon$|
|$\zeta$(zeta)|$\zeta$|
|$\eta$(eta)|$\eta$|
|$\theta$(theta)|$\theta$|
|$\vartheta$(vartheta)|$\vartheta$|
|$\Gamma$(Gamma)||
