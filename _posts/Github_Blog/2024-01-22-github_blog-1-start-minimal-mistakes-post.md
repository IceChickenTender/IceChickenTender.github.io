---
title: "[Github_Blog] 1. minimal mistakes 시작하기"
categories:
  - Github_Blog
tags:
  - Github_Blog
toc: true
toc_sticky: true
toc_label: "minimal mistakes 시작"
---

평소에 중요한 내용들은 ppt나 word로 정리를 해두는 편인데 회사에서 일을 하면서 겪는 오류들이나 환경 세팅 방법 등 자잘한 것들은 정리를 해두지 않다 보니 금새 까먹고 또 구글링하고... 이렇게 하는게 귀찮아서 이전부터 기술 블로그를 하나 만들어야지 하고 있었던 터라 이번 기회에 블로그를 만들게 되었습니다.   
그러면서 제가 직접 minimal mistakes를 이용해 Github Blog를 구축한 방법에 대해서 공유를 드리고자 포스팅 합니다.

&nbsp;

# minimal mistakes 시작하기

&nbsp;

## 1. Ruby 설치
- <https://rubyinstaller.org/downloads/>에 접속하여 Ruby 설치
- Jekyll은 32bit 이기에, 설치 시 (x64)가 아닌 (x86)으로 진행해야 함
	- x86으로 설치하지 않으면 여러 오류가 발생할 수 있음
- 버전은 원하는 버전으로 설치 진행 하시면 될 것 같고, 저는 최신 버전으로 설치를 진행 했습니다.

<div align="center">
  <img src="/assets/images/github_blog/1/ruby_install.png" width="40%" height="30%" title="ruby_install" alt="ruby_install"/>
</div>

&nbsp;

## 2. Jekyll 세팅
- 명령 프롬프트(cmd) 창을 열고 `gem install jekyll` 명령어로 설치 진행 합니다.
- 명령 프롬프트 창에서 ruby -v 와 jekyll -v 명령으로 정상적으로 설치가 되었는지 확인 해줍니다.

<div align="center">
  <img src="/assets/images/github_blog/1/install_success.png" width="60%" height="50%" title="install_success" alt="install_success"/>
</div>

&nbsp;

## 3. minimal mistakes 테마
- github blog 테마는 다양한 테마가 있지만, minimal mistakes가 커스터마이징이 쉽고, 많이 사용하는 테마라 사용하기로 결정하였습니다.
- minimal mistakes 공식 홈페이지 : <https://mmistakes.github.io/minimal-mistakes/>
- minimal mistakes 가이드 사이트 : <https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/>
- minimal mistakes Github link : <https://github.com/mmistakes/minimal-mistakes>

&nbsp;

## 4. minimal mistakes clone
- github 아이디가 없다면 회원가입을 진행하고, 로그인 후에 Repositories로 이동
- 새로운 Repository를 생성하고, github blog의 주소처럼 repository 명을 정해줍니다.
	- 이 때 생성하는 repository의 이름은 보통 `자신의 github 계정명.github.io`로 해주시면 됩니다.
- repository 생성 시 <https://github.com/mmistakes/minimal-mistakes> 이 url로 import 해서 생성해 줍니다.
	- 생각보다 import 작업이 오래 걸립니다 참고해 주세요.
- repository가 생성이 되면 github에서 제공해 주는 git tool을 사용하거나 다른 git tool을 사용해 clone을 진행해 줍니다.
	- zip 파일로 다운로드 받을 수 있지만 commit, push를 해주어야 수정한 내용들이 블로그 사이트에도 반영이 되니 git을 사용해주도록 합시다!
- 

&nbsp;

## 5. minimal mistakes 로컬 서버 실행
- 본인 로컬 pc에서 구동해 블로그를 로컬 서버처럼 띄울 수 있습니다.
	- 이런 작업을 하는 이유는 로컬 서버로 띄울 경우 git에 굳이 push를 해주지 않아도 내가 수정한 내용을 바로 바로 확인을 할 수 있기 때문입니다.
	- 본인의 github blog git에 push를 할 경우에는 자잘한 수정 사항이 있을 때마다 push를 해주어야 하고, 반영 시간이 늦어 확인이 어렵기 때문에 로컬 서버에서 띄우는 것을 추천 드립니다.
- <1.4> 챕터에서 minimal mistakes clone을 진행한 후 명령 프롬프트(cmd) 창을 열어 clone을 폴더로 간 뒤 jekyll 실행을 위한 bundle 설치 및 실행 명령을 실행해 줍니다.
	- gem install jekyll bundler
	- bundle install
	- bundle exec jekyll serve
- 명령이 정상적으로 동작했다면 clone을 받아온 폴더에 `.jekyll-cache` 폴더와 `Gemfile.lock` 파일이 생성됩니다.

<div align="center">
  <img src="/assets/images/github_blog/1/jekyll-cache_gemfile-lock.png" width="60%" height="50%" title="jekyll-cache_gemfile-lock" alt="jekyll-cache_gemfile-lock"/>
</div>

&nbsp;

이렇게 minimal mistakes를 이용해 로컬 서버에 띄워보고 웹 상에서도 아직 꾸미지 않긴 했지만 자신의 블로그를 띄워 보았습니다.
다음 시간에는 자신만의 블로그로 꾸미기 위해 \_config.yml 파일 수정하는 방법에 대해 알아보도록 하겠습니다.
오타나 수정 사항, 궁금한 것이 있다면 댓글로 작성 부탁 드립니다!