---
title: "[SublimeMerge] SublimeMerge 먹통(freezing) 현상 원인 및 해결"
categories:
  - SublimeMerge
tags:
  - SublimeMerge

use_math: true
toc: true
toc_sticky: true
toc_label: "SublimeMerge 먹통(freezing) 현상 원인 및 해결"
---

# 개요

SublimeMerge를 사용하다 보면 가끔 SublimeMerge를 켰을 때 먹통(freezing)이 되는 경우가 많이 있습니다. 이런 경우 SublimeMerge를 종료 시켜도 종료되지 않고, 작업관리자로 종료 시킨 후에 다시 켜보아도 여전히 먹통이고, 컴퓨터를 재시작해 보아도 여전히 먹통인 경우가 많습니다. 이런 현상이 자주 일어나 현상이 발생할 때마다 해결 방법을 찾아보고 해결하였습니다만 이번 기회에 관련 내용을 정리하여 추후에도 바로 바로 블로그에서 찾아볼 수 있게 하기 위해 포스트로 정리해 보았습니다.

# 문제 원인

문제의 원인은 SublimeMerge에서 직전 상태를 저장하는 세션 파일이 꼬인 것인 문제의 원인으로, 이렇게 파일이 꼬이게 되는 경우는 주로 컴퓨터의 비정상적인 종료로 SublimeMerge가 강제 종료되어 발생합니다. 또한 SublimeMerge는 그대로 켜놓은 상태에서 컴퓨터를 종료할 때에도 세션 파일이 꼬이는 경우가 생겨 이렇게 먹통이 되게 됩니다.

# 해결 방법

SublimeMerge 먹통 문제를 해결하는 방법은 꼬인 세션 파일을 제거하는 것입니다. 방법은 다음과 같습니다.

1. SublimeMerge를 작업관리자를 이용해 완전히 종료합니다.

2. 운영체제에 맞는 경로로 이동합니다.

    - Windows: "%APPDATA%\Sublime Merge\Local\" (탐색기 주소창에 복사-붙여넣기)
    - macOS: "~/Library/Application Support/Sublime Merge/Local/"
    
    아래 이미지와 같이 "Session.sublime_session" 이라는 파일이 있습니다.

<div align="center">
  <img src="/assets/images/sublimemerge/sublimemerge_freezing/session_file.png" width="55%" height="40%"/>
</div>

<br>

3. "Session.sublime_session" 파일을 삭제합니다.

4. SublimeMerge를 다시 실행합니다.

    session 파일을 삭제하고 난뒤 실행하면 기존에 저장된 정보들이 삭제되어 이미지와 같이 저장소를 여는 창이 뜨게되며 정상적으로 실행이 됩니다.

<div align="center">
  <img src="/assets/images/sublimemerge/sublimemerge_freezing/sublimemerge_normal_execution.png" width="55%" height="40%"/>
</div>

<br>