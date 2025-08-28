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
