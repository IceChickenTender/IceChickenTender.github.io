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
