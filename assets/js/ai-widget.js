const chatButton = document.getElementById('ai-widget-button');
const chatWindow = document.getElementById('ai-chat-window');
const chatClose = document.getElementById('ai-chat-close');
const chatInput = document.getElementById('ai-chat-input');
const chatMessages = document.getElementById('ai-chat-messages');

// 버튼 클릭 시 채팅창 토글
chatButton.onclick = () => {
  chatWindow.style.display = chatWindow.style.display === 'none' ? 'flex' : 'none';
};

// 닫기 버튼 클릭 시 숨김
chatClose.onclick = () => {
  chatWindow.style.display = 'none';
};

// ... (나머지 엔터키 전송 로직은 기존과 동일) ...