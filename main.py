# ============================================================
# main.py - 金融智能体应用入口
# ============================================================

import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from api.routes import router, init_orchestrator

app = FastAPI(
    title="金融多智能体系统",
    description="一个基于多 Agent 架构和 RAG 的金融智能问答与分析系统。",
    version="1.0.0",
)


# ============================================================
# 根路径：交互式聊天页面
# ============================================================
@app.get("/", include_in_schema=False)
async def root():
    """
    渲染一个完整的聊天界面。
    - 自适应布局（PC / 手机）
    - SSE 流式输出，逐字展示回复
    - 显示当前处理请求的 Agent 名称
    - 自动管理 session_id
    """
    resp = HTMLResponse(content=CHAT_HTML)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
    return resp


app.include_router(router, prefix="/api", tags=["Agent 接口"])


@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("金融多智能体系统启动中...")
    print("=" * 50)
    asyncio.create_task(init_orchestrator())
    from config import HOST, PORT
    print(f"系统已启动（知识库后台加载中）")
    print(f"  聊天页面：http://localhost:{PORT}")
    print(f"  API 文档：http://localhost:{PORT}/docs")
    print("=" * 50)


if __name__ == "__main__":
    from config import HOST, PORT
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)


# ============================================================
# 聊天页面 HTML（完整自包含，无外部依赖）
# ============================================================
CHAT_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Cache-Control" content="no-store">
<title>金融多智能体系统</title>
<style>
  /* ---- 全局重置 & 布局 ---- */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f0f2f5; height: 100vh; display: flex; justify-content: center;
    color: #1a1a2e;
  }
  .app { width: 100%; max-width: 800px; height: 100vh; display: flex; flex-direction: column; }
  .header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #fff; padding: 16px 24px; display: flex; align-items: center; gap: 12px;
    flex-shrink: 0;
  }
  .header h1 { font-size: 18px; font-weight: 600; flex: 1; }
  .header .badge {
    font-size: 11px; background: rgba(255,255,255,0.15); padding: 4px 10px;
    border-radius: 12px; white-space: nowrap;
  }
  .messages {
    flex: 1; overflow-y: auto; padding: 20px 24px; display: flex;
    flex-direction: column; gap: 16px; background: #f8f9fb;
  }
  .messages:empty::after {
    content: "\u53d1\u9001\u4e00\u6761\u6d88\u606f\u5f00\u59cb\u5bf9\u8bdd"; display: block; text-align: center;
    color: #bbb; margin-top: 60px; font-size: 15px;
  }
  .msg { display: flex; gap: 10px; max-width: 85%; animation: fadeIn .25s ease; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.agent { align-self: flex-start; }
  .msg .avatar {
    width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 14px; flex-shrink: 0;
  }
  .msg.user .avatar { background: #4361ee; color: #fff; }
  .msg.agent .avatar { background: #2ec4b6; color: #fff; }
  .msg .bubble {
    padding: 10px 14px; border-radius: 14px; font-size: 14px; line-height: 1.6;
    word-break: break-word; white-space: pre-wrap;
  }
  .msg.user .bubble { background: #4361ee; color: #fff; border-bottom-right-radius: 4px; }
  .msg.agent .bubble { background: #fff; color: #1a1a2e; border-bottom-left-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .msg .agent-label {
    font-size: 11px; color: #888; margin-top: 4px; display: block;
  }
  .msg.streaming .bubble::after {
    content: "|"; animation: blink .7s infinite; color: #4361ee;
  }
  @keyframes blink { 50% { opacity: 0; } }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .input-area {
    display: flex; gap: 8px; padding: 16px 24px; background: #fff;
    border-top: 1px solid #e8e8e8; flex-shrink: 0;
  }
  .input-area input {
    flex: 1; padding: 12px 16px; border: 1px solid #ddd; border-radius: 24px;
    font-size: 14px; outline: none; transition: border .2s;
  }
  .input-area input:focus { border-color: #4361ee; }
  .input-area button {
    width: 44px; height: 44px; border: none; border-radius: 50%;
    background: #4361ee; color: #fff; font-size: 18px; cursor: pointer;
    transition: background .2s; flex-shrink: 0;
  }
  .input-area button:hover { background: #3651d4; }
  .input-area button:disabled { background: #ccc; cursor: not-allowed; }
  .typing-indicator { display: none; align-self: flex-start; padding: 12px 18px;
    background: #fff; border-radius: 18px; box-shadow: 0 1px 3px rgba(0,0,0,.08); gap: 4px; }
  .typing-indicator.active { display: flex; }
  .typing-indicator span {
    width: 8px; height: 8px; background: #bbb; border-radius: 50%;
    animation: bounce 1.4s infinite both;
  }
  .typing-indicator span:nth-child(2) { animation-delay: .2s; }
  .typing-indicator span:nth-child(3) { animation-delay: .4s; }
  @keyframes bounce { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-8px); } }
  @media (max-width: 600px) {
    .header { padding: 12px 16px; }
    .header h1 { font-size: 15px; }
    .messages { padding: 12px 16px; }
    .msg { max-width: 92%; }
    .input-area { padding: 12px 16px; }
  }
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <h1>💰 金融多智能体系统</h1>
    <span class="badge" id="agentBadge">就绪</span>
  </div>
  <div class="messages" id="messages"></div>
  <div class="typing-indicator" id="typing"><span></span><span></span><span></span></div>
  <div class="input-area">
    <input type="text" id="input" placeholder="输入你的问题..." autofocus>
    <button id="sendBtn" onclick="send()">➤</button>
  </div>
  <div id="sessionId" style="display:none"></div>
</div>
<script>
(function() {
  var messagesEl = document.getElementById('messages');
  var inputEl = document.getElementById('input');
  var sendBtn = document.getElementById('sendBtn');
  var typingEl = document.getElementById('typing');
  var badgeEl = document.getElementById('agentBadge');
  var sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
  var isLoading = false;
  var currentBubble = null;
  var currentText = '';
  var currentAgent = '';

  function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

  function addUserMessage(text) {
    var div = document.createElement('div');
    div.className = 'msg user';
    div.innerHTML = '<div class="avatar">\u4f60</div><div class="bubble">' + escapeHtml(text) + '</div>';
    messagesEl.appendChild(div);
    scrollBottom();
  }

  function createAgentBubble(agentName) {
    var div = document.createElement('div');
    div.className = 'msg agent streaming';
    div.innerHTML = '<div class="avatar">AI</div><div class="bubble"></div>';
    messagesEl.appendChild(div);
    currentBubble = div.querySelector('.bubble');
    currentText = '';
    currentAgent = agentName || 'assistant';
    badgeEl.textContent = currentAgent === 'assistant_agent' ? '\u52a9\u7406' : currentAgent === 'decision_agent' ? '\u51b3\u7b56\u5206\u6790' : currentAgent === 'consulting_agent' ? '\u91d1\u878d\u54a8\u8be2' : currentAgent;
    scrollBottom();
    return div;
  }

  function appendStreamText(chunk) {
    if (!currentBubble) return;
    currentText += chunk;
    currentBubble.textContent = currentText;
    scrollBottom();
  }

  function finishStream(agentName) {
    if (currentBubble) {
      var msgEl = currentBubble.closest('.msg');
      if (msgEl) {
        msgEl.classList.remove('streaming');
        var label = document.createElement('span');
        label.className = 'agent-label';
        label.textContent = (agentName || currentAgent).replace('_agent', ' Agent');
        msgEl.querySelector('.bubble').after(label);
      }
    }
    currentBubble = null;
    currentText = '';
    isLoading = false;
    sendBtn.disabled = false;
    inputEl.disabled = false;
    inputEl.focus();
    typingEl.classList.remove('active');
    badgeEl.textContent = '\u5c31\u7eea';
    scrollBottom();
  }

  window.send = function() {
    var text = inputEl.value.trim();
    if (!text || isLoading) return;
    inputEl.value = '';
    addUserMessage(text);
    isLoading = true;
    sendBtn.disabled = true;
    inputEl.disabled = true;
    typingEl.classList.add('active');
    currentBubble = null;
    currentText = '';
    createAgentBubble('');

    fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message: text, session_id: sessionId }),
    })
    .then(function(response) {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      return response.json();
    })
    .then(function(data) {
      currentAgent = data.agent_name || '';
      if (data.response) {
        appendStreamText(data.response);
      }
      finishStream(data.agent_name);
    })
    .catch(function(err) {
      appendStreamText('[\u5931\u8d25] ' + err.message);
      finishStream('error');
    });
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  inputEl.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  inputEl.focus();

  var welcomeDiv = document.createElement('div');
  welcomeDiv.className = 'msg agent';
  welcomeDiv.innerHTML = '<div class="avatar">AI</div><div class="bubble">\u4f60\u597d\uff01\u6211\u662f\u91d1\u878d\u591a\u667a\u80fd\u4f53\u7cfb\u7edf\u3002\\n\\n\u6211\u53ef\u4ee5\u5e2e\u4f60\uff1a\\n- \u89e3\u91ca\u91d1\u878d\u6982\u5ff5\uff08\u5982\"\u4ec0\u4e48\u662f\u5e02\u76c8\u7387\uff1f\"\uff09\\n- \u5206\u6790\u6295\u8d44\u51b3\u7b56\uff08\u5982\"\u5b9a\u6295\u57fa\u91d1\u600e\u4e48\u6837\uff1f\"\uff09\\n- \u65e5\u5e38\u91d1\u878d\u54a8\u8be2\\n\\n\u8bf7\u8f93\u5165\u4f60\u7684\u95ee\u9898\u5f00\u59cb\u5bf9\u8bdd\u3002</div>';
  messagesEl.appendChild(welcomeDiv);
  scrollBottom();
})();
</script>
</body>
</html>
"""