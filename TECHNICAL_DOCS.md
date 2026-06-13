# 金融多智能体系统 — 技术文档

> 版本：1.0.0
> 更新时间：2026-06-11
> 技术栈：Python 3.10 + FastAPI + ChromaDB + Ollama

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [技术选型](#3-技术选型)
4. [核心组件详解](#4-核心组件详解)
5. [API 接口规范](#5-api-接口规范)
6. [数据流详解](#6-数据流详解)
7. [异步编程模型](#7-异步编程模型)
8. [流式输出机制（SSE）](#8-流式输出机制)
9. [记忆管理策略](#9-记忆管理策略)
10. [前端界面](#10-前端界面)
11. [知识库扩展：Notion 集成](#11-知识库扩展notion-集成)
12. [部署指南](#12-部署指南)
13. [配置说明](#13-配置说明)
14. [扩展开发指南](#14-扩展开发指南)
15. [常见问题与排查](#15-常见问题与排查)

---

## 1. 项目概述

### 1.1 项目背景

构建一个面向中文金融场景的多智能体问答与分析系统。用户可以通过自然语言与系统交互，获取金融知识解答、投资决策分析、日常咨询等服务。

### 1.2 核心功能

| 功能 | 说明 |
|------|------|
| 多 Agent 协同 | 助理、决策、咨询三种 Agent 按意图自动路由 |
| RAG 检索增强 | 咨询 Agent 基于 ChromaDB 向量库检索相关知识后回复 |
| 对话记忆 | 滑动窗口管理多轮上下文，超时自动清理 |
| 本地部署 | 基于 Ollama 本地模型，无需外部 API 依赖 |
| Notion 知识库 | 可选同步 Notion 页面内容到向量库 |

### 1.3 适用场景

- 金融知识问答（"什么是市盈率？"）
- 投资决策分析（"现在适合买基金吗？"）
- 日常金融咨询（"ETF 和普通基金有什么区别？"）

---

## 2. 系统架构

### 2.1 整体架构图

`
┌─────────────────────────────────────────────────────────┐
│                    客户端 (浏览器/curl)                    │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI 服务层 (端口 8000)               │
│  ┌──────────────────────────────────────────────────┐   │
│  │              api/routes.py 路由层                  │   │
│  │  POST /chat | POST /chat/stream | GET /health    │   │
│  └────────────────────┬─────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              agents/orchestrator.py 编排器                │
│              意图识别 → Agent 路由                        │
└──────┬────────────┬──────────────┬──────────────────────┘
       │            │              │
       ▼            ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────────────┐
│助理 Agent│ │决策 Agent│ │  咨询 Agent       │
│日常对话  │ │决策分析  │ │  RAG 检索增强     │
│  兜底    │ │ 风险评   │ │  知识库问答       │
│          │ │ 估       │ │                   │
└──────────┘ └──────────┘ └────────┬─────────┘
                                   │
                                   ▼
                   ┌───────────────────────────┐
                   │    knowledge/ 知识库模块    │
                   │  ChromaDB 向量检索          │
                   │  ┌─────────┐ ┌─────────┐  │
                   │  │ 向量库  │ │ 种子数据 │  │
                   │  │ + Notion│ │ seed    │  │
                   │  └────┬────┘ └─────────┘  │
                   └───────┼───────────────────┘
                           │ Ollama Embedding API
                           ▼
                   ┌──────────────────┐
                   │  Ollama 本地服务  │
                   │  (llama3.2 模型) │
                   └──────────────────┘
`

### 2.2 模块依赖关系

`
main.py
  └─ api/routes.py
       ├─ agents/orchestrator.py
       │    ├─ agents/assistant_agent.py
       │    │    └─ utils/llm_client.py ←─ config.py
       │    ├─ agents/decision_agent.py
       │    │    └─ utils/llm_client.py
       │    └─ agents/consulting_agent.py
       │         ├─ utils/llm_client.py
       │         └─ knowledge/knowledge_base.py
       │              ├─ knowledge/vector_store.py ←─ utils/llm_client.py
       │              ├─ knowledge/seed_data.py
       │              └─ knowledge/notion_sync.py   ← Notion API
       └─ memory/memory_manager.py
            └─ memory/conversation.py
`

---

## 3. 技术选型

### 3.1 选型对比表

| 组件 | 选型 | 备选方案 | 选择理由 |
|------|------|----------|----------|
| Web 框架 | **FastAPI** | Flask, Django | 原生 async 支持，自动生成 OpenAPI 文档 |
| 大模型 API | **Ollama** (OpenAI 兼容) | OpenAI API, 本地 transformers | 本地运行，数据不外传，成本低 |
| LLM 客户端 | **openai** Python 包 | LangChain, requests | 接口标准化，流式支持好 |
| 向量数据库 | **ChromaDB** | FAISS, Milvus, Qdrant | 轻量嵌入式，无需独立部署 |
| Embedding | **Ollama 自带 API** | sentence-transformers | 复用已有模型，不需额外下载 |
| 对话记忆 | **内存字典 + 滑动窗口** | Redis, SQLite | 轻量无依赖，适合单机部署 |
| 外部知识源 | **Notion API** (可选) | 本地文件, Confluence | 日常笔记直接同步 |

### 3.2 关键依赖清单

`
openai>=1.0.0       连接 Ollama 兼容 API
chromadb>=0.5.0     向量数据库
httpx>=0.27.0       HTTP 客户端（openai 的底层依赖）
sse-starlette>=1.8.0 SSE 流式响应
tiktoken>=0.7.0     Token 计数（后续扩展用）

# 可选：Notion 集成
notion-client>=2.0.0  Notion 官方 API 客户端
`

---

## 4. 核心组件详解

### 4.1 LLM 客户端

**文件**: utils/llm_client.py

#### 职责

封装对 Ollama（或任何 OpenAI 兼容 API）的调用，提供三种核心能力：

1. **非流式对话**：chat() — 发送消息列表，等待完整回复
2. **流式对话**：chat_stream() — async generator，逐 token 产出
3. **文本向量化**：get_embedding() — 将文本转为浮点数向量

#### 关键实现

`python
client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama 不校验 key
)
`

- AsyncOpenAI 是所有异步 API 调用的入口
- ase_url 配置为 Ollama 地址，完全兼容 OpenAI 协议
- stream=True 参数切换流式模式

#### Embedding 调用方式

`python
response = await client.embeddings.create(
    model=LLM_MODEL_NAME,
    input=text,
)
return response.data[0].embedding  # list[float]
`

Ollama 的 /v1/embeddings 接口返回与 OpenAI 一致的格式。

> ⚠ 注意：chat_stream() 是一个 **async generator**（sync def + yield），
> 调用方必须使用 sync for chunk in chat_stream(messages):
> 而非 wait chat_stream(messages)。

---

### 4.2 记忆管理系统

**文件**: memory/conversation.py, memory/memory_manager.py

#### 设计思路

系统采用 **会话（Session）** 作为对话单位，每个会话由一系列消息组成。

#### Conversation — 单次会话

`
Conversation
├── session_id: str         # 会话唯一标识
├── messages: list[dict]    # 消息历史
│   └── 每条: {role, content, timestamp}
├── created_at: datetime
└── updated_at: datetime
`

#### MemoryManager — 全局管理器

`
MemoryManager
└── sessions: dict[str, Conversation]
    ├── get_or_create_session(id) → Conversation
    ├── add_message(id, role, content)    # 自动裁剪
    └── get_context_messages(id, system_prompt) → list[dict]
`

#### 滑动窗口算法

`
最近 N 轮对话   ←── 保留
更早的历史      ──→ 丢弃

MEMORY_WINDOW_SIZE = 10（config.py 中配置）
保留最近 10 轮对话（10 条 user + 10 条 assistant = 20 条消息）
`

为什么不保留全部历史？
- LLM 上下文窗口有限（通常 4K-128K tokens）
- 越早的对话对当前回复帮助越小
- 减少 token 消耗，降低响应延迟

#### 会话生命周期

`
创建：用户首次发送消息时自动创建（get_or_create_session）
更新：每次消息交互后更新 updated_at
清理：超过 SESSION_TIMEOUT_SECONDS（默认 1 小时）未活跃的会话被清理
`

---

### 4.3 Agent 系统

**文件**: gents/base_agent.py, gents/assistant_agent.py, gents/decision_agent.py, gents/consulting_agent.py, gents/orchestrator.py

#### 4.3.1 抽象基类 BaseAgent

`
BaseAgent (ABC)
├── name: str                    # Agent 标识
├── system_prompt: str           # 系统提示词
├── process(user_input, session_id) → AgentResult  # 抽象方法
└── _build_system_messages(extra_context)           # 辅助方法
`

AgentResult：
`
AgentResult
├── response: str       # 生成的回复文本
├── agent_name: str     # 负责的 Agent 名称
└── metadata: dict      # 额外信息
`

#### 4.3.2 三种 Agent 对比

| 特性 | 助理 Agent | 决策 Agent | 咨询 Agent |
|------|-----------|-----------|-----------|
| **类名** | AssistantAgent | DecisionAgent | ConsultingAgent |
| **核心依赖** | LLM 直答 | LLM 分步推理 | 知识库检索 + LLM |
| **温度** | 0.7（默认） | 0.3（低） | 0.7（默认） |
| **System Prompt** | 友好闲聊 | 分步骤分析 | 基于资料回答 |
| **适用问题** | 问候、闲聊 | 投资分析、风险评估 | 概念解释、知识问答 |
| **调用链路** | user → LLM | user → LLM (CoT) | user → RAG → LLM |

#### 4.3.3 咨询 Agent 的 RAG 流程

`
ConsultingAgent.process(user_input)
  │
  ├─ (1) 检索 Retrieve
  │    └─ knowledge_base.build_rag_context(query)
  │         └─ vector_store.search(query, k=3)
  │              └─ get_embedding(query) → ChromaDB 向量检索
  │
  ├─ (2) 增强 Augment
  │    └─ 将检索结果格式化为 "参考资料"，注入 system prompt
  │
  └─ (3) 生成 Generate
       └─ LLM 基于参考资料 + 对话历史 生成回答
`

#### 4.3.4 编排器 Orchestrator — 意图识别

**双阶段路由策略**：

`
第一阶段：关键词匹配（O(1)，零成本）
  ├─ 决策关键词：投资、风险、决策、对比、推荐、该不该...
  ├─ 咨询关键词：是什么、定义、概念、区别、市盈率、ETF...
  └─ 助理关键词：你好、谢谢、再见...

第二阶段（可选）：关键词未命中时由 LLM 判断
  └─ 当前实现：未命中 → 助理 Agent（兜底）

优先顺序：决策 → 咨询 → 助理
`

关键词定义（正则表达式，忽略大小写）：
`python
DECISION_KEYWORDS = [r"投资", r"风险", r"决策", r"对比", ...]
CONSULTING_KEYWORDS = [r"是什么", r"什么是", r"定义", r"概念", ...]
ASSISTANT_KEYWORDS = [r"你好", r"您好", r"谢谢", ...]
`

---

### 4.4 RAG 知识库

**文件**: knowledge/vector_store.py, knowledge/knowledge_base.py, knowledge/seed_data.py, knowledge/notion_sync.py

#### 4.4.1 架构层级

`
KnowledgeBase        ← 高层管理器（初始化、检索）
  └─ VectorStore     ← ChromaDB 封装（增、删、查）
       └─ ChromaDB   ← 向量数据库（持久化到本地磁盘）

NotionSync (可选)     ← Notion 内容同步器
  └─ VectorStore     ← 直接调用 add_documents 写入同一 ChromaDB
`

#### 4.4.2 VectorStore — ChromaDB 封装

与 ChromaDB 默认实现的关键区别在于 **不使用 ChromaDB 自带的 ONNX embedding 模型**，而是通过 Ollama API 生成向量：

`python
# 创建集合时指定不用默认 embedding 函数
self.collection = self.client.get_or_create_collection(
    name="financial_knowledge",
    embedding_function=None,  # 禁用默认 ONNX 模型
)

# 插入文档时手动传入向量
emb = await get_embedding(text)  # 调用 Ollama
self.collection.add(
    embeddings=[emb],       # 向量（Ollama 生成）
    documents=[text],
    metadatas=[metadata],
    ids=["doc_0"],
)
`

好处：
1. 避免 ChromaDB 默认 ONNX 模型的权限和缓存问题
2. 统一使用 Ollama embedding，保持模型一致
3. 避免额外的模型下载和内存占用

#### 4.4.3 向量检索流程

`
用户问题 → get_embedding(问题) → 向量A
                                    │
ChromaDB 集合 ── 查找与向量A最相似的 ──
    │                                │
    ▼                                ▼
返回 top-K 文档 ──────────────── 距离排序（余弦相似度）
`

#### 4.4.4 种子数据

seed_data.py 内置了 19 条中文金融知识，分 6 个类别：

| 类别 | 条目数 | 示例 |
|------|--------|------|
| 股票 | 4 | 股票概念、市盈率、市净率、股息率 |
| 基金 | 3 | 基金概念、ETF、指数基金 |
| 债券 | 2 | 债券概念、国债详解 |
| 宏观指标 | 3 | GDP、CPI、LPR |
| 技术分析 | 3 | 技术分析基础、MA、MACD |
| 风险管理 | 4 | 风险概述、资产配置、定投、止损 |

每条知识包含 (content, metadata)，metadata 中有 	itle（标题）、category（分类）、source（来源）。

#### 4.4.5 Notion 集成（可选模块）


otion_sync.py 提供从 Notion 页面同步内容到 ChromaDB 的能力。

**工作流程**：

`
Notion 页面
  │  Notion API (notion-client)
  ▼
读取所有 Block → 解析为纯文本 → 按段落切片（500 字符/段）
  │
  ▼
调用 VectorStore.add_documents() → 存入 ChromaDB
  │
  ▼
查询时：种子数据 + Notion 内容 一同被检索
`

**使用方式**：

`ash
pip install notion-client
python -c "
import asyncio
from knowledge.notion_sync import NotionSync
asyncio.run(NotionSync(
    token='secret_你的Token',
    page_id='你的Notion页面ID'
).sync_to_vectorstore())
"
`

前置条件：
1. 在 https://www.notion.so/my-integrations 创建 Integration
2. 在 Notion 页面中 Connect 该 Integration
3. 从页面 URL 中提取 32 位 page_id

---

### 4.5 API 层

**文件**: pi/routes.py

#### Pydantic 请求/响应模型

`python
class ChatRequest(BaseModel):
    message: str                    # 用户消息
    session_id: str | None = None   # 会话 ID（可选）

class ChatResponse(BaseModel):
    response: str       # Agent 回复
    agent_name: str     # Agent 标识
    session_id: str     # 会话 ID
`

#### 全局实例

`python
router = APIRouter()                     # 路由分组
orchestrator = Orchestrator()            # 编排器（全局单例）
`

orchestrator 是全局单例，所有请求共享同一个编排器实例。

---

### 4.6 应用入口

**文件**: main.py

#### 主要功能

`python
app = FastAPI(title="金融多智能体系统", ...)
app.include_router(router, prefix="/api", tags=["Agent 接口"])

@app.get("/")           # 返回聊天页面（HTML + CSS + JS）
@app.on_event("startup")  # 启动时初始化知识库
`

#### 启动事件

`python
@app.on_event("startup")
async def startup_event():
    # 1. 检查 ChromaDB 集合
    # 2. 如果集合为空，灌入 seed_data.py 的种子数据
    # 3. 已有数据则跳过（幂等初始化）
    await init_orchestrator()
`

#### 前端页面

聊天页面是一个完整的自包含 HTML 文件（CSS + JavaScript 内嵌），不需要额外的静态文件服务器：

- **CSS**：全内联，响应式布局（PC / 手机自适应）
- **JavaScript**：使用 etch('/api/chat', ...) 发送 POST 请求
- **缓存控制**：Cache-Control: no-store 防止页面缓存
- **输入方式**：支持按钮点击和 Enter 键发送

---

## 5. API 接口规范

### 5.1 健康检查

`
GET /api/health
`

**响应**:
`json
{"status": "ok", "active_sessions": 0}
`

### 5.2 聊天（非流式）

`
POST /api/chat
Content-Type: application/json

{
    "message": "什么是市盈率？",
    "session_id": "abc123"
}
`

**响应**:
`json
{
    "response": "市盈率（PE）是...",
    "agent_name": "consulting_agent",
    "session_id": "abc123"
}
`

### 5.3 聊天（流式 SSE）

`
POST /api/chat/stream
Content-Type: application/json

{
    "message": "什么是市盈率？",
    "session_id": "abc123"
}
`

**响应格式**（Server-Sent Events）:
`
event: message
data: {"type": "chunk", "content": "市盈"}

event: message
data: {"type": "chunk", "content": "率（PE）是..."}

event: message
data: {"type": "done", "agent_name": "consulting_agent", "session_id": "abc123"}
`

> 浏览器端接收 SSE 时，需注意 SSE 协议使用 \r\n 换行，
> JavaScript 中需先归一化换行符再解析：
> `javascript
> buffer = buffer.replace(/\r\n/g, '\n');
> const parts = buffer.split('\n\n');
> `

### 5.4 创建会话

`
POST /api/session/new
`

**响应**:
`json
{
    "session_id": "a1b2c3d4e5f6...",
    "messages": []
}
`

### 5.5 获取会话历史

`
GET /api/session/{session_id}
`

**响应**:
`json
{
    "session_id": "abc123",
    "messages": [
        {"role": "user", "content": "你好", "timestamp": "..."},
        {"role": "assistant", "content": "你好！...", "timestamp": "..."}
    ]
}
`

### 5.6 重新加载知识库

`
POST /api/knowledge/refresh
`

**响应**:
`json
{"status": "ok", "message": "知识库已重新加载"}
`

---

## 6. 数据流详解

### 6.1 一次完整请求的数据流

以用户提问"什么是市盈率？"为例：

`
(1) 用户 → POST /api/chat (message="什么是市盈率？", session_id="s1")
         │
(2) FastAPI 路由接收请求
         │
(3) orchestrator.process("什么是市盈率？", "s1")
     ├─ 匹配 CONSULTING_KEYWORDS → consulting_agent
         │
(4) consulting_agent.process()
     ├─ memory.get_context_messages("s1")
     │    ├─ 创建/获取会话（如不存在则创建）
     │    └─ 返回最近 10 轮对话（含 system prompt）
     │
     ├─ knowledge_base.build_rag_context("什么是市盈率？")
     │    ├─ vector_store.search("什么是市盈率？", k=3)
     │    │    ├─ get_embedding("什么是市盈率？") → 向量 [0.1, -0.3, ...]
     │    │    └─ ChromaDB 语义检索 → 返回 3 篇最相关文档
     │    └─ 格式化为 "参考资料" 文本
     │
     ├─ 构建 messages = system_prompt + 参考资料 + 历史对话 + 用户问题
     │
     ├─ async for chunk in chat_stream(messages)
     │    └─ 逐 token 收集完整回答
     │
     ├─ memory.add_message("s1", "user", "什么是市盈率？")
     ├─ memory.add_message("s1", "assistant", "市盈率（PE）...")
     │
     └─ 返回 AgentResult(response="市盈率（PE）...", agent_name="consulting_agent")
         │
(5) FastAPI → JSON 回复给用户
`

---

## 7. 异步编程模型

### 7.1 Python asyncio 基础

整个系统基于 Python 的 syncio 异步框架。

| 概念 | 说明 | 代码形态 |
|------|------|----------|
| 协程 | 可暂停执行的函数 | sync def func() |
| 等待 | 暂停当前协程，等待另一个协程完成 | wait other_func() |
| 异步生成器 | 逐段产出结果 | sync def gen(): + yield |
| 事件循环 | 调度所有协程的运行时 | syncio.run(main()) |
| 异步 for | 消费异步生成器 | sync for chunk in gen(): |

### 7.2 系统中的异步调用链

`
API 请求处理（FastAPI 自动在事件循环中运行）
  │
  await orchestrator.process()
    │
    await consulting_agent.process()
      │
      await knowledge_base.build_rag_context()
        │
        await vector_store.search()
          │
          await get_embedding(query)  ←─ HTTP 请求到 Ollama
      │
      async for chunk in chat_stream(messages)
        │
        await client.chat.completions.create(stream=True)
          │
          async for chunk in response:  ←─ 逐 token 读取
            yield chunk
`

每次 wait 处，当前协程暂停，事件循环可以处理其他请求。
**一个请求在等待 LLM 回复时，另一个请求可以同时被处理**。

### 7.3 ⚠ 常见陷阱：async generator 不能 await

`python
# ❌ 错误写法
response = await chat_stream(messages)     # chat_stream 是 async generator
async for chunk in response:               # 这行不会执行！
    ...

# ✅ 正确写法
async for chunk in chat_stream(messages):   # 直接在 async for 中调用
    ...
`

chat_stream() 返回的是 AsyncGenerator 对象，不是 Coroutine。
wait 只能用于协程（sync def 返回的），不能用于生成器。

---

## 8. 流式输出机制（SSE）

### 8.1 SSE 协议简介

Server-Sent Events（SSE）是 HTML5 标准协议，允许服务端向客户端推送事件。

| 特性 | SSE | WebSocket |
|------|-----|-----------|
| 方向 | 服务端 → 客户端 | 双向 |
| 协议 | HTTP | 独立协议（ws://） |
| 自动重连 | ✅ 内置 | ❌ 需自行实现 |

### 8.2 sse-starlette 实现

`python
from sse_starlette.sse import EventSourceResponse

async def event_generator():
    result = await orchestrator.process(message, session_id)
    full_text = result.response
    
    # 按 10 个字符一组输出（模拟打字机效果）
    for i in range(0, len(full_text), 10):
        yield {
            "event": "message",
            "data": json.dumps({
                "type": "chunk",
                "content": full_text[i:i+10],
            }),
        }
        await asyncio.sleep(0.05)

    yield {
        "event": "message",
        "data": json.dumps({"type": "done", "agent_name": "...", "session_id": "..."}),
    }

return EventSourceResponse(event_generator())
`

> 前端默认使用非流式 POST /api/chat 接口（更可靠）。
> /api/chat/stream 流式端点保留供需要打字机效果的场景调用。

---

## 9. 记忆管理策略

### 9.1 滑动窗口

`python
MEMORY_WINDOW_SIZE = 10

保留：最近 10 轮对话（10 条 user + 10 条 assistant = 20 条消息）
丢弃：更早的历史
时机：每次 add_message() 后自动裁剪
`

### 9.2 超时清理

`python
def cleanup_expired_sessions(self):
    now = time.time()
    for session_id, conv in self.sessions.items():
        if (now - conv.updated_at.timestamp()) > SESSION_TIMEOUT_SECONDS:
            del self.sessions[session_id]
`

---

## 10. 前端界面

### 10.1 页面架构

聊天页面是一个完整的自包含 HTML 文件（约 8KB），内嵌在 main.py 的 CHAT_HTML 字符串中：

`
CHAT_HTML (约 11168 字节)
├── HTML 结构
│   ├── .header     顶栏（标题 + Agent 名称标签）
│   ├── .messages   消息列表（滚动区域）
│   ├── .typing-indicator   "对方正在输入"
│   └── .input-area 输入框 + 发送按钮
├── CSS（内联 styles）
│   ├── 响应式布局（flex + max-width）
│   ├── 消息气泡样式
│   ├── 输入区域样式
│   └── 移动端适配（@media max-width: 600px）
└── JavaScript（内联，IIFE 模式）
    ├── 会话管理（session_id 自动生成）
    ├── 消息渲染（用户/Agent 气泡）
    ├── send() 函数 → POST /api/chat
    ├── appendStreamText / finishStream 回调
    └── Enter 键发送
`

### 10.2 请求方式

前端使用 etch('/api/chat', ...) 发送 POST 请求，这是**非流式**方式：

`javascript
fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ message: text, session_id: sessionId }),
})
.then(response => response.json())
.then(data => {
    // data.response → 完整回复文本
    // data.agent_name → 处理请求的 Agent
})
`

### 10.3 缓存控制

HTML 响应包含 Cache-Control: no-store 头，
页面内也有 <meta http-equiv="Cache-Control" content="no-store">，
确保浏览器始终获取最新版本。

---

## 11. 知识库扩展：Notion 集成

### 11.1 概述

knowledge/notion_sync.py 模块允许将 Notion 页面内容同步到 ChromaDB 向量库，
与种子数据一起被检索。

### 11.2 架构位置

`
ChromaDB 集合 (financial_knowledge)
├── seed_data.py → 19 条内置知识（初始化时自动灌入）
└── NotionSync   → 用户按需同步的 Notion 内容
`

### 11.3 同步流程

`
Step 1: Notion Integration Token
        └─ 在 https://www.notion.so/my-integrations 创建
        └─ token 格式: secret_xxxxxxxxxxxx

Step 2: 在 Notion 页面中 Share → Connect 该 Integration

Step 3: 提取 page_id (URL 中 32 位字符)

Step 4: 运行同步脚本
        python -c "
        import asyncio
        from knowledge.notion_sync import NotionSync
        asyncio.run(NotionSync(token='secret_xxx', page_id='xxx').sync_to_vectorstore())
        "
`

### 11.4 NotionSync 类结构

`
NotionSync
├── __init__(token, page_id, chunk_size)
│   └─ 初始化 NotionClient + VectorStore
│
├── sync_to_vectorstore(page_ids)
│   └─ 主入口：读取页面 → 切片 → 存入 ChromaDB
│
├── _read_page_blocks(page_id)
│   └─ 递归读取 Notion 页面下所有 Block
│
├── _blocks_to_text(blocks)
│   └─ 将 Notion Block → 纯文本
│   支持类型：heading / bulleted_list / code / quote / to_do / callout
│
└── _chunk_text(text, chunk_size)
    └─ 按字符数切片，尽量在句号/换行处分割
`

---

## 12. 部署指南

### 12.1 环境要求

| 组件 | 版本要求 | 说明 |
|------|----------|------|
| Python | >= 3.10 | 异步语法兼容 |
| Ollama | >= 0.1.0 | 需加载 llama3.2 模型 |
| 磁盘空间 | >= 1GB | 模型文件 + 代码 |
| 内存 | >= 8GB | 推荐 16GB 以上 |

### 12.2 安装步骤

`ash
# 1. 安装 Ollama（如未安装）
#    访问 https://ollama.com 下载安装包

# 2. 拉取模型
ollama pull llama3.2

# 3. 激活 conda 环境
conda activate ai_project

# 4. 安装 Python 依赖
cd financial_agent
pip install -r requirements.txt

# 5. 启动 Ollama 服务（新开终端）
ollama serve

# 6. 启动系统
python main.py
# 或：python -m uvicorn main:app --host 0.0.0.0 --port 8000
`

### 12.3 验证部署

`ash
# 检查服务是否启动
curl http://localhost:8000/api/health

# 发送测试请求
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是市盈率？", "session_id": "test1"}'

# 测试流式
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是市盈率？", "session_id": "test1"}'

# 查看 API 文档
# 浏览器打开 http://localhost:8000/docs
`

### 12.4 Notion 集成（可选）

`ash
conda activate ai_project
pip install notion-client
python -c "
import asyncio
from knowledge.notion_sync import NotionSync
asyncio.run(NotionSync(
    token='secret_你的Token',
    page_id='你的页面ID'
).sync_to_vectorstore())
"
`

---

## 13. 配置说明

所有可配置项集中在 config.py 中：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| LLM_BASE_URL | http://localhost:11434/v1 | Ollama API 地址 |
| LLM_API_KEY | ollama | API Key（Ollama 不校验） |
| LLM_MODEL_NAME | llama3.2 | 对话模型名称 |
| EMBEDDING_MODEL_NAME | llama3.2 | Embedding 模型名称 |
| CHROMA_PERSIST_DIR | ./chroma_db | 向量数据库持久化路径 |
| CHROMA_COLLECTION_NAME | inancial_knowledge | 集合名称 |
| RETRIEVAL_TOP_K | 3 | 每次检索返回文档数 |
| MEMORY_WINDOW_SIZE | 10 | 保留最近 N 轮对话 |
| SESSION_TIMEOUT_SECONDS | 3600 | 会话超时时间（秒） |
| HOST |  .0.0.0 | 监听地址 |
| PORT | 8000 | 监听端口 |

配置支持环境变量覆盖（os.getenv），适合容器化部署。

---

## 14. 扩展开发指南

### 14.1 添加新的 Agent

1. 在 gents/ 下创建新类，继承 BaseAgent：

`python
from agents.base_agent import BaseAgent, AgentResult
from utils.llm_client import chat_stream
from memory.memory_manager import memory_manager

class NewAgent(BaseAgent):
    name = "new_agent"
    system_prompt = "你是...请用中文回答。"

    async def process(self, user_input: str, session_id: str) -> AgentResult:
        messages = self.memory.get_context_messages(
            session_id=session_id,
            system_prompt=self.system_prompt,
        )
        messages.append({"role": "user", "content": user_input})

        full_response = ""
        async for chunk in chat_stream(messages):
            full_response += chunk

        memory_manager.add_message(session_id, "user", user_input)
        memory_manager.add_message(session_id, "assistant", full_response)

        return AgentResult(response=full_response, agent_name=self.name)
`

2. 在 orchestrator.py 中注册：

`python
self.new_agent = NewAgent()
NEW_KEYWORDS = [r"关键词1", r"关键词2"]
# 在 _route_intent 中添加规则
`

### 14.2 扩展知识库

`python
# 方法 A：补充种子数据
# seed_data.py 中添加：
SEED_KNOWLEDGE.append((
    "知识文本...",
    {"title": "标题", "category": "分类", "source": "来源"},
))

# 方法 B：同步 Notion
# 运行 notion_sync.py

# 方法 C：热加载
# POST /api/knowledge/refresh
`

### 14.3 替换 LLM 模型

修改 config.py：

`python
LLM_MODEL_NAME = "qwen2.5:7b"         # 聊天模型
EMBEDDING_MODEL_NAME = "qwen2.5:7b"   # 需支持 embeddings
`

### 14.4 接入外部 API

替换 utils/llm_client.py 中的 base_url：

`python
# OpenAI
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = "sk-..."

# 通义千问
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = "sk-..."
`

---

## 15. 常见问题与排查

### 15.1 服务无法启动

**问题**：Attribute "app" not found in module "main"

**排查**：
1. 确认在 inancial_agent/ 目录下运行 python main.py
2. 检查语法：python -c "import ast; ast.parse(open('main.py').read())"
3. 检查文件编码：确保是 UTF-8 无 BOM

### 15.2 Ollama 连接失败

**问题**：APIConnectionError: Connection error

**排查**：
1. 确认 Ollama 已启动：ollama serve
2. 检查端口：curl http://localhost:11434/api/tags
3. 确认模型已下载：ollama list
4. 检查 config.py 中的 LLM_BASE_URL

### 15.3 页面打不开 / 一直转圈

**原因**：多数情况下是浏览器缓存了旧版页面。

**解决**：
1. 按 Ctrl+F5 或 Cmd+Shift+R 强制刷新
2. 关闭标签页，重新打开 http://localhost:8000
3. 用系统默认浏览器（Chrome/Edge）而非 in-app 浏览器

如果页面完全无法加载，检查：
`ash
# 确认服务器在运行
curl http://localhost:8000/api/health

# 确认 HTML 正常返回
curl http://localhost:8000/ | head -5
`

### 15.4 前端发不出消息

**如果页面能加载但发不出消息**，通常是 JavaScript 语法错误：

1. 打开浏览器开发者工具（F12）→ Console，看有没有红字报错
2. 最常见原因：Python """...""" 字符串中的 \n 被解释为实际换行符，
   导致 JavaScript 字符串字面量跨行断裂
3. 检查欢迎消息里是否使用了 \\n（双反斜杠）而非 \n

### 15.5 中文字符显示异常

`ash
# Windows PowerShell 中设置 UTF-8 编码
 = "utf-8"
python main.py
`

---

## 附录

### A. 文件清单

`
financial_agent/
├── main.py                   # 应用入口（含内嵌 HTML）
├── config.py                 # 全局配置
├── requirements.txt          # 依赖清单
├── TECHNICAL_DOCS.md         # 本技术文档
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py         # 抽象基类
│   ├── assistant_agent.py    # 助理 Agent
│   ├── consulting_agent.py   # 咨询 Agent（RAG）
│   ├── decision_agent.py     # 决策 Agent
│   └── orchestrator.py       # 编排器
│
├── memory/
│   ├── __init__.py
│   ├── conversation.py       # 会话数据结构
│   └── memory_manager.py     # 记忆管理器
│
├── knowledge/
│   ├── __init__.py
│   ├── vector_store.py       # ChromaDB 封装
│   ├── knowledge_base.py     # 知识库管理器
│   ├── seed_data.py          # 种子数据（19 条金融知识）
│   └── notion_sync.py        # Notion 同步器（可选）
│
├── utils/
│   ├── __init__.py
│   └── llm_client.py         # LLM 客户端
│
└── api/
    ├── __init__.py
    └── routes.py             # API 路由

chroma_db/                    # ChromaDB 持久化数据（自动生成）
`

### B. 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0.0 | 2026-06-10 | 初始版本 |
| 1.0.1 | 2026-06-11 | 修复 async generator 错误用法 |
| 1.0.2 | 2026-06-11 | 修复前端 SSE 换行符兼容问题 |
| 1.0.3 | 2026-06-11 | 前端改为非流式 POST 请求 |
| 1.0.4 | 2026-06-11 | 修复 JS 字符串转义问题，添加缓存控制 |
| 1.0.5 | 2026-06-11 | 添加 Notion 集成模块 |

---
