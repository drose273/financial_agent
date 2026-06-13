# ============================================================
# api/routes.py — FastAPI 路由定义
# ============================================================
# 这个文件定义了所有的 HTTP API 端点。
# 包括：
#   1. POST /chat — 普通聊天（返回 JSON）
#   2. POST /chat/stream — 流式聊天（SSE 格式）
#   3. POST /session/new — 创建新会话
#   4. GET /session/{session_id} — 查看会话历史
#   5. GET /health — 健康检查
#
# 什么是 SSE（Server-Sent Events）？
# - 服务端向客户端推送事件的技术
# - 客户端通过 EventSource API 接收
# - 比 WebSocket 轻量，只支持服务端→客户端单向推送
# - 适合流式输出场景

import json
import uuid  # 用于生成唯一会话 ID
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# SSE 流式响应支持
# sse-starlette 提供了 EventSourceResponse 类
# 配合 async generator 可以方便地实现 SSE
from sse_starlette.sse import EventSourceResponse

from agents.orchestrator import Orchestrator
from memory.memory_manager import memory_manager


# ============================================================
# 请求/响应模型（Pydantic）
# ============================================================
# Pydantic 是 Python 的数据验证库
# 通过定义类（继承 BaseModel）来自动生成请求/响应格式
# FastAPI 会基于这些类做：
# - 请求体验证（类型检查、必填项检查）
# - 自动生成 OpenAPI 文档

class ChatRequest(BaseModel):
    """普通聊天请求体。"""
    message: str                           # 用户消息
    session_id: Optional[str] = None       # 会话 ID（可选，不传则创建新会话）


class ChatResponse(BaseModel):
    """普通聊天响应体。"""
    response: str          # Agent 回复
    agent_name: str        # 处理请求的 Agent 名称
    session_id: str        # 会话 ID


class SessionResponse(BaseModel):
    """会话响应体。"""
    session_id: str
    messages: list[dict]


# ============================================================
# 创建路由和编排器
# ============================================================
# APIRouter 是 FastAPI 的路由分组
# 我们可以把路由定义在这里，然后在 main.py 中挂载到主应用
router = APIRouter()

# 创建编排器实例（全局共享）
orchestrator = Orchestrator()


# ============================================================
# 初始化事件（在应用启动时调用）
# ============================================================
async def init_orchestrator():
    """初始化编排器。异常时不影响服务，首次请求会触发按需初始化。"""
    try:
        await orchestrator.initialize()
    except Exception as e:
        print(f"[警告] 知识库初始化失败: {e}")
        print("[警告] 首次咨询请求将触发按需初始化")


# ============================================================
# API 端点 1：健康检查
# ============================================================
@router.get("/health")
async def health_check():
    """
    健康检查接口。
    用于验证服务是否正常运行。
    """
    return {
        "status": "ok",
        "active_sessions": memory_manager.active_sessions_count,
    }


# ============================================================
# API 端点 2：创建新会话
# ============================================================
@router.post("/session/new", response_model=SessionResponse)
async def create_session():
    """
    创建一个新的会话。

    流程：
    1. 生成一个唯一的 session_id（UUID）
    2. 在记忆管理器中创建会话
    3. 返回 session_id

    为什么要手动创建会话？
    - 不需要。get_or_create_session 会自动创建。
    - 但有时候客户端需要事先知道 session_id。
    - 这个端点为客户端提供主动获取新 session_id 的能力。
    """
    # uuid.uuid4() 生成一个随机的 UUID
    # hex 转为纯字符串（不带短横线）
    session_id = uuid.uuid4().hex

    # 在记忆管理器中创建会话
    memory_manager.get_or_create_session(session_id)

    return SessionResponse(
        session_id=session_id,
        messages=[],
    )


# ============================================================
# API 端点 3：查看会话历史
# ============================================================
@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    获取指定会话的完整历史。

    参数：
        session_id: 会话 ID（路径参数）

    返回：
        session_id + 消息列表

    用途：
    - 前端页面刷新时恢复对话
    - 调试查看对话记录
    """
    messages = memory_manager.get_session_history(session_id)

    return SessionResponse(
        session_id=session_id,
        messages=messages,
    )


# ============================================================
# API 端点 4：普通聊天（非流式）
# ============================================================
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    普通聊天接口（非流式一次性返回）。

    请求体：
        {"message": "什么是市盈率？", "session_id": "xxx"}

    处理流程：
    1. 获取或创建会话
    2. 编排器进行意图识别和 Agent 路由
    3. Agent 处理请求
    4. 返回完整回复

    适用场景：
    - 不需要流式效果的客户端
    - 后台脚本调用
    """
    # 如果没有传 session_id，自动生成一个
    session_id = request.session_id or uuid.uuid4().hex

    # 编排器处理请求
    result = await orchestrator.process(request.message, session_id)

    return ChatResponse(
        response=result.response,
        agent_name=result.agent_name,
        session_id=session_id,
    )


# ============================================================
# API 端点 5：流式聊天（SSE）
# ============================================================
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口（SSE，Server-Sent Events）。

    请求体：
        {"message": "讲一个故事", "session_id": "xxx"}

    响应格式（SSE）：
        data: {"type": "chunk", "content": "在"}

        data: {"type": "chunk", "content": "一"}

        data: {"type": "chunk", "content": "个"}

        data: {"type": "done", "agent_name": "consulting_agent"}

    什么是 EventSourceResponse？
    - sse-starlette 提供的 SSE 响应类
    - 接受一个 async generator
    - 自动设置正确的 Content-Type（text/event-stream）
    - 浏览器端可以用 EventSource API 接收

    流式处理流程：
    1. 接收用户请求
    2. 编排器路由到对应 Agent
    3. Agent 调用 LLM 的流式 API
    4. 每个 chunk 立即通过 SSE 推送给客户端
    5. 完成后发送 done 信号

    ⚠ 注意：目前简化版实现，先做非流式路由
    后续可以扩展为真正逐 chunk 流式输出
    """
    session_id = request.session_id or uuid.uuid4().hex

    # 定义异步生成器：逐块产出 SSE 事件
    async def event_generator():
        """
        SSE 事件生成器（async generator）。

        每次都 yield 一个 dict，sse-starlette 会自动
        把它格式化为 SSE 协议格式：

        data: {...}\n\n

        浏览器端用 EventSource 接收时，可以通过
        event.data 获取内容。
        """
        try:
            # 对编排器进行一次处理
            # 注：这是非流式处理的简化版本
            # 真正的流式需要每个 Agent 支持流式输出
            result = await orchestrator.process(request.message, session_id)

            # 逐字符输出模拟流式效果
            # 在实践中，这里应该调用 Agent 的流式 API
            # 但为了演示 SSE 机制，我们先获取完整回复再逐段输出
            full_text = result.response

            # 按句子拆分，模拟逐句输出
            # 也可以用更细的粒度（如逐字）
            # 这里我们按 10 个字符一组输出
            chunk_size = 10
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i + chunk_size]
                # yield 给 sse-starlette，格式化为 SSE 事件
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "type": "chunk",
                        "content": chunk,
                    }),
                }

                # 模拟延迟（让客户端看到"打字效果"）
                # 在实际应用中，这里应该是等待 LLM 的下一个 chunk
                import asyncio
                await asyncio.sleep(0.05)

            # 发送完成事件
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "done",
                    "agent_name": result.agent_name,
                    "session_id": session_id,
                }),
            }

        except Exception as e:
            # 错误处理：发送错误事件
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "content": str(e),
                }),
            }

    # EventSourceResponse 包装 async generator
    # 自动设置：
    # - Content-Type: text/event-stream
    # - Cache-Control: no-cache
    # - Connection: keep-alive
    return EventSourceResponse(event_generator())


# ============================================================
# API 端点 6：重新加载知识库
# ============================================================
@router.post("/knowledge/refresh")
async def refresh_knowledge():
    """
    重新加载知识库。

    用途：
    - 知识库更新后，热加载新数据
    - 不需要重启服务

    注意：当前实现是重新运行 initialize()
    initialize() 是幂等的，已有数据不会重复添加
    """
    # 重新获取知识库实例
    from knowledge.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    await kb.initialize()

    return {"status": "ok", "message": "知识库已重新加载"}

