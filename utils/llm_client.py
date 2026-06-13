# ============================================================
# utils/llm_client.py — LLM 客户端封装
# ============================================================
# 这个文件封装了对 Ollama（或任何 OpenAI 兼容 API）的调用。
# 提供两种调用方式：
#   1. chat() — 普通非流式调用，等待完整返回
#   2. chat_stream() — 流式调用，逐 chunk 产出（async generator）
#
# ⚠ 关于异步编程（asyncio）的重要概念：
#   - async def 定义异步函数，await 暂停执行等待 I/O
#   - async generator（yield 在 async def 中）：逐段产出结果
#   - 两者都通过事件循环（event loop）调度，不会阻塞主线程
#   这使得在处理 LLM 请求的同时还能处理其他用户请求

import json
from typing import AsyncGenerator, Optional

# openai 库是 OpenAI 官方 Python SDK
# AsyncOpenAI 是它的异步版本，所有方法都返回 awaitable 对象
from openai import AsyncOpenAI

# 导入项目配置
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME

# ============================================================
# 1. 创建异步客户端实例
# ============================================================
# AsyncOpenAI 默认连接 https://api.openai.com/v1
# 我们把 base_url 改成 Ollama 的地址：http://localhost:11434/v1
# Ollama 不校验 API key，但 SDK 要求传一个，随便填
client = AsyncOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)


# ============================================================
# 2. 非流式调用（完整返回）
# ============================================================
async def chat(
    messages: list[dict],
    model: str = LLM_MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    非流式 LLM 调用：发送消息列表，等待完整回复。

    参数：
        messages: 消息列表，格式为 [{"role": "user", "content": "你好"}, ...]
                  role 可以是 "system"（系统指令）、"user"（用户）、"assistant"（助手）
        model:    模型名称，默认从 config 读取
        temperature: 温度参数，0~2，越低回答越确定，越高越有创造性
        max_tokens: 最大生成长度

    返回：
        str: 模型生成的完整回答文本

    使用示例：
        reply = await chat([{"role": "user", "content": "什么是市盈率？"}])
        print(reply)
    """
    # await 关键字会暂停当前协程，等待 API 返回
    # 在此期间事件循环可以处理其他任务
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,  # 非流式：一次性返回完整结果
    )

    # 从响应对象中提取文本内容
    # response.choices[0] 是第一个（也是唯一一个）候选回答
    # .message.content 就是模型生成的文本
    return response.choices[0].message.content


# ============================================================
# 3. 流式调用（逐 chunk 产出，SSE 风格）
# ============================================================
async def chat_stream(
    messages: list[dict],
    model: str = LLM_MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> AsyncGenerator[str, None]:
    """
    流式 LLM 调用：逐 chunk 产出文本。

    什么是流式（Streaming）？
    - 传统 API 等 LLM 生成完所有文字才一次性返回
    - 流式 API 在 LLM 每生成一小段文字后就立即返回
    - 适合聊天场景，给用户"实时打字"的体验

    什么是 AsyncGenerator（异步生成器）？
    - 函数里有 yield 关键字，说明它是生成器
    - async def + yield = 异步生成器
    - 调用方用 async for chunk in chat_stream(...) 来逐段消费
    - 每次 yield 后，函数暂停，事件循环可以去做其他事

    参数：
        同 chat() 函数

    产出：
        str: 每次 yield 一小段文本（通常是一个 token 或几个字符）

    使用示例：
        async for chunk in chat_stream([{"role": "user", "content": "讲个故事"}]):
            print(chunk, end="")  # 逐段打印，像 LLM 在打字
    """
    # stream=True 告诉 API 我们想要流式响应
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,  # ⚠ 流式模式
    )

    # async for 是异步 for 循环
    # 每次迭代 = 等 API 返回一个新的 chunk
    # 用 stream=True 时，API 会逐个返回 delta（增量）
    async for chunk in response:
        # 每个 chunk 的结构：
        # chunk.choices[0].delta 包含本次增量
        # .delta.content 是本 chunk 新增的文本（可能是 None）
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content  # 产出这段文本


# ============================================================
# 4. 获取 Embedding（文本向量化）
# ============================================================
async def get_embedding(text: str) -> list[float]:
    """
    将文本转换为向量（embedding）。

    什么是 Embedding？
    - 把一段文字转换成一个固定长度的数字列表（向量）
    - 语义相近的文本，在向量空间中的距离也更近
    - 这是 RAG（检索增强生成）的基石

    参数：
        text: 要向量化的文本

    返回：
        list[float]: 浮点数列表，即文本的向量表示

    使用场景：
        knowledge/vector_store.py 中会用到它来把文档转成向量存入数据库
    """
    # requests 超时时间设长一点，embedding 通常比对话慢
    response = await client.embeddings.create(
        model=LLM_MODEL_NAME,
        input=text,
    )
    # embedding 在 response.data[0].embedding 中
    return response.data[0].embedding

