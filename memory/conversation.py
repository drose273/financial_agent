# ============================================================
# memory/conversation.py — 单次会话的数据结构
# ============================================================
# 这个文件定义了"一次会话"（Conversation）类。
# 每个用户与系统的对话被组织成一个会话（session），
# 会话由一系列消息组成，并维护滑动窗口以控制上下文长度。

# dataclass 是 Python 3.7+ 引入的装饰器
# 它自动生成 __init__、__repr__ 等方法，适合作为纯数据容器
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ============================================================
# 消息类型：单条对话记录
# ============================================================
# 为什么用 dict 而不是类？
# - OpenAI/Ollama API 要求的消息格式就是 list[dict]
# - 所以我们直接用 dict[role, content] 存储，省去转换
# - 格式: {"role": "user", "content": "你好"}
# - role 可以是: "system", "user", "assistant"
# - 为了方便追踪时间，我们加了一个 timestamp 字段


# ============================================================
# Conversation 类：一次完整的对话会话
# ============================================================
@dataclass
class Conversation:
    """
    表示用户的一次对话会话。

    属性：
        session_id: 会话唯一标识（由调用方生成，如 UUID）
        messages:    完整的消息历史列表（包含 system prompt + 所有轮次）
                     格式: [{"role": "...", "content": "...", "timestamp": ...}]
        created_at:  会话创建时间
        updated_at:  最后更新时间
        metadata:    额外信息（如用户标签、来源等）

    滑动窗口机制（在 memory_manager.py 中实现）：
        messages 保留全部历史用于展示，但传给 LLM 时只取最近 N 轮
    """
    session_id: str
    messages: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def add_message(self, role: str, content: str):
        """
        添加一条消息到会话历史。

        参数：
            role: 消息角色 ("user" 或 "assistant" 或 "system")
            content: 消息文本内容

        注意：这里只是追加到列表末尾
        滑动窗口裁剪在 memory_manager 里做
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.updated_at = datetime.now()

    def to_llm_messages(self) -> list[dict]:
        """
        转换成 LLM API 所需的消息格式（不含 timestamp）。

        返回：
            list[dict]: 格式为 [{"role": "...", "content": "..."}, ...]
            注意：去掉了我们额外加的 timestamp 字段
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.messages
        ]

