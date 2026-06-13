# ============================================================
# memory/memory_manager.py — 上下文记忆管理器
# ============================================================
# 这个文件管理所有对话会话的"记忆"。
# 核心功能：
#   1. 创建/获取会话（session）
#   2. 滑动窗口裁剪（只保留最近 N 轮对话）
#   3. 超时会话自动清理
#   4. 提供带上下文的 LLM 输入

import time
from typing import Optional

from memory.conversation import Conversation
from config import MEMORY_WINDOW_SIZE, SESSION_TIMEOUT_SECONDS


# ============================================================
# MemoryManager 类：全局记忆管理器
# ============================================================
# 设计模式：单例模式（Singleton）
# - 整个程序只有一个 MemoryManager 实例
# - 所有 Agent 共享同一个记忆管理器
class MemoryManager:
    """
    全局对话记忆管理器。

    职责：
    - 创建新会话、获取现有会话
    - 管理滑动窗口（只保留近期对话，节省 token 和上下文空间）
    - 清理长时间不活跃的会话

    工作原理：
    - sessions 是一个字典：{session_id: Conversation 对象}
    - 每个 session_id 对应一个独立对话
    """
    def __init__(self):
        # 核心数据：所有会话的字典
        # key：session_id (str)
        # value：Conversation 对象
        self.sessions: dict[str, Conversation] = {}

    def get_or_create_session(self, session_id: str) -> Conversation:
        """
        获取已有会话，或创建新会话。

        参数：
            session_id: 会话唯一标识

        返回：
            Conversation: 对应的会话对象

        逻辑：
        - 如果 session_id 已存在，返回会话
        - 如果不存在，创建新会话并存入字典
        - 每次访问都会更新 updated_at
        """
        if session_id not in self.sessions:
            # 创建新会话
            # Conversation 的 dataclass 会自动生成 created_at
            self.sessions[session_id] = Conversation(session_id=session_id)
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str) -> Conversation:
        """
        在指定会话中添加一条消息，并做滑动窗口裁剪。

        参数：
            session_id: 会话标识
            role:  消息角色 ("user" | "assistant")
            content: 消息内容

        返回：
            Conversation: 更新后的会话对象
        """
        conv = self.get_or_create_session(session_id)
        conv.add_message(role, content)

        # ⚠ 滑动窗口裁剪（Sliding Window）
        # =================================
        # 为什么需要滑动窗口？
        # - LLM 的上下文窗口有限（如 8K tokens）
        # - 对话越长，token 消耗越大，成本越高
        # - 很久以前的对话对当前回复帮助不大
        #
        # 怎么做？
        # - 保留系统 prompt（role == "system"）
        # - 保留最近 N 轮对话（user + assistant 成对出现）
        # - 丢弃更早的历史
        #
        # 注意：我们裁剪的是 messages 列表，不是数据库
        # 裁剪后旧消息就丢了，因为我们是内存存储
        self._trim_conversation(conv)

        return conv

    def get_context_messages(
        self,
        session_id: str,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """
        获取适合传入 LLM 的消息列表（带系统提示 + 滑动窗口）。

        参数：
            session_id: 会话标识
            system_prompt: 可选的系统提示词（system message）

        返回：
            list[dict]: 格式为 [{"role": "...", "content": "..."}, ...]
                        ready to be passed to LLM API

        流程：
        1. 如果传了 system_prompt，作为第一条消息
        2. 从会话中获取历史消息（已裁剪过的）
        3. 合并返回
        """
        conv = self.get_or_create_session(session_id)
        messages = conv.to_llm_messages()

        # 如果传了 system prompt，加到最前面
        # system prompt 是 LLM 的系统指令，优先级最高
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return messages

    def _trim_conversation(self, conv: Conversation):
        """
        滑动窗口裁剪：只保留最近 N 轮对话。

        参数：
            conv: 要裁剪的会话对象

        实现细节：
        - MEMORY_WINDOW_SIZE 在 config.py 中定义
        - 从后往前保留 N 对 (user + assistant) 消息
        - system prompt 不受影响（在调用时动态传入，不在 messages 里）
        """
        window = MEMORY_WINDOW_SIZE

        # ---- 只考虑 user 和 assistant 的消息（排除特殊消息） ----
        # 实际所有消息都在 messages 列表中
        # 我们直接保留 messages[-window*2:] 即可
        # 因为每条 user 消息通常对应一条 assistant 回复
        if len(conv.messages) > window * 2:
            # 保留最新的 window*2 条消息（user + assistant 成对）
            conv.messages = conv.messages[-(window * 2):]

    def clear_session(self, session_id: str):
        """
        清空指定会话。

        参数：
            session_id: 要清空的会话标识
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_session_history(self, session_id: str) -> list[dict]:
        """
        获取会话完整历史（用于前端展示或调试）。

        参数：
            session_id: 会话标识

        返回：
            list[dict]: 包含 timestamp 的完整消息列表
        """
        conv = self.get_or_create_session(session_id)
        return conv.messages

    def cleanup_expired_sessions(self):
        """
        清理超时会话（定期调用）。

        逻辑：
        - 遍历所有会话
        - 如果 updated_at 距今超过 SESSION_TIMEOUT_SECONDS，删除
        - 防止内存泄漏
        """
        now = time.time()
        expired_ids = []

        for session_id, conv in self.sessions.items():
            # 将 datetime 转为 timestamp 进行比较
            age = now - conv.updated_at.timestamp()
            if age > SESSION_TIMEOUT_SECONDS:
                expired_ids.append(session_id)

        for session_id in expired_ids:
            del self.sessions[session_id]

    @property
    def active_sessions_count(self) -> int:
        """
        当前活跃会话数量（只读属性）。
        用于监控和调试。
        """
        return len(self.sessions)


# ============================================================
# 全局单例：整个应用共享同一个记忆管理器
# ============================================================
# 为什么是全局变量？
# - 所有模块都导入同一个 memory_manager 实例
# - 不需要手动传递对象，方便使用
memory_manager = MemoryManager()

