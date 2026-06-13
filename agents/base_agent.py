# ============================================================
# agents/base_agent.py — Agent 抽象基类
# ============================================================
# 这个文件定义了所有 Agent 的公共接口。
# 什么是 Agent（智能体）？
# - Agent 是一个能独立处理特定类型任务的"大脑"
# - 每个 Agent 有自己的 System Prompt、处理逻辑
# - 在这套系统中，Agent 的核心工作就是：
#   接收用户消息 → 结合上下文 → 调用 LLM → 返回回复
#
# 为什么要用抽象基类？
# - 保证所有 Agent 有统一的接口（多态）
# - 公共逻辑写在基类里，子类只需关注差异部分

from abc import ABC, abstractmethod
from typing import Optional

from memory.memory_manager import memory_manager
from memory.conversation import Conversation


# ============================================================
# AgentResult 数据类：Agent 处理结果
# ============================================================
class AgentResult:
    """
    Agent 处理返回的结果对象。

    Agent 的处理结果不仅仅是文本回复，还包括：
    - response:  生成的回复文本
    - agent_name: 处理该请求的 Agent 名称（用于标识）
    - metadata:  额外信息（如检索到的知识列表）

    为什么不用 dataclass？
    - 为了灵活性，允许子类添加更多字段
    """
    def __init__(
        self,
        response: str,
        agent_name: str = "unknown",
        metadata: Optional[dict] = None,
    ):
        self.response = response      # Agent 生成的回复文本
        self.agent_name = agent_name  # 负责的 Agent 名称
        self.metadata = metadata or {}  # 额外信息


# ============================================================
# BaseAgent 抽象基类
# ============================================================
class BaseAgent(ABC):
    """
    所有 Agent 的抽象基类。

    子类必须实现：
    - process(): 处理消息的核心方法

    子类可以选择重写：
    - system_prompt: Agent 的系统提示词

    方法调用流程：
    1. process(user_input, session_id) — 入口
    2. 子类在 process 中构建自己的处理逻辑
    3. 调用 LLM 获取回复
    4. 将对话存入记忆管理器
    5. 返回 AgentResult
    """

    # Agent 名称（子类覆盖）
    name: str = "base_agent"

    # 系统提示词（System Prompt）
    # System Prompt 是 LLM 的系统级指令
    # 它告诉 LLM 它应该扮演什么角色、如何回答问题
    # 每个 Agent 有不同的 System Prompt
    system_prompt: str = "你是一个智能助手，请用中文回答用户的问题。"

    def __init__(self):
        # 注入记忆管理器（方便调试和替换）
        self.memory = memory_manager

    @abstractmethod
    async def process(
        self,
        user_input: str,
        session_id: str,
    ) -> AgentResult:
        """
        处理用户输入并生成回复。

        参数：
            user_input: 用户输入的文本
            session_id: 会话唯一标识（用于记忆管理）

        返回：
            AgentResult: 包含回复文本、Agent 名称和元数据

        这是抽象方法，子类必须实现。
        """
        pass

    def _build_system_messages(
        self,
        extra_context: str = "",
    ) -> list[dict]:
        """
        构建系统消息列表。

        参数：
            extra_context: 额外的上下文信息（如 RAG 检索结果）

        返回：
            list[dict]: 格式 [{"role": "system", "content": "...", ...}]

        这个方法用于在 system prompt 中注入额外信息。
        例如咨询 Agent 会把 RAG 检索到的知识注入到 system prompt 中。
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if extra_context:
            # 将额外上下文追加到 system prompt 后面
            messages.append({
                "role": "system",
                "content": f"以下是参考资料，请基于这些内容回答用户问题：\n{extra_context}",
            })

        return messages

