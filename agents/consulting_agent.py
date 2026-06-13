# ============================================================
# agents/consulting_agent.py — 咨询 Agent（RAG 检索增强）
# ============================================================
# 咨询 Agent 是使用 RAG（检索增强生成）的专业问答 Agent。
# 它的工作方式是：
#   1. 用户提问 → 2. 从知识库检索相关知识 →
#   3. 将知识注入 prompt → 4. LLM 基于知识回答
#
# 适用场景：
# - 概念解释："什么是市盈率？"
# - 知识问答："ETF 和普通基金有什么区别？"
# - 术语解释："MACD 指标是什么？"
# - 任何需要"事实性知识"的问题

from typing import Optional

from agents.base_agent import BaseAgent, AgentResult
from utils.llm_client import chat_stream
from memory.memory_manager import memory_manager
from knowledge.knowledge_base import KnowledgeBase


class ConsultingAgent(BaseAgent):
    """
    咨询 Agent：基于 RAG 的专业金融知识问答。

    核心流程（RAG）：
    1. 接收用户问题
    2. 从知识库（ChromaDB）检索最相关文档
    3. 将检索结果构建为"参考资料"文本
    4. 将参考资料注入到 system prompt 中
    5. 调用 LLM 生成基于知识的回答
    6. 返回结果

    为什么 RAG 比纯 LLM 回答更好？
    - 减少幻觉：LLM 有参考资料，不会无中生有
    - 知识可控：回答基于我们灌入的知识库
    - 可追溯：可以标注信息来源（引用）  # 感谢您的指正！
    """

    name = "consulting_agent"

    # 基础 system prompt（不包含 RAG 检索结果）
    system_prompt = """
        你是金融知识咨询专家。
        你的知识来源于内部的金融知识库，请基于提供的参考资料回答用户问题。

        回答原则：
        1. 只能基于参考资料中的内容回答问题
        2. 如果参考资料中没有相关信息，明确告诉用户"我没找到相关信息"
        3. 在回答中引用参考资料中的内容时，标注对应的编号 [1][2] 等
        4. 用通俗易懂的语言解释金融概念，举生活中的例子帮助理解
        5. 如果你需要更多信息才能回答，请主动询问用户
    """

    def __init__(self):
        super().__init__()
        # 初始化知识库管理器
        # 注意：知识库的 initialize() 在应用启动时由 main.py 调用
        self.knowledge_base = KnowledgeBase()

    async def initialize_knowledge(self):
        """初始化知识库（在应用启动时调用）。"""
        await self.knowledge_base.initialize()

    async def process(
        self,
        user_input: str,
        session_id: str,
    ) -> AgentResult:
        """
        处理用户输入（RAG 流程）。

        这是 RAG 的核心实现：
        1. 检索 → 2. 增强 → 3. 生成

        步骤详解：

        【检索 Retrieve】
        - 把用户问题传给 KnowledgeBase
        - KnowledgeBase 调用 VectorStore 做语义检索
        - 返回最相关的 3 条知识

        【增强 Augment】
        - 把检索结果格式化为"参考资料"
        - 注入到 system prompt 中
        - 形成：system prompt + 参考资料 + 对话历史 + 当前问题

        【生成 Generate】
        - 把增强后的 messages 发给 LLM
        - LLM 基于参考资料生成回答
        - 回答会更准确、更有依据
        """
        # ---- 【检索】从知识库中查找相关信息 ----
        # await 因为 search 是异步的（调用 Ollama embedding API）
        rag_context = await self.knowledge_base.build_rag_context(
            user_input,
            k=3,  # 检索最相关的 3 条知识
        )

        # ---- 【增强】构建 LLM 输入 ----
        # 获取对话上下文
        messages = self.memory.get_context_messages(
            session_id=session_id,
            system_prompt=self.system_prompt,
        )

        # 如果有检索到相关知识，注入到上下文中
        # extra_context 是一个 system 消息，告诉 LLM "请参考以下资料"
        if rag_context:
            # 把参考资料作为 system message 注入
            messages.append({
                "role": "system",
                "content": f"以下是来自金融知识库的参考资料，请基于这些内容回答：\n{rag_context}",
            })

        # 添加用户问题
        messages.append({"role": "user", "content": user_input})

        # ---- 【生成】调用 LLM ----

        # 收集完整回复
        full_response = ""
        async for chunk in chat_stream(messages):
            full_response += chunk

        # 存入记忆管理器
        memory_manager.add_message(session_id, "user", user_input)
        memory_manager.add_message(session_id, "assistant", full_response)

        # 在元数据中返回检索到的知识列表（方便调试）
        return AgentResult(
            response=full_response,
            agent_name=self.name,
            metadata={
                "rag_context": rag_context,
                "has_knowledge": bool(rag_context),
            },
        )

