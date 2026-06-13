# ============================================================
# agents/assistant_agent.py — 助理 Agent
# ============================================================
# 助理 Agent 是用户的日常对话助手。
# 它处理：
# - 一般问候和闲聊（"你好"、"今天天气如何"）
# - 不需要专业知识的问题
# - 当其他 Agent 都不匹配时的兜底（fallback）
#
# 特点：
# - 不依赖知识库
# - System Prompt 比较开放，允许自由对话
# - 可以记录用户偏好等简单记忆

from typing import Optional

from agents.base_agent import BaseAgent, AgentResult
from utils.llm_client import chat_stream
from memory.memory_manager import memory_manager


class AssistantAgent(BaseAgent):
    """
    助理 Agent：日常对话助手。

    适用场景：
    - 用户问候："你好"、"早上好"
    - 闲聊："你叫什么名字"
    - 一般咨询："帮我解释一下这个"
    - 其他 Agent 无法确定的兜底情况

    System Prompt 设计：
    - 友好、乐于助人的语气
    - 明确职责范围
    - 提示用户在需要专业知识时可以请求其他服务
    """

    name = "assistant_agent"  # Agent 标识

    system_prompt = """
        你是一个友好的金融助理助手。
        你可以帮用户处理日常对话，回答一般性问题。

        在对话中请遵循以下原则：
        1. 用中文回答，语气友好、专业
        2. 如果用户问到专业的金融问题（如投资决策、风险评估），
           请提醒用户可以切换到"决策分析"或"金融咨询"模式获取更专业的回答
        3. 不要编造事实，如果不确定就说不知道
        4. 回复简洁明了，不要太啰嗦
    """

    async def process(
        self,
        user_input: str,
        session_id: str,
    ) -> AgentResult:
        """
        处理用户输入：

        1. 从记忆管理器获取对话上下文（含滑动窗口）
        2. 构建系统消息 + 上下文
        3. 调用 LLM 生成回复（非流式）
        4. 将对话存入记忆管理器
        5. 返回结果

        参数：
            user_input: 用户输入
            session_id: 会话 ID

        返回：
            AgentResult
        """
        # ---- 步骤1：获取对话上下文 ----
        # memory_manager.get_context_messages 会：
        # - 自动创建/获取会话
        # - 应用滑动窗口裁剪
        # - 注入 system prompt
        messages = self.memory.get_context_messages(
            session_id=session_id,
            system_prompt=self.system_prompt,
        )

        # ---- 步骤2：添加用户消息 ----
        messages.append({"role": "user", "content": user_input})

        # ---- 步骤3：调用 LLM ----

        full_response = ""
        async for chunk in chat_stream(messages):
            full_response += chunk

        # ---- 步骤5：存入记忆管理器 ----
        # 保存 user 消息和 assistant 回复
        memory_manager.add_message(session_id, "user", user_input)
        memory_manager.add_message(session_id, "assistant", full_response)

        # ---- 步骤6：返回结果 ----
        return AgentResult(
            response=full_response,
            agent_name=self.name,
        )

