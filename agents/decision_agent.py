# ============================================================
# agents/decision_agent.py — 决策 Agent
# ============================================================
# 决策 Agent 专注于金融决策分析。
# 它处理：
# - 投资风险评估（"这只股票风险大吗？"）
# - 方案对比（"A 基金和 B 基金哪个好？"）
# - 趋势分析（"现在适合入场吗？"）
# - 配置建议（"我月薪 1 万怎么配置？"）
#
# 特点：
# - System Prompt 要求分步骤推理，输出决策过程
# - 温度参数较低（temperature=0.3），回复更"谨慎"
# - 强调免责声明（不构成投资建议）

from typing import Optional

from agents.base_agent import BaseAgent, AgentResult
from utils.llm_client import chat_stream
from memory.memory_manager import memory_manager


class DecisionAgent(BaseAgent):
    """
    决策 Agent：金融分析与决策支持。

    处理逻辑：
    - 接收用户关于金融决策的咨询
    - 结合对话历史进行分析
    - 要求 LLM 分步骤输出推理过程
    - 最后给出结论和建议

    设计思路：
    - 决策类问题需要考虑多种因素
    - 分步骤推理（Chain-of-Thought）让回答更可靠
    - 明确的免责声明：不构成投资建议
    """

    name = "decision_agent"

    system_prompt = """
        你是金融决策分析专家。你的职责是帮用户分析金融决策问题。

        在处理每一条请求时，请严格按以下步骤进行：

        第一步【信息收集】：明确用户的需求和当前状况
        第二步【因素分析】：列出影响决策的关键因素（风险、收益、流动性等）
        第三步【方案对比】：如果有多个方案，客观对比各方案的优缺点
        第四步【风险评估】：分析每种选择可能的风险
        第五步【结论建议】：给出你的分析结论和倾向性建议

        重要原则：
        1. 保持客观中立，不偏向任何具体产品
        2. 必须包含风险提示：说明你的分析仅作为参考，不构成投资建议
        3. 用中文回答，表达清晰
        4. 如果信息不足，明确告诉用户还需要什么信息
        5. temperature=0.3：回复要谨慎、理性，不要过度乐观
    """

    async def process(
        self,
        user_input: str,
        session_id: str,
    ) -> AgentResult:
        """
        处理决策分析请求。

        与 AssistantAgent 类似，但有两点不同：
        1. System Prompt 不同（要求分步骤分析）
        2. 温度参数较低（temperature=0.3）
        3. 元数据会标记为解决类型为"decision"
        """
        # 获取对话上下文（含 system prompt）
        messages = self.memory.get_context_messages(
            session_id=session_id,
            system_prompt=self.system_prompt,
        )

        # 添加用户消息
        messages.append({"role": "user", "content": user_input})

        # 调用 LLM（决策 Agent 使用较低温度）
        # temperature=0.3：回答更确定、更保守

        # 收集完整回复
        full_response = ""
        async for chunk in chat_stream(messages, temperature=0.3):
            full_response += chunk

        # 存入记忆管理器
        memory_manager.add_message(session_id, "user", user_input)
        memory_manager.add_message(session_id, "assistant", full_response)

        return AgentResult(
            response=full_response,
            agent_name=self.name,
            metadata={"analysis_type": "decision"},
        )

