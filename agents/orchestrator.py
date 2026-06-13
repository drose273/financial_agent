# ============================================================
# agents/orchestrator.py — Agent 编排器
# ============================================================
# 编排器（Orchestrator）是系统的"交通指挥员"。
# 它的职责：
#   1. 接收用户消息
#   2. 判断应该由哪个 Agent 处理（意图识别）
#   3. 将请求转发给对应的 Agent
#   4. 返回处理结果
#
# 意图识别策略（双阶段）：
#   第一阶段：关键词匹配（快速、低开销）
#    - 包含"投资、风险、决策、对比"等 → 决策 Agent
#    - 包含"是什么、定义、概念、区别"等 → 咨询 Agent
#    - 其他 → 助理 Agent（兜底）
#
#   第二阶段（可选）：LLM 辅助判断
#    - 当关键词匹配不明确时
#    - 让 LLM 自己判断应该由哪个 Agent 处理
#    - 这里出于简化，我们用关键词 + 简单规则

import re  # 正则表达式，用于关键词匹配
from typing import Optional

from agents.base_agent import AgentResult
from agents.assistant_agent import AssistantAgent
from agents.decision_agent import DecisionAgent
from agents.consulting_agent import ConsultingAgent
from memory.memory_manager import memory_manager


# ============================================================
# 关键词路由规则
# ============================================================
# 定义：某些关键词 → 路由到某个 Agent
# 用正则表达式做匹配，忽略大小写

# 决策 Agent 触发关键词（投资分析、决策类问题）
DECISION_KEYWORDS = [
    r"投资", r"风险", r"决策", r"对比", r"推荐",
    r"收益.*?风险", r"风险评估", r"配置", r"方案",
    r"该不该", r"值不值得", r"好不好", r"建仓",
    r"止损", r"止盈", r"仓位", r"抄底", r"追涨",
]

# 咨询 Agent 触发关键词（知识问答类问题）
CONSULTING_KEYWORDS = [
    r"是什么", r"什么是", r"定义", r"概念",
    r"区别", r"关系", r"作用", r"含义",
    r"解释", r"说明", r"举例", r"意思",
    r"如何", r"怎么", r"哪些", r"哪几种",
    r"ETF", r"市盈率", r"市净率", r"MACD",
    r"KDJ", r"RSI", r"股息", r"债券", r"GDP",
    r"CPI", r"LPR", r"基金",
]

# 助理 Agent 触发关键词（日常对话）
ASSISTANT_KEYWORDS = [
    r"你好", r"您好", r"嗨", r"早上好", r"晚上好",
    r"谢谢", r"再见", r"你是谁", r"你能做什么",
]


# ============================================================
# Orchestrator 类：Agent 编排器
# ============================================================
class Orchestrator:
    """
    Agent 编排器。

    职责：
    - 意图识别：判断用户想要什么
    - Agent 路由：把请求发给正确的 Agent
    - 结果整合：包装返回结果

    使用示例：
        orchestrator = Orchestrator()
        result = await orchestrator.process("什么是市盈率？", "session_001")
        print(result.response)

    路由流程图：
        用户输入
            │
            ▼
        ┌─────────────────┐
        │  关键词匹配      │
        │  第一阶段        │
        └─────┬───────────┘
              │
        ┌─────┴─────┬──────────┬──────────┐
        │           │          │          │
        ▼           ▼          ▼          ▼
    咨询Agent   决策Agent  助理Agent  助理Agent
    (知识问答)  (决策分析)  (日常对话)  (兜底)
    """

    def __init__(self):
        # 实例化所有 Agent
        # 注意：这些 Agent 共享同一个 memory_manager
        # 因为它们都在基类中通过 memory_manager 访问内存
        self.assistant = AssistantAgent()
        self.decision = DecisionAgent()
        self.consulting = ConsultingAgent()

        # 知识库初始化在 main.py 中单独调用
        self._knowledge_initialized = False

    async def initialize(self):
        """
        初始化编排器（在应用启动时调用）。
        主要是初始化咨询 Agent 的知识库。
        """
        await self.consulting.initialize_knowledge()
        self._knowledge_initialized = True

    async def process(
        self,
        user_input: str,
        session_id: str,
    ) -> AgentResult:
        """
        处理用户输入：意图识别 → 路由 → 返回结果。

        参数：
            user_input: 用户输入文本
            session_id: 会话唯一标识

        返回：
            AgentResult: 包含回复文本和 Agent 标识

        意图识别流程：
        1. 先将输入标准化（去空格、转小写）
        2. 依次匹配：决策关键词 → 咨询关键词 → 助理关键词
        3. 匹配到第一组关键词就路由到对应 Agent
        4. 如果都没匹配到，默认走助理 Agent（兜底）
        """
        # ---- 步骤1：意图识别 ----
        agent = self._route_intent(user_input)

        # ---- 步骤2：路由到对应的 Agent ----
        # 注意：所有 Agent 的 process 方法都是 async 的
        result = await agent.process(user_input, session_id)

        # ---- 步骤3：在结果中添加 Agent 标识 ----
        # 让调用方知道是哪个 Agent 处理的
        result.agent_name = agent.name

        return result

    def _route_intent(self, user_input: str):
        """
        意图识别：根据用户输入选择对应的 Agent。

        参数：
            user_input: 用户输入

        返回：
            BaseAgent: 对应的 Agent 实例

        逻辑：
        1. 先用正则匹配 DECISION_KEYWORDS
        2. 如果匹配到，返回 DecisionAgent
        3. 否则匹配 CONSULTING_KEYWORDS
        4. 如果匹配到，返回 ConsultingAgent
        5. 否则返回 AssistantAgent（兜底）
        """
        # 标准化输入：去空格（保留中文标点）
        text = user_input.strip()

        # 检查决策关键词
        for pattern in DECISION_KEYWORDS:
            if re.search(pattern, text, re.IGNORECASE):
                return self.decision

        # 检查咨询关键词
        for pattern in CONSULTING_KEYWORDS:
            if re.search(pattern, text, re.IGNORECASE):
                return self.consulting

        # 检查助理关键词
        for pattern in ASSISTANT_KEYWORDS:
            if re.search(pattern, text, re.IGNORECASE):
                return self.assistant

        # 默认使用助理 Agent（兜底）
        return self.assistant

    async def process_stream(
        self,
        user_input: str,
        session_id: str,
    ):
        """
        流式处理用户输入。

        注意：这个版本为了简化，流式输出由 API 层负责。
        这里我们只做路由和非流式处理。
        流式输出的实现在 api/routes.py 中。

        后续可以扩展：
        - 让每个 Agent 都支持流式输出
        - 在编排器中实现更复杂的流式路由
        """
        # 目前直接调用 process 方法
        # 流式输出由 API 层的 SSE 实现
        result = await self.process(user_input, session_id)
        return result

