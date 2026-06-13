#
# knowledge/knowledge_base.py — 知识库管理器
#

import asyncio

from knowledge.vector_store import VectorStore
from knowledge.seed_data import SEED_KNOWLEDGE


class KnowledgeBase:

    def __init__(self):
        self.store = VectorStore()
        self.initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """幂等初始化（双重检查锁定，防止并发重复初始化）。"""
        if self.initialized:
            return
        async with self._lock:
            if self.initialized:
                return
            self.store.create_collection()
            info = self.store.get_collection_info()

            if info["count"] == 0:
                print(f"知识库为空，正在灌入 {len(SEED_KNOWLEDGE)} 条种子知识...")
                await self.store.add_documents(SEED_KNOWLEDGE)
                print(f"知识库初始化完成，共 {len(SEED_KNOWLEDGE)} 条知识")
            else:
                print(f"知识库已有 {info['count']} 条数据，跳过初始化")

            self.initialized = True

    async def search_knowledge(self, query: str, k: int = 3) -> list[dict]:
        if not self.initialized:
            await self.initialize()
        return await self.store.search(query, k=k)

    async def build_rag_context(self, query: str, k: int = 3) -> str:
        results = await self.search_knowledge(query, k=k)
        if not results:
            return ""

        context_parts = ["\n=== 参考资料 ==="]
        for i, doc in enumerate(results, 1):
            title = doc["metadata"].get("title", "未命名")
            source = doc["metadata"].get("source", "未知来源")
            context_parts.append(
                f"[{i}] 标题：{title} | 来源：{source}\n{doc['content'].strip()}"
            )
        context_parts.append("=== 参考资料结束 ===\n")
        return "\n\n".join(context_parts)
