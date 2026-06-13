# vector_store.py — ChromaDB 向量数据库封装
# 使用 Ollama embedding API，不依赖 ChromaDB 默认的 ONNX 模型

import chromadb
from chromadb.config import Settings
from typing import Optional
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, RETRIEVAL_TOP_K
from utils.llm_client import get_embedding


class VectorStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None

    def create_collection(self):
        # 创建集合，明确指定不用默认 embedding 函数
        # 我们后续手动传 embedding 向量
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=None,      # 重要：不用 ChromaDB 自带的 ONNX 模型
            metadata={"hnsw:space": "cosine"},  # 使用余弦距离
        )

    async def add_documents(
        self,
        documents: list[tuple[str, dict]],
    ):
        if self.collection is None:
            self.create_collection()

        texts = []
        metadatas = []
        ids = []
        embeddings = []

        for i, (content, metadata) in enumerate(documents):
            text = content.strip()
            texts.append(text)
            metadatas.append(metadata)
            ids.append(f"doc_{i}")
            # 使用 Ollama 生成 embedding 向量
            emb = await get_embedding(text)
            embeddings.append(emb)

        # 传向量，不传 documents（避免 ChromaDB 再用默认函数 embed）
        self.collection.add(
            embeddings=embeddings,   # 直接传向量
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

    async def search(
        self,
        query: str,
        k: int = RETRIEVAL_TOP_K,
    ) -> list[dict]:
        if self.collection is None:
            self.create_collection()

        # 用 Ollama 把查询转成向量
        query_embedding = await get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "id": results["ids"][0][i],
                })

        return formatted_results

    def get_collection_info(self) -> dict:
        if self.collection is None:
            self.create_collection()
        try:
            count = self.collection.count()
        except:
            count = 0
        return {"name": self.collection.name, "count": count}
