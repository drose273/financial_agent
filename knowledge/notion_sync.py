"""
knowledge/notion_sync.py - Notion -> ChromaDB 知识同步器
"""
import re
from typing import Optional
from notion_client import Client as NotionClient
from knowledge.vector_store import VectorStore


class NotionSync:
    def __init__(self, token: str, page_id: str = None, chunk_size: int = 500):
        self.client = NotionClient(auth=token)
        self.page_id = page_id
        self.chunk_size = chunk_size
        self.store = VectorStore()
        self.store.create_collection()

    async def sync_to_vectorstore(self, page_ids: list = None):
        ids = page_ids or [self.page_id]
        all_docs = []
        for pid in ids:
            print(f"[NotionSync] 读取页面: {pid}")
            blocks = self._read_page_blocks(pid)
            text = self._blocks_to_text(blocks)
            chunks = self._chunk_text(text, self.chunk_size)
            for i, chunk in enumerate(chunks):
                all_docs.append((
                    chunk,
                    {"title": f"Notion-{pid[:8]}", "source": f"notion_page_{pid}", "chunk": i, "total_chunks": len(chunks)},
                ))
            print(f"  -> 切出 {len(chunks)} 段, 共 {len(text)} 字符")
        if all_docs:
            await self.store.add_documents(all_docs)
            print(f"[NotionSync] 完成！新增 {len(all_docs)} 条知识")
        else:
            print("[NotionSync] 未读取到内容")

    def _read_page_blocks(self, page_id: str) -> list:
        blocks = []
        cursor = None
        while True:
            resp = self.client.blocks.children.list(block_id=page_id, start_cursor=cursor, page_size=100)
            blocks.extend(resp["results"])
            if not resp.get("has_more"):
                break
            cursor = resp["next_cursor"]
        return blocks

    def _blocks_to_text(self, blocks: list) -> str:
        lines = []
        for block in blocks:
            bt = block.get("type", "")
            rt_list = block.get(bt, {}).get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rt_list)
            if not text.strip():
                continue
            if bt in ("heading_1", "heading_2", "heading_3"):
                level = bt[-1]
                lines.append(f"\\n{"#" * int(level)} {text}\\n")
            elif bt in ("bulleted_list_item", "numbered_list_item"):
                lines.append(f"  - {text}")
            elif bt == "code":
                lines.append(f"\\n```\\n{text}\\n```\\n")
            elif bt == "quote":
                lines.append(f"> {text}")
            elif bt == "to_do":
                checked = block["to_do"].get("checked", False)
                prefix = "[x]" if checked else "[ ]"
                lines.append(f"{prefix} {text}")
            else:
                lines.append(text)
        return "\\n\\n".join(lines)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list:
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                search = text.rfind(chr(12290), max(end-50, start), end)
                if search > max(end-50, start):
                    end = search + 1
            chunks.append(text[start:end].strip())
            start = end
        return chunks
