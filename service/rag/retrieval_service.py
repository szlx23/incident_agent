import logging

from config import Config
from service.rag.embedding_service import EmbeddingService
from service.rag.vector_store_service import VectorStoreService


class RetrievalService:
    def __init__(self):
        self.vector_store_service = VectorStoreService(
            embedding=EmbeddingService(
                api_key=Config.ark_api_key,
                base_url=Config.doubao_embedding_base_url,
                model=Config.doubao_embedding_model_id,
                dimension=Config.embedding_dimension
            ),
            milvus_host=Config.milvus_host,
            milvus_port=Config.milvus_port
        )

    def retrieve_knowledge(self, query: str, k: int = 4) -> str:
        if not query or not query.strip():
            return "检索失败：查询内容为空。"

        try:
            docs = self.vector_store_service.vector_store.similarity_search(
                query=query.strip(),
                k=k
            )
            if not docs:
                return "未检索到相关知识。"

            contexts = []
            for idx, doc in enumerate(docs, start=1):
                source = ""
                if isinstance(doc.metadata, dict):
                    source = doc.metadata.get("_source", "")

                snippet = doc.page_content.strip()
                if len(snippet) > 500:
                    snippet = f"{snippet[:500]}..."

                contexts.append(
                    f"[片段{idx}] 来源: {source or '未知'}\n{snippet}"
                )
            return "\n\n".join(contexts)
        except Exception as e:
            logging.exception("知识库检索失败")
            return f"知识库检索失败：{e}"
