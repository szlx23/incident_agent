import logging
import time
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus


class VectorStoreService:
    def __init__(self, embedding: Embeddings, milvus_host: str, milvus_port: int):
        # 统一约定集合字段名，保证写入、检索、删除使用同一套 schema。
        self.vector_store = Milvus(
            embedding_function=embedding,
            collection_name="incident_agent_rag",
            connection_args={
                "host": milvus_host,
                "port": milvus_port
            },
            auto_id=False,
            drop_old=False,
            text_field="content",
            vector_field="vector",
            primary_field="id",
            metadata_field="metadata"
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        start_time = time.time()

        # 显式生成主键，便于后续排障、去重和针对性删除。
        ids = [str(uuid.uuid4()) for _ in documents]

        result_ids = self.vector_store.add_documents(documents, ids=ids)

        elapsed = time.time() - start_time
        logging.info(
            f"批量添加 {len(documents)} 个文档完成,"
            f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
        )

        return result_ids

    def delete_by_resource(self, file_path: str) -> int:
        # 通过 metadata 中的源文件路径删除旧索引，确保重建索引时数据单一且可预测。
        expression = f'metadata["_source"] == "{file_path}"'
        result = self.vector_store.delete(expr=expression)
        delete_count = result.delete_count if hasattr(result, "delete_count") else 0
        logging.info(f"删除文件旧数据: {file_path}, 删除数量: {delete_count}")
        return delete_count