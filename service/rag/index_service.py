import logging
from pathlib import Path

from config import Config
from service.rag.embedding_service import EmbeddingService
from service.rag.split_service import SplitService
from service.rag.vector_store_service import VectorStoreService


class IndexService:
    def __init__(self):
        # 向量库服务：负责按文档分片写入、以及按来源路径删除旧向量。
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
        # 分片服务：先按标题结构切，再按长度切，兼顾语义边界与模型上下文窗口。
        self.split_store_service = SplitService(
            chunk_max_size=Config.chunk_max_size,
            chunk_overlap=Config.chunk_overlap,
            chunk_min_size=Config.chunk_min_size
        )

    def index_single_file(self, file_path: str):
        path = Path(file_path).resolve()

        # 读取原始文本，作为后续切分和 embedding 的输入源。
        content = path.read_text(encoding="utf-8")
        logging.info(f"读取文件: {path}, 内容长度: {len(content)} 字符")

        normalized_path = path.as_posix()
        # 先删旧索引再写新索引，避免重复文档导致召回结果污染。
        self.vector_store_service.delete_by_resource(normalized_path)

        documents = self.split_store_service.split_markdown(content=content, file_path=file_path)

        if documents:
            # 空文档不入库，减少无效向量与噪音数据。
            self.vector_store_service.add_documents(documents)
            logging.info(f"文档索引完成: {file_path} -> {len(documents)} 个分片")

