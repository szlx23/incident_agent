import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # 文本分片参数：chunk_max_size 越大上下文越完整，chunk_overlap 用于相邻分片语义衔接。
    chunk_max_size = 800
    chunk_overlap = 100
    chunk_min_size = 200

    # 向量维度需与 embedding 模型输出完全一致，否则入库或检索会报错。
    embedding_dimension = 1024

    # 上传白名单扩展名：只允许可解析的文本文件进入索引流程。
    allow_extension = ["md", "txt"]

    ark_api_key = os.getenv("ARK_API_KEY")

    doubao_model_id = os.getenv("DOUBAO_MODEL_ID")
    doubao_base_url = os.getenv("DOUBAO_BASE_URL")

    doubao_embedding_model_id = os.getenv("DOUBAO_EMBEDDING_MODEL_ID")
    doubao_embedding_base_url = os.getenv("DOUBAO_EMBEDDING_BASE_URL")

    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    milvus_host = os.getenv("MILVUS_HOST")
    # 统一为 int，避免下游客户端收到字符串端口。
    milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

    # 统一上传目录，便于后续索引、覆盖写入和历史清理策略落地。
    upload_dir = "uploads/"