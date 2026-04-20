import os

from dotenv import load_dotenv

load_dotenv()
class Config:
    chunk_max_size = 800
    chunk_overlap = 100
    chunk_min_size = 200

    ark_api_key= os.getenv("ARK_API_KEY")

    doubao_model_id=os.getenv("DOUBAO_MODEL_ID")
    doubao_base_url=os.getenv("DOUBAO_BASE_URL")

    doubao_embedding_model_id=os.getenv("DOUBAO_EMBEDDING_MODEL_ID")
    doubao_embedding_base_url=os.getenv("DOUBAO_EMBEDDING_BASE_URL")

    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    milvus_host = os.getenv("MILVUS_HOST")
    milvus_port = os.getenv("MILVUS_PORT")
