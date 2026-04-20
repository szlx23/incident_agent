from langchain_core.embeddings import Embeddings
from openai import OpenAI


class EmbeddingService(Embeddings):
    def __init__(self, api_key: str, base_url: str, model: str, dimension: int = 1024):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimension,
            encoding_format="float"
        )

        return [item.embedding for item in result.data]

    def embed_query(self, text: str) -> list[float]:
        result = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimension,
            encoding_format="float"
        )

        return result.data[0].embedding
