import json
from abc import abstractmethod
from typing import Any

from pgvector_template.core.embedder import BaseEmbeddingProvider

from utils.api_bedrock import get_bedrock_client_from_environ


class BedrockEmbeddingProvider(BaseEmbeddingProvider):
    """Abstract base for Bedrock embedding providers. Handles client setup and shared embed logic."""

    def __init__(self, model_id: str, verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.verbose = verbose
        self.bedrock_client: Any = get_bedrock_client_from_environ()

    @abstractmethod
    def _build_payload(self, text: str) -> dict: ...

    @abstractmethod
    def _parse_response(self, response: dict) -> list[float]: ...

    def _invoke(self, text: str) -> list[float]:
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(self._build_payload(text)),
            contentType="application/json",
            accept="application/json",
        )
        return self._parse_response(json.loads(response["body"].read()))

    def embed_text(self, text: str) -> list[float]:
        vector = self._invoke(text)
        if self.verbose:
            print(f"Embedding vector for '{text}': {vector}")
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = [self._invoke(t) for t in texts]
        if self.verbose:
            for text, vector in zip(texts, vectors, strict=True):
                print(f"Embedding vector for '{text}': {vector}")
        return vectors

    def get_dimensions(self) -> int:
        return 1024


class TitanEmbeddingProvider(BedrockEmbeddingProvider):
    """Embedding provider for Amazon Titan models (amazon.titan-embed-*).

    Only supports Titan models — payload shape and response parsing are
    hard-coded to the Titan API. Kept as a legacy fallback; prefer CohereEmbeddingProvider.
    """

    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0", **kwargs):
        super().__init__(model_id=model_id, **kwargs)

    def _build_payload(self, text: str) -> dict:
        return {"inputText": text}

    def _parse_response(self, response: dict) -> list[float]:
        return response["embedding"]


class CohereEmbeddingProvider(BedrockEmbeddingProvider):
    """Embedding provider for Cohere Embed v4 (cohere.embed-v4:0).

    Supports input_type separation: use "search_document" when embedding corpus
    content (upload pipeline), and "search_query" when embedding user queries
    (retrieval side in logseq-mcp).
    """

    def __init__(
        self,
        model_id: str = "cohere.embed-v4:0",
        input_type: str = "search_document",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.input_type = input_type

    def _build_payload(self, text: str) -> dict:
        return {
            "texts": [text],
            "input_type": self.input_type,
            "embedding_types": ["float"],
        }

    def _parse_response(self, response: dict) -> list[float]:
        return response["embeddings"]["float"][0]
