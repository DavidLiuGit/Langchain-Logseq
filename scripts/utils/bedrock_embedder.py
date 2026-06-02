import json
from typing import Any

from pgvector_template.core.embedder import BaseEmbeddingProvider

from utils.api_bedrock import get_bedrock_client_from_environ


class BedrockEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider for Amazon Bedrock Titan embedding models.

    Only supports Titan models (amazon.titan-embed-*). The request payload and
    response parsing are hard-coded to the Titan API shape ({\"inputText\": ...},
    response[\"embedding\"]). Passing a non-Titan model_id will produce a bad
    request or a KeyError at runtime.
    """

    def __init__(
        self, model_id: str = "amazon.titan-embed-text-v2:0", verbose=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.verbose = verbose
        self.bedrock_client: Any = get_bedrock_client_from_environ()

    def _invoke(self, text: str) -> list[float]:
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"inputText": text}),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["embedding"]

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
