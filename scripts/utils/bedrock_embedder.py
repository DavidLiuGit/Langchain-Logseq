from langchain_aws.embeddings import BedrockEmbeddings
from pgvector_template.core.embedder import BaseEmbeddingProvider

from utils.api_bedrock import get_bedrock_client_from_environ


class BedrockEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider for Amazon Bedrock"""

    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0", verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.bedrock_client = get_bedrock_client_from_environ()
        self.bedrock_embeddings = BedrockEmbeddings(
            model_id=model_id,
            client=self.bedrock_client,
        )

    def embed_text(self, text: str) -> list[float]:
        """Get embedding for text"""
        vector = self.bedrock_embeddings.embed_documents([text])[0]
        if self.verbose:
            print(f"Embedding vector for '{text}': {vector}")
        return vector
    
    def embed_batch(self, texts) -> list[list[float]]:
        """Generate embeddings for multiple texts"""
        vectors = self.bedrock_embeddings.embed_documents(texts)
        if self.verbose:
            for i, text in enumerate(texts):
                print(f"Embedding vector for '{text}': {vectors[i]}")
        return vectors

    def get_dimensions(self):
        return 1024
