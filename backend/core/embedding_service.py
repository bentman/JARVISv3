"""
Feature Hashing Embedding Service for JARVISv3
Provides zero-dependency, deterministic embeddings as fallback for semantic search reliability.
"""
import re
import hashlib
import numpy as np
from typing import List, Optional
from ..ai.context.schemas import TaskContext


class FeatureHashingEmbeddingService:
    """
    Lightweight, local-first embedding service using feature hashing.
    Provides deterministic embeddings suitable for approximate semantic search.
    Zero external dependencies - works offline immediately.
    """

    def __init__(self, dim: int = 768):
        """
        Initialize feature hashing embedding service.

        Args:
            dim: Embedding dimension (default matches transformer embeddings for compatibility)
        """
        self.dim = dim

    def tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace + punctuation tokenization.
        Lowercases for normalization, filters out empty tokens.
        """
        tokens = re.split(r"[^a-z0-9]+", text.lower())
        return [t for t in tokens if t]

    def hash_token(self, token: str) -> int:
        """
        Hash token to dimension index using SHA1 for stable hashing.
        Maps to vector dimension range for consistent embedding size.
        """
        # Use sha1 for stable hashing across runs/platforms
        hash_obj = hashlib.sha1(token.encode("utf-8"))
        hash_int = int(hash_obj.hexdigest(), 16)
        return hash_int % self.dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using feature hashing.

        Returns:
            numpy array of shape (len(texts), dim) with L2-normalized embeddings
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Initialize embedding matrix
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = self.tokenize(text)

            # Feature hashing: increment dimension for each token
            for token in tokens:
                dim_idx = self.hash_token(token)
                embeddings[i, dim_idx] += 1.0

        # L2 normalize each embedding vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero for empty texts
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string.

        Returns:
            numpy array of shape (dim,) with L2-normalized embedding
        """
        embeddings = self.embed_texts([query])
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self.dim, dtype=np.float32)

    def get_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity between query and document embeddings.

        Args:
            query_embedding: Query embedding vector
            doc_embedding: Document embedding vector

        Returns:
            Cosine similarity score (higher = more similar)
        """
        # Cosine similarity = dot product (since vectors are L2 normalized)
        similarity = np.dot(query_embedding, doc_embedding)
        return float(similarity)

    def search_similar(
        self,
        query: str,
        doc_embeddings: np.ndarray,
        doc_texts: List[str],
        top_k: int = 5
    ) -> List[tuple[float, str]]:
        """
        Find most similar documents to query using feature hashing embeddings.

        Args:
            query: Search query string
            doc_embeddings: Pre-computed document embeddings (shape: n_docs, dim)
            doc_texts: Corresponding document texts
            top_k: Number of top results to return

        Returns:
            List of (similarity_score, text) tuples, sorted by similarity (descending)
        """
        if len(doc_embeddings) == 0 or len(doc_texts) != len(doc_embeddings):
            return []

        query_embedding = self.embed_query(query)

        # Calculate similarities for all documents
        similarities = np.dot(doc_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return (similarity, text) pairs
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity > 0:  # Only include positive similarities
                results.append((float(similarity), doc_texts[idx]))

        return results


# Global instance for backward compatibility
feature_hashing_service = FeatureHashingEmbeddingService()


class EmbeddingService:
    """
    Unified embedding service that supports multiple embedding strategies.
    Provides seamless fallback from transformer embeddings to feature hashing.
    """

    def __init__(self, primary_dim: int = 768):
        """
        Initialize unified embedding service.

        Args:
            primary_dim: Primary embedding dimension (matches transformer embeddings)
        """
        self.primary_dim = primary_dim
        self.feature_hashing = FeatureHashingEmbeddingService(dim=primary_dim)
        self._transformer_available = False
        self._sentence_transformers = None

    def _initialize_transformer_embeddings(self):
        """Lazy initialization of transformer embeddings"""
        if self._transformer_available:
            return True

        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight model for reliability
            self._sentence_transformers = SentenceTransformer('all-MiniLM-L6-v2')
            self._transformer_available = True
            return True
        except ImportError:
            # sentence-transformers not available, will use feature hashing
            return False

    def embed_texts(self, texts: List[str], use_transformer: bool = True) -> np.ndarray:
        """
        Generate embeddings with automatic fallback to feature hashing.

        Args:
            texts: List of texts to embed
            use_transformer: Whether to try transformer embeddings first

        Returns:
            numpy array of embeddings
        """
        if use_transformer and self._initialize_transformer_embeddings():
            try:
                # Use transformer embeddings (higher quality)
                embeddings = self._sentence_transformers.encode(texts, convert_to_numpy=True)
                # Ensure consistent dimension and normalization
                if embeddings.shape[1] != self.primary_dim:
                    # Resize if needed (simple approach - could be improved)
                    if embeddings.shape[1] > self.primary_dim:
                        embeddings = embeddings[:, :self.primary_dim]
                    else:
                        # Pad with zeros
                        padding = np.zeros((embeddings.shape[0], self.primary_dim - embeddings.shape[1]))
                        embeddings = np.concatenate([embeddings, padding], axis=1)

                # L2 normalize
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                embeddings = embeddings / norms

                return embeddings.astype(np.float32)

            except Exception as e:
                # Fallback to feature hashing on any error
                print(f"Transformer embeddings failed ({e}), falling back to feature hashing")

        # Use feature hashing (deterministic, zero dependencies)
        return self.feature_hashing.embed_texts(texts)

    def embed_query(self, query: str, use_transformer: bool = True) -> np.ndarray:
        """
        Generate embedding for a single query with fallback support.
        """
        embeddings = self.embed_texts([query], use_transformer=use_transformer)
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self.primary_dim, dtype=np.float32)

    def get_embedding_info(self) -> dict:
        """
        Get information about current embedding configuration.
        """
        return {
            "primary_dimension": self.primary_dim,
            "transformer_available": self._transformer_available,
            "feature_hashing_available": True,
            "current_strategy": "transformer" if self._transformer_available else "feature_hashing"
        }


# Global unified embedding service instance
embedding_service = EmbeddingService()
