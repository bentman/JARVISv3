"""
Enhanced Vector Store Service for JARVISv3
Implements semantic search using FAISS with multiple embedding strategies and fallback support.
"""
import os
import pickle
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from .embedding_service import embedding_service
from .config import settings

# Optional imports for vector store
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Enhanced vector storage and retrieval using FAISS with embedding fallback support.
    Automatically falls back from transformer embeddings to feature hashing for reliability.
    """

    def __init__(
        self,
        index_path: str = "./vector_store.index",
        metadata_path: str = "./vector_metadata.pkl",
        embedding_dim: int = 768
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = embedding_dim
        self.index = None
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.embedding_strategy = "auto"  # "transformer", "feature_hashing", or "auto"
        self._initialized = False

    def initialize(self):
        """Initialize the vector store with embedding service"""
        if self._initialized:
            return

        if not HAS_FAISS:
            logger.warning("FAISS not found. Vector store will be disabled.")
            return

        try:
            # Load or create FAISS index
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded vector store with {self.index.ntotal} items")
            else:
                # Use cosine similarity (IndexFlatIP) for normalized embeddings
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = {}
                logger.info("Created new vector store")

            # Get embedding service info
            embedding_info = embedding_service.get_embedding_info()
            self.embedding_strategy = embedding_info["current_strategy"]

            self._initialized = True
            logger.info(f"Vector store initialized with {self.embedding_strategy} embeddings")

        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """
        Add texts and metadata to the store using current embedding strategy.
        Automatically falls back to feature hashing if transformer embeddings fail.
        """
        if not self._initialized:
            self.initialize()

        if not self._initialized or not self.index or not texts:
            return

        try:
            # Use available embeddings (will fallback to feature hashing if transformer unavailable)
            embeddings = embedding_service.embed_texts(texts, use_transformer=True)

            # Ensure embeddings are the right shape and normalized
            if embeddings.shape[1] != self.dimension:
                logger.warning(f"Embedding dimension mismatch: got {embeddings.shape[1]}, expected {self.dimension}")
                # Truncate or pad as needed
                if embeddings.shape[1] > self.dimension:
                    embeddings = embeddings[:, :self.dimension]
                else:
                    padding = np.zeros((embeddings.shape[0], self.dimension - embeddings.shape[1]))
                    embeddings = np.concatenate([embeddings, padding], axis=1)

            start_idx = self.index.ntotal
            self.index.add(embeddings)

            # Store metadata with embedding strategy info
            for i, meta in enumerate(metadatas):
                enhanced_meta = dict(meta)
                enhanced_meta["_embedding_strategy"] = self.embedding_strategy
                enhanced_meta["_embedding_dim"] = self.dimension
                self.metadata[start_idx + i] = enhanced_meta

            self.save()
            logger.debug(f"Added {len(texts)} texts to vector store")

        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search for similar texts using current embedding strategy.
        Returns similarity scores and metadata.
        """
        if not self._initialized:
            self.initialize()

        if not self._initialized or not self.index or self.index.ntotal == 0:
            return []

        try:
            # Generate query embedding using same strategy as stored documents
            query_embedding = embedding_service.embed_query(query, use_transformer=True)
            query_embedding = query_embedding.reshape(1, -1)

            # Ensure dimension matches
            if query_embedding.shape[1] != self.dimension:
                if query_embedding.shape[1] > self.dimension:
                    query_embedding = query_embedding[:, :self.dimension]
                else:
                    padding = np.zeros((1, self.dimension - query_embedding.shape[1]))
                    query_embedding = np.concatenate([query_embedding, padding], axis=1)

            # Search the index
            scores, indices = self.index.search(query_embedding.astype(np.float32), min(k, self.index.ntotal))

            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx in self.metadata:
                    # Convert distance to similarity (higher = more similar)
                    similarity = float(scores[0][i])
                    results.append((similarity, self.metadata[idx]))

            # Sort by similarity (descending)
            results.sort(key=lambda x: x[0], reverse=True)
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def search_with_fallback(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search with explicit fallback to feature hashing if transformer search fails.
        This ensures search reliability even when transformer embeddings are unavailable.
        """
        # Try primary search first
        results = self.search(query, k)
        if results:
            return results

        # Fallback to feature hashing if no results or error
        logger.info("Primary search failed, attempting fallback with feature hashing")

        try:
            # Generate query embedding using feature hashing only
            query_embedding = embedding_service.embed_texts([query], use_transformer=False)[0]
            query_embedding = query_embedding.reshape(1, -1)

            if query_embedding.shape[1] != self.dimension:
                if query_embedding.shape[1] > self.dimension:
                    query_embedding = query_embedding[:, :self.dimension]
                else:
                    padding = np.zeros((1, self.dimension - query_embedding.shape[1]))
                    query_embedding = np.concatenate([query_embedding, padding], axis=1)

            scores, indices = self.index.search(query_embedding.astype(np.float32), min(k, self.index.ntotal))

            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx in self.metadata:
                    similarity = float(scores[0][i])
                    results.append((similarity, self.metadata[idx]))

            results.sort(key=lambda x: x[0], reverse=True)
            return results

        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return []

    def save(self):
        """Save index and metadata to disk"""
        if not self._initialized or not self.index:
            return

        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_documents": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.dimension,
            "embedding_strategy": self.embedding_strategy,
            "index_path": self.index_path,
            "metadata_path": self.metadata_path,
            "initialized": self._initialized
        }

    def clear(self):
        """Clear all data from the vector store"""
        if self.index:
            self.index.reset()
        self.metadata = {}
        self.save()
        logger.info("Vector store cleared")


# Global instance - use configured data directory
data_dir = Path(settings.JARVIS_DATA_DIR)
vector_store = VectorStore(
    index_path=str(data_dir / "vector_store.index"),
    metadata_path=str(data_dir / "vector_metadata.pkl")
)
