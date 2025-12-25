"""
Vector Store Service for JARVISv3
Implements semantic search using FAISS and Sentence Transformers.
"""
import os
import pickle
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Optional imports for vector store
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_VECTOR_DEPS = True
except ImportError:
    HAS_VECTOR_DEPS = False

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector storage and retrieval using FAISS.
    """
    
    def __init__(self, index_path: str = "./vector_store.index", metadata_path: str = "./vector_metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.index = None
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.model = None
        self._initialized = False
        
    def initialize(self):
        """Initialize the vector store and model"""
        if self._initialized:
            return
            
        if not HAS_VECTOR_DEPS:
            logger.warning("Vector store dependencies not found. Semantic search will be disabled.")
            return

        try:
            # Check if using CPU or GPU (defaulting to CPU for now as per requirements)
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded vector store with {self.index.ntotal} items")
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
                self.metadata = {}
                logger.info("Created new vector store")
                
            self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Add texts and metadata to the store"""
        if not self._initialized:
            self.initialize()
            
        if not self._initialized or not self.index:
            return
            
        try:
            embeddings = self.model.encode(texts)
            faiss.normalize_L2(embeddings)
            
            start_idx = self.index.ntotal
            self.index.add(embeddings)
            
            for i, meta in enumerate(metadatas):
                self.metadata[start_idx + i] = meta
                
            self.save()
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar texts"""
        if not self._initialized:
            self.initialize()
            
        if not self._initialized or not self.index or self.index.ntotal == 0:
            return []
            
        try:
            query_vector = self.model.encode([query])
            faiss.normalize_L2(query_vector)
            
            distances, indices = self.index.search(query_vector, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx in self.metadata:
                    results.append((float(distances[0][i]), self.metadata[idx]))
                    
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
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

# Global instance
vector_store = VectorStore()
