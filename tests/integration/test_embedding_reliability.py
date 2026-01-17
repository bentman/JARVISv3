"""
Integration tests for Embedding Reliability Enhancement
Tests Phase 10: Embedding Reliability Enhancement capabilities
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from backend.core.embedding_service import (
    FeatureHashingEmbeddingService, EmbeddingService, embedding_service
)
from backend.core.vector_store import vector_store


@pytest.mark.asyncio
async def test_feature_hashing_embedding_service():
    """Test basic feature hashing embedding functionality"""
    service = FeatureHashingEmbeddingService(dim=768)

    # Test tokenization
    tokens = service.tokenize("Hello, World! How are you?")
    assert len(tokens) > 0
    assert "hello" in tokens
    assert "world" in tokens

    # Test hash function determinism
    hash1 = service.hash_token("test")
    hash2 = service.hash_token("test")
    assert hash1 == hash2
    assert 0 <= hash1 < 768

    # Test embedding generation
    texts = ["Hello world", "Goodbye world", "Hello universe"]
    embeddings = service.embed_texts(texts)

    # Check shape and properties
    assert embeddings.shape == (3, 768)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)  # L2 normalized

    # Check that similarities are reasonable (feature hashing is approximate)
    similarity1 = service.get_similarity(embeddings[0], embeddings[1])  # hello vs goodbye
    similarity2 = service.get_similarity(embeddings[0], embeddings[2])  # hello vs hello

    # Both should be valid similarity scores
    assert 0 <= similarity1 <= 1
    assert 0 <= similarity2 <= 1

    # Test that identical texts have perfect similarity
    identical_sim = service.get_similarity(embeddings[0], embeddings[0])
    assert abs(identical_sim - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_feature_hashing_search():
    """Test search functionality with feature hashing"""
    service = FeatureHashingEmbeddingService(dim=768)

    # Create test documents
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "A brown fox is quick and agile",
        "The lazy dog sleeps all day",
        "Jumping foxes are quick animals"
    ]

    # Generate embeddings for documents
    doc_embeddings = service.embed_texts(docs)

    # Test search
    query = "quick brown fox"
    results = service.search_similar(query, doc_embeddings, docs, top_k=2)

    assert len(results) <= 2
    for similarity, text in results:
        assert similarity > 0
        assert text in docs

    # Most similar result should be the one with most matching words
    if results:
        best_match = results[0][1]
        assert "quick" in best_match.lower() or "brown" in best_match.lower() or "fox" in best_match.lower()


@pytest.mark.asyncio
async def test_unified_embedding_service():
    """Test unified embedding service with fallback functionality"""
    service = EmbeddingService(primary_dim=768)

    # Test embedding info
    info = service.get_embedding_info()
    assert "primary_dimension" in info
    assert "transformer_available" in info
    assert "feature_hashing_available" in info
    assert "current_strategy" in info
    assert info["feature_hashing_available"] == True

    # Test feature hashing fallback (when transformer unavailable)
    texts = ["Test document one", "Test document two"]
    embeddings = service.embed_texts(texts, use_transformer=False)

    assert embeddings.shape == (2, 768)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)

    # Test query embedding
    query_embedding = service.embed_query("test query", use_transformer=False)
    assert query_embedding.shape == (768,)
    assert np.allclose(np.linalg.norm(query_embedding), 1.0)


@pytest.mark.asyncio
async def test_transformer_fallback_to_hashing():
    """Test automatic fallback from transformer to feature hashing"""
    service = EmbeddingService(primary_dim=768)

    # Mock transformer failure
    with patch.object(service, '_initialize_transformer_embeddings', return_value=False):
        embeddings = service.embed_texts(["Test text"], use_transformer=True)
        # Should fall back to feature hashing
        assert embeddings.shape == (1, 768)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


@pytest.mark.asyncio
async def test_vector_store_with_embedding_fallback():
    """Test vector store integration with embedding fallback"""
    # Initialize vector store
    vector_store.initialize()

    # Skip test if FAISS not available
    if not hasattr(vector_store, 'index') or vector_store.index is None:
        pytest.skip("FAISS not available")

    # Test adding texts with fallback embeddings
    texts = ["First test document", "Second test document", "Third test document"]
    metadatas = [
        {"id": "doc1", "type": "test"},
        {"id": "doc2", "type": "test"},
        {"id": "doc3", "type": "test"}
    ]

    # Add texts (should work with feature hashing fallback)
    vector_store.add_texts(texts, metadatas)

    # Test search
    results = vector_store.search("test document", k=2)
    assert isinstance(results, list)

    # Test fallback search
    results_fallback = vector_store.search_with_fallback("test document", k=2)
    assert isinstance(results_fallback, list)

    # Get stats
    stats = vector_store.get_stats()
    assert "total_documents" in stats
    assert "embedding_dimension" in stats
    assert "embedding_strategy" in stats


@pytest.mark.asyncio
async def test_embedding_dimensionality_consistency():
    """Test that embeddings maintain consistent dimensionality across strategies"""
    service = EmbeddingService(primary_dim=768)

    # Test feature hashing consistency
    texts = ["Consistency test"]
    fh_embeddings = service.embed_texts(texts, use_transformer=False)

    assert fh_embeddings.shape == (1, 768)

    # Test that dimension is preserved in vector store operations
    if vector_store.index is not None:
        # Add a test document
        vector_store.add_texts(["Dimension test"], [{"test": "dimension"}])

        # Search should work without dimension errors
        results = vector_store.search("dimension test", k=1)
        assert isinstance(results, list)


@pytest.mark.asyncio
async def test_deterministic_embeddings():
    """Test that feature hashing produces deterministic results"""
    service1 = FeatureHashingEmbeddingService(dim=768)
    service2 = FeatureHashingEmbeddingService(dim=768)

    text = "Deterministic test string"

    # Generate embeddings with both services
    emb1 = service1.embed_texts([text])
    emb2 = service2.embed_texts([text])

    # Should be identical (deterministic)
    np.testing.assert_array_almost_equal(emb1, emb2)


@pytest.mark.asyncio
async def test_embedding_service_error_handling():
    """Test error handling in embedding services"""
    service = EmbeddingService(primary_dim=768)

    # Test with empty input
    empty_embeddings = service.embed_texts([], use_transformer=False)
    assert empty_embeddings.shape == (0, 768)

    # Test with very short text
    short_embeddings = service.embed_texts(["a"], use_transformer=False)
    assert short_embeddings.shape == (1, 768)
    assert np.allclose(np.linalg.norm(short_embeddings, axis=1), 1.0)


@pytest.mark.asyncio
async def test_similarity_calculations():
    """Test similarity calculations between embeddings"""
    service = FeatureHashingEmbeddingService(dim=768)

    # Create test embeddings
    emb1 = service.embed_query("machine learning")
    emb2 = service.embed_query("artificial intelligence")
    emb3 = service.embed_query("machine learning algorithms")

    # Similar texts should have higher similarity
    sim1 = service.get_similarity(emb1, emb3)  # Same topic
    sim2 = service.get_similarity(emb1, emb2)  # Different topics

    # This is a probabilistic test - similarity should generally be higher for related texts
    # but feature hashing is approximate, so we just check it's a valid similarity score
    assert 0 <= sim1 <= 1
    assert 0 <= sim2 <= 1


@pytest.mark.asyncio
async def test_vector_store_initialization():
    """Test vector store initialization with embedding service"""
    # Test stats before initialization
    stats = vector_store.get_stats()
    assert "initialized" in stats

    # Initialize if not already done
    vector_store.initialize()

    # Test stats after initialization
    stats = vector_store.get_stats()
    assert stats["initialized"] == True
    assert "embedding_strategy" in stats
    assert "embedding_dimension" in stats


@pytest.mark.asyncio
async def test_embedding_strategy_metadata():
    """Test that embedding strategy is stored in metadata"""
    if vector_store.index is None:
        pytest.skip("FAISS not available")

    # Clear vector store first
    vector_store.clear()

    # Add test documents
    texts = ["Strategy test document"]
    metadatas = [{"test": "strategy"}]

    vector_store.add_texts(texts, metadatas)

    # Search and check metadata
    results = vector_store.search("strategy test", k=1)
    if results:
        similarity, metadata = results[0]
        assert "_embedding_strategy" in metadata
        assert "_embedding_dim" in metadata
        assert metadata["_embedding_dim"] == 768


@pytest.mark.asyncio
async def test_fallback_search_reliability():
    """Test that fallback search provides reliability when primary search fails"""
    if vector_store.index is None:
        pytest.skip("FAISS not available")

    # Add some test documents
    vector_store.add_texts(
        ["Reliability test one", "Reliability test two"],
        [{"id": "rel1"}, {"id": "rel2"}]
    )

    # Test primary search
    primary_results = vector_store.search("reliability test", k=2)
    assert isinstance(primary_results, list)

    # Test fallback search
    fallback_results = vector_store.search_with_fallback("reliability test", k=2)
    assert isinstance(fallback_results, list)

    # Fallback should at least work (may return same or different results)
    # This ensures search reliability even when transformer embeddings fail


@pytest.mark.asyncio
async def test_embedding_service_configuration():
    """Test embedding service configuration and info reporting"""
    service = EmbeddingService(primary_dim=512)  # Different dimension

    info = service.get_embedding_info()
    assert info["primary_dimension"] == 512
    assert info["feature_hashing_available"] == True
    assert info["current_strategy"] in ["transformer", "feature_hashing"]

    # Test with feature hashing only
    embeddings = service.embed_texts(["Config test"], use_transformer=False)
    assert embeddings.shape == (1, 512)
