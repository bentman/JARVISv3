import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.memory import MemoryService
from backend.core.database import DatabaseManager

@pytest.fixture
def mock_db_manager():
    return MagicMock(spec=DatabaseManager)

@pytest.fixture
def memory_service(mock_db_manager):
    with patch("backend.core.memory.database_manager", mock_db_manager):
        service = MemoryService()
        service.db = mock_db_manager
        return service

@pytest.mark.asyncio
async def test_memory_tags(memory_service, mock_db_manager):
    """Test setting tags via MemoryService"""
    mock_db_manager.initialize = AsyncMock()
    mock_db_manager.set_conversation_tags = AsyncMock(return_value=True)
    
    result = await memory_service.set_conversation_tags("conv_123", ["important", "todo"])
    
    assert result is True
    mock_db_manager.set_conversation_tags.assert_called_once_with("conv_123", ["important", "todo"])

@pytest.mark.asyncio
async def test_memory_stats(memory_service, mock_db_manager):
    """Test getting conversation stats via MemoryService"""
    mock_db_manager.initialize = AsyncMock()
    mock_db_manager.get_conversation_stats = AsyncMock(return_value={"message_count": 5, "token_count": 100})
    
    stats = await memory_service.get_conversation_stats("conv_123")
    
    assert stats["message_count"] == 5
    assert stats["token_count"] == 100
    mock_db_manager.get_conversation_stats.assert_called_once_with("conv_123")

@pytest.mark.asyncio
async def test_memory_export(memory_service, mock_db_manager):
    """Test exporting data via MemoryService"""
    mock_db_manager.initialize = AsyncMock()
    mock_db_manager.export_all_data = AsyncMock(return_value={"users": [], "conversations": []})
    
    data = await memory_service.export_all_data()
    
    assert "users" in data
    mock_db_manager.export_all_data.assert_called_once()

@pytest.mark.asyncio
async def test_memory_caching(memory_service):
    """Test semantic search caching"""
    with patch("backend.core.memory.cache_service") as mock_cache:
        # 1. Test Cache Hit
        mock_cache.healthy = AsyncMock(return_value=True)
        mock_cache.get_json = AsyncMock(return_value=[{"content": "cached result"}])
        
        # We need to mock vector_store too to ensure it's NOT called on hit
        memory_service.vector_store = MagicMock()
        memory_service.vector_store.search = MagicMock()
        
        results = await memory_service.semantic_search("test query")
        
        assert len(results) == 1
        assert results[0]["content"] == "cached result"
        assert mock_cache.get_json.called
        # vector_store.search should NOT be called
        memory_service.vector_store.search.assert_not_called()
        
        # 2. Test Cache Miss
        mock_cache.get_json = AsyncMock(return_value=None)
        mock_cache.set_json = AsyncMock()
        memory_service.vector_store.search.return_value = [(0.9, {"content": "fresh result"})]
        
        results_fresh = await memory_service.semantic_search("new query")
        
        assert len(results_fresh) == 1
        assert results_fresh[0]["content"] == "fresh result"
        assert memory_service.vector_store.search.called
        assert mock_cache.set_json.called
