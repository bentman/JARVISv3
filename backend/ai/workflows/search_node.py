"""Search Node for JARVISv3
Integrates DuckDuckGo search into the agentic workflow.
"""
import logging
from typing import Dict, Any, List, Optional
from ..context.schemas import TaskContext
from ...core.memory import memory_service
from ...core.privacy import privacy_service
from ...core.cache_service import cache_service
from ...core.config import settings
from ...core.search_providers import (
    WebSearchProvider, 
    DuckDuckGoProvider, 
    BingProvider, 
    GoogleProvider, 
    TavilyProvider
)
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

class SearchNode:
    """
    Workflow node for performing web searches.
    """

    def __init__(self):
        self.providers: Dict[str, WebSearchProvider] = self._init_providers()

    def _init_providers(self) -> Dict[str, WebSearchProvider]:
        """Initialize search providers based on settings"""
        providers = {}
        enabled_list = [p.strip().lower() for p in settings.SEARCH_PROVIDERS.split(",")]
        
        if "duckduckgo" in enabled_list:
            providers["duckduckgo"] = DuckDuckGoProvider()
        
        if "bing" in enabled_list and settings.SEARCH_BING_API_KEY:
            providers["bing"] = BingProvider(settings.SEARCH_BING_API_KEY, settings.SEARCH_BING_ENDPOINT)
            
        if "google" in enabled_list and settings.SEARCH_GOOGLE_API_KEY and settings.SEARCH_GOOGLE_CX:
            providers["google"] = GoogleProvider(settings.SEARCH_GOOGLE_API_KEY, settings.SEARCH_GOOGLE_CX)
            
        if "tavily" in enabled_list and settings.SEARCH_TAVILY_API_KEY:
            providers["tavily"] = TavilyProvider(settings.SEARCH_TAVILY_API_KEY)
            
        return providers

    async def execute(self, context: TaskContext, node_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a unified search across local memory and the web.
        """
        query = context.workflow_context.initiating_query

        # If a previous node (like a planner) refined the search query, use that
        if "search_query" in context.additional_context:
            query = context.additional_context["search_query"]

        logger.info(f"Performing unified search for: {query}")

        # Check Cache
        import hashlib
        cache_key = f"search:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
        cached_data = await cache_service.get_json(cache_key)
        if cached_data:
            logger.info(f"Cache hit for search: {query}")
            return cached_data

        search_results = {
            "local_memory": [],
            "web": [],
            "success": False,
            "privacy_assessment": {},
            "retrieval_stats": {}
        }

        try:
            # 1. Privacy Assessment
            privacy_assessment = privacy_service.classify_data(query)
            search_results["privacy_assessment"] = {
                "classification": str(privacy_assessment),
                "should_process_locally": privacy_service.should_process_locally(query, context.system_context.user_preferences.privacy_level),
                "timestamp": datetime.now(UTC).isoformat()
            }

            # 2. Local Memory Search (Semantic)
            memory_start_time = datetime.now(UTC)
            memory_hits = await memory_service.semantic_search(query, k=3)
            memory_end_time = datetime.now(UTC)

            search_results["local_memory"] = [
                {
                    "content": hit.get("content"),
                    "role": hit.get("role"),
                    "timestamp": hit.get("timestamp"),
                    "conversation_id": hit.get("conversation_id"),
                    "relevance_score": 0.9  # Placeholder for actual relevance score
                }
                for hit in memory_hits
            ]

            # 3. Web Search - only if privacy allows
            web_results = []
            web_start_time = datetime.now(UTC)
            web_end_time = web_start_time
            
            if not search_results["privacy_assessment"]["should_process_locally"] and settings.SEARCH_ENABLED:
                web_start_time = datetime.now(UTC)
                
                # Redact query for privacy
                safe_query = privacy_service.redact_sensitive_data(query)
                logger.info(f"Using redacted query for web search: {safe_query}")

                # Iterate through providers until we get results
                for name, provider in self.providers.items():
                    try:
                        hits = await provider.search(safe_query, max_results=settings.SEARCH_MAX_RESULTS)
                        if hits:
                            for r in hits:
                                web_results.append({
                                    "title": r.get("title"),
                                    "href": r.get("url"),
                                    "body": r.get("snippet"),
                                    "timestamp": datetime.now(UTC).isoformat(),
                                    "source": r.get("source", "web")
                                })
                            # If we found results, we can stop or continue based on policy. 
                            # v2 stopped after first successful provider.
                            break
                    except Exception as e:
                        logger.error(f"Search provider {name} failed: {e}")
                        continue
                        
                web_end_time = datetime.now(UTC)

            search_results["web"] = web_results

            # 4. Update retrieval statistics
            search_results["retrieval_stats"] = {
                "local_memory_count": len(search_results["local_memory"]),
                "web_results_count": len(web_results),
                "memory_retrieval_time_ms": (memory_end_time - memory_start_time).total_seconds() * 1000,
                "web_retrieval_time_ms": (web_end_time - web_start_time).total_seconds() * 1000 if web_results else 0,
                "total_results": len(search_results["local_memory"]) + len(web_results),
                "timestamp": datetime.now(UTC).isoformat()
            }

            search_results["success"] = True

            # Update context budget for tool usage (conceptually)
            context.workflow_context.add_artifact(f"unified_search_results_{len(search_results['web']) + len(search_results['local_memory'])}")

            # Store in cache
            if search_results["success"]:
                await cache_service.set_json(cache_key, search_results, ttl_seconds=3600)

            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "error": str(e),
                "success": False
            }

    async def execute_advanced_search(self, context: TaskContext, node_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute advanced search with query refinement and result ranking
        """
        query = context.workflow_context.initiating_query

        # Query refinement based on context
        refined_query = await self._refine_query(query, context)
        logger.info(f"Refined search query: {refined_query}")

        # Perform unified search
        search_results = await self.execute(context, node_results)

        # Rank and filter results
        ranked_results = await self._rank_results(search_results, refined_query)

        return {
            **search_results,
            "refined_query": refined_query,
            "ranked_results": ranked_results
        }

    async def _refine_query(self, original_query: str, context: TaskContext) -> str:
        """
        Refine query based on context and previous results
        """
        # Simple refinement logic - could be enhanced with LLM-based query expansion
        refined_query = original_query

        # Add context from workflow if available
        if "search_context" in context.additional_context:
            refined_query = f"{context.additional_context['search_context']} {original_query}"

        return refined_query

    async def _rank_results(self, search_results: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """
        Rank search results based on relevance and quality
        """
        ranked_results = []

        # Combine all results
        all_results = search_results["local_memory"] + search_results["web"]

        # Simple ranking based on content length and source
        for result in all_results:
            # Calculate a simple relevance score
            content_length = len(result.get("content", "") or result.get("body", ""))
            source_weight = 1.5 if result.get("source") == "local_memory" else 1.0

            # Basic relevance score (could be enhanced with proper ranking algorithms)
            relevance_score = min(content_length / 100 * source_weight, 1.0)

            ranked_results.append({
                **result,
                "relevance_score": relevance_score,
                "rank": len(ranked_results) + 1
            })

        # Sort by relevance score (descending)
        ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return ranked_results

# Global instance
search_node = SearchNode()
