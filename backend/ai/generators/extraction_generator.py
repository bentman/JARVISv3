"""
Structured Extraction Context Generator for JARVISv3
"""
import re
import logging
from typing import Dict, Any, List
from .base import ContextGenerator
from ..context.schemas import TaskContext

logger = logging.getLogger(__name__)

class StructuredExtractionGenerator(ContextGenerator):
    """Extracts structured information from the query using patterns"""
    
    def __init__(self):
        self.patterns = {
            "date": r'\b(?:\d{4}-\d{2}-\d{2}|today|tomorrow|yesterday)\b',
            "time": r'\b(?:\d{1,2}:\d{2}(?:\s?[ap]m)?)\b',
            "url": r'https?://[^\s]+',
            "code_block": r'```(?:[\s\S]*?)```'
        }
        
    @property
    def name(self) -> str:
        return "extraction_generator"
        
    async def generate(self, context: TaskContext, **kwargs) -> TaskContext:
        try:
            query = context.workflow_context.initiating_query
            extracted = {}
            
            for key, pattern in self.patterns.items():
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    extracted[key] = matches
            
            # Identify potential task keywords
            task_keywords = ["summarize", "search", "write", "debug", "create", "analyze"]
            found_keywords = [kw for kw in task_keywords if kw in query.lower()]
            if found_keywords:
                extracted["task_keywords"] = found_keywords
                
            # Add to additional_context
            if extracted:
                context.additional_context["extracted_entities"] = extracted
                logger.debug(f"Extracted {len(extracted)} entities from query")
                
        except Exception as e:
            logger.error(f"Error in extraction generator: {e}")
            
        return context
