"""Voice Node for JARVISv3
Integrates Voice capabilities (STT/TTS) into the agentic workflow.
"""
import logging
from typing import Dict, Any, Optional
from ..context.schemas import TaskContext, VoiceContext
from ...core.voice import voice_service
from ...core.config import settings

logger = logging.getLogger(__name__)

class VoiceNode:
    """
    Workflow node for handling Voice interactions (STT and TTS).
    Designed to be used as executable functions in WorkflowNode.
    """

    async def execute_stt(self, context: TaskContext, node_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Speech-to-Text:
        - Reads audio from context.voice_context.audio_input
        - Transcribes to text
        - Updates context.workflow_context.initiating_query (or outputs transcription)
        """
        logger.info("Executing Voice STT node")
        
        if not context.voice_context or not context.voice_context.audio_input:
            logger.warning("No audio input found in VoiceContext")
            return {"error": "No audio input provided", "success": False}

        try:
            # Call Voice Service
            text, confidence = await voice_service.speech_to_text(context.voice_context.audio_input)
            
            # Update Context
            context.voice_context.transcription = text
            # Optionally update initiating query if this is the start of a flow
            if not context.workflow_context.initiating_query:
                context.workflow_context.initiating_query = text
                
            logger.info(f"STT Complete: '{text[:50]}...' (Confidence: {confidence})")
            
            return {
                "transcription": text,
                "confidence": confidence,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"STT execution failed: {e}")
            return {"error": str(e), "success": False}

    async def execute_tts(self, context: TaskContext, node_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Text-to-Speech:
        - Reads text from input_context or previous node results
        - Generates audio
        - Updates context.voice_context.audio_output
        """
        logger.info("Executing Voice TTS node")
        
        # Determine text source
        text_to_speak = ""
        
        # 1. Check explicit input
        if context.node_context and context.node_context.input_context:
            text_to_speak = context.node_context.input_context.get("text", "")
            
        # 2. Fallback to previous node response (e.g. LLM)
        if not text_to_speak:
            # Try to find a 'response' key in node_results
            for node_id, result in node_results.items():
                if isinstance(result, dict) and "response" in result:
                    text_to_speak = result["response"]
                    # We might want the *last* response, but this is a simple heuristic
                    
        if not text_to_speak:
            logger.warning("No text found to speak")
            return {"error": "No text input provided", "success": False}

        try:
            # Call Voice Service
            audio_bytes = await voice_service.text_to_speech(text_to_speak)
            
            # Update Context
            if not context.voice_context:
                context.voice_context = VoiceContext()
                
            context.voice_context.audio_output = audio_bytes
            
            logger.info(f"TTS Complete: {len(audio_bytes)} bytes generated")
            
            return {
                "audio_size_bytes": len(audio_bytes),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"TTS execution failed: {e}")
            return {"error": str(e), "success": False}

# Global instance
voice_node = VoiceNode()
