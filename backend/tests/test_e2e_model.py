"""
End-to-End Intelligence Test for JARVISv3
Verifies that the system can actually perform AI inference with a real model.
"""
import asyncio
import logging
from ..core.model_router import model_router
from ..ai.context.schemas import TaskContext, TaskType

logger = logging.getLogger(__name__)

async def test_real_inference():
    """
    Attempts to perform a real inference using the model router.
    This goes beyond 'skeleton' testing by checking if a model is actually reachable and responding.
    """
    print("\nChecking for real-world intelligence (LLM connection)...")
    
    # Check if any provider is available
    providers_available = []
    for provider_name, provider in model_router.providers.items():
        if await provider.is_available():
            providers_available.append(provider_name)
    
    if not providers_available:
        print("⚠ SKIP: No local model providers (Ollama/llama.cpp) are currently running.")
        print("  (This indicates the system skeleton is fine, but the 'brain' is offline)")
        return True # Skip doesn't mean fail in this context, but it's a warning

    print(f"✓ Found available providers: {', '.join(providers_available)}")
    
    # Try a simple generation
    prompt = "Reply with exactly one word: 'Validated'."
    try:
        print(f"Attempting real-world generation with {providers_available[0]}...")
        result = await model_router.generate_response(prompt)
        
        if result and result.response:
            print(f"✓ Real intelligence verified! Response: '{result.response.strip()}'")
            print(f"  Tokens used: {result.tokens_used}, Provider: {result.provider}")
            return True
        else:
            print("✗ Generation failed: Received empty response from model.")
            return False
            
    except Exception as e:
        print(f"✗ Generation failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = asyncio.run(test_real_inference())
    exit(0 if success else 1)
