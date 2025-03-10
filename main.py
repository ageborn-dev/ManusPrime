# main.py for ManusPrime
import asyncio

from app.agent.manusprime import ManusPrime
from app.logger import logger
from app.memory.embeddings import EmbeddingClient
from app.memory.manager import MemoryManager
from app.memory.vector_store import VectorMemory


async def main():
    # Create local embedding client for semantic memory (no API calls)
    embedding_client = EmbeddingClient(embedding_dimension=1536)
    
    # Create vector memory
    vector_memory = VectorMemory(embedding_dimension=1536)
    
    # Create memory manager
    memory_manager = MemoryManager(
        embedding_client=embedding_client,
        vector_memory=vector_memory
    )
    
    # Create ManusPrime agent with memory manager
    agent = ManusPrime(memory_manager=memory_manager)
    
    print("\n🚀 ManusPrime with semantic memory initialized!")
    print("💡 Type 'exit' or 'quit' to end the session\n")
    
    while True:
        try:
            prompt = input("\n🧠 Enter your prompt: ")
            prompt_lower = prompt.lower()
            
            if prompt_lower in ["exit", "quit"]:
                logger.info("Goodbye!")
                break
                
            if not prompt.strip():
                logger.warning("Skipping empty prompt.")
                continue
                
            logger.info("Processing your request with ManusPrime...")
            result = await agent.run(prompt)
            
            print("\n✅ Task completed!")
            
            # Display model usage statistics
            if hasattr(agent.llm, "model_usage") and agent.llm.model_usage:
                print("\n📊 Model usage statistics:")
                for model, count in agent.llm.model_usage.items():
                    print(f"  - {model}: {count} calls")
                
        except KeyboardInterrupt:
            logger.warning("Session terminated by user.")
            break
        except Exception as e:
            logger.error(f"Error processing request: {e}")


if __name__ == "__main__":
    asyncio.run(main())
