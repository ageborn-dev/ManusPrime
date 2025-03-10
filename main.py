# main.py for ManusPrime
import asyncio

from app.agent.manusprime import ManusPrime
from app.logger import logger


async def main():
    # Create ManusPrime agent
    agent = ManusPrime()
    
    while True:
        try:
            prompt = input("Enter your prompt (or 'exit'/'quit' to quit): ")
            prompt_lower = prompt.lower()
            if prompt_lower in ["exit", "quit"]:
                logger.info("Goodbye!")
                break
            if not prompt.strip():
                logger.warning("Skipping empty prompt.")
                continue
                
            logger.info("Processing your request with ManusPrime...")
            result = await agent.run(prompt)
            
            # Show model usage statistics
            if hasattr(agent.llm, "model_usage") and agent.llm.model_usage:
                print("\nModel usage statistics:")
                for model, count in agent.llm.model_usage.items():
                    print(f"- {model}: {count} calls")
                
        except KeyboardInterrupt:
            logger.warning("Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
