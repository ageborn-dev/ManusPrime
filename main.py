# main.py for ManusPrime
import asyncio

# This will be implemented later
from app.agent.manusprime import ManusPrime
from app.logger import logger


async def main():
    # Create ManusPrime agent (will be implemented later)
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
            await agent.run(prompt)
        except KeyboardInterrupt:
            logger.warning("Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
