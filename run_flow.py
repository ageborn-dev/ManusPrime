# run_flow.py for ManusPrime
import asyncio
import time

# These will be implemented later
from app.agent.manusprime import ManusPrime
from app.flow.base import FlowType
from app.flow.flow_factory import FlowFactory
from app.logger import logger


async def run_flow():
    # Initialize with our main ManusPrime agent and any specialized agents
    agents = {
        "manusprime": ManusPrime(),
        # We'll add specialized agents later
    }

    while True:
        try:
            prompt = input("Enter your prompt (or 'exit' to quit): ")
            if prompt.lower() == "exit":
                logger.info("Goodbye!")
                break

            # Create a flow using our agents
            flow = FlowFactory.create_flow(
                flow_type=FlowType.PLANNING,
                agents=agents,
            )
            
            if prompt.strip().isspace():
                logger.warning("Skipping empty prompt.")
                continue
                
            logger.info("Processing your request with ManusPrime...")

            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    flow.execute(prompt),
                    timeout=3600,  # 60 minute timeout for the entire execution
                )
                elapsed_time = time.time() - start_time
                logger.info(f"Request processed in {elapsed_time:.2f} seconds")
                logger.info(result)
            except asyncio.TimeoutError:
                logger.error("Request processing timed out after 1 hour")
                logger.info(
                    "Operation terminated due to timeout. Please try a simpler request."
                )

        except KeyboardInterrupt:
            logger.info("Operation cancelled by user.")
        except Exception as e:
            logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_flow())
