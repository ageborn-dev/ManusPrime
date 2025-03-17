# main.py
import asyncio
import argparse
import uuid
from typing import Optional

from core.agent import ManusPrime
from core.plugin_manager import plugin_manager
from utils.logger import logger
from utils.monitor import resource_monitor

async def run_agent(prompt: str, model: Optional[str] = None):
    """Run the agent with a prompt.
    
    Args:
        prompt: The prompt to execute
        model: Optional specific model to use
    """
    # Initialize plugin manager and agent
    await plugin_manager.initialize()
    agent = ManusPrime()
    await agent.initialize()
    
    try:
        # Create task ID
        task_id = str(uuid.uuid4())
        logger.info(f"Task ID: {task_id}")
        
        # Start resource monitoring
        resource_monitor.start_session(task_id=task_id)
        
        # Execute task
        logger.info(f"Executing: {prompt}")
        result = await agent.execute_task(prompt, model=model)
        
        # Check result
        if result["success"]:
            logger.info("Task completed successfully:")
            print("\n" + result["content"])
        else:
            logger.error(f"Task failed: {result.get('error', 'Unknown error')}")
        
        # Show usage statistics
        usage = resource_monitor.get_summary()
        logger.info(f"Tokens used: {usage['tokens']['total']} (prompt: {usage['tokens']['prompt']}, completion: {usage['tokens']['completion']})")
        logger.info(f"Cost: ${usage['cost']:.4f}")
        logger.info(f"Models used: {usage['models']}")
        logger.info(f"Execution time: {result['execution_time']:.2f}s")
        
    finally:
        # Clean up
        await agent.cleanup()
        resource_monitor.end_session()

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="ManusPrime CLI")
    parser.add_argument("prompt", nargs="?", help="Prompt to execute")
    parser.add_argument("--model", help="Specific model to use")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        asyncio.run(interactive_mode())
    elif args.prompt:
        asyncio.run(run_agent(args.prompt, args.model))
    else:
        parser.print_help()

async def interactive_mode():
    """Run in interactive mode."""
    print("ManusPrime Interactive Mode (type 'exit' to quit)")
    
    # Initialize plugin manager
    await plugin_manager.initialize()
    
    try:
        while True:
            prompt = input("\nEnter prompt: ").strip()
            
            if not prompt:
                continue
                
            if prompt.lower() in ["exit", "quit"]:
                break
                
            await run_agent(prompt)
            
    finally:
        await plugin_manager.cleanup()

if __name__ == "__main__":
    main()
