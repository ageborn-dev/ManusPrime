# main.py for ManusPrime
import asyncio

from app.agent.manusprime import ManusPrime
from app.logger import logger
from app.utils.monitor import resource_monitor


async def main():
    # Create ManusPrime agent with budget limit
    agent = ManusPrime(budget_limit=0.5)  # 50 cent budget limit
    
    print("\n🚀 ManusPrime initialized with resource monitoring!")
    print("📊 Budget limit set to $0.50")
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
            
            # Show command to display stats
            if prompt_lower in ["stats", "usage", "resources"]:
                # Get resource usage summary
                usage = resource_monitor.get_summary()
                
                print("\n📊 Resource Usage Statistics:")
                print(f"  Tokens: {usage['tokens']['total']} total ({usage['tokens']['prompt']} prompt, {usage['tokens']['completion']} completion)")
                print(f"  API Calls: {usage['api_calls']['total']} total ({usage['api_calls']['errors']} errors)")
                print(f"  Models Used:")
                for model, count in usage['models'].items():
                    print(f"    - {model}: {count} calls")
                print(f"  Tools Used:")
                for tool, count in usage['tools'].items():
                    print(f"    - {tool}: {count} calls")
                print(f"  Estimated Cost: ${usage['cost']:.4f}")
                print(f"  Current Session:")
                print(f"    - Tokens: {usage['current_session']['tokens']}")
                print(f"    - Cost: ${usage['current_session']['cost']:.4f}")
                print(f"    - Execution Time: {usage['current_session']['execution_time']:.2f} seconds")
                continue
                
            # Start timing the request
            start_time = asyncio.get_event_loop().time()
            
            logger.info("Processing your request with ManusPrime...")
            result = await agent.run(prompt)
            
            # Calculate elapsed time
            elapsed_time = asyncio.get_event_loop().time() - start_time
            
            print("\n✅ Task completed!")
            print(f"⏱️ Time taken: {elapsed_time:.2f} seconds")
            
            # Display model usage statistics
            if hasattr(agent.llm, "model_usage") and agent.llm.model_usage:
                print("\n📊 Model usage statistics:")
                for model, count in agent.llm.model_usage.items():
                    print(f"  - {model}: {count} calls")
            
            # Display cost information
            usage = resource_monitor.get_task_summary("complete_run")
            if usage:
                print(f"\n💰 Request cost: ${usage.get('cost', 0):.4f}")
                print(f"📝 Tokens used: {usage.get('tokens', {}).get('total', 0)}")
                
        except KeyboardInterrupt:
            logger.warning("Session terminated by user.")
            break
        except Exception as e:
            logger.error(f"Error processing request: {e}")


if __name__ == "__main__":
    asyncio.run(main())
