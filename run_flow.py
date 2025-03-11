# run_flow.py for ManusPrime
import asyncio
import time
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.agent.manusprime import ManusPrime
from app.database.session import SessionLocal, create_tables
from app.database import crud
from app.flow.base import FlowType
from app.flow.flow_factory import FlowFactory
from app.logger import logger
from app.utils.monitor import resource_monitor


async def run_flow():
    # Ensure database tables are created
    create_tables()
    
    # Initialize with our main ManusPrime agent and any specialized agents
    agents = {
        "manusprime": ManusPrime(),
        # We'll add specialized agents later
    }

    while True:
        try:
            prompt = input("\n🧠 Enter your prompt (or 'exit' to quit): ")
            if prompt.lower() == "exit":
                logger.info("Goodbye!")
                break
                
            if prompt.lower() == "history":
                show_task_history()
                continue

            if prompt.strip() == "" or prompt.isspace():
                logger.warning("Skipping empty prompt.")
                continue
                
            # Generate a task ID and store in database
            task_id = str(uuid.uuid4())
            db = SessionLocal()
            try:
                task = crud.create_task(db, task_id, prompt)
                logger.info(f"Created task with ID: {task_id}")
            finally:
                db.close()
                
            # Create a flow using our agents
            flow = FlowFactory.create_flow(
                flow_type=FlowType.PLANNING,
                agents=agents,
            )
            
            logger.info("Processing your request with ManusPrime flow...")

            try:
                # Start tracking resources
                resource_monitor.start_task(f"flow_{task_id}")
                
                # Update task status
                db = SessionLocal()
                try:
                    crud.update_task_status(db, task_id, "running")
                finally:
                    db.close()
                
                # Execute the flow
                start_time = time.time()
                result = await asyncio.wait_for(
                    flow.execute(prompt),
                    timeout=3600,  # 60 minute timeout for the entire execution
                )
                elapsed_time = time.time() - start_time
                
                # Get resource usage
                usage = resource_monitor.get_task_summary(f"flow_{task_id}")
                resource_monitor.end_task()
                
                # Update task in database
                db = SessionLocal()
                try:
                    # Store the result
                    crud.add_task_step(db, task_id, 1, result, "result")
                    
                    # Update task status
                    crud.update_task_status(db, task_id, "completed")
                    
                    # Store resource usage
                    if usage:
                        models_used = {}
                        for agent_name, agent in agents.items():
                            if hasattr(agent, 'llm') and hasattr(agent.llm, 'model_usage'):
                                for model, count in agent.llm.model_usage.items():
                                    models_used[model] = models_used.get(model, 0) + count
                        
                        crud.update_resource_usage(
                            db,
                            task_id,
                            total_tokens=usage.get('tokens', {}).get('total', 0),
                            prompt_tokens=usage.get('tokens', {}).get('prompt', 0),
                            completion_tokens=usage.get('tokens', {}).get('completion', 0),
                            cost=usage.get('cost', 0),
                            execution_time=elapsed_time,
                            models_used=models_used,
                            tools_used=usage.get('tools', {})
                        )
                finally:
                    db.close()
                
                # Display results
                logger.info(f"✅ Task completed in {elapsed_time:.2f} seconds")
                if usage:
                    logger.info(f"💰 Estimated cost: ${usage.get('cost', 0):.4f}")
                    logger.info(f"🔤 Total tokens used: {usage.get('tokens', {}).get('total', 0)}")
                    
                    # Show models used
                    if models_used:
                        logger.info(f"🧠 Models used: {', '.join([f'{m}={c}' for m, c in models_used.items()])}")
                    
                logger.info("\n" + result)
                
            except asyncio.TimeoutError:
                logger.error("⚠️ Request processing timed out after 1 hour")
                logger.info("Operation terminated due to timeout. Please try a simpler request.")
                
                # Update task status
                db = SessionLocal()
                try:
                    crud.update_task_status(db, task_id, "failed: Timeout after 1 hour")
                finally:
                    db.close()
                    
                # End resource monitoring
                resource_monitor.end_task()

        except KeyboardInterrupt:
            logger.info("Operation cancelled by user.")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            
            # Try to update task status if we have a task_id
            if 'task_id' in locals():
                try:
                    db = SessionLocal()
                    crud.update_task_status(db, task_id, f"failed: {str(e)}")
                    db.close()
                except Exception:
                    pass


def show_task_history():
    """Display recent tasks from the database"""
    try:
        db = SessionLocal()
        tasks = crud.get_all_tasks(db, limit=10)
        
        if not tasks:
            print("\n📋 No tasks found in history")
            return
            
        print("\n📋 Recent tasks:")
        print("--------------------------------------------------")
        for i, task in enumerate(tasks):
            # Format the created_at date
            created_at = task.created_at.strftime("%Y-%m-%d %H:%M")
            
            # Get the status with proper emoji
            if task.status == "completed":
                status = "✅ Completed"
            elif task.status == "running":
                status = "⚙️ Running"
            elif task.status.startswith("failed"):
                status = f"❌ Failed: {task.status[7:]}"
            else:
                status = f"⏳ {task.status}"
                
            # Get resource usage if available
            cost = "N/A"
            tokens = "N/A"
            try:
                resource_usage = crud.get_resource_usage(db, task.id)
                if resource_usage:
                    cost = f"${resource_usage.cost:.4f}"
                    tokens = str(resource_usage.total_tokens)
            except Exception:
                pass
                
            # Truncate prompt if too long
            prompt = task.prompt
            if len(prompt) > 70:
                prompt = prompt[:67] + "..."
                
            print(f"{i+1}. [{created_at}] {status}")
            print(f"   Prompt: {prompt}")
            print(f"   Cost: {cost} | Tokens: {tokens}")
            print("--------------------------------------------------")
    except Exception as e:
        print(f"Error retrieving task history: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("\n🚀 ManusPrime Flow Interface")
    print("Type 'exit' to quit or 'history' to see recent tasks")
    asyncio.run(run_flow())
