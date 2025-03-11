import asyncio
import uuid
from datetime import datetime
from json import dumps

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database.session import create_tables, get_db
from app.database.task_manager import DatabaseTaskManager
from app.agent.manusprime import ManusPrime
from app.utils.monitor import resource_monitor

# Create app and setup routes
app = FastAPI()

# Create database tables when app starts
create_tables()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database task manager
task_manager = DatabaseTaskManager()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/tasks")
async def create_task(prompt: str = Body(..., embed=True), db: Session = Depends(get_db)):
    task_id = str(uuid.uuid4())
    task = task_manager.create_task(db, prompt, task_id)
    asyncio.create_task(run_task(task_id, prompt))
    return {"task_id": task_id}

# Import both Manus and ManusPrime to maintain compatibility
from app.agent.manus import Manus

async def run_task(task_id: str, prompt: str, db: Session = Depends(get_db)):
    """Run a task using ManusPrime agent with database persistence."""
    try:
        # Update task status
        await task_manager.update_task_status(db, task_id, "running")

        # Use ManusPrime instead of Manus for multi-model capabilities
        agent = ManusPrime(
            name="ManusPrime",
            description="A versatile multi-model agent that can solve various tasks efficiently",
            max_steps=30
        )

        from app.logger import logger

        class SSELogHandler:
            def __init__(self, task_id, db_session):
                self.task_id = task_id
                self.db = db_session

            async def __call__(self, message):
                import re

                # Extract content after - 
                cleaned_message = re.sub(r'^.*? - ', '', message)

                event_type = "log"
                if "✨ Manus's thoughts:" in cleaned_message or "✨ ManusPrime's thoughts:" in cleaned_message:
                    event_type = "think"
                elif "🛠️ Manus selected" in cleaned_message or "🛠️ ManusPrime selected" in cleaned_message:
                    event_type = "tool"
                elif "🎯 Tool" in cleaned_message:
                    event_type = "act"
                elif "📝 Oops!" in cleaned_message:
                    event_type = "error"
                elif "🏁 Special tool" in cleaned_message:
                    event_type = "complete"
                elif "Using model:" in cleaned_message or "Identified" in cleaned_message:
                    event_type = "model"  # New event type for model selection
                elif "Budget alert:" in cleaned_message:
                    event_type = "budget"  # New event type for budget alerts

                await task_manager.update_task_step(self.db, self.task_id, 0, cleaned_message, event_type)

        # Create a database session for the log handler
        db_session = next(get_db())
        sse_handler = SSELogHandler(task_id, db_session)
        logger.add(sse_handler)

        # Start resource monitoring for this task
        resource_monitor.start_task(f"task_{task_id}")
        
        # Record start time for execution duration tracking
        start_time = asyncio.get_event_loop().time()

        # Run the agent
        result = await agent.run(prompt)

        # Calculate execution time
        execution_time = asyncio.get_event_loop().time() - start_time

        # Add model usage information
        if hasattr(agent.llm, "model_usage") and agent.llm.model_usage:
            model_usage_info = ", ".join([f"{model}: {count}" for model, count in agent.llm.model_usage.items()])
            await task_manager.update_task_step(
                db_session, task_id, 0, 
                f"Models used: {model_usage_info}", 
                "info"
            )

        # Add resource usage information
        usage = resource_monitor.get_task_summary(f"task_{task_id}")
        if usage:
            cost_info = f"Request cost: ${usage.get('cost', 0):.4f}, Tokens used: {usage.get('tokens', {}).get('total', 0)}"
            await task_manager.update_task_step(
                db_session, task_id, 0, 
                cost_info, 
                "resource"
            )

            # Store resource usage in database
            await task_manager.update_resource_usage(
                db_session,
                task_id,
                usage_data={
                    'total_tokens': usage.get('tokens', {}).get('total', 0),
                    'prompt_tokens': usage.get('tokens', {}).get('prompt', 0),
                    'completion_tokens': usage.get('tokens', {}).get('completion', 0),
                    'cost': usage.get('cost', 0),
                    'execution_time': execution_time,
                    'models_used': agent.llm.model_usage if hasattr(agent.llm, "model_usage") else {},
                    'tools_used': usage.get('tools', {})
                }
            )

        # End resource monitoring for this task
        resource_monitor.end_task()

        # Add final result
        await task_manager.update_task_step(db_session, task_id, 1, result, "result")
        
        # Mark task as complete
        await task_manager.complete_task(db_session, task_id)

    except Exception as e:
        # Log the exception
        logger.error(f"Error in run_task: {str(e)}")
        
        # End resource monitoring if needed
        if resource_monitor.current_task and resource_monitor.current_task.startswith(f"task_{task_id}"):
            resource_monitor.end_task()
            
        # Notify the client
        db_session = next(get_db())
        await task_manager.fail_task(db_session, task_id, str(e))

@app.get("/tasks/{task_id}/events")
async def task_events(task_id: str, db: Session = Depends(get_db)):
    """Stream task events to the client."""
    
    # Check if task exists
    task = task_manager.get_task(db, task_id)
    if not task:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Task with ID {task_id} not found"}
        )
    
    async def event_generator():
        if task_id not in task_manager.queues:
            task_manager.queues[task_id] = asyncio.Queue()
            
        queue = task_manager.queues[task_id]

        # Send initial task status
        steps = [
            {"step": s.step, "result": s.result, "type": s.type} 
            for s in task.steps
        ]
        
        yield f"event: status\ndata: {dumps({
            'type': 'status',
            'status': task.status,
            'steps': steps
        })}\n\n"

        while True:
            try:
                event = await queue.get()
                formatted_event = dumps(event)

                yield ": heartbeat\n\n"  # Keep connection alive

                if event["type"] == "complete":
                    yield f"event: complete\ndata: {formatted_event}\n\n"
                    break
                elif event["type"] == "error":
                    yield f"event: error\ndata: {formatted_event}\n\n"
                    break
                elif event["type"] == "status":
                    yield f"event: status\ndata: {formatted_event}\n\n"
                elif event["type"] in ["think", "tool", "act", "run", "model", "info", "budget", "resource"]:
                    # Support for all event types
                    yield f"event: {event['type']}\ndata: {formatted_event}\n\n"
                else:
                    yield f"event: {event['type']}\ndata: {formatted_event}\n\n"

            except asyncio.CancelledError:
                print(f"Client disconnected for task {task_id}")
                break
            except Exception as e:
                print(f"Error in event stream: {str(e)}")
                yield f"event: error\ndata: {dumps({'message': str(e)})}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/tasks")
async def get_tasks(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all tasks with pagination."""
    tasks = task_manager.get_all_tasks(db, skip, limit)
    return tasks

@app.get("/tasks/{task_id}")
async def get_task(task_id: str, db: Session = Depends(get_db)):
    """Get a specific task by ID."""
    task = task_manager.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/tasks/{task_id}/resource-usage")
async def get_task_resource_usage(task_id: str, db: Session = Depends(get_db)):
    """Get resource usage metrics for a specific task."""
    from app.database import crud
    
    resource_usage = crud.get_resource_usage(db, task_id)
    if not resource_usage:
        raise HTTPException(status_code=404, detail="Resource usage data not found")
    
    return resource_usage.to_dict()

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"message": f"Server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5172)
