# api/endpoints.py
import uuid
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from db.session import get_db
from db import crud
from core.agent import ManusPrime
from utils.monitor import resource_monitor
from utils.logger import logger

router = APIRouter()

# Store task event queues
task_queues: Dict[str, asyncio.Queue] = {}

# API endpoints
@router.post("/tasks")
async def create_task(
    prompt: str = Body(..., embed=True),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Create a new task.
    
    Args:
        prompt: The task prompt
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        dict: Task ID
    """
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Create task in database
    task = crud.create_task(db, task_id, prompt)
    
    # Create event queue for this task
    task_queues[task_id] = asyncio.Queue()
    
    # Start task execution in background
    background_tasks.add_task(run_task, task_id, prompt)
    
    return {"task_id": task_id}

@router.get("/tasks")
async def get_tasks(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get a list of tasks.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        list: List of tasks
    """
    tasks = crud.get_tasks(db, skip, limit)
    return [task.to_dict() for task in tasks]

@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    db: Session = Depends(get_db)
):
    """Get a task by ID.
    
    Args:
        task_id: Task ID
        db: Database session
        
    Returns:
        dict: Task details
    """
    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Get task results
    results = crud.get_task_results(db, task_id)
    
    # Get resource usage
    resource_usage = crud.get_resource_usage(db, task_id)
    
    # Build response
    response = task.to_dict()
    response["results"] = [result.to_dict() for result in results]
    if resource_usage:
        response["resource_usage"] = resource_usage.to_dict()
    
    return response

@router.get("/tasks/{task_id}/events")
async def get_task_events(task_id: str, db: Session = Depends(get_db)):
    """Get a stream of task events.
    
    Args:
        task_id: Task ID
        db: Database session
        
    Returns:
        StreamingResponse: Server-sent events stream
    """
    # Verify task exists
    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Create queue if it doesn't exist
    if task_id not in task_queues:
        task_queues[task_id] = asyncio.Queue()
    
    async def event_generator():
        # Send initial task status
        yield f"event: status\ndata: {json.dumps({'status': task.status})}\n\n"
        
        queue = task_queues[task_id]
        
        try:
            while True:
                # Get event from queue with timeout
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield ": heartbeat\n\n"
                    continue
                
                # Format event as SSE
                event_type = event.get("type", "message")
                event_data = json.dumps(event)
                yield f"event: {event_type}\ndata: {event_data}\n\n"
                
                # Exit loop if task is completed or failed
                if event_type in ["complete", "error"]:
                    break
                    
        except asyncio.CancelledError:
            logger.info(f"Client disconnected from events for task {task_id}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Background task function
async def run_task(task_id: str, prompt: str):
    """Run a task in the background.
    
    Args:
        task_id: Task ID
        prompt: Task prompt
    """
    db = next(get_db())
    
    try:
        # Update task status
        crud.update_task_status(db, task_id, "running")
        await _send_event(task_id, "status", {"status": "running"})
        
        # Initialize agent
        agent = ManusPrime()
        await agent.initialize()
        
        # Start resource monitoring
        resource_monitor.start_session(task_id=task_id)
        
        # Execute task
        result = await agent.execute_task(prompt)
        
        # Store result
        if result["success"]:
            content = result["content"]
            crud.create_task_result(db, task_id, content, "text")
            crud.update_task_status(db, task_id, "completed")
            
            # Send result event
            await _send_event(task_id, "result", {"content": content})
            
            # Send complete event
            await _send_event(task_id, "complete", {
                "message": "Task completed successfully",
                "execution_time": result["execution_time"]
            })
        else:
            error = result.get("error", "Unknown error")
            crud.create_task_result(db, task_id, error, "error")
            crud.update_task_status(db, task_id, f"failed: {error}")
            
            # Send error event
            await _send_event(task_id, "error", {"message": error})
        
        # Store resource usage
        usage_summary = resource_monitor.get_summary()
        crud.update_resource_usage(db, task_id, {
            "total_tokens": usage_summary["tokens"]["total"],
            "prompt_tokens": usage_summary["tokens"]["prompt"],
            "completion_tokens": usage_summary["tokens"]["completion"],
            "cost": usage_summary["cost"],
            "execution_time": result["execution_time"],
            "models_used": usage_summary["models"]
        })
        
        # Send resource usage event
        await _send_event(task_id, "resource", {
            "tokens": usage_summary["tokens"],
            "cost": usage_summary["cost"],
            "models": usage_summary["models"]
        })
        
    except Exception as e:
        logger.error(f"Error running task {task_id}: {e}")
        
        # Update task status
        crud.update_task_status(db, task_id, f"failed: {str(e)}")
        
        # Send error event
        await _send_event(task_id, "error", {"message": str(e)})
        
    finally:
        # Clean up agent
        await agent.cleanup()
        
        # End resource monitoring
        resource_monitor.end_session()

async def _send_event(task_id: str, event_type: str, data: Dict):
    """Send an event to a task's event queue.
    
    Args:
        task_id: Task ID
        event_type: Event type
        data: Event data
    """
    if task_id in task_queues:
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await task_queues[task_id].put(event)