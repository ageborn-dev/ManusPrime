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

@router.post("/tasks/{task_id}/continue")
async def continue_task(
    task_id: str,
    prompt: str = Body(..., embed=True),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Continue an existing task.
    
    Args:
        task_id: Task ID
        prompt: The continuation prompt
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        dict: Task ID
    """
    # Verify task exists
    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Create event queue if it doesn't exist
    if task_id not in task_queues:
        task_queues[task_id] = asyncio.Queue()
    
    # Start task execution in background with continue flag
    background_tasks.add_task(run_task, task_id, prompt, continue_task=True)
    
    return {"task_id": task_id}

@router.post("/tasks/{task_id}/save")
async def save_task_state(
    task_id: str,
    db: Session = Depends(get_db)
):
    """Save the current state of a task.
    
    Args:
        task_id: Task ID
        db: Database session
        
    Returns:
        dict: Success message
    """
    # Verify task exists
    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update task timestamp
    task.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Task state saved successfully"}

@router.post("/tasks/{task_id}/cleanup")
async def cleanup_task(
    task_id: str,
    db: Session = Depends(get_db)
):
    """Clean up task resources.
    
    Args:
        task_id: Task ID
        db: Database session
        
    Returns:
        dict: Success message
    """
    # Verify task exists
    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    agent = None
    try:
        # Initialize agent for cleanup
        agent = ManusPrime()
        await agent.initialize()
        
        # Clean up sandbox session if exists
        if task.sandbox_session_id:
            await agent.sandbox_manager.cleanup(task_id)
        
        # Update task status
        crud.update_task_status(db, task_id, "completed")
        
        return {"message": "Task resources cleaned up successfully"}
        
    except Exception as e:
        logger.error(f"Error cleaning up task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if agent:
            await agent.cleanup()

@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    db: Session = Depends(get_db)
):
    """Delete a task by ID.
    
    Args:
        task_id: Task ID
        db: Database session
        
    Returns:
        dict: Success message
    """
    # Verify task exists
    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Delete task
    success = crud.delete_task(db, task_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete task")
    
    return {"message": "Task deleted successfully"}

# New endpoints for capabilities and metrics
@router.get("/capabilities")
async def get_capabilities():
    """Get available plugin capabilities."""
    registry_instance = registry.PluginRegistry()
    capabilities = {}
    
    for plugin_name, plugin_class in registry_instance.plugin_classes.items():
        caps = registry_instance.get_plugin_capabilities(plugin_name)
        if caps:
            capabilities[plugin_name] = list(caps)
    
    return {"capabilities": capabilities}

@router.get("/plugins/metrics")
async def get_plugin_metrics():
    """Get plugin performance metrics."""
    registry_instance = registry.PluginRegistry()
    metrics = {}
    
    for plugin_name, plugin in registry_instance.plugin_instances.items():
        if hasattr(plugin, 'info'):
            info = plugin.info
            if 'performance' in info:
                metrics[plugin_name] = info['performance']
    
    return {"metrics": metrics}

@router.get("/tasks/{task_id}/analysis")
async def get_task_analysis(task_id: str, db: Session = Depends(get_db)):
    """Get AI task analysis results."""
    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    analysis = crud.get_task_analysis(db, task_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Task analysis not found")
    
    return analysis

@router.get("/tasks/{task_id}/steps")
async def get_task_steps(task_id: str, db: Session = Depends(get_db)):
    """Get parallel execution step status."""
    task = crud.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    steps = crud.get_task_steps(db, task_id)
    return {"steps": steps}

async def run_task(task_id: str, prompt: str, continue_task: bool = False):
    """Run a task in the background.
    
    Args:
        task_id: Task ID
        prompt: Task prompt
    """
    db = next(get_db())
    agent = None
    
    try:
        # Update task status
        crud.update_task_status(db, task_id, "running")
        await _send_event(task_id, "status", {"status": "running"})
        
        # Initialize agent
        agent = ManusPrime()
        await agent.initialize()
        
        # Start resource monitoring
        resource_monitor.start_session(task_id=task_id)
        
        # Get provider for task analysis
        provider = registry.get_active_plugin(PluginCategory.PROVIDER)
        if not provider:
            raise ValueError("No provider plugin active")
            
        # Analyze task using AI
        task_analysis = await agent.ai_planner.create_execution_plan(prompt, provider)
        
        # Store task analysis
        crud.store_task_analysis(db, task_id, task_analysis)
        
        # Execute task with AI analysis
        result = await agent.execute_task(
            prompt, 
            task_id=task_id,
            continue_task=continue_task,
            task_analysis=task_analysis
        )
        
        # Store results with steps information
        if result["success"]:
            content = result["content"]
            steps_executed = result.get("steps_executed", 0)
            execution_pattern = result.get("execution_pattern", "sequential")
            
            crud.create_task_result(db, task_id, content, "text")
            crud.update_task_status(db, task_id, "completed")
            
            # Store step results if available
            if "step_results" in result:
                for step in result["step_results"]:
                    crud.store_task_step(db, task_id, step)
            
            # Send enhanced result event
            await _send_event(task_id, "result", {
                "content": content,
                "steps_executed": steps_executed,
                "execution_pattern": execution_pattern
            })
            
            # Send complete event with additional metrics
            await _send_event(task_id, "complete", {
                "message": "Task completed successfully",
                "execution_time": result["execution_time"],
                "performance_metrics": result.get("performance_metrics", {})
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
