import asyncio
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Dict, Optional

from app.database import crud, models

class DatabaseTaskManager:
    """Task manager that persists data to a database."""
    
    def __init__(self):
        """Initialize the task manager."""
        self.queues: Dict[str, asyncio.Queue] = {}
    
    def create_task(self, db: Session, prompt: str, task_id: str) -> models.Task:
        """Create a new task in the database."""
        task = crud.create_task(db, task_id, prompt)
        self.queues[task_id] = asyncio.Queue()
        return task
    
    def get_task(self, db: Session, task_id: str) -> Optional[models.Task]:
        """Get a task by ID."""
        return crud.get_task(db, task_id)
    
    def get_all_tasks(self, db: Session, skip: int = 0, limit: int = 100) -> list:
        """Get all tasks with pagination."""
        return crud.get_all_tasks(db, skip, limit)
    
    async def update_task_step(self, db: Session, task_id: str, step: int, result: str, step_type: str = "step"):
        """Add a step to a task and notify listeners."""
        if task_id not in self.queues:
            self.queues[task_id] = asyncio.Queue()
            
        # Add step to database
        crud.add_task_step(db, task_id, step, result, step_type)
        
        # Send event to queue
        await self.queues[task_id].put({
            "type": step_type,
            "step": step,
            "result": result
        })
        
        # Send updated status for clients
        task = crud.get_task(db, task_id)
        steps = [
            {"step": s.step, "result": s.result, "type": s.type} 
            for s in crud.get_task_steps(db, task_id)
        ]
        
        await self.queues[task_id].put({
            "type": "status",
            "status": task.status,
            "steps": steps
        })
    
    async def update_task_status(self, db: Session, task_id: str, status: str):
        """Update a task's status and notify listeners."""
        if task_id not in self.queues:
            self.queues[task_id] = asyncio.Queue()
            
        # Update database
        task = crud.update_task_status(db, task_id, status)
        
        # Get steps for the response
        steps = [
            {"step": s.step, "result": s.result, "type": s.type} 
            for s in crud.get_task_steps(db, task_id)
        ]
        
        # Send update to queue
        await self.queues[task_id].put({
            "type": "status",
            "status": status,
            "steps": steps
        })
    
    async def complete_task(self, db: Session, task_id: str):
        """Mark a task as completed and notify listeners."""
        if task_id not in self.queues:
            return
            
        # Update database
        task = crud.update_task_status(db, task_id, "completed")
        
        # Get steps for the response
        steps = [
            {"step": s.step, "result": s.result, "type": s.type} 
            for s in crud.get_task_steps(db, task_id)
        ]
        
        # Send updates to queue
        await self.queues[task_id].put({
            "type": "status",
            "status": "completed",
            "steps": steps
        })
        
        await self.queues[task_id].put({"type": "complete"})
    
    async def fail_task(self, db: Session, task_id: str, error: str):
        """Mark a task as failed and notify listeners."""
        if task_id not in self.queues:
            return
            
        # Update database
        task = crud.update_task_status(db, task_id, f"failed: {error}")
        
        # Send error to queue
        await self.queues[task_id].put({
            "type": "error",
            "message": error
        })
    
    async def update_resource_usage(self, db: Session, task_id: str, usage_data: dict):
        """Update resource usage metrics for a task."""
        crud.update_resource_usage(
            db, 
            task_id,
            total_tokens=usage_data.get('total_tokens'),
            prompt_tokens=usage_data.get('prompt_tokens'),
            completion_tokens=usage_data.get('completion_tokens'),
            cost=usage_data.get('cost'),
            execution_time=usage_data.get('execution_time'),
            models_used=usage_data.get('models_used'),
            tools_used=usage_data.get('tools_used')
        )
