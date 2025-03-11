from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session

from app.database import models

# Task CRUD operations
def create_task(db: Session, task_id: str, prompt: str) -> models.Task:
    """Create a new task record in the database."""
    db_task = models.Task(
        id=task_id,
        prompt=prompt,
        status="pending"
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

def get_task(db: Session, task_id: str) -> Optional[models.Task]:
    """Get a task by its ID."""
    return db.query(models.Task).filter(models.Task.id == task_id).first()

def get_all_tasks(db: Session, skip: int = 0, limit: int = 100) -> List[models.Task]:
    """Get all tasks with pagination support."""
    return db.query(models.Task).order_by(models.Task.created_at.desc()).offset(skip).limit(limit).all()

def update_task_status(db: Session, task_id: str, status: str) -> Optional[models.Task]:
    """Update a task's status."""
    db_task = get_task(db, task_id)
    if db_task:
        db_task.status = status
        if status == "completed":
            db_task.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(db_task)
    return db_task

# Task Step CRUD operations
def add_task_step(db: Session, task_id: str, step: int, result: str, step_type: str = "step") -> models.TaskStep:
    """Add a step to a task."""
    db_step = models.TaskStep(
        task_id=task_id,
        step=step,
        result=result,
        type=step_type
    )
    db.add(db_step)
    db.commit()
    db.refresh(db_step)
    return db_step

def get_task_steps(db: Session, task_id: str) -> List[models.TaskStep]:
    """Get all steps for a specific task."""
    return db.query(models.TaskStep).filter(models.TaskStep.task_id == task_id).order_by(models.TaskStep.id).all()

# Resource Usage CRUD operations
def create_resource_usage(db: Session, task_id: str) -> models.ResourceUsage:
    """Create a new resource usage record for a task."""
    db_usage = models.ResourceUsage(
        task_id=task_id
    )
    db.add(db_usage)
    db.commit()
    db.refresh(db_usage)
    return db_usage

def get_resource_usage(db: Session, task_id: str) -> Optional[models.ResourceUsage]:
    """Get resource usage for a specific task."""
    return db.query(models.ResourceUsage).filter(models.ResourceUsage.task_id == task_id).first()

def update_resource_usage(
    db: Session, 
    task_id: str, 
    total_tokens: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    cost: Optional[float] = None,
    execution_time: Optional[float] = None,
    models_used: Optional[Dict[str, int]] = None,
    tools_used: Optional[Dict[str, int]] = None
) -> Optional[models.ResourceUsage]:
    """Update resource usage metrics for a task."""
    db_usage = get_resource_usage(db, task_id)
    
    if not db_usage:
        db_usage = create_resource_usage(db, task_id)
    
    # Update fields if provided
    if total_tokens is not None:
        db_usage.total_tokens = total_tokens
    if prompt_tokens is not None:
        db_usage.prompt_tokens = prompt_tokens
    if completion_tokens is not None:
        db_usage.completion_tokens = completion_tokens
    if cost is not None:
        db_usage.cost = cost
    if execution_time is not None:
        db_usage.execution_time = execution_time
    
    # Update model and tool usage
    if models_used:
        current_models = db_usage.models_used or {}
        for model, count in models_used.items():
            current_models[model] = current_models.get(model, 0) + count
        db_usage.models_used = current_models
    
    if tools_used:
        current_tools = db_usage.tools_used or {}
        for tool, count in tools_used.items():
            current_tools[tool] = current_tools.get(tool, 0) + count
        db_usage.tools_used = current_tools
    
    db.commit()
    db.refresh(db_usage)
    return db_usage
