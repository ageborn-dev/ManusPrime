# db/crud.py
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime

from db.models import Task, TaskResult, ResourceUsage

# Task CRUD operations
def create_task(db: Session, task_id: str, prompt: str) -> Task:
    """Create a new task.
    
    Args:
        db: Database session
        task_id: Unique task identifier
        prompt: Task prompt
        
    Returns:
        Task: The created task
    """
    db_task = Task(
        id=task_id,
        prompt=prompt,
        status="pending"
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

def get_task(db: Session, task_id: str) -> Optional[Task]:
    """Get a task by ID.
    
    Args:
        db: Database session
        task_id: Task ID
        
    Returns:
        Optional[Task]: The task if found, None otherwise
    """
    return db.query(Task).filter(Task.id == task_id).first()

def get_tasks(db: Session, skip: int = 0, limit: int = 100) -> List[Task]:
    """Get a list of tasks with pagination.
    
    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List[Task]: List of tasks
    """
    return db.query(Task).order_by(Task.created_at.desc()).offset(skip).limit(limit).all()

def update_task_status(db: Session, task_id: str, status: str) -> Optional[Task]:
    """Update a task's status.
    
    Args:
        db: Database session
        task_id: Task ID
        status: New status
        
    Returns:
        Optional[Task]: Updated task if found, None otherwise
    """
    db_task = get_task(db, task_id)
    if db_task:
        db_task.status = status
        if status == "completed":
            db_task.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(db_task)
    return db_task

# Task Result CRUD operations
def create_task_result(db: Session, task_id: str, content: str, result_type: str = "text") -> TaskResult:
    """Create a task result.
    
    Args:
        db: Database session
        task_id: Task ID
        content: Result content
        result_type: Result type
        
    Returns:
        TaskResult: Created result
    """
    db_result = TaskResult(
        task_id=task_id,
        content=content,
        result_type=result_type
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    return db_result

def get_task_results(db: Session, task_id: str) -> List[TaskResult]:
    """Get all results for a task.
    
    Args:
        db: Database session
        task_id: Task ID
        
    Returns:
        List[TaskResult]: List of task results
    """
    return db.query(TaskResult).filter(TaskResult.task_id == task_id).order_by(TaskResult.created_at).all()

# Resource Usage CRUD operations
def create_resource_usage(db: Session, task_id: str) -> ResourceUsage:
    """Create a resource usage record.
    
    Args:
        db: Database session
        task_id: Task ID
        
    Returns:
        ResourceUsage: Created resource usage record
    """
    db_usage = ResourceUsage(task_id=task_id)
    db.add(db_usage)
    db.commit()
    db.refresh(db_usage)
    return db_usage

def get_resource_usage(db: Session, task_id: str) -> Optional[ResourceUsage]:
    """Get resource usage for a task.
    
    Args:
        db: Database session
        task_id: Task ID
        
    Returns:
        Optional[ResourceUsage]: Resource usage if found, None otherwise
    """
    return db.query(ResourceUsage).filter(ResourceUsage.task_id == task_id).first()

def update_resource_usage(
    db: Session,
    task_id: str,
    usage_data: Dict[str, Any]
) -> Optional[ResourceUsage]:
    """Update resource usage for a task.
    
    Args:
        db: Database session
        task_id: Task ID
        usage_data: Resource usage data
        
    Returns:
        Optional[ResourceUsage]: Updated resource usage if found, None otherwise
    """
    db_usage = get_resource_usage(db, task_id)
    
    if not db_usage:
        # Create new resource usage if it doesn't exist
        db_usage = ResourceUsage(task_id=task_id)
        db.add(db_usage)
    
    # Update fields based on provided data
    for key, value in usage_data.items():
        if hasattr(db_usage, key):
            setattr(db_usage, key, value)
    
    db.commit()
    db.refresh(db_usage)
    return db_usage