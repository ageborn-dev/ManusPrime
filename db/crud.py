# db/crud.py
import logging
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

from db.models import (
    Task, TaskResult, ResourceUsage, TaskAnalysis, 
    ExecutionStep, PluginMetrics, StepStatus
)

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

# Task Analysis operations
def store_task_analysis(db: Session, task_id: str, analysis: Dict[str, Any]) -> TaskAnalysis:
    """Store task analysis results.
    
    Args:
        db: Database session
        task_id: Task ID
        analysis: Analysis results from AI planner
        
    Returns:
        TaskAnalysis: Created analysis record
    """
    db_analysis = TaskAnalysis(
        task_id=task_id,
        task_type=analysis.get("analysis", {}).get("task_type", "default"),
        categories=analysis.get("analysis", {}).get("categories", []),
        capabilities_needed=analysis.get("analysis", {}).get("capabilities_needed", []),
        complexity_assessment=analysis.get("analysis", {}).get("complexity_assessment", {}),
        execution_plan=analysis.get("execution_plan", {})
    )
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis

def get_task_analysis(db: Session, task_id: str) -> Optional[TaskAnalysis]:
    """Get task analysis by task ID.
    
    Args:
        db: Database session
        task_id: Task ID
        
    Returns:
        Optional[TaskAnalysis]: Analysis if found, None otherwise
    """
    return db.query(TaskAnalysis).filter(TaskAnalysis.task_id == task_id).first()

# Execution Step operations
def store_task_step(db: Session, task_id: str, step_data: Dict[str, Any]) -> ExecutionStep:
    """Store execution step information.
    
    Args:
        db: Database session
        task_id: Task ID
        step_data: Step information
        
    Returns:
        ExecutionStep: Created step record
    """
    db_step = ExecutionStep(
        task_id=task_id,
        step_id=step_data.get("id"),
        description=step_data.get("description"),
        model=step_data.get("model"),
        plugins=step_data.get("plugins", []),
        requires_ui=step_data.get("requires_ui", False),
        expected_output=step_data.get("expected_output"),
        dependencies=step_data.get("dependencies", [])
    )
    db.add(db_step)
    db.commit()
    db.refresh(db_step)
    return db_step

def get_task_steps(db: Session, task_id: str) -> List[ExecutionStep]:
    """Get all execution steps for a task.
    
    Args:
        db: Database session
        task_id: Task ID
        
    Returns:
        List[ExecutionStep]: List of execution steps
    """
    return db.query(ExecutionStep).filter(ExecutionStep.task_id == task_id).all()

def update_step_status(
    db: Session,
    task_id: str,
    step_id: str,
    status: StepStatus,
    result: Optional[Dict] = None,
    error: Optional[str] = None
) -> Optional[ExecutionStep]:
    """Update execution step status.
    
    Args:
        db: Database session
        task_id: Task ID
        step_id: Step ID
        status: New status
        result: Step result data
        error: Error message if failed
        
    Returns:
        Optional[ExecutionStep]: Updated step if found
    """
    db_step = (
        db.query(ExecutionStep)
        .filter(ExecutionStep.task_id == task_id, ExecutionStep.step_id == step_id)
        .first()
    )
    
    if db_step:
        db_step.status = status
        if status == StepStatus.RUNNING:
            db_step.started_at = datetime.utcnow()
        elif status in [StepStatus.COMPLETED, StepStatus.FAILED]:
            db_step.completed_at = datetime.utcnow()
            if db_step.started_at:
                db_step.execution_time = (
                    db_step.completed_at - db_step.started_at
                ).total_seconds()
        
        if result is not None:
            db_step.result = result
        if error is not None:
            db_step.error = error
            
        db.commit()
        db.refresh(db_step)
    return db_step

# Plugin Metrics operations
def update_plugin_metrics(
    db: Session,
    plugin_name: str,
    success: bool,
    execution_time: float,
    error: Optional[str] = None
) -> PluginMetrics:
    """Update plugin performance metrics.
    
    Args:
        db: Database session
        plugin_name: Plugin name
        success: Whether operation was successful
        execution_time: Time taken for operation
        error: Error message if failed
        
    Returns:
        PluginMetrics: Updated metrics
    """
    db_metrics = (
        db.query(PluginMetrics)
        .filter(PluginMetrics.plugin_name == plugin_name)
        .first()
    )
    
    if not db_metrics:
        db_metrics = PluginMetrics(plugin_name=plugin_name)
        db.add(db_metrics)
    
    db_metrics.calls += 1
    if success:
        db_metrics.success_count += 1
        db_metrics.last_success = datetime.utcnow()
    else:
        db_metrics.error_count += 1
        db_metrics.last_error = error
        
    db_metrics.total_execution_time += execution_time
    db_metrics.avg_response_time = db_metrics.total_execution_time / db_metrics.calls
    
    db.commit()
    db.refresh(db_metrics)
    return db_metrics

def get_plugin_metrics(db: Session, plugin_name: Optional[str] = None) -> Union[Optional[PluginMetrics], List[PluginMetrics]]:
    """Get plugin metrics.
    
    Args:
        db: Database session
        plugin_name: Optional plugin name to filter by
        
    Returns:
        Union[Optional[PluginMetrics], List[PluginMetrics]]: Plugin metrics
    """
    if plugin_name:
        return db.query(PluginMetrics).filter(PluginMetrics.plugin_name == plugin_name).first()
    return db.query(PluginMetrics).all()

def delete_task(db: Session, task_id: str) -> bool:
    """Delete a task by ID.
    
    Args:
        db: Database session
        task_id: Task ID
        
    Returns:
        bool: True if deletion was successful
    """
    db_task = get_task(db, task_id)
    if not db_task:
        return False
        
    try:

        db.delete(db_task)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting task {task_id}: {e}")
        return False
