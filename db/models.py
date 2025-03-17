# db/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Boolean, Enum
import enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Task(Base):
    """Model for storing task information."""
    
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True)
    prompt = Column(Text, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    sandbox_session_id = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    results = relationship("TaskResult", back_populates="task", cascade="all, delete-orphan")
    resource_usage = relationship("ResourceUsage", back_populates="task", uselist=False, cascade="all, delete-orphan")
    
    # Additional relationships
    analysis = relationship("TaskAnalysis", back_populates="task", uselist=False, cascade="all, delete-orphan")
    steps = relationship("ExecutionStep", back_populates="task", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "sandbox_session_id": self.sandbox_session_id,
            "is_active": self.is_active,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "steps": [step.to_dict() for step in self.steps] if self.steps else []
        }


class TaskResult(Base):
    """Model for storing task execution results."""
    
    __tablename__ = "task_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, ForeignKey("tasks.id", ondelete="CASCADE"))
    content = Column(Text)
    result_type = Column(String, default="text")  # text, tool_result, error
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    task = relationship("Task", back_populates="results")
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "content": self.content,
            "result_type": self.result_type,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class StepStatus(enum.Enum):
    """Execution step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TaskAnalysis(Base):
    """Model for storing AI task analysis results."""
    
    __tablename__ = "task_analysis"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, ForeignKey("tasks.id", ondelete="CASCADE"), unique=True)
    task_type = Column(String)
    categories = Column(JSON)
    capabilities_needed = Column(JSON)
    complexity_assessment = Column(JSON)
    execution_plan = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    task = relationship("Task", back_populates="analysis")
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "categories": self.categories or [],
            "capabilities_needed": self.capabilities_needed or [],
            "complexity_assessment": self.complexity_assessment or {},
            "execution_plan": self.execution_plan or {},
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class ExecutionStep(Base):
    """Model for storing execution step information."""
    
    __tablename__ = "execution_steps"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, ForeignKey("tasks.id", ondelete="CASCADE"))
    step_id = Column(String)  # ID from execution plan
    description = Column(Text)
    model = Column(String)
    plugins = Column(JSON)
    requires_ui = Column(Boolean, default=False)
    expected_output = Column(String)
    status = Column(Enum(StepStatus), default=StepStatus.PENDING)
    error = Column(Text, nullable=True)
    result = Column(JSON, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    execution_time = Column(Float, nullable=True)
    dependencies = Column(JSON)  # List of step_ids this step depends on
    
    # Relationship
    task = relationship("Task", back_populates="steps")
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "description": self.description,
            "model": self.model,
            "plugins": self.plugins or [],
            "requires_ui": self.requires_ui,
            "expected_output": self.expected_output,
            "status": self.status.value,
            "error": self.error,
            "result": self.result,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": self.execution_time,
            "dependencies": self.dependencies or []
        }

class PluginMetrics(Base):
    """Model for storing plugin performance metrics."""
    
    __tablename__ = "plugin_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    plugin_name = Column(String, unique=True)
    calls = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    total_execution_time = Column(Float, default=0.0)
    avg_response_time = Column(Float, default=0.0)
    last_error = Column(Text, nullable=True)
    last_success = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "plugin_name": self.plugin_name,
            "calls": self.calls,
            "success_rate": (self.success_count / self.calls) if self.calls > 0 else 0.0,
            "error_count": self.error_count,
            "avg_response_time": self.avg_response_time,
            "last_error": self.last_error,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class ResourceUsage(Base):
    """Model for tracking resource usage."""
    
    __tablename__ = "resource_usage"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, ForeignKey("tasks.id", ondelete="CASCADE"), unique=True)
    total_tokens = Column(Integer, default=0)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    execution_time = Column(Float, default=0.0)  # in seconds
    models_used = Column(JSON, default=dict)
    
    # Relationship
    task = relationship("Task", back_populates="resource_usage")
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
            "execution_time": self.execution_time,
            "models_used": self.models_used or {}
        }
