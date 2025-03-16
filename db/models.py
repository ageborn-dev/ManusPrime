# db/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Boolean
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
            "is_active": self.is_active
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
