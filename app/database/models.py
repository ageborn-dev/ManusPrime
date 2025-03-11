from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Task(Base):
    """Model representing a user task."""
    __tablename__ = "tasks"

    id = Column(String, primary_key=True)
    prompt = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String, default="pending")
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    steps = relationship("TaskStep", back_populates="task", cascade="all, delete-orphan")
    resource_usage = relationship("ResourceUsage", back_populates="task", uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "prompt": self.prompt,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "steps": [step.to_dict() for step in self.steps]
        }


class TaskStep(Base):
    """Model representing an individual step in a task's execution."""
    __tablename__ = "task_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, ForeignKey("tasks.id", ondelete="CASCADE"))
    step = Column(Integer)
    result = Column(Text)
    type = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    task = relationship("Task", back_populates="steps")

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "step": self.step,
            "result": self.result,
            "type": self.type,
            "timestamp": self.timestamp.isoformat()
        }


class ResourceUsage(Base):
    """Model tracking resource usage for a task."""
    __tablename__ = "resource_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, ForeignKey("tasks.id", ondelete="CASCADE"), unique=True)
    total_tokens = Column(Integer, default=0)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    execution_time = Column(Float, default=0.0)  # in seconds
    
    # Store model usage counts as JSON
    models_used = Column(JSON, default=dict)
    tools_used = Column(JSON, default=dict)
    
    # Relationship
    task = relationship("Task", back_populates="resource_usage")

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
            "execution_time": self.execution_time,
            "models_used": self.models_used,
            "tools_used": self.tools_used
        }
