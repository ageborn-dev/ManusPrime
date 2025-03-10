import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from app.logger import logger


@dataclass
class ResourceUsage:
    """Data class to track resource usage."""
    
    # API usage tracking
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # API calls
    api_calls: int = 0
    api_errors: int = 0
    
    # Model usage
    model_usage: Dict[str, int] = field(default_factory=dict)
    
    # Time tracking
    start_time: float = field(default_factory=time.time)
    total_execution_time: float = 0
    
    # Tools usage
    tool_usage: Dict[str, int] = field(default_factory=dict)
    
    # Budget tracking
    estimated_cost: float = 0
    budget_limit: Optional[float] = None
    
    def reset(self) -> None:
        """Reset all counters."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0
        self.api_errors = 0
        self.model_usage = {}
        self.start_time = time.time()
        self.total_execution_time = 0
        self.tool_usage = {}
        self.estimated_cost = 0


class ResourceMonitor:
    """
    Monitor and track resource usage in ManusPrime.
    
    This class tracks API calls, token usage, execution time, and estimated
    costs to help optimize performance and manage budget.
    """
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ResourceMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the resource monitor."""
        if not getattr(self, "_initialized", False):
            # Overall usage stats
            self.usage = ResourceUsage()
            
            # Session-specific usage stats
            self.session_usage = ResourceUsage()
            
            # For task-specific tracking
            self.current_task: Optional[str] = None
            self.task_usage: Dict[str, ResourceUsage] = {}
            
            # Model cost estimates ($ per 1K tokens)
            self.model_costs = {
                "gpt-4o": 0.015,
                "gpt-4o-mini": 0.005,
                "deepseek-chat": 0.003,
                "deepseek-r1": 0.007,
            }
            
            # Budget alert listeners
            self.budget_listeners: Set[callable] = set()
            
            # Execution timing
            self.timers: Dict[str, float] = {}
            
            logger.info("Resource monitor initialized")
            self._initialized = True
    
    def start_session(self, budget_limit: Optional[float] = None) -> None:
        """
        Start a new monitoring session.
        
        Args:
            budget_limit: Optional budget limit in dollars
        """
        self.session_usage = ResourceUsage(
            start_time=time.time(),
            budget_limit=budget_limit
        )
        logger.info("Resource monitoring session started")
    
    def start_task(self, task_name: str) -> None:
        """
        Start tracking a specific task.
        
        Args:
            task_name: Name of the task to track
        """
        self.current_task = task_name
        self.task_usage[task_name] = ResourceUsage(start_time=time.time())
        logger.info(f"Started monitoring task: {task_name}")
    
    def end_task(self) -> Optional[ResourceUsage]:
        """
        End tracking the current task.
        
        Returns:
            ResourceUsage for the completed task, or None if no task was active
        """
        if not self.current_task:
            return None
            
        task_name = self.current_task
        task_data = self.task_usage.get(task_name)
        
        if task_data:
            execution_time = time.time() - task_data.start_time
            task_data.total_execution_time = execution_time
            logger.info(f"Task {task_name} completed in {execution_time:.2f} seconds")
            
        self.current_task = None
        return task_data
    
    def track_api_call(
        self, 
        model: str, 
        prompt_tokens: int = 0, 
        completion_tokens: int = 0,
        success: bool = True
    ) -> None:
        """
        Track an API call to a language model.
        
        Args:
            model: Name of the model
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens generated
            success: Whether the call was successful
        """
        # Update overall stats
        self.usage.api_calls += 1
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.total_tokens += (prompt_tokens + completion_tokens)
        
        if not success:
            self.usage.api_errors += 1
            
        # Update model usage
        self.usage.model_usage[model] = self.usage.model_usage.get(model, 0) + 1
        
        # Update session stats
        self.session_usage.api_calls += 1
        self.session_usage.prompt_tokens += prompt_tokens
        self.session_usage.completion_tokens += completion_tokens
        self.session_usage.total_tokens += (prompt_tokens + completion_tokens)
        
        if not success:
            self.session_usage.api_errors += 1
            
        # Update model usage for session
        self.session_usage.model_usage[model] = self.session_usage.model_usage.get(model, 0) + 1
        
        # Update task stats if we're tracking a task
        if self.current_task and self.current_task in self.task_usage:
            task_data = self.task_usage[self.current_task]
            task_data.api_calls += 1
            task_data.prompt_tokens += prompt_tokens
            task_data.completion_tokens += completion_tokens
            task_data.total_tokens += (prompt_tokens + completion_tokens)
            
            if not success:
                task_data.api_errors += 1
                
            # Update model usage for task
            task_data.model_usage[model] = task_data.model_usage.get(model, 0) + 1
        
        # Calculate cost
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        self.usage.estimated_cost += cost
        self.session_usage.estimated_cost += cost
        
        if self.current_task and self.current_task in self.task_usage:
            self.task_usage[self.current_task].estimated_cost += cost
        
        # Check if we've exceeded budget limit
        if (self.session_usage.budget_limit is not None and 
            self.session_usage.estimated_cost > self.session_usage.budget_limit):
            self._trigger_budget_alert(
                self.session_usage.estimated_cost, 
                self.session_usage.budget_limit
            )
    
    def track_tool_usage(self, tool_name: str) -> None:
        """
        Track usage of a tool.
        
        Args:
            tool_name: Name of the tool
        """
        # Update overall stats
        self.usage.tool_usage[tool_name] = self.usage.tool_usage.get(tool_name, 0) + 1
        
        # Update session stats
        self.session_usage.tool_usage[tool_name] = self.session_usage.tool_usage.get(tool_name, 0) + 1
        
        # Update task stats if we're tracking a task
        if self.current_task and self.current_task in self.task_usage:
            task_data = self.task_usage[self.current_task]
            task_data.tool_usage[tool_name] = task_data.tool_usage.get(tool_name, 0) + 1
    
    def start_timer(self, timer_name: str) -> None:
        """
        Start a named timer.
        
        Args:
            timer_name: Name for the timer
        """
        self.timers[timer_name] = time.time()
    
    def end_timer(self, timer_name: str) -> Optional[float]:
        """
        End a named timer and return elapsed time.
        
        Args:
            timer_name: Name of the timer to end
            
        Returns:
            Elapsed time in seconds, or None if timer not found
        """
        if timer_name not in self.timers:
            return None
            
        elapsed = time.time() - self.timers[timer_name]
        del self.timers[timer_name]
        return elapsed
    
    def get_summary(self) -> Dict:
        """
        Get a summary of resource usage.
        
        Returns:
            Dictionary with usage summary
        """
        return {
            "tokens": {
                "total": self.usage.total_tokens,
                "prompt": self.usage.prompt_tokens,
                "completion": self.usage.completion_tokens
            },
            "api_calls": {
                "total": self.usage.api_calls,
                "errors": self.usage.api_errors
            },
            "models": self.usage.model_usage,
            "tools": self.usage.tool_usage,
            "cost": self.usage.estimated_cost,
            "current_session": {
                "tokens": self.session_usage.total_tokens,
                "api_calls": self.session_usage.api_calls,
                "models": self.session_usage.model_usage,
                "cost": self.session_usage.estimated_cost,
                "execution_time": time.time() - self.session_usage.start_time
            }
        }
    
    def get_task_summary(self, task_name: Optional[str] = None) -> Dict:
        """
        Get a summary of resource usage for a specific task.
        
        Args:
            task_name: Name of task, or None for current task
            
        Returns:
            Dictionary with task usage summary or empty dict if task not found
        """
        task_name = task_name or self.current_task
        
        if not task_name or task_name not in self.task_usage:
            return {}
            
        task_data = self.task_usage[task_name]
        
        return {
            "name": task_name,
            "tokens": {
                "total": task_data.total_tokens,
                "prompt": task_data.prompt_tokens,
                "completion": task_data.completion_tokens
            },
            "api_calls": {
                "total": task_data.api_calls,
                "errors": task_data.api_errors
            },
            "models": task_data.model_usage,
            "tools": task_data.tool_usage,
            "cost": task_data.estimated_cost,
            "execution_time": task_data.total_execution_time or (time.time() - task_data.start_time)
        }
    
    def add_budget_listener(self, callback) -> None:
        """
        Add a callback function to be notified when budget limit is reached.
        
        Args:
            callback: Function to call when budget is exceeded
        """
        self.budget_listeners.add(callback)
    
    def remove_budget_listener(self, callback) -> None:
        """
        Remove a budget limit callback.
        
        Args:
            callback: Function to remove
        """
        if callback in self.budget_listeners:
            self.budget_listeners.remove(callback)
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the estimated cost for a model API call.
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Estimated cost in dollars
        """
        # Get cost per 1K tokens for this model (default to cheapest if unknown)
        model_cost = self.model_costs.get(model, min(self.model_costs.values()) if self.model_costs else 0.001)
        
        # Calculate cost (convert to per token cost)
        prompt_cost = (prompt_tokens / 1000) * model_cost
        completion_cost = (completion_tokens / 1000) * model_cost
        
        return prompt_cost + completion_cost
    
    def _trigger_budget_alert(self, current_cost: float, budget_limit: float) -> None:
        """
        Trigger budget alert callbacks.
        
        Args:
            current_cost: Current estimated cost
            budget_limit: Budget limit that was exceeded
        """
        logger.warning(f"Budget alert: ${current_cost:.2f} exceeds limit of ${budget_limit:.2f}")
        
        for callback in self.budget_listeners:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Create a task to run the callback
                    asyncio.create_task(callback(current_cost, budget_limit))
                else:
                    callback(current_cost, budget_limit)
            except Exception as e:
                logger.error(f"Error in budget alert callback: {e}")


# Create a singleton instance
resource_monitor = ResourceMonitor()
