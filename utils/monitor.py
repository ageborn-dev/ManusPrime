# utils/monitor.py
import time
import logging
from typing import Dict, Optional, Any, Set, Callable

logger = logging.getLogger("manusprime.utils.monitor")

class ResourceMonitor:
    """Monitor resource usage, including tokens and costs."""
    
    def __init__(self):
        """Initialize the resource monitor."""
        # Session tracking
        self.active_session = False
        self.current_task_id = None
        self.session_start_time = 0.0
        
        # Token usage
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
        # Cost tracking
        self.total_cost = 0.0
        self.budget_limit = 0.0
        
        # Model usage
        self.model_usage = {}
        
        # API calls
        self.api_calls = 0
        self.api_errors = 0
        
        # Timers
        self.timers = {}
        
        # Budget alert listeners
        self.budget_listeners: Set[Callable[[float, float], None]] = set()
    
    def start_session(self, task_id: Optional[str] = None, budget_limit: Optional[float] = None):
        """Start a new monitoring session.
        
        Args:
            task_id: Optional task ID for the session
            budget_limit: Optional budget limit in dollars
        """
        self.active_session = True
        self.current_task_id = task_id
        self.session_start_time = time.time()
        
        # Reset counters
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        self.model_usage = {}
        self.api_calls = 0
        self.api_errors = 0
        self.timers = {}
        
        # Set budget limit if provided
        if budget_limit is not None:
            self.budget_limit = budget_limit
            
        logger.info(f"Started resource monitoring session: {task_id or 'unnamed'}")
    
    def end_session(self):
        """End the current monitoring session."""
        if not self.active_session:
            return
            
        self.active_session = False
        duration = time.time() - self.session_start_time
        
        logger.info(
            f"Ended monitoring session after {duration:.2f}s. "
            f"Tokens: {self.total_tokens}, Cost: ${self.total_cost:.4f}"
        )
        
        self.current_task_id = None
    
    def track_tokens(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Track token usage.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            model: Model name
        """
        if not self.active_session:
            return
            
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        
        # Track model usage
        self.model_usage[model] = self.model_usage.get(model, 0) + 1
    
    def track_cost(self, cost: float):
        """Track cost.
        
        Args:
            cost: Cost in dollars
        """
        if not self.active_session:
            return
            
        self.total_cost += cost
        
        # Check if budget limit is exceeded
        if self.budget_limit > 0 and self.total_cost > self.budget_limit:
            logger.warning(f"Budget limit exceeded: ${self.total_cost:.4f} > ${self.budget_limit:.4f}")
            self._notify_budget_exceeded()
    
    def track_api_call(self, success: bool = True):
        """Track API call.
        
        Args:
            success: Whether the call was successful
        """
        if not self.active_session:
            return
            
        self.api_calls += 1
        if not success:
            self.api_errors += 1
    
    def start_timer(self, name: str):
        """Start a named timer.
        
        Args:
            name: Timer name
        """
        if not self.active_session:
            return
            
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> Optional[float]:
        """End a named timer and return elapsed time.
        
        Args:
            name: Timer name
            
        Returns:
            Optional[float]: Elapsed time in seconds, or None if timer not found
        """
        if not self.active_session or name not in self.timers:
            return None
            
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of resource usage.
        
        Returns:
            Dict: Resource usage summary
        """
        return {
            "tokens": {
                "total": self.total_tokens,
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens
            },
            "cost": self.total_cost,
            "api_calls": {
                "total": self.api_calls,
                "errors": self.api_errors
            },
            "models": self.model_usage,
            "task_id": self.current_task_id,
            "session_duration": time.time() - self.session_start_time if self.active_session else 0
        }
    
    def add_budget_listener(self, listener: Callable[[float, float], None]):
        """Add a budget limit listener.
        
        Args:
            listener: Callback function that takes current cost and budget limit
        """
        self.budget_listeners.add(listener)
    
    def remove_budget_listener(self, listener: Callable[[float, float], None]):
        """Remove a budget limit listener.
        
        Args:
            listener: Listener to remove
        """
        if listener in self.budget_listeners:
            self.budget_listeners.remove(listener)
    
    def _notify_budget_exceeded(self):
        """Notify all budget listeners that the budget was exceeded."""
        for listener in self.budget_listeners:
            try:
                listener(self.total_cost, self.budget_limit)
            except Exception as e:
                logger.error(f"Error in budget listener: {e}")


# Create global instance
resource_monitor = ResourceMonitor()