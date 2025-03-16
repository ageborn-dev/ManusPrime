from typing import List, Dict, Optional, Tuple
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger("manusprime.fallback")

@dataclass
class ModelConfig:
    """Configuration for a model in the fallback chain."""
    name: str
    provider: str
    timeout: float
    max_retries: int
    cost_per_token: float
    max_tokens: Optional[int] = None
    warm_up: bool = False

class FallbackChain:
    """Manages model fallback chains with intelligent retry logic."""
    
    def __init__(self, 
                 models: List[ModelConfig],
                 max_total_retries: int = 5,
                 cost_threshold: float = 1.0):
        """Initialize fallback chain.
        
        Args:
            models: List of models in fallback order
            max_total_retries: Maximum total retries across all models
            cost_threshold: Maximum cost before trying cheaper models
        """
        self.models = models
        self.max_total_retries = max_total_retries
        self.cost_threshold = cost_threshold
        
        # Track model performance
        self.success_rates: Dict[str, float] = {
            model.name: 1.0 for model in models
        }
        self.error_counts: Dict[str, int] = {
            model.name: 0 for model in models
        }
        self.avg_latency: Dict[str, float] = {
            model.name: 0.0 for model in models
        }
        
        # Backoff tracking
        self.last_error: Dict[str, datetime] = {}
        self.backoff_until: Dict[str, datetime] = {}
        
        # Initialize model connections
        self._warm_up_models()
    
    def _warm_up_models(self):
        """Pre-warm models that have warm_up enabled."""
        for model in self.models:
            if model.warm_up:
                try:
                    # Send minimal prompt to warm up model
                    logger.info(f"Warming up model {model.name}")
                    # Actual warm-up would happen here through provider
                except Exception as e:
                    logger.warning(f"Error warming up {model.name}: {e}")
    
    def _calculate_backoff(self, error_count: int) -> float:
        """Calculate exponential backoff time in seconds."""
        return min(300, 2 ** error_count)  # Max 5 minutes
    
    def _update_metrics(self, 
                       model: str, 
                       success: bool, 
                       latency: float,
                       error: Optional[str] = None):
        """Update performance metrics for a model."""
        # Update success rate
        old_rate = self.success_rates[model]
        self.success_rates[model] = (old_rate * 0.9) + (float(success) * 0.1)
        
        # Update latency
        old_latency = self.avg_latency[model]
        self.avg_latency[model] = (old_latency * 0.9) + (latency * 0.1)
        
        if not success:
            self.error_counts[model] += 1
            self.last_error[model] = datetime.now()
            
            # Calculate backoff
            backoff = self._calculate_backoff(self.error_counts[model])
            self.backoff_until[model] = datetime.now() + timedelta(seconds=backoff)
    
    def _should_skip_model(self, model: ModelConfig) -> Tuple[bool, Optional[str]]:
        """Determine if a model should be skipped based on current state."""
        now = datetime.now()
        
        # Check if model is in backoff
        if model.name in self.backoff_until:
            if now < self.backoff_until[model.name]:
                return True, "backoff"
        
        # Check success rate
        if self.success_rates[model.name] < 0.5:  # Less than 50% success
            return True, "low_success_rate"
        
        # Check error threshold
        if self.error_counts[model.name] > model.max_retries:
            return True, "max_retries_exceeded"
            
        return False, None
    
    async def execute(self, 
                     prompt: str,
                     agent,
                     task_type: Optional[str] = None,
                     **kwargs) -> Dict:
        """Execute prompt with fallback chain.
        
        Args:
            prompt: The prompt to execute
            agent: ManusPrime agent instance
            task_type: Optional task type for model selection
            **kwargs: Additional execution parameters
            
        Returns:
            Dict: Execution result
        """
        total_retries = 0
        total_cost = 0.0
        errors = []
        
        for model in self.models:
            # Check total retries
            if total_retries >= self.max_total_retries:
                break
                
            # Check if we should skip this model
            should_skip, reason = self._should_skip_model(model)
            if should_skip:
                logger.info(f"Skipping {model.name}: {reason}")
                continue
                
            # Check cost threshold
            estimated_cost = len(prompt.split()) * model.cost_per_token
            if total_cost + estimated_cost > self.cost_threshold:
                logger.info(f"Cost threshold would be exceeded with {model.name}")
                continue
            
            try:
                start_time = datetime.now()
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    agent.execute_task(
                        prompt,
                        model=model.name,
                        **kwargs
                    ),
                    timeout=model.timeout
                )
                
                latency = (datetime.now() - start_time).total_seconds()
                
                # Update metrics
                self._update_metrics(
                    model.name,
                    success=result["success"],
                    latency=latency,
                    error=result.get("error")
                )
                
                if result["success"]:
                    # Reset error count on success
                    self.error_counts[model.name] = 0
                    return result
                
                # Track cost
                total_cost += result.get("cost", 0.0)
                
                # Collect error
                errors.append({
                    "model": model.name,
                    "error": result.get("error", "Unknown error")
                })
                
            except asyncio.TimeoutError:
                latency = model.timeout
                self._update_metrics(
                    model.name,
                    success=False,
                    latency=latency,
                    error="Timeout"
                )
                errors.append({
                    "model": model.name,
                    "error": "Timeout"
                })
                
            except Exception as e:
                latency = (datetime.now() - start_time).total_seconds()
                self._update_metrics(
                    model.name,
                    success=False,
                    latency=latency,
                    error=str(e)
                )
                errors.append({
                    "model": model.name,
                    "error": str(e)
                })
            
            total_retries += 1
        
        # All models failed
        return {
            "success": False,
            "error": "All models failed",
            "errors": errors,
            "total_retries": total_retries,
            "total_cost": total_cost
        }
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics for all models.
        
        Returns:
            Dict: Model performance metrics
        """
        return {
            model.name: {
                "success_rate": self.success_rates[model.name],
                "error_count": self.error_counts[model.name],
                "avg_latency": self.avg_latency[model.name],
                "last_error": self.last_error.get(model.name),
                "backoff_until": self.backoff_until.get(model.name)
            }
            for model in self.models
        }
