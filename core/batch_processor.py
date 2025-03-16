import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from utils.monitor import resource_monitor

logger = logging.getLogger("manusprime.batch")

@dataclass
class BatchTask:
    """Represents a task in the batch queue."""
    id: str
    prompt: str
    model: Optional[str] = None
    priority: int = 0
    max_retries: int = 3
    retries: int = 0
    result: Optional[Dict] = None
    error: Optional[str] = None

class BatchProcessor:
    """Handles batch processing of multiple tasks."""
    
    def __init__(self, 
                 max_batch_size: int = 10,
                 max_concurrent: int = 3,
                 cost_threshold: float = 1.0):
        """Initialize batch processor.
        
        Args:
            max_batch_size: Maximum tasks per batch
            max_concurrent: Maximum concurrent batches
            cost_threshold: Maximum cost per batch
        """
        self.max_batch_size = max_batch_size
        self.max_concurrent = max_concurrent
        self.cost_threshold = cost_threshold
        
        self.queue: List[BatchTask] = []
        self.processing: List[BatchTask] = []
        self.completed: List[BatchTask] = []
        
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._lock = asyncio.Lock()
        
    async def add_task(self, task: BatchTask) -> str:
        """Add a task to the queue.
        
        Args:
            task: The task to add
            
        Returns:
            str: Task ID
        """
        async with self._lock:
            self.queue.append(task)
            # Sort by priority (higher first)
            self.queue.sort(key=lambda x: x.priority, reverse=True)
            
        return task.id
        
    async def process_batch(self, tasks: List[BatchTask], agent) -> List[Dict]:
        """Process a batch of tasks.
        
        Args:
            tasks: List of tasks to process
            agent: ManusPrime agent instance
            
        Returns:
            List[Dict]: Results for each task
        """
        results = []
        current_cost = 0.0
        
        for task in tasks:
            # Check if batch cost threshold exceeded
            if current_cost >= self.cost_threshold:
                # Move remaining tasks back to queue
                async with self._lock:
                    self.queue.extend(tasks[tasks.index(task):])
                break
                
            try:
                result = await agent.execute_task(task.prompt, model=task.model)
                
                # Update cost tracking
                current_cost += result.get("cost", 0.0)
                
                if result["success"]:
                    task.result = result
                    results.append(result)
                else:
                    if task.retries < task.max_retries:
                        # Retry failed task
                        task.retries += 1
                        async with self._lock:
                            self.queue.append(task)
                    else:
                        task.error = result.get("error", "Unknown error")
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"Error processing task {task.id}: {e}")
                task.error = str(e)
                results.append({"success": False, "error": str(e)})
                
            # Add to completed list
            if task.result or task.error:
                self.completed.append(task)
                
        return results
        
    async def process_queue(self, agent) -> List[Dict]:
        """Process all tasks in the queue.
        
        Args:
            agent: ManusPrime agent instance
            
        Returns:
            List[Dict]: Results for all processed tasks
        """
        all_results = []
        
        while self.queue:
            async with self._lock:
                # Get next batch
                batch = self.queue[:self.max_batch_size]
                self.queue = self.queue[self.max_batch_size:]
                self.processing.extend(batch)
            
            # Process batch
            results = await self.process_batch(batch, agent)
            all_results.extend(results)
            
            # Remove from processing
            for task in batch:
                if task in self.processing:
                    self.processing.remove(task)
                    
        return all_results
        
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific task.
        
        Args:
            task_id: The task ID to check
            
        Returns:
            Optional[Dict]: Task status or None if not found
        """
        # Check completed tasks
        for task in self.completed:
            if task.id == task_id:
                return {
                    "status": "completed",
                    "result": task.result,
                    "error": task.error
                }
                
        # Check processing tasks
        for task in self.processing:
            if task.id == task_id:
                return {"status": "processing"}
                
        # Check queued tasks
        for task in self.queue:
            if task.id == task_id:
                return {"status": "queued"}
                
        return None
        
    def get_queue_stats(self) -> Dict[str, int]:
        """Get current queue statistics.
        
        Returns:
            Dict[str, int]: Queue statistics
        """
        return {
            "queued": len(self.queue),
            "processing": len(self.processing),
            "completed": len(self.completed)
        }
        
    def clear_completed(self):
        """Clear completed tasks from memory."""
        self.completed.clear()
