import logging
from typing import Dict, List

logger = logging.getLogger("manusprime.core.task_analyzer")

class TaskAnalyzer:
    """Handles task analysis and model selection."""
    
    def __init__(self, task_patterns: Dict[str, List[str]], task_models: Dict[str, str]):
        """Initialize TaskAnalyzer.
        
        Args:
            task_patterns: Dictionary of task patterns
            task_models: Dictionary mapping task types to models
        """
        self.task_patterns = task_patterns
        self.task_models = task_models
    
    def analyze_task(self, task: str) -> str:
        """Analyze a task to determine its category.
        
        Args:
            task: The task description
            
        Returns:
            str: The task category ('code', 'planning', etc.)
        """
        logger.debug(f"Analyzing task: {task[:100]}...")
        try:
            task_lower = task.lower()
            
            for category, patterns in self.task_patterns.items():
                for pattern in patterns:
                    if pattern in task_lower:
                        logger.debug(f"Task categorized as: {category}")
                        return category
            
            logger.debug("Task categorized as: default")
            return 'default'
        except Exception as e:
            logger.error(f"Error in analyze_task: {e}")
            logger.error(f"Traceback:", exc_info=True)
            return 'default'
    
    def select_model(self, task: str) -> str:
        """Select the appropriate model for a given task.
        
        Args:
            task: The task description
            
        Returns:
            str: The selected model name
        """
        logger.debug(f"Selecting model for task: {task[:100]}...")
        try:
            category = self.analyze_task(task)
            selected_model = self.task_models.get(category, self.task_models['default'])
            logger.debug(f"Selected model: {selected_model} for category: {category}")
            return selected_model
        except Exception as e:
            logger.error(f"Error in select_model: {e}")
            logger.error(f"Traceback:", exc_info=True)
            return self.task_models['default']
