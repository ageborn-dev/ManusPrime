# core/model_selector.py
import logging
from typing import Dict, List, Optional

from config import config

logger = logging.getLogger("manusprime.model_selector")

class ModelSelector:
    """Selects the most appropriate model for a given task based on patterns and config."""
    
    def __init__(self):
        """Initialize the model selector with task patterns and model mappings."""
        # Task analysis patterns
        self.task_patterns = {
            'code': [
                'code', 'program', 'function', 'script', 'python', 'javascript', 'java', 
                'cpp', 'c++', 'html', 'css', 'api', 'algorithm', 'debug', 'coding'
            ],
            'planning': [
                'plan', 'organize', 'strategy', 'workflow', 'complex', 'steps',
                'roadmap', 'schedule', 'project', 'manage', 'coordinate', 'outline'
            ],
            'tool_use': [
                'search', 'browse', 'file', 'execute', 'run', 'automate', 'zapier',
                'find', 'lookup', 'google', 'web', 'internet', 'download'
            ],
            'creative': [
                'story', 'poem', 'creative', 'write', 'essay', 'blog', 'fiction',
                'article', 'narrative', 'novel', 'scene', 'dialogue', 'character'
            ],
            'reasoning': [
                'analyze', 'evaluate', 'compare', 'critique', 'solve', 'puzzle',
                'logic', 'reason', 'deduce', 'inference', 'argument', 'philosophy'
            ]
        }
        
        # Load task-specific models from config
        self.task_models = {
            'code': config.get_value('task_models.code_generation', 'codestral-latest'),
            'planning': config.get_value('task_models.planning', 'claude-3.7-sonnet'),
            'tool_use': config.get_value('task_models.tool_use', 'claude-3.7-sonnet'),
            'creative': config.get_value('task_models.creative', 'claude-3.7-sonnet'),
            'reasoning': config.get_value('task_models.reasoning', 'claude-3.7-sonnet'),
            'default': config.get_value('task_models.default', 'mistral-small-latest')
        }
        
        # Models by provider
        self.provider_models = self._load_provider_models()
        
        # Provider-based cost optimization
        self.provider_preference = config.get_value('model_selection.provider_preference', ['anthropic', 'openai', 'mistral'])
    
    def _load_provider_models(self) -> Dict[str, List[str]]:
        """Load available models by provider from configuration."""
        provider_models = {}
        for provider_name, provider_config in config.providers.items():
            if provider_name != 'default' and isinstance(provider_config, dict):
                models = provider_config.get('models', [])
                if models:
                    provider_models[provider_name] = models
        return provider_models
    
    def analyze_task(self, task: str) -> str:
        """Analyze a task to determine the best category.
        
        Args:
            task: The task description
            
        Returns:
            str: The task category ('code', 'planning', etc., or 'default')
        """
        task_lower = task.lower()
        
        # Count pattern matches for each category
        category_matches = {category: 0 for category in self.task_patterns}
        
        for category, patterns in self.task_patterns.items():
            for pattern in patterns:
                if pattern in task_lower:
                    category_matches[category] += 1
        
        # Find category with most matches
        best_category = 'default'
        max_matches = 0
        
        for category, matches in category_matches.items():
            if matches > max_matches:
                max_matches = matches
                best_category = category
        
        logger.debug(f"Task categorized as '{best_category}' with {max_matches} pattern matches")
        return best_category
    
    def select_model(self, task: str, provider: Optional[str] = None) -> str:
        """Select the appropriate model for a given task.
        
        Args:
            task: The task description
            provider: Optional provider to use (overrides provider preference)
            
        Returns:
            str: The selected model name
        """
        # Get task category
        category = self.analyze_task(task)
        
        # Get recommended model for this category
        recommended_model = self.task_models.get(category, self.task_models['default'])
        
        # If provider specified, find best matching model from that provider
        if provider:
            provider_models = self.provider_models.get(provider, [])
            if not provider_models:
                logger.warning(f"No models found for provider '{provider}', using recommended model")
                return recommended_model
                
            # Find model with name closest to the recommended model
            for model in provider_models:
                if recommended_model in model:
                    return model
                
            # Fall back to first model from provider
            return provider_models[0]
        
        # Otherwise return the recommended model directly
        return recommended_model
    
    def get_fallback_model(self, provider: Optional[str] = None) -> str:
        """Get a fallback model in case the primary selection fails.
        
        Args:
            provider: Optional provider to use
            
        Returns:
            str: The fallback model name
        """
        if provider:
            # Get first model from specified provider
            provider_models = self.provider_models.get(provider, [])
            if provider_models:
                return provider_models[0]
        
        # Default fallback model
        return self.task_models['default']
    
    def optimize_for_cost(self, task: str, budget_constraint: bool = False) -> str:
        """Select a model optimized for cost while still suitable for the task.
        
        Args:
            task: The task description
            budget_constraint: Whether to prioritize budget constraints
            
        Returns:
            str: The selected model name
        """
        category = self.analyze_task(task)
        
        if budget_constraint:
            # Use the lowest-cost option: task-specific "small" model if available
            for provider in self.provider_preference:
                provider_models = self.provider_models.get(provider, [])
                for model in provider_models:
                    if "small" in model.lower() or "mini" in model.lower():
                        return model
        
        # Fall back to default model
        return self.task_models['default']


# Create a global instance
model_selector = ModelSelector()