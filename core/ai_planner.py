import logging
import json
from typing import Dict, List, Optional, Any

from config import config

logger = logging.getLogger("manusprime.core.ai_planner")

class AIPlannerException(Exception):
    """Exception raised for AI planner errors."""
    pass

class AIPlanner:
    """Handles AI-driven task analysis and planning."""
    
    def __init__(self):
        """Initialize the AI planner."""
        # Get available models from configured providers
        self.available_models = self._get_available_models()
        self.model_capabilities = config.get_value("model_capabilities", {})
        
    def _get_available_models(self) -> Dict[str, List[str]]:
        """Get available models from providers with valid API keys."""
        available = {}
        for provider_name, provider_config in config.providers.items():
            if provider_name == "default":
                continue
                
            # Check if provider has API key in environment
            api_key = provider_config.get("api_key", "")
            if isinstance(api_key, str) and api_key.startswith("$"):
                env_var = api_key[1:]
                api_key = os.environ.get(env_var)
            
            if api_key:
                available[provider_name] = provider_config.get("models", [])
                logger.info(f"Provider {provider_name} available with {len(available[provider_name])} models")
        
        return available
    
    def _get_best_reasoning_model(self) -> Optional[str]:
        """Get the best available model for task analysis."""
        reasoning_models = self.model_capabilities.get("reasoning", [])
        
        # Find first available reasoning model
        for model in reasoning_models:
            for provider, models in self.available_models.items():
                if model in models:
                    return model
        
        return None
    
    def _create_analysis_prompt(self, task: str) -> str:
        """Create the prompt for task analysis."""
        # Flatten available models into a list
        model_list = []
        for provider, models in self.available_models.items():
            model_list.extend(models)
        
        return f'''
Given this task: "{task}"

Create a step-by-step execution plan using only these available models:
{json.dumps(model_list, indent=2)}

Each step should be the smallest logical unit of work.

Output a JSON object with this exact structure:
{{
    "steps": [
        {{
            "description": "step description",
            "model": "specific model to use from available list",
            "plugins": ["plugin1", "plugin2"],  // from: {list(config.get_value("plugins.active", {}).keys())}
            "requires_ui": boolean,  // true if step needs visible browser window
            "expected_output": "output type (text, code, visualization, etc)"
        }}
    ]
}}

Important:
1. Only use models from the provided list
2. Break complex tasks into smaller steps
3. Specify exact plugin names from the list provided
4. Set requires_ui true for any step needing visual display
'''
    
    async def create_execution_plan(self, task: str, provider: Any) -> Dict[str, Any]:
        """Create an execution plan for a task.
        
        Args:
            task: The task to analyze
            provider: Provider plugin instance
            
        Returns:
            Dict: Execution plan with steps
            
        Raises:
            AIPlannerException: If planning fails
        """
        try:
            # Get best model for analysis
            model = self._get_best_reasoning_model()
            if not model:
                raise AIPlannerException("No suitable model available for task analysis")
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(task)
            
            # Get analysis from model
            response = await provider.generate(
                prompt=prompt,
                model=model,
                temperature=0.7,
                response_format={"type": "json"}
            )
            
            # Parse and validate plan
            try:
                plan = json.loads(response.get("content", "{}"))
                if not isinstance(plan, dict) or "steps" not in plan:
                    raise ValueError("Invalid plan format")
                
                # Validate each step
                for step in plan["steps"]:
                    required_fields = ["description", "model", "plugins", "requires_ui", "expected_output"]
                    if not all(field in step for field in required_fields):
                        raise ValueError(f"Step missing required fields: {required_fields}")
                
                logger.info(f"Created execution plan with {len(plan['steps'])} steps")
                return plan
                
            except json.JSONDecodeError as e:
                raise AIPlannerException(f"Failed to parse plan: {e}")
            except ValueError as e:
                raise AIPlannerException(f"Invalid plan structure: {e}")
                
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            raise AIPlannerException(f"Planning failed: {e}")
    
    def find_provider_for_model(self, model: str) -> Optional[str]:
        """Find provider that has the specified model.
        
        Args:
            model: Model name to find
            
        Returns:
            Optional[str]: Provider name if found, None otherwise
        """
        for provider, models in self.available_models.items():
            if model in models:
                return provider
        return None
    
    def find_fallback_model(self, original_model: str) -> Optional[tuple]:
        """Find fallback model with similar capabilities.
        
        Args:
            original_model: Original model name
            
        Returns:
            Optional[tuple]: (provider, model) if found, None otherwise
        """
        # Find capability category of original model
        category = None
        for cap, models in self.model_capabilities.items():
            if original_model in models:
                category = cap
                break
        
        if category:
            # Try other models with same capability
            for model in self.model_capabilities[category]:
                if model != original_model:
                    provider = self.find_provider_for_model(model)
                    if provider:
                        return (provider, model)
        
        return None
