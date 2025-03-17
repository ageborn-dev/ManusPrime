import logging
import json
import os
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
    
    async def get_analysis_model(self, provider: Any) -> Optional[str]:
        """Get the best model for task analysis based on capabilities.
        
        Args:
            provider: Provider plugin instance
            
        Returns:
            Optional[str]: Best model for analysis, or None if none available
        """
        try:
            # Get models with reasoning capabilities
            reasoning_models = self.model_capabilities.get("reasoning", [])
            if not reasoning_models:
                logger.warning("No models configured for reasoning capabilities")
                return None
                
            # Score each model based on capabilities
            model_scores = {}
            for model in reasoning_models:
                if not any(model in models for models in self.available_models.values()):
                    continue
                    
                # Get model capabilities from provider
                capabilities = await provider.get_model_capabilities(model)
                
                # Score based on relevant capabilities
                score = 0
                if capabilities.get("context_length", 0) >= 8000:  # Prefer models with larger context
                    score += 2
                if capabilities.get("function_calling", False):  # Prefer models with function calling
                    score += 2
                if capabilities.get("json_mode", False):  # Prefer models with JSON output
                    score += 1
                    
                model_scores[model] = score
            
            if not model_scores:
                return None
                
            # Return model with highest score
            return max(model_scores.items(), key=lambda x: x[1])[0]
            
        except Exception as e:
            logger.error(f"Error getting analysis model: {e}")
            return None
    
    def _create_analysis_prompt(self, task: str, available_plugins: List[str]) -> str:
        """Create the prompt for task analysis."""
        # Flatten available models into a list
        model_list = []
        for provider, models in self.available_models.items():
            model_list.extend(models)
        
        return f'''
Analyze this task in detail: "{task}"

Available Models:
{json.dumps(model_list, indent=2)}

Available Plugins:
{json.dumps(available_plugins, indent=2)}

Analyze and provide a structured execution plan that considers:

1. Task Analysis:
- Primary task category and subcategories
- Required capabilities and skills
- Complexity assessment
- Potential challenges or constraints

2. Resource Requirements:
- Required model capabilities
- Necessary plugins and tools
- External dependencies
- Memory or processing needs

3. Execution Strategy:
- Parallel vs sequential processing opportunities
- Dependencies between subtasks
- Error handling considerations
- Performance optimization opportunities

Output a comprehensive JSON plan with this structure:
{{
    "analysis": {{
        "task_type": "primary category of task",
        "categories": ["relevant", "task", "categories"],
        "capabilities_needed": ["capability1", "capability2"],
        "complexity_assessment": {{
            "level": "simple|moderate|complex",
            "factors": ["factor1", "factor2"]
        }},
        "challenges": ["challenge1", "challenge2"]
    }},
    "resources": {{
        "required_capabilities": ["capability1", "capability2"],
        "suggested_models": ["model1", "model2"],
        "required_plugins": ["plugin1", "plugin2"],
        "memory_requirements": "low|medium|high",
        "external_dependencies": ["dependency1", "dependency2"]
    }},
    "execution_plan": {{
        "parallel_execution": boolean,
        "steps": [
            {{
                "id": "step-1",
                "description": "detailed step description",
                "model": "specific model name from available list",
                "plugins": ["required plugin names"],
                "requires_ui": boolean,
                "expected_output": "output type",
                "error_handling": {{
                    "retry_strategy": "none|simple|exponential",
                    "max_retries": number,
                    "fallback_action": "description of fallback"
                }},
                "dependencies": ["step-ids this step depends on"],
                "estimated_complexity": "low|medium|high"
            }}
        ]
    }}
}}

Important:
1. Only reference available models and plugins
2. Break complex operations into atomic steps
3. Include error handling for critical steps
4. Specify clear dependencies between steps
5. Consider resource efficiency and parallel execution opportunities
'''
    
    async def create_execution_plan(self, task: str, provider: Any, cache: Optional[Any] = None) -> Dict[str, Any]:
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
            # Get best model for analysis with caching
            cache_key = f"analysis_model_{provider.__class__.__name__}"
            if cache:
                model = cache.get(cache_key)
            
            if not model:
                model = await self.get_analysis_model(provider)
                if cache:
                    cache.put(cache_key, model)
            if not model:
                raise AIPlannerException("No suitable model available for task analysis")
            
            # Get available plugins
            available_plugins = list(config.get_value("plugins.active", {}).keys())
            
            # Create detailed analysis prompt
            prompt = self._create_analysis_prompt(task, available_plugins)
            
            # Get analysis from model
            response = await provider.generate(
                prompt=prompt,
                model=model,
                temperature=0.7,
                response_format={"type": "json"}
            )
            
            # Parse and enhance plan
            try:
                plan = json.loads(response.get("content", "{}"))
                if not isinstance(plan, dict):
                    raise ValueError("Invalid plan format")
                
                # Validate and enhance plan sections
                required_sections = ["analysis", "resources", "execution_plan"]
                if not all(section in plan for section in required_sections):
                    raise ValueError(f"Plan missing required sections: {required_sections}")
                
                # Validate execution steps
                if "steps" in plan.get("execution_plan", {}):
                    for step in plan["execution_plan"]["steps"]:
                        required_fields = [
                            "id", "description", "model", "plugins",
                            "requires_ui", "expected_output", "error_handling",
                            "dependencies", "estimated_complexity"
                        ]
                        if not all(field in step for field in required_fields):
                            raise ValueError(f"Step missing required fields: {required_fields}")
                
                # Enhance plan with performance estimates
                plan["performance_estimates"] = {
                    "expected_duration": "short|medium|long",
                    "resource_intensity": "low|medium|high",
                    "parallelization_potential": plan.get("execution_plan", {}).get("parallel_execution", False)
                }
                
                logger.info(f"Created enhanced execution plan with {len(plan.get('execution_plan', {}).get('steps', []))} steps")
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
