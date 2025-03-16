# plugins/providers/mistral.py
import logging
from typing import Dict, List, Optional, Any, ClassVar

from mistralai.client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from plugins.base import Plugin, PluginCategory, ProviderPlugin
from config import config

logger = logging.getLogger("manusprime.plugins.mistral")

class MistralProvider(ProviderPlugin):
    """Provider plugin for Mistral AI models."""
    
    name: ClassVar[str] = "mistral"
    description: ClassVar[str] = "Mistral AI provider for natural language tasks"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.PROVIDER
    supported_models: ClassVar[List[str]] = [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "mistral-tiny-latest",
        "codestral-latest"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Mistral provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self.client = None
        self.api_key = None
        self.endpoint = "https://api.mistral.ai/v1"
        self.model_costs = {
            "mistral-large-latest": 0.008,   # $0.008 per 1K tokens
            "mistral-medium-latest": 0.005,  # $0.005 per 1K tokens
            "mistral-small-latest": 0.002,   # $0.002 per 1K tokens
            "mistral-tiny-latest": 0.0002,   # $0.0002 per 1K tokens
            "codestral-latest": 0.008        # $0.008 per 1K tokens
        }
        self.default_model = "mistral-large-latest"
    
    async def initialize(self) -> bool:
        """Initialize the Mistral client.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Extract config
            self.api_key = self.config.get("api_key", "")
            endpoint = self.config.get("endpoint", self.endpoint)
            
            if not self.api_key:
                logger.error("Mistral API key not provided")
                return False
            
            # Initialize client
            self.client = MistralAsyncClient(api_key=self.api_key, endpoint=endpoint)
            logger.info("Mistral provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Mistral provider: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """Generate a response from Mistral.
        
        Args:
            prompt: The prompt to send to the model
            model: The specific model to use
            temperature: The sampling temperature
            max_tokens: The maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict: The response data
        """
        if not self.client:
            raise ValueError("Mistral provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.supported_models else self.default_model
        
        try:
            # Prepare messages format
            messages = [ChatMessage(role="user", content=prompt)]
            
            # Make the API call
            response = await self.client.chat(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract content
            content = response.choices[0].message.content if response.choices else ""
            
            # Calculate cost (per 1K tokens)
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            model_cost = self.get_model_cost(model_name)
            cost = (total_tokens / 1000) * model_cost
            
            # Return standardized response
            return {
                "content": content,
                "model": model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response from Mistral: {e}")
            raise
    
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict:
        """Generate a response from Mistral that may include tool calls.
        
        Args:
            prompt: The prompt to send to the model
            tools: The tools available to the model
            model: The specific model to use
            temperature: The sampling temperature
            tool_choice: How to choose tools ("none", "auto", "required")
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict: The response data
        """
        if not self.client:
            raise ValueError("Mistral provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.supported_models else self.default_model
        
        try:
            # Prepare messages format
            messages = [ChatMessage(role="user", content=prompt)]
            
            # Skip tools if tool_choice is "none"
            mistral_tools = None
            if tool_choice != "none":
                mistral_tools = tools
                
            # Handle tool_choice
            mistral_kwargs = {}
            if tool_choice == "required" and mistral_tools:
                mistral_kwargs["tool_choice"] = "any"  # Mistral doesn't have exact equivalent to "required"
            
            # Make the API call
            response = await self.client.chat(
                model=model_name,
                messages=messages,
                temperature=temperature,
                tools=mistral_tools,
                **{**kwargs, **mistral_kwargs}
            )
            
            # Extract content
            message = response.choices[0].message if response.choices else None
            content = message.content if message else ""
            
            # Extract tool calls
            tool_calls = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                for idx, call in enumerate(message.tool_calls):
                    tool_calls.append({
                        'id': call.id if hasattr(call, 'id') else f"call_{idx}",
                        'type': 'function',
                        'function': {
                            'name': call.function.name,
                            'arguments': call.function.arguments
                        }
                    })
            
            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # Calculate cost (per 1K tokens)
            model_cost = self.get_model_cost(model_name)
            cost = (total_tokens / 1000) * model_cost
            
            # Return standardized response
            return {
                "content": content if not tool_calls else None,
                "tool_calls": tool_calls,
                "model": model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response with tools from Mistral: {e}")
            raise
    
    async def execute(self, **kwargs) -> Dict:
        """Execute the provider's primary function (generate).
        
        Args:
            **kwargs: Arguments for the generate method
            
        Returns:
            Dict: The generation result
        """
        if "tools" in kwargs:
            return await self.generate_with_tools(**kwargs)
        else:
            return await self.generate(**kwargs)
    
    def get_default_model(self) -> str:
        """Get the default model for this provider.
        
        Returns:
            str: The name of the default model
        """
        return self.default_model
    
    def get_model_cost(self, model: str) -> float:
        """Get the cost per 1K tokens for a specific model.
        
        Args:
            model: The model name
            
        Returns:
            float: The cost per 1K tokens
        """
        return self.model_costs.get(model, 0.008)
    
    async def cleanup(self) -> None:
        """Clean up resources used by the provider."""
        self.client = None