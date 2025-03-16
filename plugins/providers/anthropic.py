# plugins/providers/anthropic.py
import logging
from typing import Dict, List, Optional, Any, ClassVar

from anthropic import AsyncAnthropic
from plugins.base import Plugin, PluginCategory, ProviderPlugin
from config import config

logger = logging.getLogger("manusprime.plugins.anthropic")

class AnthropicProvider(ProviderPlugin):
    """Provider plugin for Anthropic's Claude models."""
    
    name: ClassVar[str] = "anthropic"
    description: ClassVar[str] = "Anthropic Claude provider for natural language tasks"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.PROVIDER
    supported_models: ClassVar[List[str]] = [
        "claude-3.7-sonnet", 
        "claude-3.5-haiku", 
        "claude-3-opus", 
        "claude-3-sonnet"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Anthropic provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self.client = None
        self.api_key = None
        self.base_url = "https://api.anthropic.com/v1"
        self.model_costs = {
            "claude-3.7-sonnet": 0.015,  # $0.015 per 1K tokens
            "claude-3.5-haiku": 0.003,   # $0.003 per 1K tokens
            "claude-3-opus": 0.030,      # $0.030 per 1K tokens
            "claude-3-sonnet": 0.015     # $0.015 per 1K tokens
        }
        self.default_model = "claude-3.7-sonnet"
    
    async def initialize(self) -> bool:
        """Initialize the Anthropic client.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Extract config
            self.api_key = self.config.get("api_key", "")
            base_url = self.config.get("base_url", self.base_url)
            
            if not self.api_key:
                logger.error("Anthropic API key not provided")
                return False
            
            # Initialize client
            self.client = AsyncAnthropic(api_key=self.api_key, base_url=base_url)
            logger.info("Anthropic provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Anthropic provider: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """Generate a response from Claude.
        
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
            raise ValueError("Anthropic provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.supported_models else self.default_model
        
        try:
            # Prepare messages format
            messages = [{"role": "user", "content": prompt}]
            
            # Set reasonable default for max_tokens if not provided
            max_tokens_value = max_tokens or 4096
            
            # Make the API call
            response = await self.client.messages.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens_value,
                **kwargs
            )
            
            # Extract content
            content = response.content[0].text if response.content else ""
            
            # Calculate cost
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens
            
            # Calculate cost (per 1K tokens)
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
            logger.error(f"Error generating response from Anthropic: {e}")
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
        """Generate a response from Claude that may include tool calls.
        
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
            raise ValueError("Anthropic provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.supported_models else self.default_model
        
        try:
            # Prepare messages format
            messages = [{"role": "user", "content": prompt}]
            
            # Skip tools if tool_choice is "none"
            anthropic_tools = []
            if tool_choice != "none":
                # Convert tools to Anthropic format
                anthropic_tools = []
                for tool in tools:
                    if tool["type"] == "function":
                        anthropic_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool["function"]["name"],
                                "description": tool["function"].get("description", ""),
                                "parameters": tool["function"].get("parameters", {})
                            }
                        })
            
            # Make the API call
            response = await self.client.messages.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                tools=anthropic_tools if anthropic_tools else None,
                **kwargs
            )
            
            # Extract content
            content = response.content[0].text if response.content else ""
            
            # Extract tool calls if present
            tool_calls = []
            if response.tool_calls:
                for idx, call in enumerate(response.tool_calls):
                    tool_calls.append({
                        'id': call.id if hasattr(call, 'id') else f"call_{idx}",
                        'type': 'function',
                        'function': {
                            'name': call.function.name,
                            'arguments': call.function.arguments
                        }
                    })
            
            # Calculate cost
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens
            
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
            logger.error(f"Error generating response with tools from Anthropic: {e}")
            raise
    
    async def execute(self, **kwargs) -> Dict:
        """Execute the provider's primary function (generate).
        
        Args:
            **kwargs: Arguments for the generate method
            
        Returns:
            Dict: The generation result
        """
        if kwargs.get("tools"):
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
        return self.model_costs.get(model, 0.015)
    
    async def cleanup(self) -> None:
        """Clean up resources used by the provider."""
        self.client = None