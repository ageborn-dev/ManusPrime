# plugins/providers/openai.py
import logging
import json
from typing import Dict, List, Optional, Any, ClassVar

from openai import AsyncOpenAI
from plugins.base import Plugin, PluginCategory, ProviderPlugin
from manusprime.config import config

logger = logging.getLogger("manusprime.plugins.openai")

class OpenAIProvider(ProviderPlugin):
    """Provider plugin for OpenAI's GPT models."""
    
    name: ClassVar[str] = "openai"
    description: ClassVar[str] = "OpenAI GPT provider for natural language tasks"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.PROVIDER
    supported_models: ClassVar[List[str]] = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the OpenAI provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self.client = None
        self.api_key = None
        self.base_url = "https://api.openai.com/v1"
        self.model_costs = {
            "gpt-4o": 0.010,        # $0.010 per 1K tokens
            "gpt-4o-mini": 0.005,   # $0.005 per 1K tokens
            "gpt-4-turbo": 0.010,   # $0.010 per 1K tokens
            "gpt-3.5-turbo": 0.001  # $0.001 per 1K tokens
        }
        self.default_model = "gpt-4o"
    
    async def initialize(self) -> bool:
        """Initialize the OpenAI client.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Extract config
            self.api_key = self.config.get("api_key", "")
            base_url = self.config.get("base_url", self.base_url)
            
            if not self.api_key:
                logger.error("OpenAI API key not provided")
                return False
            
            # Initialize client
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
            logger.info("OpenAI provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI provider: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """Generate a response from GPT.
        
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
            raise ValueError("OpenAI provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.supported_models else self.default_model
        
        try:
            # Prepare messages format
            messages = [{"role": "user", "content": prompt}]
            
            # Make the API call
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract content
            content = response.choices[0].message.content if response.choices else ""
            
            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
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
            logger.error(f"Error generating response from OpenAI: {e}")
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
        """Generate a response from GPT that may include tool calls.
        
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
            raise ValueError("OpenAI provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.supported_models else self.default_model
        
        try:
            # Prepare messages format
            messages = [{"role": "user", "content": prompt}]
            
            # Map tool_choice to OpenAI format
            openai_tool_choice = tool_choice
            if tool_choice == "required" and tools:
                # If required, specify first tool
                openai_tool_choice = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}
            
            # Make the API call
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                tools=tools if tools and tool_choice != "none" else None,
                tool_choice=openai_tool_choice if tool_choice != "none" else "none",
                **kwargs
            )
            
            # Extract content
            message = response.choices[0].message if response.choices else None
            content = message.content if message else ""
            
            # Extract tool calls
            tool_calls = []
            if message and message.tool_calls:
                for tool_call in message.tool_calls:
                    # Convert to standard format
                    tool_calls.append({
                        'id': tool_call.id,
                        'type': 'function',
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments
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
            logger.error(f"Error generating response with tools from OpenAI: {e}")
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
        return self.model_costs.get(model, 0.010)
    
    async def cleanup(self) -> None:
        """Clean up resources used by the provider."""
        self.client = None
