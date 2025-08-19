# plugins/providers/openai.py
import logging
import json
from typing import Dict, List, Optional, Any, ClassVar

from openai import AsyncOpenAI
from openai._exceptions import APIError, RateLimitError, APITimeoutError, APIConnectionError
from plugins.base import Plugin, PluginCategory, ProviderPlugin
from config import config
from utils.retry import retry_on_failure, ProviderError, RateLimitError as RetryRateLimitError, ServiceUnavailableError, AuthenticationError

logger = logging.getLogger("manusprime.plugins.openai")

class OpenAIProvider(ProviderPlugin):
    """Provider plugin for OpenAI's GPT models."""
    
    name: ClassVar[str] = "openai"
    description: ClassVar[str] = "OpenAI GPT provider for natural language tasks"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.PROVIDER
    supported_models: ClassVar[List[str]] = [
        "gpt-4o",
        "gpt-4o-mini"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the OpenAI provider."""
        super().__init__(config)
        self.client = None
        self.api_key = None
        self.base_url = "https://api.openai.com/v1"
        self.model_costs = {
            "gpt-4o": 0.010,
            "gpt-4o-mini": 0.005,
        }
        self.default_model = "gpt-4o-mini"
    
    async def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        try:
            # Extract config
            self.api_key = self.config.get("api_key", "")
            base_url = self.config.get("base_url", self.base_url)
            
            if not self.api_key:
                logger.error("OpenAI API key not provided")
                return False
            
            # Initialize client
            self.client = AsyncOpenAI(
                api_key=self.api_key, 
                base_url=base_url,
                timeout=30.0  # Set reasonable timeout
            )
            logger.info("OpenAI provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI provider: {e}")
            return False
    
    async def has_valid_api_key(self) -> bool:
        """Check if the API key is valid.
        
        Returns:
            bool: True if API key is valid
        """
        if not self.api_key or not self.client:
            return False
        
        try:
            # Make a minimal test request
            await self.client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.warning(f"OpenAI API key validation failed: {e}")
            return False
    
    def _handle_openai_error(self, error: Exception) -> Exception:
        """Convert OpenAI errors to our custom exceptions.
        
        Args:
            error: The original OpenAI error
            
        Returns:
            Exception: Converted exception
        """
        if isinstance(error, RateLimitError):
            return RetryRateLimitError(f"OpenAI rate limit exceeded: {error}")
        elif isinstance(error, APITimeoutError):
            return ServiceUnavailableError(f"OpenAI API timeout: {error}")
        elif isinstance(error, APIConnectionError):
            return ServiceUnavailableError(f"OpenAI connection error: {error}")
        elif isinstance(error, APIError):
            if error.status_code == 401:
                return AuthenticationError(f"OpenAI authentication failed: {error}")
            elif error.status_code == 503:
                return ServiceUnavailableError(f"OpenAI service unavailable: {error}")
            else:
                return ProviderError(f"OpenAI API error: {error}")
        else:
            return error
    
    @retry_on_failure(
        max_attempts=3,
        base_delay=1.0,
        exceptions=(RetryRateLimitError, ServiceUnavailableError, ConnectionError, TimeoutError)
    )
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """Generate a response using the Chat Completions API."""
        if not self.client:
            raise ProviderError("OpenAI provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.supported_models else self.default_model
        
        try:
            # Prepare messages with system instruction for JSON output
            messages = [
                {
                    "role": "system",
                    "content": "You must follow the exact output format specified in the prompt. Provide your response in plain text without any additional formatting or explanations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Prepare base request parameters
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature
            }
            
            # Add max_tokens if provided
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            # Filter kwargs to only include supported parameters
            supported_params = {
                "frequency_penalty", "logit_bias", "max_tokens",
                "n", "presence_penalty", "stop", "stream", 
                "temperature", "top_p", "user"
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
            params.update(filtered_kwargs)
            
            # Make the API call
            response = await self.client.chat.completions.create(**params)
            
            # Extract and validate content
            raw_content = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI Raw Response Content: {raw_content}")
            
            # Return response without JSON parsing
            return {
                "content": raw_content,
                "model": model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": response.usage.total_tokens / 1000 * self.get_model_cost(model_name)
                }
            }
            
        except Exception as e:
            # Convert to our custom exceptions
            converted_error = self._handle_openai_error(e)
            logger.error(f"Error generating response from OpenAI: {converted_error}")
            raise converted_error
    
    @retry_on_failure(
        max_attempts=3,
        base_delay=1.0,
        exceptions=(RetryRateLimitError, ServiceUnavailableError, ConnectionError, TimeoutError)
    )
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict:
        """Generate a response that may include tool calls using the Chat Completions API."""
        if not self.client:
            raise ProviderError("OpenAI provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.supported_models else self.default_model
        
        try:
            # Prepare messages for tool usage
            messages = [
                {
                    "role": "system",
                    "content": "You must follow the exact output format specified in the prompt. When using tools, follow the tool-specific format requirements."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Prepare base request parameters
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature
            }
            
            # Add tools configuration
            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice if tool_choice in ["none", "auto"] else "auto"
                if tool_choice == "required" and tools:
                    params["tool_choice"] = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}
            
            # Filter kwargs to only include supported parameters
            supported_params = {
                "frequency_penalty", "logit_bias", "max_tokens",
                "n", "presence_penalty", "stop", "stream", 
                "temperature", "top_p", "user"
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
            params.update(filtered_kwargs)
            
            # Make the API call
            response = await self.client.chat.completions.create(**params)
            
            # Extract response content and tool calls
            message = response.choices[0].message
            raw_content = message.content.strip() if message.content else "{}"
            
            # Extract tool calls if present
            tool_calls = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                for call in message.tool_calls:
                    tool_calls.append({
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments
                        }
                    })
            
            return {
                "content": raw_content,
                "tool_calls": tool_calls,
                "model": model_name,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": response.usage.total_tokens / 1000 * self.get_model_cost(model_name)
                }
            }
            
        except Exception as e:
            # Convert to our custom exceptions
            converted_error = self._handle_openai_error(e)
            logger.error(f"Error generating response with tools from OpenAI: {converted_error}")
            raise converted_error
    
    async def execute(self, **kwargs) -> Dict:
        """Execute the provider's primary function (generate)."""
        try:
            if "tools" in kwargs:
                return await self.generate_with_tools(**kwargs)
            else:
                return await self.generate(**kwargs)
        except Exception as e:
            logger.error(f"Error in OpenAI provider execute: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": kwargs.get("model", self.default_model),
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0
                }
            }
    
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return self.default_model
    
    def get_model_cost(self, model: str) -> float:
        """Get the cost per 1K tokens for a specific model."""
        return self.model_costs.get(model, 0.010)
    
    async def cleanup(self) -> None:
        """Clean up resources used by the provider."""
        if self.client:
            try:
                await self.client.close()
            except Exception as e:
                logger.warning(f"Error closing OpenAI client: {e}")
            finally:
                self.client = None
                logger.info("OpenAI client closed")
                