from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
from openai import AsyncOpenAI, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import BaseProvider, ProviderConfig, ProviderResponse, ModelUsage
from app.logger import logger

class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, config: Union[Dict[str, Any], ProviderConfig]):
        super().__init__(config)
        
        # Initialize client
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
        
        # Set default costs per 1K tokens
        self.set_model_cost("gpt-4o", 0.01)
        self.set_model_cost("gpt-4o-mini", 0.005)
        
        # Track usage
        self.total_tokens_used = 0
        self.requests_made = 0

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
    )
    async def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ProviderResponse, AsyncGenerator[str, None]]:
        """Generate a response using OpenAI's API."""
        try:
            # Validate and get model
            model = self.validate_model(model)
            
            # Prepare messages
            if isinstance(prompt, list):
                messages = self.format_messages(prompt)
            else:
                messages = [{"role": "user", "content": prompt}]

            if stream:
                return self._stream_response(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                stream=False,
                **kwargs
            )

            # Track usage
            self.total_tokens_used += response.usage.total_tokens
            self.requests_made += 1

            usage = ModelUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cost=self.calculate_cost(ModelUsage(**response.usage.model_dump()), model)
            )

            return ProviderResponse(
                content=response.choices[0].message.content,
                model=model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response.model_dump()
            )

        except RateLimitError as e:
            logger.warning(f"Rate limit hit for model {model}. Retrying...")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate: {str(e)}")
            raise

    async def _stream_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                stream=True,
                **kwargs
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in stream response: {str(e)}")
            raise

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        tool_choice: str = "auto",
        max_tokens: Optional[int] = None,
        extended_thinking: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response that may include tool calls."""
        try:
            model = self.validate_model(model)
            
            if isinstance(prompt, list):
                messages = self.format_messages(prompt)
            else:
                messages = [{"role": "user", "content": prompt}]

            # Map tool_choice to OpenAI format
            openai_tool_choice = tool_choice
            if tool_choice == "required" and tools:
                openai_tool_choice = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}
            elif tool_choice == "none":
                openai_tool_choice = "none"

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                tools=tools,
                tool_choice=openai_tool_choice,
                **kwargs
            )

            # Track usage
            self.total_tokens_used += response.usage.total_tokens
            self.requests_made += 1

            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": self.calculate_cost(
                        ModelUsage(**response.usage.model_dump()),
                        model
                    )
                }
            }

        except Exception as e:
            logger.error(f"Error in generate_with_tools: {str(e)}")
            raise

    def supports_tools(self) -> bool:
        """OpenAI supports tool/function calling."""
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and limitations."""
        return {
            **super().get_capabilities(),
            "max_tokens": {
                "gpt-4o": 8192,
                "gpt-4o-mini": 4096
            },
            "supports_vision": False,
            "supports_embeddings": True,
            "cost_per_1k": self._model_costs
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "total_requests": self.requests_made,
            "estimated_cost": self.calculate_cost(
                ModelUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=self.total_tokens_used
                ),
                self.get_default_model()
            )
        }