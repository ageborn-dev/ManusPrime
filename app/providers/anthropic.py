from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from anthropic import AsyncAnthropic, HUMAN_PROMPT, ASSISTANT_PROMPT, APIError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import BaseProvider, ProviderConfig, ProviderResponse, ModelUsage
from app.logger import logger

class AnthropicProvider(BaseProvider):
    """Anthropic API provider implementation."""
    
    def __init__(self, config: Union[Dict[str, Any], ProviderConfig]):
        super().__init__(config)
        
        # Initialize Anthropic client with beta features
        headers = {
            "output-128k-2025-02-19": "true"  # Enable increased output length
        }
        
        self.client = AsyncAnthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            default_headers=headers
        )
        
        # Set default costs per 1K tokens (based on latest pricing)
        self.set_model_cost("claude-3.7-sonnet", 0.015)  # $0.015/1K input, $0.075/1K output
        self.set_model_cost("claude-3.5-haiku", 0.003)   # $0.003/1K input, $0.015/1K output
        
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
        extended_thinking: bool = False,  # New parameter for Claude 3.7 Sonnet
        **kwargs
    ) -> Union[ProviderResponse, AsyncGenerator[str, None]]:
        """Generate a response using Anthropic's API."""
        try:
            # Validate and get model
            model = self.validate_model(model)
            
            # Format prompt if it's a message list
            if isinstance(prompt, list):
                messages = self._format_messages(prompt)
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

            # Set max tokens based on model and mode
            if model == "claude-3.7-sonnet":
                if extended_thinking:
                    default_max_tokens = 64000  # Extended thinking mode
                else:
                    default_max_tokens = 8192   # Normal mode
            else:
                default_max_tokens = 8192 if model == "claude-3.5-haiku" else 4096
            
            # Generate response
            response = await self.client.messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or default_max_tokens,
                stream=False,
                **kwargs
            )

            # Track usage
            self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens
            self.requests_made += 1

            usage = ModelUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cost=self.calculate_cost(
                    ModelUsage(
                        prompt_tokens=response.usage.input_tokens,
                        completion_tokens=response.usage.output_tokens,
                        total_tokens=response.usage.input_tokens + response.usage.output_tokens
                    ),
                    model
                )
            )

            return ProviderResponse(
                content=response.content[0].text,
                model=model,
                usage=usage,
                finish_reason=response.stop_reason,
                raw_response=response.model_dump()
            )

        except APIError as e:
            logger.error(f"Anthropic API error: {str(e)}")
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
        """Stream response from Anthropic API."""
        try:
            stream = await self.client.messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                stream=True,
                **kwargs
            )

            async for chunk in stream:
                if chunk.content and chunk.content[0].text:
                    yield chunk.content[0].text

        except Exception as e:
            logger.error(f"Error in stream response: {str(e)}")
            raise

    async def generate_with_tools(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response that may include tool calls."""
        try:
            model = self.validate_model(model)
            
            # Format prompt if it's a message list
            if isinstance(prompt, list):
                messages = self._format_messages(prompt)
            else:
                messages = [{"role": "user", "content": prompt}]

            # Convert tools to Anthropic's tool format
            tools_config = self._convert_tools_to_anthropic(tools)

            response = await self.client.messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                tools=tools_config,
                **kwargs
            )

            # Track usage
            self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens
            self.requests_made += 1

            # Extract tool calls if present
            tool_calls = []
            if response.tool_calls:
                tool_calls = [
                    {
                        "name": call.function.name,
                        "parameters": call.function.arguments
                    }
                    for call in response.tool_calls
                ]

            return {
                "content": response.content[0].text if not tool_calls else None,
                "tool_calls": tool_calls if tool_calls else None,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                    "cost": self.calculate_cost(
                        ModelUsage(
                            prompt_tokens=response.usage.input_tokens,
                            completion_tokens=response.usage.output_tokens,
                            total_tokens=response.usage.input_tokens + response.usage.output_tokens
                        ),
                        model
                    )
                }
            }

        except Exception as e:
            logger.error(f"Error in generate_with_tools: {str(e)}")
            raise

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Anthropic API."""
        formatted = []
        for msg in messages:
            if msg["role"] == "system":
                formatted.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "user":
                formatted.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                formatted.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "tool":
                # Add tool responses as assistant messages with a tool prefix
                formatted.append({
                    "role": "assistant",
                    "content": f"Tool Response: {msg['content']}"
                })
        return formatted

    def _convert_tools_to_anthropic(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Anthropic format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
                }
            }
            for tool in tools
        ]

    def supports_tools(self) -> bool:
        """Anthropic supports tool/function calling."""
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and limitations."""
        return {
            **super().get_capabilities(),
            "max_tokens": {
                "claude-3.7-sonnet": {
                    "normal": 8192,
                    "extended": 64000,
                    "beta": 128000  # With beta header
                },
                "claude-3.5-haiku": 8192
            },
            "supports_vision": True,
            "supports_embeddings": False,  # Anthropic doesn't provide embeddings yet
            "cost_per_1k": self._model_costs,
            "extended_thinking": ["claude-3.7-sonnet"]  # Models that support extended thinking
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
