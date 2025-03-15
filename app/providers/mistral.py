from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from mistralai.client import MistralAsyncClient
from mistralai.exceptions import MistralAPIError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import BaseProvider, ProviderConfig, ProviderResponse, ModelUsage
from app.logger import logger

class MistralProvider(BaseProvider):
    """Mistral AI provider implementation."""
    
    def __init__(self, config: Union[Dict[str, Any], ProviderConfig]):
        super().__init__(config)
        
        # Initialize Mistral client
        self.client = MistralAsyncClient(
            api_key=self.config.api_key,
            endpoint=self.config.base_url
        )
        
        # Premier models
        self.set_model_cost("mistral-large-latest", 0.008)
        self.set_model_cost("pixtral-large-latest", 0.008)
        self.set_model_cost("codestral-latest", 0.008)
        self.set_model_cost("mistral-saba-latest", 0.008)
        self.set_model_cost("ministral-8b-latest", 0.004)
        self.set_model_cost("ministral-3b-latest", 0.002)
        # Free models
        self.set_model_cost("mistral-small-latest", 0.002)
        self.set_model_cost("pixtral-12b-2409", 0.002)
        
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
        """Generate a response using Mistral's API."""
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

            # Set max tokens based on model capabilities
            model_max_tokens = self.get_capabilities()["max_tokens"].get(model, 4096)
            
            # Generate response
            response = await self.client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or model_max_tokens,
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
                cost=self.calculate_cost(
                    ModelUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens
                    ),
                    model
                )
            )

            return ProviderResponse(
                content=response.choices[0].message.content,
                model=model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response.model_dump()
            )

        except MistralAPIError as e:
            logger.error(f"Mistral API error: {str(e)}")
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
        """Stream response from Mistral API."""
        try:
            model_max_tokens = self.get_capabilities()["max_tokens"].get(model, 4096)
            
            stream = await self.client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or model_max_tokens,
                stream=True,
                **kwargs
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
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
                messages = self._format_messages(prompt)
            else:
                messages = [{"role": "user", "content": prompt}]

            # Convert tools to Mistral's format
            mistral_tools = self._convert_tools_to_mistral(tools)
            
            # Skip tools if tool_choice is "none"
            if tool_choice == "none":
                mistral_tools = []

            # Handle tool_choice
            mistral_kwargs = {}
            if tool_choice == "required" and mistral_tools:
                mistral_kwargs["tool_choice"] = "any"  # Mistral doesn't have exact equivalent to "required"

            # Set max tokens based on model capabilities
            model_max_tokens = self.get_capabilities()["max_tokens"].get(model, 4096)

            response = await self.client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or model_max_tokens,
                tools=mistral_tools if mistral_tools else None,
                **{**kwargs, **mistral_kwargs}
            )

# Track usage
            self.total_tokens_used += response.usage.total_tokens
            self.requests_made += 1

            # Extract tool calls if present
            tool_calls = []
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                for idx, call in enumerate(response.choices[0].message.tool_calls):
                    tool_calls.append({
                        'id': call.id if hasattr(call, 'id') else f"call_{idx}",
                        'type': 'function',
                        'function': {
                            'name': call.function.name,
                            'arguments': call.function.arguments
                        }
                    })

            return {
                "content": response.choices[0].message.content if not tool_calls else None,
                "tool_calls": tool_calls if tool_calls else None,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": self.calculate_cost(
                        ModelUsage(
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens
                        ),
                        model
                    )
                }
            }

        except Exception as e:
            logger.error(f"Error in generate_with_tools: {str(e)}")
            raise

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Mistral API."""
        formatted = []
        system_content = None
        
        for msg in messages:
            if msg["role"] == "system":
                # Mistral doesn't have explicit system messages, collect for prepending
                system_content = msg["content"]
            elif msg["role"] in ["user", "assistant"]:
                formatted.append({"role": msg["role"], "content": msg["content"]})
            elif msg["role"] == "tool":
                # Add tool responses in a format Mistral understands
                formatted.append({
                    "role": "user",
                    "content": f"Tool Response from {msg.get('name', 'tool')}: {msg['content']}"
                })
        
        # Prepend system message to first user message if present
        if system_content and formatted and formatted[0]["role"] == "user":
            formatted[0]["content"] = f"{system_content}\n\n{formatted[0]['content']}"
        elif system_content:
            formatted.append({"role": "user", "content": system_content})
            
        return formatted

    def _convert_tools_to_mistral(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Mistral format."""
        mistral_tools = []
        for tool in tools:
            if tool["type"] == "function":
                mistral_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"].get("parameters", {})
                    }
                })
        return mistral_tools

    def supports_tools(self) -> bool:
        """Mistral supports tool/function calling."""
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and limitations."""
        return {
            **super().get_capabilities(),
            "max_tokens": {
                # Premier models
                "mistral-large-latest": 131072,  # 131K
                "pixtral-large-latest": 131072,  # 131K
                "codestral-latest": 262144,      # 256K
                "mistral-saba-latest": 32768,    # 32K
                "ministral-8b-latest": 131072,   # 131K
                "ministral-3b-latest": 131072,   # 131K
                # Free models
                "mistral-small-latest": 32768,   # 32K
                "pixtral-12b-2409": 131072      # 131K
            },
            "supports_vision": True,  # Pixtral models support vision
            "supports_embeddings": True,
            "cost_per_1k": self._model_costs,
            "model_deprecation": {
                "open-mistral-7b": {
                    "legacy_date": "2024/11/25",
                    "deprecation_date": "2024/11/30",
                    "retirement_date": "2025/03/30",
                    "alternative_model": "ministral-8b-latest"
                },
                "open-mixtral-8x7b": {
                    "legacy_date": "2024/11/25",
                    "deprecation_date": "2024/11/30",
                    "retirement_date": "2025/03/30",
                    "alternative_model": "mistral-small-latest"
                }
            }
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