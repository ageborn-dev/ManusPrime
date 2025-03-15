from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import aiohttp
from tenacity import retry, stop_after_attempt, wait_random_exponential
from datetime import datetime, timezone

from .base import BaseProvider, ProviderConfig, ProviderResponse, ModelUsage
from app.logger import logger

class DeepSeekProvider(BaseProvider):
    """DeepSeek API provider implementation."""
    
    def __init__(self, config: Union[Dict[str, Any], ProviderConfig]):
        super().__init__(config)
        
        # Initialize session
        self.session = None
        
        self.set_model_cost("deepseek-chat", {
            "cache_hit": 0.07,
            "cache_miss": 0.27,
            "output": 1.10 
        })
        self.set_model_cost("deepseek-reasoner", {
            "cache_hit": 0.14,
            "cache_miss": 0.55,
            "output": 2.19
        })
        
        # Track usage
        self.total_tokens_used = 0
        self.requests_made = 0

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                base_url=self.config.base_url,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

    async def _close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    def _is_discount_period(self) -> bool:
        """Check if current time is in discount period (UTC 16:30-00:30)."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute
        current_minutes = hour * 60 + minute
        
        start_discount = 16 * 60 + 30
        end_discount = 24 * 60 + 30
        
        return current_minutes >= start_discount or current_minutes < (end_discount % (24 * 60))

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
        """Generate a response using DeepSeek's API."""
        try:
            # Validate and get model
            model = self.validate_model(model)
            
            # Format prompt if it's a message list
            if isinstance(prompt, list):
                messages = self._format_messages(prompt)
            else:
                messages = [{"role": "user", "content": prompt}]

            await self._ensure_session()

            if stream:
                return self._stream_response(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

            # Set max tokens based on model capabilities
            model_caps = self.get_capabilities()["max_tokens"][model]
            default_max_tokens = model_caps["output"]
            
            # Prepare request payload with context caching
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or default_max_tokens,
                "stream": False,
                "context_caching": True,
                **kwargs
            }

            if model == "deepseek-reasoner":
                payload["cot_tokens"] = model_caps["cot"]

            async with self.session.post("/v1/chat/completions", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek API error: {error_text}")

                data = await response.json()


            is_discount = self._is_discount_period()
            discount_factor = 0.5 if model == "deepseek-chat" else 0.25 

            usage = ModelUsage(
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
                total_tokens=data["usage"]["total_tokens"],
                cost=self.calculate_cost(
                    ModelUsage(**data["usage"]),
                    model
                ) * (discount_factor if is_discount else 1.0)
            )

            # Update total usage
            self.total_tokens_used += data["usage"]["total_tokens"]
            self.requests_made += 1

            return ProviderResponse(
                content=data["choices"][0]["message"]["content"],
                model=model,
                usage=usage,
                finish_reason=data["choices"][0]["finish_reason"],
                raw_response=data
            )

        except Exception as e:
            logger.error(f"DeepSeek generation error: {str(e)}")
            raise
        finally:
            if not stream:
                await self._close_session()

    async def _stream_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from DeepSeek API."""
        try:
            # Set max tokens based on model capabilities
            model_caps = self.get_capabilities()["max_tokens"][model]
            default_max_tokens = model_caps["output"]
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or default_max_tokens,
                "stream": True,
                "context_caching": True,
                **kwargs
            }

            if model == "deepseek-reasoner":
                payload["cot_tokens"] = model_caps["cot"]

            async with self.session.post("/v1/chat/completions", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek API error: {error_text}")

                async for line in response.content.iter_any():
                    line = line.decode('utf-8').strip()
                    if line:
                        if line.startswith("data: "):
                            line = line[6:] 
                            try:
                                chunk = eval(line)
                                if chunk["choices"][0]["delta"].get("content"):
                                    yield chunk["choices"][0]["delta"]["content"]
                            except Exception as e:
                                logger.warning(f"Failed to parse chunk: {line}")

        except Exception as e:
            logger.error(f"Error in stream response: {str(e)}")
            raise
        finally:
            await self._close_session()

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

            await self._ensure_session()

            model_caps = self.get_capabilities()["max_tokens"][model]
            default_max_tokens = model_caps["output"]

            # Map tool_choice to DeepSeek format (which uses OpenAI format)
            ds_tool_choice = tool_choice
            if tool_choice == "required" and tools:
                ds_tool_choice = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}
            elif tool_choice == "none":
                ds_tool_choice = "none"
                tools = []

            # Prepare request payload
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": default_max_tokens,
                "tools": tools if tools else None,
                "tool_choice": ds_tool_choice,
                "context_caching": True
            }

            if model == "deepseek-reasoner":
                payload["cot_tokens"] = model_caps["cot"]

            async with self.session.post("/v1/chat/completions", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek API error: {error_text}")

                data = await response.json()

            # Track usage with discount if applicable
            is_discount = self._is_discount_period()
            discount_factor = 0.5 if model == "deepseek-chat" else 0.25

            self.total_tokens_used += data["usage"]["total_tokens"]
            self.requests_made += 1

            # Extract tool calls if present
            tool_calls = []
            message = data["choices"][0]["message"]
            if "tool_calls" in message:
                for idx, call in enumerate(message["tool_calls"]):
                    tool_calls.append({
                        'id': call.get('id', f"call_{idx}"),
                        'type': 'function',
                        'function': {
                            'name': call["function"]["name"],
                            'arguments': call["function"]["arguments"]
                        }
                    })

            return {
                "content": message["content"] if not tool_calls else None,
                "tool_calls": tool_calls if tool_calls else None,
                "model": model,
                "usage": {
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                    "total_tokens": data["usage"]["total_tokens"],
                    "cost": self.calculate_cost(
                        ModelUsage(**data["usage"]),
                        model
                    ) * (discount_factor if is_discount else 1.0)
                }
            }

        except Exception as e:
            logger.error(f"Error in generate_with_tools: {str(e)}")
            raise
        finally:
            await self._close_session()

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for DeepSeek API."""
        formatted = []
        for msg in messages:
            if msg["role"] == "system":
                formatted.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "user":
                formatted.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                formatted.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "tool":
                # Add tool responses as assistant messages
                formatted.append({
                    "role": "assistant",
                    "content": f"Tool Response from {msg.get('name', 'tool')}: {msg['content']}"
                })
        return formatted

    def supports_tools(self) -> bool:
        """DeepSeek supports tool/function calling."""
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and limitations."""
        return {
            **super().get_capabilities(),
            "max_tokens": {
                "deepseek-chat": {
                    "context": 64000,
                    "output": 8000
                },
                "deepseek-reasoner": {
                    "context": 64000,
                    "output": 8000,
                    "cot": 32000 
                }
            },
            "supports_vision": False,
            "supports_embeddings": False,
            "cost_per_1k": self._model_costs,
            "features": {
                "context_caching": True,
                "chain_of_thought": ["deepseek-reasoner"] 
            },
            "pricing_tiers": {
                "standard": {
                    "time": "UTC 00:30-16:30",
                    "discount": 0
                },
                "discount": {
                    "time": "UTC 16:30-00:30",
                    "discount": 0.50
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