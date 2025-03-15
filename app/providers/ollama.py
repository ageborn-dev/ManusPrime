from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import aiohttp
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import BaseProvider, ProviderConfig, ProviderResponse, ModelUsage
from app.logger import logger

class OllamaProvider(BaseProvider):
    """Ollama API provider for local model deployment."""
    
    def __init__(self, config: Union[Dict[str, Any], ProviderConfig]):
        super().__init__(config)
        self.session = None
        self.total_tokens_used = 0
        self.requests_made = 0

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                base_url=self.config.base_url,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

    async def _close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

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
        """Generate a response using Ollama's API."""
        try:
            model = self.validate_model(model)
            await self._ensure_session()

            # Format messages if needed
            if isinstance(prompt, list):
                prompt = self._format_chat_prompt(prompt)

            # Prepare request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    **({"num_tokens": max_tokens} if max_tokens else {})
                }
            }

            if stream:
                return self._stream_response(payload)

            async with self.session.post("/api/generate", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Ollama API error: {error_text}")

                data = await response.json()
                
                # Estimate token usage (Ollama doesn't provide this directly)
                prompt_tokens = len(prompt.split()) if isinstance(prompt, str) else sum(len(msg.get("content", "").split()) for msg in prompt)
                completion_tokens = len(data["response"].split())
                total_tokens = prompt_tokens + completion_tokens
                
                # Track usage
                self.total_tokens_used += total_tokens
                self.requests_made += 1
                
                usage = ModelUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=0.0  # Local models have no cost
                )
                
                return ProviderResponse(
                    content=data["response"],
                    model=model,
                    usage=usage,
                    finish_reason="stop",  # Ollama doesn't provide this info
                    raw_response=data
                )

        except Exception as e:
            logger.error(f"Ollama generation error: {str(e)}")
            raise
        finally:
            if not stream:
                await self._close_session()

    async def _stream_response(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream response from Ollama API."""
        try:
            async with self.session.post("/api/generate", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Ollama API error: {error_text}")

                # Ollama streams JSON lines
                async for line in response.content:
                    if not line:
                        continue
                    try:
                        line_str = line.decode('utf-8').strip()
                        if not line_str:
                            continue
                        data = json.loads(line_str)
                        if "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode Ollama response line: {line}")

        except Exception as e:
            logger.error(f"Ollama streaming error: {str(e)}")
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
            
            # Format tools into prompt for Ollama (which doesn't natively support tools)
            tools_description = "Available tools:\n"
            for tool in tools:
                if tool["type"] == "function":
                    func_info = tool["function"]
                    tools_description += f"- {func_info['name']}: {func_info.get('description', '')}\n"
                    if "parameters" in func_info and "properties" in func_info["parameters"]:
                        tools_description += "  Parameters:\n"
                        for param_name, param_info in func_info["parameters"]["properties"].items():
                            tools_description += f"    - {param_name}: {param_info.get('description', '')}\n"

            # Format messages or prompt
            if isinstance(prompt, list):
                content = self._format_chat_prompt(prompt)
            else:
                content = prompt

            # Create structured prompt
            structured_prompt = f"""
{tools_description}

Respond using the following JSON format if you want to use a tool:
{{
    "tool_calls": [{{
        "name": "tool_name",
        "parameters": {{
            "param1": "value1",
            ...
        }}
    }}]
}}

Or respond normally if no tool is needed.

User query: {content}
Assistant:"""

            # Skip tools if tool_choice is "none"
            if tool_choice == "none":
                structured_prompt = content

            # Call generate with the structured prompt
            response = await self.generate(
                prompt=structured_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Try to parse response as JSON for tool calls
            tool_calls = []
            content = response.content
            try:
                parsed_content = json.loads(response.content)
                if "tool_calls" in parsed_content:
                    tool_calls = []
                    for idx, tool_call in enumerate(parsed_content["tool_calls"]):
                        tool_calls.append({
                            "id": f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["parameters"])
                            }
                        })
                    # If we successfully parsed tool calls, set content to None
                    content = None
            except (json.JSONDecodeError, KeyError, TypeError):
                # Not JSON or doesn't have expected structure, treat as regular response
                pass

            return {
                "content": content,
                "tool_calls": tool_calls,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": 0.0  # Local models have no cost
                }
            }

        except Exception as e:
            logger.error(f"Ollama tool generation error: {str(e)}")
            raise

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages for Ollama."""
        formatted = []
        
        # Group messages by role for better formatting
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                formatted.append(f"<system>\n{content}\n</system>")
            elif role == "user":
                formatted.append(f"<human>\n{content}\n</human>")
            elif role == "assistant":
                formatted.append(f"<assistant>\n{content}\n</assistant>")
            elif role == "tool":
                formatted.append(f"<tool>\n{content}\n</tool>")
        
        return "\n".join(formatted)

    def supports_tools(self) -> bool:
        """Ollama has basic tool support through our implementation."""
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and limitations."""
        return {
            **super().get_capabilities(),
            "max_tokens": 4096,  # Conservative default
            "supports_vision": False,  # Most local models don't support vision yet
            "supports_embeddings": True,
            "cost_per_1k": {model: 0.0 for model in self.models}  # Local models are free
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "total_requests": self.requests_made,
            "estimated_cost": 0.0  # Local models have no cost
        }