from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import aiohttp
import json
from .base import BaseProvider, ProviderConfig, ProviderResponse
from app.logger import logger

class OllamaProvider(BaseProvider):
    """Ollama API provider for local model deployment."""
    
    def __init__(self, config: Union[Dict[str, Any], ProviderConfig]):
        super().__init__(config)
        self.session = None

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

    async def generate(
        self,
        prompt: str,
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
                prompt = self.format_prompt(prompt)

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
                prompt_tokens = len(prompt.split())
                completion_tokens = len(data["response"].split())
                
                return ProviderResponse(
                    content=data["response"],
                    model=model,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    },
                    finish_reason="stop"  # Ollama doesn't provide this info
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
                        data = json.loads(line)
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
        prompt: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Ollama doesn't support native function calling - we implement a basic version."""
        try:
            # Format tools into prompt
            tools_description = "Available tools:\n"
            for tool in tools:
                tools_description += f"- {tool['name']}: {tool['description']}\n"
                if "parameters" in tool:
                    tools_description += "  Parameters:\n"
                    for param in tool["parameters"].get("properties", {}).items():
                        tools_description += f"    - {param[0]}: {param[1].get('description', '')}\n"

            # Create a structured prompt
            structured_prompt = f"""
{tools_description}

Respond in the following JSON format if you want to use a tool:
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

User: {prompt}
Assistant:"""

            response = await self.generate(
                prompt=structured_prompt,
                model=model,
                temperature=temperature,
                **kwargs
            )

            # Try to parse response as JSON for tool calls
            try:
                tool_data = json.loads(response.content)
                if "tool_calls" in tool_data:
                    return {
                        "content": None,
                        "tool_calls": tool_data["tool_calls"],
                        "model": response.model,
                        "usage": response.usage
                    }
            except json.JSONDecodeError:
                # Not JSON, treat as regular response
                pass

            return {
                "content": response.content,
                "tool_calls": None,
                "model": response.model,
                "usage": response.usage
            }

        except Exception as e:
            logger.error(f"Ollama tool generation error: {str(e)}")
            raise

    def supports_tools(self) -> bool:
        """Ollama has basic tool support through our implementation."""
        return True
