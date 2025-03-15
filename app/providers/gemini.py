from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import BaseProvider, ProviderConfig, ProviderResponse, ModelUsage
from app.logger import logger

class GeminiProvider(BaseProvider):
    """Google Gemini API provider implementation."""
    
    def __init__(self, config: Union[Dict[str, Any], ProviderConfig]):
        super().__init__(config)
        
        # Initialize Gemini client
        genai.configure(api_key=self.config.api_key)
        
        # Configure generation settings
        self.generation_config = {
            "candidate_count": 1,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 1,
            "stop_sequences": [],
            "max_output_tokens": None,
            # Required for multimodal Live API
            "multimodal_live": True
        }

        # Set default costs per 1K tokens (based on latest pricing)
        # Input/Output costs
        self.set_model_cost("gemini-2.0-flash", 0.007)      # $0.007/1K input, $0.014/1K output
        self.set_model_cost("gemini-2.0-flash-lite", 0.003) # $0.003/1K input, $0.006/1K output
        self.set_model_cost("gemini-1.5-pro", 0.005)        # $0.005/1K input, $0.010/1K output
        
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
        """Generate a response using Gemini's API."""
        try:
            # Validate and get model
            model = self.validate_model(model)
            
            # Format prompt if it's a message list
            if isinstance(prompt, list):
                formatted_prompt = self._format_chat_prompt(prompt)
            else:
                formatted_prompt = prompt

            # Get the model instance
            model_instance = genai.GenerativeModel(model)

            if stream:
                return self._stream_response(
                    model_instance=model_instance,
                    prompt=formatted_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

            # Generate response
            response = await model_instance.generate_content_async(
                formatted_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens or 8192,
                    **kwargs
                }
            )

            # Track usage
            prompt_tokens = response.usage.prompt_token_count
            completion_tokens = response.usage.completion_token_count
            total_tokens = prompt_tokens + completion_tokens
            
            self.total_tokens_used += total_tokens
            self.requests_made += 1

            usage = ModelUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=self.calculate_cost(
                    ModelUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens
                    ),
                    model
                )
            )

            return ProviderResponse(
                content=response.text,
                model=model,
                usage=usage,
                finish_reason=response.prompt_feedback.stop_reason if hasattr(response, 'prompt_feedback') else None,
                raw_response=response.candidates[0] if hasattr(response, 'candidates') else None
            )

        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            raise

    async def _stream_response(
        self,
        model_instance: Any,
        prompt: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from Gemini API."""
        try:
            response = await model_instance.generate_content_async(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens or 8192,
                    **kwargs
                },
                stream=True
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

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
                formatted_prompt = self._format_chat_prompt(prompt)
            else:
                formatted_prompt = prompt

            # Convert tools to Gemini's function declarations format
            function_declarations = self._convert_tools_to_functions(tools)
            
            # Skip tools if tool_choice is "none"
            if tool_choice == "none":
                function_declarations = []
            
            # Get model instance
            model_instance = genai.GenerativeModel(model)

            response = await model_instance.generate_content_async(
                formatted_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens or 8192,
                    **kwargs
                },
                tools=function_declarations if function_declarations else None
            )

            # Track usage
            prompt_tokens = response.usage.prompt_token_count
            completion_tokens = response.usage.completion_token_count
            total_tokens = prompt_tokens + completion_tokens
            
            self.total_tokens_used += total_tokens
            self.requests_made += 1

            # Extract tool calls if present
            tool_calls = []
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for idx, call in enumerate(response.tool_calls):
                    tool_calls.append({
                        'id': f"call_{idx}",
                        'type': 'function',
                        'function': {
                            'name': call.function_name,
                            'arguments': call.function_parameters
                        }
                    })

            return {
                "content": response.text if not tool_calls else None,
                "tool_calls": tool_calls if tool_calls else None,
                "model": model,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": self.calculate_cost(
                        ModelUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens
                        ),
                        model
                    )
                }
            }

        except Exception as e:
            logger.error(f"Error in generate_with_tools: {str(e)}")
            raise

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages for Gemini."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            elif role == "tool":
                formatted.append(f"Tool Response from {msg.get('name', 'tool')}: {content}")
                
        return "\n".join(formatted)

    def _convert_tools_to_functions(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tools to Gemini function declarations."""
        function_declarations = []
        for tool in tools:
            if tool["type"] == "function":
                function_declarations.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"].get("parameters", {})
                })
        return function_declarations

    def supports_tools(self) -> bool:
        """Gemini supports tool/function calling."""
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and limitations."""
        return {
            **super().get_capabilities(),
            "max_tokens": {
                "gemini-2.0-flash": {
                    "input": 1048576,  # 1M tokens
                    "output": {
                        "normal": 8192,
                        "beta": 128000  # With beta flag
                    }
                },
                "gemini-2.0-flash-lite": {
                    "input": 1048576,  # 1M tokens
                    "output": 8192
                },
                "gemini-1.5-pro": {
                    "input": 1048576,  # 1M tokens
                    "output": 8192
                }
            },
            "supports_vision": True,
            "supports_embeddings": True,
            "supports_audio": True,
            "supports_video": True,
            "supports_image_generation": True,
            "cost_per_1k": self._model_costs,
            "features": {
                "multimodal_live": True,
                "native_tool_use": True,
                "code_execution": True,
                "structured_output": True
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