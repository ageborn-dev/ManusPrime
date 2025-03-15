from typing import Dict, List, Literal, Optional, Union, Any, AsyncGenerator
from openai import AsyncOpenAI

# Optional provider imports with fallbacks
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    from mistralai.client import MistralAsyncClient
except ImportError:
    MistralAsyncClient = None

try:
    from google import generativeai as genai
except ImportError:
    genai = None

from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config import config
from app.logger import logger
from app.schema import Message, ToolCall
from app.utils.monitor import resource_monitor


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class ProviderNotAvailableError(ProviderError):
    """Exception raised when a provider is not available."""
    pass


class ModelNotAvailableError(ProviderError):
    """Exception raised when a model is not available."""
    pass


class LLM:
    """Main LLM interface that handles multiple providers."""
    
    _instances: Dict[str, "LLM"] = {}

    def __new__(cls, config_name: str = "default", llm_config: Optional[Dict[str, Any]] = None) -> "LLM":
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(self, config_name: str = "default", llm_config: Optional[Dict[str, Any]] = None) -> None:
        if not hasattr(self, "clients"):  # Only initialize if not already initialized
            self.provider = config.providers.providers.get(
                config_name, 
                config.providers.providers[config.providers.default_provider]
            )
            self.model = None  # Will be set per request
            self.temperature = 0.7
            
            # Initialize provider-specific clients
            self.clients = {}
            self._initialize_clients()
            
            # Track usage
            self.model_usage = {}

    def _initialize_clients(self) -> None:
        """Initialize provider-specific clients."""
        if self.provider.type != "cloud":
            return

        base_url = self.provider.base_url

        # Initialize OpenAI-compatible client (OpenAI, DeepSeek)
        if "openai" in base_url or "deepseek" in base_url:
            self.clients["openai"] = AsyncOpenAI(
                api_key=self.provider.api_key,
                base_url=self.provider.base_url,
                timeout=self.provider.timeout
            )
        
        # Initialize Anthropic client
        if "anthropic" in base_url:
            if not AsyncAnthropic:
                raise ProviderNotAvailableError("Anthropic client not available. Install with 'pip install anthropic'")
            
            headers = {}
            if self.provider.capabilities.beta_features:
                headers["output-128k-2025-02-19"] = "true"
            
            self.clients["anthropic"] = AsyncAnthropic(
                api_key=self.provider.api_key,
                base_url=self.provider.base_url,
                timeout=self.provider.timeout,
                default_headers=headers
            )
        
        # Initialize Mistral client
        if "mistral" in base_url:
            if not MistralAsyncClient:
                raise ProviderNotAvailableError("Mistral client not available. Install with 'pip install mistralai'")
            
            self.clients["mistral"] = MistralAsyncClient(
                api_key=self.provider.api_key,
                endpoint=self.provider.base_url
            )
        
        # Initialize Gemini client
        if "googleapis" in base_url:
            if not genai:
                raise ProviderNotAvailableError("Gemini client not available. Install with 'pip install google-generativeai'")
            
            genai.configure(api_key=self.provider.api_key)
            self.clients["gemini"] = genai

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """Format messages for LLM by converting them to OpenAI message format."""
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages
        for msg in formatted_messages:
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError("Message must contain either 'content' or 'tool_calls'")

        return formatted_messages

    def _get_token_limits(self, model: str) -> Dict[str, int]:
        """Get input/output token limits for a model."""
        if "claude" in model:
            if model == "claude-3.7-sonnet" and self.provider.capabilities.extended_thinking:
                return {
                    "input": 200000,  # 200K context
                    "output": 128000 if "output-128k-2025-02-19" in self.provider.capabilities.beta_features else 64000
                }
            return {"input": 200000, "output": 8192}  # Standard limits
        elif "gemini" in model:
            return {
                "input": 1048576,  # 1M tokens
                "output": 128000 if self.provider.capabilities.beta_features else 8192
            }
        elif any(x in model for x in ["mistral", "pixtral", "codestral"]):
            if "codestral" in model:
                return {"input": 262144, "output": 8192}  # 256K context
            return {"input": 131072, "output": 8192}  # 131K context
        elif "deepseek" in model:
            return {"input": 64000, "output": 8000}  # 64K context
        
        return {"input": 4096, "output": 4096}  # Default fallback

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int, cache_hit: bool = False) -> float:
        """Calculate cost based on token usage and model pricing."""
        costs = config.monitoring.costs
        
        input_cost = costs.get_cost(model, "input", cache_hit) * (prompt_tokens / 1000)
        output_cost = costs.get_cost(model, "output") * (completion_tokens / 1000)
        
        if self.provider.time_based_pricing and self.provider.is_discount_period():
            if "deepseek" in model:
                discount = 0.5 if "chat" in model else 0.25
                input_cost *= discount
                output_cost *= discount
        
        return input_cost + output_cost

    def validate_model(self, model: Optional[str] = None) -> str:
        """Validate and return the model name."""
        model_name = model or self.provider.models[0]
        if model_name not in self.provider.models:
            raise ModelNotAvailableError(
                f"Model {model_name} not available for provider {self.provider.__class__.__name__}"
            )
        return model_name

    def get_client(self, provider_type: str) -> Any:
        """Get provider client with validation."""
        client = self.clients.get(provider_type)
        if not client:
            raise ProviderNotAvailableError(f"{provider_type} client not initialized")
        return client

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        extended_thinking: bool = False,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Send a prompt to the LLM and get the response."""
        try:
            resource_monitor.start_timer("llm_call")
            
            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Validate model and get limits
            model_name = self.validate_model(model)
            limits = self._get_token_limits(model_name)
            max_tokens = limits["output"]
            
            # Track usage
            self.model_usage[model_name] = self.model_usage.get(model_name, 0) + 1
            logger.info(f"Using model: {model_name}")

            # Get appropriate response using implementations
            if "anthropic" in self.provider.base_url:
                response = await _ask_anthropic(
                    self, messages, model_name, temperature, max_tokens, stream, extended_thinking
                )
            elif "mistral" in self.provider.base_url:
                response = await _ask_mistral(
                    self, messages, model_name, temperature, max_tokens, stream
                )
            elif "googleapis" in self.provider.base_url:
                response = await _ask_gemini(
                    self, messages, model_name, temperature, max_tokens, stream
                )
            elif "deepseek" in self.provider.base_url:
                response = await _ask_deepseek(
                    self, messages, model_name, temperature, max_tokens, stream
                )
            else:  # OpenAI-compatible API
                response = await _ask_openai(
                    self, messages, model_name, temperature, max_tokens, stream
                )

            # Handle usage tracking
            if not stream:
                if hasattr(response, 'usage'):
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                else:
                    prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
                    completion_tokens = len(str(response.choices[0].message.content).split())

                cost = self._calculate_cost(
                    model_name,
                    prompt_tokens,
                    completion_tokens,
                    cache_hit=getattr(response, 'cache_hit', False)
                )

                resource_monitor.track_api_call(
                    model=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=cost,
                    success=True
                )

                elapsed = resource_monitor.end_timer("llm_call")
                if elapsed:
                    logger.debug(f"LLM call completed in {elapsed:.2f} seconds")

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty response from LLM")
                return response.choices[0].message.content

            return response  # Return AsyncGenerator for streaming

        except Exception as e:
            logger.error(f"Error in ask: {str(e)}")
            resource_monitor.end_timer("llm_call")
            resource_monitor.track_api_call(model=model or self.provider.models[0], success=False)
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        tools: List[dict],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        extended_thinking: bool = False,
    ) -> Any:
        """Send a prompt to the LLM and get a response that may include tool calls."""
        try:
            resource_monitor.start_timer("llm_tool_call")
            
            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Validate model and get limits
            model_name = self.validate_model(model)
            limits = self._get_token_limits(model_name)
            max_tokens = limits["output"]
            
            # Track usage
            self.model_usage[model_name] = self.model_usage.get(model_name, 0) + 1
            logger.info(f"Using model with tools: {model_name}")

            # Get appropriate response using provider-specific implementation
            if "anthropic" in self.provider.base_url:
                response = await _ask_anthropic_tool(
                    self, messages, tools, model_name, temperature, max_tokens, tool_choice, extended_thinking
                )
            elif "mistral" in self.provider.base_url:
                response = await _ask_mistral_tool(
                    self, messages, tools, model_name, temperature, max_tokens, tool_choice
                )
            elif "googleapis" in self.provider.base_url:
                response = await _ask_gemini_tool(
                    self, messages, tools, model_name, temperature, max_tokens, tool_choice
                )
            elif "deepseek" in self.provider.base_url:
                response = await _ask_deepseek_tool(
                    self, messages, tools, model_name, temperature, max_tokens, tool_choice
                )
            else:  # OpenAI-compatible API
                response = await _ask_openai_tool(
                    self, messages, tools, model_name, temperature, max_tokens, tool_choice
                )

            # Handle usage tracking
            if hasattr(response, 'usage'):
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            else:
                prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
                completion_tokens = len(str(getattr(response, 'content', '')).split())

            cost = self._calculate_cost(
                model_name,
                prompt_tokens,
                completion_tokens,
                cache_hit=getattr(response, 'cache_hit', False)
            )

            resource_monitor.track_api_call(
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                success=True
            )

            elapsed = resource_monitor.end_timer("llm_tool_call")
            if elapsed:
                logger.debug(f"LLM tool call completed in {elapsed:.2f} seconds")

            # Standardize and format tool calls
            tool_calls = []
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for call in response.tool_calls:
                    # Handle different formats from different providers
                    if isinstance(call, dict):
                        # Convert to ToolCall format
                        tool_calls.append(ToolCall(
                            id=call.get('id', f"call_{len(tool_calls)}"),
                            type="function",
                            function={"name": call.get('name', ''), "arguments": call.get('parameters', '')}
                        ))
                    else:
                        # Assume it's already in the right format or close to it
                        tool_calls.append(call)

            # Create standardized response object
            result = type('ToolResponse', (), {
                'content': getattr(response, 'content', None),
                'tool_calls': tool_calls,
                'model': model_name,
                'choices': getattr(response, 'choices', [])
            })

            return result

        except Exception as e:
            logger.error(f"Error in ask_tool: {str(e)}")
            resource_monitor.end_timer("llm_tool_call")
            resource_monitor.track_api_call(model=model or self.provider.models[0], success=False)
            raise


# Provider-specific implementations

async def _ask_openai(llm, messages, model, temperature, max_tokens, stream):
    """Implementation for OpenAI API call."""
    client = llm.get_client("openai")
    
    temp = temperature if temperature is not None else llm.temperature
    
    if stream:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
            stream=True
        )
        
        async def stream_generator():
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        return stream_generator()
    
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        stream=False
    )
    
    return response


async def _ask_anthropic(llm, messages, model, temperature, max_tokens, stream, extended_thinking=False):
    """Implementation for Anthropic API call."""
    client = llm.get_client("anthropic")
    
    temp = temperature if temperature is not None else llm.temperature
    
    # Format messages for Anthropic
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "system":
            formatted_messages.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user":
            formatted_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            formatted_messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "tool":
            # Add tool responses as assistant messages with a tool prefix
            formatted_messages.append({
                "role": "assistant",
                "content": f"Tool Response: {msg['content']}"
            })
    
    if stream:
        response = await client.messages.create(
            model=model,
            messages=formatted_messages,
            temperature=temp,
            max_tokens=max_tokens,
            stream=True
        )
        
        async def stream_generator():
            async for chunk in response:
                if chunk.content and chunk.content[0].text:
                    yield chunk.content[0].text
        
        return stream_generator()
    
    kwargs = {}
    if extended_thinking and model == "claude-3.7-sonnet":
        kwargs["system"] = "Take a deep breath and work on this step-by-step."
    
    response = await client.messages.create(
        model=model,
        messages=formatted_messages,
        temperature=temp,
        max_tokens=max_tokens,
        stream=False,
        **kwargs
    )
    
    # Convert to OpenAI-like format for consistency
    openai_format = type('AnthropicResponse', (), {
        'choices': [type('Choice', (), {
            'message': type('Message', (), {
                'content': response.content[0].text
            }),
            'finish_reason': response.stop_reason
        })],
        'usage': type('Usage', (), {
            'prompt_tokens': response.usage.input_tokens,
            'completion_tokens': response.usage.output_tokens,
            'total_tokens': response.usage.input_tokens + response.usage.output_tokens
        })
    })
    
    return openai_format


async def _ask_mistral(llm, messages, model, temperature, max_tokens, stream):
    """Implementation for Mistral API call."""
    client = llm.get_client("mistral")
    
    temp = temperature if temperature is not None else llm.temperature
    
    # Format messages for Mistral
    formatted_messages = []
    system_content = None
    
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] in ["user", "assistant"]:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        elif msg["role"] == "tool":
            # Add tool responses in a format Mistral understands
            formatted_messages.append({
                "role": "assistant",
                "content": f"Tool Response: {msg['content']}"
            })
    
    # Prepend system message to first user message if present
    if system_content and formatted_messages and formatted_messages[0]["role"] == "user":
        formatted_messages[0]["content"] = f"{system_content}\n\n{formatted_messages[0]['content']}"
    elif system_content:
        formatted_messages.insert(0, {"role": "user", "content": system_content})
    
    if stream:
        response = await client.chat(
            model=model,
            messages=formatted_messages,
            temperature=temp,
            max_tokens=max_tokens,
            stream=True
        )
        
        async def stream_generator():
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        return stream_generator()
    
    response = await client.chat(
        model=model,
        messages=formatted_messages,
        temperature=temp,
        max_tokens=max_tokens,
        stream=False
    )
    
    return response


async def _ask_gemini(llm, messages, model, temperature, max_tokens, stream):
    """Implementation for Google Gemini API call."""
    gemini = llm.get_client("gemini")
    
    temp = temperature if temperature is not None else llm.temperature
    
    # Format chat messages for Gemini
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
            formatted.append(f"Tool Response: {content}")
                
    prompt = "\n".join(formatted)
    
    # Get the model instance
    model_instance = gemini.GenerativeModel(model)
    
    if stream:
        response = await model_instance.generate_content_async(
            prompt,
            generation_config={
                "temperature": temp,
                "max_output_tokens": max_tokens,
            },
            stream=True
        )
        
        async def stream_generator():
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        
        return stream_generator()
    
    response = await model_instance.generate_content_async(
        prompt,
        generation_config={
            "temperature": temp,
            "max_output_tokens": max_tokens,
        }
    )
    
    # Convert to OpenAI-like format for consistency
    openai_format = type('GeminiResponse', (), {
        'choices': [type('Choice', (), {
            'message': type('Message', (), {
                'content': response.text
            }),
            'finish_reason': getattr(response.prompt_feedback, 'stop_reason', None)
        })],
        'usage': type('Usage', (), {
            'prompt_tokens': response.usage.prompt_token_count,
            'completion_tokens': response.usage.completion_token_count,
            'total_tokens': response.usage.prompt_token_count + response.usage.completion_token_count
        })
    })
    
    return openai_format


async def _ask_deepseek(llm, messages, model, temperature, max_tokens, stream):
    """Implementation for DeepSeek API call."""
    client = llm.get_client("openai")  # DeepSeek uses OpenAI-compatible API
    
    temp = temperature if temperature is not None else llm.temperature
    
    # Format messages for DeepSeek
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "system":
            formatted_messages.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user":
            formatted_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            formatted_messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "tool":
            # Add tool responses as assistant messages
            formatted_messages.append({
                "role": "assistant",
                "content": f"Tool Response: {msg['content']}"
            })
    
    if stream:
        response = await client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            temperature=temp,
            max_tokens=max_tokens,
            stream=True,
            context_caching=True  # DeepSeek specific feature
        )
        
        async def stream_generator():
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        return stream_generator()
    
    params = {
        "model": model,
        "messages": formatted_messages,
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False,
        "context_caching": True  # DeepSeek specific feature
    }
    
    # Add chain-of-thought tokens if using the reasoner model
    if model == "deepseek-reasoner":
        params["cot_tokens"] = 32000
    
    response = await client.chat.completions.create(**params)
    
    return response


# Tool call implementations

async def _ask_openai_tool(llm, messages, tools, model, temperature, max_tokens, tool_choice):
    """Implementation for OpenAI API tool call."""
    client = llm.get_client("openai")
    
    temp = temperature if temperature is not None else llm.temperature
    
    # Map tool_choice to OpenAI format
    openai_tool_choice = "auto"
    if tool_choice == "required":
        openai_tool_choice = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}
    elif tool_choice == "none":
        openai_tool_choice = "none"
    
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=openai_tool_choice
    )
    
    return response


async def _ask_anthropic_tool(llm, messages, tools, model, temperature, max_tokens, tool_choice, extended_thinking=False):
    """Implementation for Anthropic API tool call."""
    client = llm.get_client("anthropic")
    
    temp = temperature if temperature is not None else llm.temperature
    
    # Format messages for Anthropic
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "system":
            formatted_messages.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user":
            formatted_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            if "tool_calls" in msg:
                # Convert OpenAI tool_calls format to Anthropic
                assistant_content = msg.get("content", "")
                formatted_messages.append({"role": "assistant", "content": assistant_content})
            else:
                formatted_messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "tool":
            # Add tool responses as assistant messages with a tool prefix
            formatted_messages.append({
                "role": "user",
                "content": f"Tool Response from {msg.get('name', 'tool')}: {msg['content']}"
            })
    
    # Convert tools to Anthropic's format
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
    
    kwargs = {}
    if extended_thinking and model == "claude-3.7-sonnet":
        kwargs["system"] = "Take a deep breath and work on this step-by-step."
    
    response = await client.messages.create(
        model=model,
        messages=formatted_messages,
        temperature=temp,
        max_tokens=max_tokens,
        tools=anthropic_tools,
        **kwargs
    )
    
    # Convert to standardized format
    content = response.content[0].text if response.content and response.content[0].type == "text" else None
    
    # Handle Anthropic tool_calls
    tool_calls = []
    if response.tool_calls:
        for idx, call in enumerate(response.tool_calls):
            tool_calls.append({
                'id': f"call_{idx}",
                'type': 'function',
                'function': {
                    'name': call.function.name,
                    'arguments': call.function.arguments
                }
            })
    
    # Create standardized response object
    result = type('AnthropicToolResponse', (), {
        'content': content,
        'tool_calls': tool_calls,
        'usage': type('Usage', (), {
            'prompt_tokens': response.usage.input_tokens,
            'completion_tokens': response.usage.output_tokens,
            'total_tokens': response.usage.input_tokens + response.usage.output_tokens
        })
    })
    
    return result


async def _ask_mistral_tool(llm, messages, tools, model, temperature, max_tokens, tool_choice):
    """Implementation for Mistral API tool call."""
    client = llm.get_client("mistral")
    
    temp = temperature if temperature is not None else llm.temperature
    
    # Format messages for Mistral
    formatted_messages = []
    system_content = None
    
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] in ["user", "assistant"]:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                # Skip tool_calls for now as Mistral handles them differently
                if msg.get("content"):
                    formatted_messages.append({"role": "assistant", "content": msg["content"]})
            else:
                formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        elif msg["role"] == "tool":
            # Add tool responses in a format Mistral understands
            formatted_messages.append({
                "role": "user",
                "content": f"Tool Response: {msg['content']}"
            })
    
# Prepend system message to first user message if present
    if system_content and formatted_messages and formatted_messages[0]["role"] == "user":
        formatted_messages[0]["content"] = f"{system_content}\n\n{formatted_messages[0]['content']}"
    elif system_content:
        formatted_messages.insert(0, {"role": "user", "content": system_content})
    
    # Convert tools to Mistral's format
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
    
    # Handle tool_choice
    mistral_kwargs = {}
    if tool_choice == "required" and mistral_tools:
        mistral_kwargs["tool_choice"] = "any"  # Mistral doesn't have exact equivalent to "required"
    elif tool_choice == "none":
        mistral_tools = []  # Simply don't include tools
    
    response = await client.chat(
        model=model,
        messages=formatted_messages,
        temperature=temp,
        max_tokens=max_tokens,
        tools=mistral_tools if mistral_tools else None,
        **mistral_kwargs
    )
    
    # Convert to standardized format
    content = response.choices[0].message.content if hasattr(response.choices[0].message, 'content') else None
    
    # Handle Mistral tool_calls
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
    
    # Create standardized response object
    result = type('MistralToolResponse', (), {
        'content': content,
        'tool_calls': tool_calls,
        'usage': response.usage
    })
    
    return result


async def _ask_gemini_tool(llm, messages, tools, model, temperature, max_tokens, tool_choice):
    """Implementation for Google Gemini API tool call."""
    gemini = llm.get_client("gemini")
    
    temp = temperature if temperature is not None else llm.temperature
    
    # Format chat messages for Gemini
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
            formatted.append(f"Tool Response: {content}")
                
    prompt = "\n".join(formatted)
    
    # Convert tools to Gemini's function declarations format
    function_declarations = []
    for tool in tools:
        if tool["type"] == "function":
            function_declarations.append({
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "parameters": tool["function"].get("parameters", {})
            })
    
    # Skip tools if tool_choice is "none"
    if tool_choice == "none":
        function_declarations = []
    
    # Get model instance
    model_instance = gemini.GenerativeModel(model)
    
    response = await model_instance.generate_content_async(
        prompt,
        generation_config={
            "temperature": temp,
            "max_output_tokens": max_tokens,
        },
        tools=function_declarations if function_declarations else None
    )
    
    # Handle Gemini tool calls
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
    
    # Create standardized response object
    result = type('GeminiToolResponse', (), {
        'content': response.text if not tool_calls else None,
        'tool_calls': tool_calls,
        'usage': type('Usage', (), {
            'prompt_tokens': response.usage.prompt_token_count,
            'completion_tokens': response.usage.completion_token_count,
            'total_tokens': response.usage.prompt_token_count + response.usage.completion_token_count
        })
    })
    
    return result


async def _ask_deepseek_tool(llm, messages, tools, model, temperature, max_tokens, tool_choice):
    """Implementation for DeepSeek API tool call."""
    client = llm.get_client("openai")  # DeepSeek uses OpenAI-compatible API
    
    temp = temperature if temperature is not None else llm.temperature
    
    # Format messages for DeepSeek
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "system":
            formatted_messages.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user":
            formatted_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            if "tool_calls" in msg:
                # Skip tool_calls for now as we'll handle them separately
                if msg.get("content"):
                    formatted_messages.append({"role": "assistant", "content": msg["content"]})
            else:
                formatted_messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "tool":
            # Add tool responses as assistant messages
            formatted_messages.append({
                "role": "assistant",
                "content": f"Tool Response: {msg['content']}"
            })
    
    # Map tool_choice to DeepSeek format (which uses OpenAI format)
    ds_tool_choice = "auto"
    if tool_choice == "required":
        ds_tool_choice = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}
    elif tool_choice == "none":
        ds_tool_choice = "none"
    
    params = {
        "model": model,
        "messages": formatted_messages,
        "temperature": temp,
        "max_tokens": max_tokens,
        "tools": tools,
        "tool_choice": ds_tool_choice,
        "context_caching": True  # DeepSeek specific feature
    }
    
    # Add chain-of-thought tokens if using the reasoner model
    if model == "deepseek-reasoner":
        params["cot_tokens"] = 32000
    
    response = await client.chat.completions.create(**params)
    
    return response