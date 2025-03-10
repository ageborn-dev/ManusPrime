from typing import Dict, List, Literal, Optional, Union

from openai import (
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config import LLMSettings, config
from app.logger import logger
from app.schema import Message
from app.utils.monitor import resource_monitor


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            provider_config = llm_config.get(config_name, llm_config["default"])
            self.model = provider_config.model
            self.max_tokens = provider_config.max_tokens
            self.temperature = provider_config.temperature
            self.api_key = provider_config.api_key
            self.base_url = provider_config.base_url
            
            # Initialize OpenAI client
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            
            # Used for multi-model tracking
            self.model_usage = {}

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """Format messages for LLM by converting them to OpenAI message format."""
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                # If message is already a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                # If message is a Message object, convert it to dict
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "Message must contain either 'content' or 'tool_calls'"
                )

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response
            model (str): Optional model override (gpt-4o, gpt-4o-mini, deepseek-chat, deepseek-r1)

        Returns:
            str: The generated response
        """
        try:
            # Start timer for API call
            resource_monitor.start_timer("llm_call")
            
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Use specified model or default
            model_name = model or self.model
            
            # Track model usage
            self.model_usage[model_name] = self.model_usage.get(model_name, 0) + 1
            
            logger.info(f"Using model: {model_name}")

            if not stream:
                # Non-streaming request
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature or self.temperature,
                    stream=False,
                )
                
                # Track API usage
                resource_monitor.track_api_call(
                    model=model_name,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    success=True
                )
                
                # End timer
                elapsed = resource_monitor.end_timer("llm_call")
                if elapsed:
                    logger.debug(f"LLM call completed in {elapsed:.2f} seconds")
                
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")
                return response.choices[0].message.content

            # Streaming request
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature or self.temperature,
                stream=True,
            )
            
            # For streaming, we need to estimate token counts
            # A simple approximation: count tokens based on whitespace-split
            prompt_tokens = sum(len(m.get("content", "").split()) for m in messages if m.get("content"))
            
            collected_messages = []
            completion_tokens = 0
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                # Count completion tokens as we receive chunks
                completion_tokens += len(chunk_message.split())
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            
            # Track API usage for streaming request
            resource_monitor.track_api_call(
                model=model_name,
                # Use approximated token counts
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                success=True
            )
            
            # End timer
            elapsed = resource_monitor.end_timer("llm_call")
            if elapsed:
                logger.debug(f"LLM streaming call completed in {elapsed:.2f} seconds")
            
            if not full_response:
                raise ValueError("Empty response from streaming LLM")
            return full_response

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            # End timer and track error
            resource_monitor.end_timer("llm_call")
            resource_monitor.track_api_call(model=model or self.model, success=False)
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            # End timer and track error
            resource_monitor.end_timer("llm_call")
            resource_monitor.track_api_call(model=model or self.model, success=False)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            # End timer and track error
            resource_monitor.end_timer("llm_call")
            resource_monitor.track_api_call(model=model or self.model, success=False)
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 60,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            model: Optional model override (gpt-4o, gpt-4o-mini, deepseek-chat, deepseek-r1)
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response
        """
        try:
            # Start timer for API call
            resource_monitor.start_timer("llm_tool_call")
            
            # Validate tool_choice
            if tool_choice not in ["none", "auto", "required"]:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")
                        
            # Use specified model or default
            model_name = model or self.model
            
            # Track model usage
            self.model_usage[model_name] = self.model_usage.get(model_name, 0) + 1
            
            logger.info(f"Using model: {model_name}")

            # Set up the completion request
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                **kwargs,
            )
            
            # Track API usage
            if hasattr(response, 'usage'):
                resource_monitor.track_api_call(
                    model=model_name,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    success=True
                )
            else:
                # Estimate token usage if not provided
                prompt_tokens = sum(len(m.get("content", "").split()) for m in messages if m.get("content"))
                completion_tokens = len(response.choices[0].message.content.split()) if response.choices[0].message.content else 0
                resource_monitor.track_api_call(
                    model=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    success=True
                )
            
            # End timer
            elapsed = resource_monitor.end_timer("llm_tool_call")
            if elapsed:
                logger.debug(f"LLM tool call completed in {elapsed:.2f} seconds")

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                print(response)
                raise ValueError("Invalid or empty response from LLM")

            return response.choices[0].message

        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            # End timer and track error
            resource_monitor.end_timer("llm_tool_call")
            resource_monitor.track_api_call(model=model or self.model, success=False)
            raise
        except OpenAIError as oe:
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            # End timer and track error
            resource_monitor.end_timer("llm_tool_call")
            resource_monitor.track_api_call(model=model or self.model, success=False)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            # End timer and track error
            resource_monitor.end_timer("llm_tool_call")
            resource_monitor.track_api_call(model=model or self.model, success=False)
            raise
