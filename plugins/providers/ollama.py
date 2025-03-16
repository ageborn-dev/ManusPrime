# plugins/providers/ollama.py
import json
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, ClassVar

from plugins.base import Plugin, PluginCategory, ProviderPlugin
from manusprime.config import config

logger = logging.getLogger("manusprime.plugins.ollama")

class OllamaProvider(ProviderPlugin):
    """Provider plugin for local Ollama models."""
    
    name: ClassVar[str] = "ollama"
    description: ClassVar[str] = "Ollama provider for local AI models"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.PROVIDER
    supported_models: ClassVar[List[str]] = [
        "llama2",
        "llama3",
        "mistral",
        "mixtral",
        "codellama",
        "phi",
        "gemma"
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Ollama provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self.session = None
        self.base_url = "http://localhost:11434/api"
        self.available_models = []
        self.default_model = "llama2"
    
    async def initialize(self) -> bool:
        """Initialize the Ollama client and check available models.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Extract config
            base_url = self.config.get("base_url", self.base_url)
            self.base_url = base_url
            
            # Create session
            self.session = aiohttp.ClientSession()
            
            # Get available models
            async with self.session.get(f"{self.base_url}/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    self.available_models = [model["name"] for model in data.get("models", [])]
                    
                    if not self.available_models:
                        logger.warning("No Ollama models found")
                    else:
                        logger.info(f"Found {len(self.available_models)} Ollama models: {', '.join(self.available_models)}")
                        # Set default model to first available if it exists
                        self.default_model = self.available_models[0]
                else:
                    logger.error(f"Failed to get Ollama models: {response.status}")
                    return False
            
            logger.info("Ollama provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Ollama provider: {e}")
            await self.cleanup()
            return False
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """Generate a response from an Ollama model.
        
        Args:
            prompt: The prompt to send to the model
            model: The specific model to use
            temperature: The sampling temperature
            max_tokens: The maximum number of tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict: The response data
        """
        if not self.session:
            raise ValueError("Ollama provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.available_models else self.default_model
        
        try:
            # Prepare request payload
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    **({"num_predict": max_tokens} if max_tokens else {})
                }
            }
            
            # Make the API call
            async with self.session.post(f"{self.base_url}/generate", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Ollama API error: {error_text}")
                    
                data = await response.json()
            
            # Extract content
            content = data.get("response", "")
            
            # Get token usage (Ollama provides eval_count and eval_duration)
            # These are approximations since Ollama doesn't expose detailed token counts
            eval_count = data.get("eval_count", 0)  # This is completion tokens
            prompt_tokens = len(prompt.split()) // 2  # Very rough estimate
            
            # Return standardized response
            return {
                "content": content,
                "model": model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": eval_count,
                    "total_tokens": prompt_tokens + eval_count,
                    "cost": 0.0  # Local models have no API cost
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            raise
    
    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict:
        """Generate a response from Ollama that may include tool calls.
        
        Ollama doesn't natively support tool calls, so we format the tools
        as part of the prompt and parse the response.
        
        Args:
            prompt: The prompt to send to the model
            tools: The tools available to the model
            model: The specific model to use
            temperature: The sampling temperature
            tool_choice: How to choose tools ("none", "auto", "required")
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict: The response data
        """
        if not self.session:
            raise ValueError("Ollama provider not initialized")
        
        # Validate and select model
        model_name = model if model in self.available_models else self.default_model
        
        # Skip tools if tool_choice is "none"
        if tool_choice == "none":
            return await self.generate(prompt, model, temperature, **kwargs)
        
        try:
            # Format the available tools into the prompt
            tools_prompt = "Available tools:\n"
            for i, tool in enumerate(tools):
                if tool["type"] == "function":
                    function = tool["function"]
                    tools_prompt += f"{i+1}. {function['name']}: {function.get('description', '')}\n"
                    
                    # Add parameters if available
                    if "parameters" in function and "properties" in function["parameters"]:
                        tools_prompt += "   Parameters:\n"
                        for param_name, param_info in function["parameters"]["properties"].items():
                            tools_prompt += f"     - {param_name}: {param_info.get('description', '')}\n"
            
            # Add instructions for response format
            tools_prompt += "\nTo use a tool, respond in the following JSON format:\n"
            tools_prompt += "```json\n{\"tool\": \"tool_name\", \"parameters\": {\"param1\": \"value1\", ...}}\n```\n"
            tools_prompt += "If you don't need to use a tool, respond normally."
            
            # Combine with original prompt
            enhanced_prompt = f"{prompt}\n\n{tools_prompt}"
            
            # Generate response
            response = await self.generate(
                prompt=enhanced_prompt,
                model=model_name,
                temperature=temperature,
                **kwargs
            )
            
            content = response["content"]
            
            # Try to parse tool calls from the response
            tool_calls = []
            try:
                # Look for JSON blocks
                json_pattern_start = content.find("```json")
                if json_pattern_start == -1:
                    json_pattern_start = content.find("{\"tool\":")
                
                if json_pattern_start != -1:
                    # Find the JSON content
                    json_start = content.find("{", json_pattern_start)
                    json_end = content.find("}", json_start) + 1
                    json_str = content[json_start:json_end]
                    
                    # Parse JSON
                    tool_data = json.loads(json_str)
                    if "tool" in tool_data and "parameters" in tool_data:
                        tool_calls.append({
                            'id': f"call_0",
                            'type': 'function',
                            'function': {
                                'name': tool_data["tool"],
                                'arguments': json.dumps(tool_data["parameters"])
                            }
                        })
                        # Remove tool call from content
                        content = None
            except Exception as e:
                logger.warning(f"Failed to parse tool calls from Ollama response: {e}")
            
            # Update response with parsed tool calls
            response["content"] = content
            response["tool_calls"] = tool_calls
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response with tools from Ollama: {e}")
            raise
    
    async def execute(self, **kwargs) -> Dict:
        """Execute the provider's primary function (generate).
        
        Args:
            **kwargs: Arguments for the generate method
            
        Returns:
            Dict: The generation result
        """
        if "tools" in kwargs:
            return await self.generate_with_tools(**kwargs)
        else:
            return await self.generate(**kwargs)
    
    def get_default_model(self) -> str:
        """Get the default model for this provider.
        
        Returns:
            str: The name of the default model
        """
        return self.default_model
    
    def get_model_cost(self, model: str) -> float:
        """Get the cost per 1K tokens for a specific model.
        
        Since Ollama uses local models, the cost is 0.
        
        Args:
            model: The model name
            
        Returns:
            float: The cost per 1K tokens (always 0)
        """
        return 0.0
    
    async def cleanup(self) -> None:
        """Clean up resources used by the provider."""
        if self.session:
            await self.session.close()
            self.session = None
