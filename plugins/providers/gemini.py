# plugins/providers/gemini.py
import os
from typing import Dict, List, Optional, Any, Union
import asyncio
import base64
from datetime import datetime
import mimetypes
import json

# Updated imports for the new Gemini SDK
from google import genai
from plugins.base import BaseProvider, PluginCategory
from config import config as app_config  # Renamed to avoid conflicts
from utils.logger import logger
from utils.monitor import resource_monitor

class GeminiProvider(BaseProvider):
    """Provider plugin for Google's Gemini AI models."""
    
    name = "gemini"
    description = "Google Gemini AI provider with support for multimodal models"
    category = PluginCategory.PROVIDER
    version = "1.0.0"
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = os.getenv("GEMINI_API_KEY") or config.get("api_key")
        self.base_url = config.get("base_url", "https://generativelanguage.googleapis.com/v1")
        self.models = config.get("models", [])
        self.model_configs = config.get("model_configs", {})
        self.initialized = False
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
    async def initialize(self) -> bool:
        """Initialize the provider."""
        if self.initialized:
            return True
            
        try:
            # Validate API key
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
                
            # Test connection
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-2.0-flash-lite",
                contents="Test connection"
            )
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini provider: {e}")
            return False
            
    def _get_model_config(self, model: str) -> Dict:
        """Get configuration for specific model."""
        return self.model_configs.get(model, {})
        
    def _encode_image(self, image_path: str) -> Dict:
        """Encode image file to base64."""
        with open(image_path, "rb") as img_file:
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "application/octet-stream"
            return {
                "mime_type": mime_type,
                "data": base64.b64encode(img_file.read()).decode()
            }
            
    async def _run_with_retry(self, func, *args, max_retries: int = 3, **kwargs):
        """Execute function with retries."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        raise last_error
        
    def _prepare_function_declarations(self, tools: List[Dict]) -> List[Dict]:
        """Convert tool definitions to Gemini function declarations."""
        declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                declarations.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func["parameters"]
                })
        return declarations
        
    async def generate(self,
                      prompt: str,
                      model: str = "gemini-2.0-flash-lite",
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None,
                      **kwargs) -> Dict:
        """Generate content using specified model."""
        start_time = datetime.now()
        
        try:
            # Get model configuration
            model_config = self._get_model_config(model)
            
            # Create generation config
            config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens or model_config.get("output_tokens", 8192),
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 40),
            }
            
            # Create content
            content = kwargs.get("content", prompt)
            
            # Handle multimodal input if provided
            if "images" in kwargs and model_config.get("multimodal"):
                contents = [content]
                for image_path in kwargs["images"]:
                    # For the new SDK, we'll need to open the image file instead of encoding
                    contents.append(open(image_path, "rb"))
                
                response = await self._run_with_retry(
                    self.client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config
                )
            else:
                response = await self._run_with_retry(
                    self.client.models.generate_content,
                    model=model,
                    contents=content,
                    config=config
                )
            
            # Extract usage information from response
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0)
            }
            
            # Calculate cost
            cost_per_1k = app_config.get_value(f"costs.{model}", 0.0)
            cost = (usage["total_tokens"] / 1000) * cost_per_1k
            
            return {
                "success": True,
                "content": response.text,
                "usage": {
                    **usage,
                    "cost": cost
                },
                "model": model,
                "finish_reason": getattr(response.candidates[0] if response.candidates else None, "finish_reason", ""),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error generating content with Gemini: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
    async def generate_with_tools(self,
                                prompt: str,
                                tools: List[Dict],
                                model: str = "gemini-2.0-flash",
                                temperature: float = 0.7,
                                tool_choice: str = "auto",
                                **kwargs) -> Dict:
        """Generate content with function calling capability."""
        start_time = datetime.now()
        
        try:
            # Verify model supports function calling
            model_config = self._get_model_config(model)
            if not model_config.get("function_calling"):
                raise ValueError(f"Model {model} does not support function calling")
                
            # Create generation config with tools
            config = {
                "temperature": temperature,
                "max_output_tokens": kwargs.get("max_tokens", model_config.get("output_tokens", 8192)),
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 40),
                "tools": self._prepare_function_declarations(tools)
            }
            
            # Set tool choice settings
            if tool_choice == "none":
                config["automatic_function_calling"] = {"disable": True}
            elif tool_choice == "required":
                config["automatic_function_calling"] = {"required": True}
            
            # Generate response
            response = await self._run_with_retry(
                self.client.models.generate_content,
                model=model,
                contents=prompt,
                config=config
            )
            
            # Extract tool calls if any
            tool_calls = []
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            tool_calls.append({
                                "type": "function",
                                "function": {
                                    "name": part.function_call.name,
                                    "arguments": json.loads(part.function_call.args)
                                }
                            })
            
            # Calculate usage and cost
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0)
            }
            
            cost_per_1k = app_config.get_value(f"costs.{model}", 0.0)
            cost = (usage["total_tokens"] / 1000) * cost_per_1k
            
            return {
                "success": True,
                "content": response.text if not tool_calls else None,
                "tool_calls": tool_calls,
                "usage": {
                    **usage,
                    "cost": cost
                },
                "model": model,
                "finish_reason": getattr(response.candidates[0] if response.candidates else None, "finish_reason", ""),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error generating content with tools: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
    async def cleanup(self):
        """Clean up resources."""
        self.initialized = False
