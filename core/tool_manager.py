import logging
from typing import Dict, List
import json

from plugins.registry import registry
from plugins.base import PluginCategory

logger = logging.getLogger("manusprime.core.tool_manager")

class ToolManager:
    """Manages tool preparation and execution."""
    
    def __init__(self):
        """Initialize ToolManager."""
        pass
    
    def prepare_tools(self, prompt: str) -> List[Dict]:
        """Prepare tool schemas based on prompt.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            List[Dict]: List of tool schemas
        """
        logger.debug(f"Preparing tools for prompt: {prompt[:100]}...")
        tools = []
        try:
            # Check for sandbox-related keywords
            sandbox_keywords = ['simulation', 'interactive', 'web app', 'application', 'browser', 'visualization']
            is_sandbox_task = any(word in prompt.lower() for word in sandbox_keywords)
            
            # Add sandbox tool if relevant
            if is_sandbox_task:
                logger.debug("Sandbox-related keywords found in prompt")
                sandbox_plugin = registry.get_plugin("selenium_sandbox")
                if sandbox_plugin:
                    logger.debug("Adding selenium_sandbox tool")
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": "selenium_sandbox",
                            "description": "Executes and renders HTML, CSS, and JavaScript code in a browser sandbox",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "code": {
                                        "type": "string",
                                        "description": "The code to execute (HTML, JavaScript, etc.)"
                                    },
                                    "code_type": {
                                        "type": "string",
                                        "enum": ["html", "javascript", "react"],
                                        "default": "html",
                                        "description": "The type of code to execute"
                                    },
                                    "width": {
                                        "type": "integer",
                                        "default": 1024,
                                        "description": "Viewport width"
                                    },
                                    "height": {
                                        "type": "integer",
                                        "default": 768,
                                        "description": "Viewport height"
                                    }
                                },
                                "required": ["code"]
                            }
                        }
                    })
            
            # Add other tools based on keywords
            if any(word in prompt.lower() for word in ['tool', 'search', 'browse', 'crawl', 'scrape', 'website']):
                logger.debug("Tool-related keywords found in prompt")
                for category, plugin in registry.active_plugins.items():
                    if category != PluginCategory.PROVIDER and plugin.name != "selenium_sandbox":
                        logger.debug(f"Adding tool for plugin: {plugin.name}")
                        
                        if category == PluginCategory.WEB_CRAWLER:
                            tools.append(self._get_crawler_tool_schema(plugin))
                        else:
                            tools.append(self._get_default_tool_schema(plugin))
            
            logger.debug(f"Prepared {len(tools)} tools")
            return tools
            
        except Exception as e:
            logger.error(f"Error preparing tools: {e}")
            logger.error("Traceback:", exc_info=True)
            return []
    
    def _get_crawler_tool_schema(self, plugin) -> Dict:
        """Get the tool schema for a web crawler plugin."""
        return {
            "type": "function",
            "function": {
                "name": plugin.name,
                "description": plugin.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the webpage to crawl"
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["html", "markdown", "json"],
                            "default": "markdown",
                            "description": "Output format for crawled content"
                        },
                        "schema": {
                            "type": "object",
                            "description": "Schema for structured data extraction",
                            "optional": True
                        }
                    },
                    "required": ["url"]
                }
            }
        }
    
    def _get_default_tool_schema(self, plugin) -> Dict:
        """Get the default tool schema for a plugin."""
        return {
            "type": "function",
            "function": {
                "name": plugin.name,
                "description": plugin.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "object",
                            "description": f"Arguments for the {plugin.name} plugin"
                        }
                    },
                    "required": ["args"]
                }
            }
        }
    
    async def execute_tool(self, tool_call: Dict, plugins_registry) -> Dict:
        """Execute a tool call.
        
        Args:
            tool_call: The tool call from the provider
            plugins_registry: The plugin registry
            
        Returns:
            Dict: Tool execution result
        """
        try:
            if tool_call.get('type') != 'function':
                logger.debug(f"Skipping non-function tool call: {tool_call.get('type')}")
                return {"error": "Invalid tool call type"}
            
            function = tool_call.get('function', {})
            plugin_name = function.get('name')
            
            # Parse arguments
            arguments_raw = function.get('arguments', {})
            if isinstance(arguments_raw, str):
                try:
                    arguments = json.loads(arguments_raw)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse arguments string: {arguments_raw}")
                    arguments = {"args": arguments_raw}
            else:
                arguments = arguments_raw
            
            # Get and execute plugin
            plugin = plugins_registry.get_plugin(plugin_name)
            if not plugin:
                return {"error": f"Plugin {plugin_name} not found"}
            
            try:
                if plugin_name == "selenium_sandbox":
                    plugin_result = await plugin.execute(**arguments)
                elif plugin_name == "crawl4ai":
                    # Special handling for web crawler
                    use_llm = arguments.get("use_llm", False) or "extraction_prompt" in arguments
                    arguments["use_llm"] = use_llm
                    arguments["from_agent"] = use_llm
                    plugin_result = await plugin.execute(**arguments)
                else:
                    plugin_result = await plugin.execute(**arguments)
                
                logger.debug(f"Plugin {plugin_name} execution successful")
                return plugin_result
                
            except Exception as e:
                logger.error(f"Error executing plugin {plugin_name}: {e}")
                logger.error("Traceback:", exc_info=True)
                return {"error": str(e)}
                
        except Exception as e:
            logger.error(f"Error in execute_tool: {e}")
            logger.error("Traceback:", exc_info=True)
            return {"error": str(e)}
