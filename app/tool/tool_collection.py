from typing import Dict, List, Optional, Type
import importlib
import pkgutil
import inspect
from pathlib import Path

from app.tool.base import BaseTool
from app.tool.zapier_tool import ZapierTool
from app.logger import logger
from app.config import config

class ToolCollection:
    """Collection of available tools."""
    
    def __init__(self, *tools):
        self._tools: Dict[str, BaseTool] = {}
        
        # Register initial tools if provided
        for tool in tools:
            self.add_tool(tool)
            
        # Load additional tools
        self._load_tools()

    def _load_tools(self):
        """Load all available tools."""
        try:
            # Load built-in tools
            self._register_builtin_tools()
            
            # Load tools from tool directory
            self._register_tools_from_directory()
            
            logger.info(f"Loaded {len(self._tools)} tools")
            
        except Exception as e:
            logger.error(f"Error loading tools: {str(e)}")
            raise

    def _register_builtin_tools(self):
        """Register built-in tools."""
        builtin_tools = []
        
        # Add Zapier tool if enabled in config
        if config.zapier.enabled:
            builtin_tools.append(ZapierTool())
            logger.info("Zapier integration enabled - registered Zapier tool")
        
        # Register all built-in tools
        for tool in builtin_tools:
            self.register_tool(tool)

    def _register_tools_from_directory(self):
        """Dynamically register tools from the tool directory."""
        try:
            # Get the directory containing tool modules
            tool_dir = Path(__file__).parent
            
            # Iterate through all Python files in the directory
            for module_info in pkgutil.iter_modules([str(tool_dir)]):
                if module_info.name == "__init__":
                    continue
                    
                try:
                    # Import the module
                    module = importlib.import_module(f"app.tool.{module_info.name}")
                    
                    # Find all classes that inherit from BaseTool
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseTool) and 
                            obj != BaseTool):
                            try:
                                # Check if tool should be registered based on config
                                if name == "ZapierTool" and not config.zapier.enabled:
                                    logger.debug("Skipping ZapierTool as it's disabled in config")
                                    continue
                                    
                                # Instantiate and register the tool
                                tool = obj()
                                self.register_tool(tool)
                            except Exception as e:
                                logger.error(f"Error instantiating tool {name}: {str(e)}")
                                
                except Exception as e:
                    logger.error(f"Error loading module {module_info.name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error scanning tool directory: {str(e)}")

    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        if not isinstance(tool, BaseTool):
            raise ValueError("Tool must inherit from BaseTool")
            
        if tool.name in self._tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")
            
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
        
    def add_tool(self, tool: BaseTool):
        """Alias for register_tool."""
        self.register_tool(tool)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
        
    @property
    def tool_map(self) -> Dict[str, BaseTool]:
        """Get a mapping of tool names to tools."""
        return self._tools.copy()

    def list_tools(self) -> List[Dict]:
        """List all available tools with their schemas."""
        return [tool.to_param() for tool in self._tools.values()]
        
    def to_params(self) -> List[Dict]:
        """Convert tools to function call format for LLM."""
        return [tool.to_param() for tool in self._tools.values()]

    def get_tool_help(self, name: str) -> Optional[str]:
        """Get help text for a specific tool."""
        tool = self.get_tool(name)
        if tool:
            return tool.help_text if hasattr(tool, 'help_text') else tool.description
        return None

    def get_tool_example(self, name: str) -> Optional[str]:
        """Get example usage for a specific tool."""
        tool = self.get_tool(name)
        if tool:
            return tool.example_usage if hasattr(tool, 'example_usage') else None
        return None

    async def execute(self, name: str, tool_input: Dict) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
            
        return await tool.execute(**tool_input)