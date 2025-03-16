# plugins/code_execution/python_execute.py
import asyncio
import logging
import sys
import time
from io import StringIO
from typing import Dict, ClassVar, Optional, Any
import threading

from plugins.base import Plugin, PluginCategory

logger = logging.getLogger("manusprime.plugins.python_execute")

class PythonExecutePlugin(Plugin):
    """Plugin for executing Python code safely."""
    
    name: ClassVar[str] = "python_execute"
    description: ClassVar[str] = "Safely executes Python code with timeout and output capture"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.CODE_EXECUTION
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Python execution plugin.
        
        Args:
            config: Plugin configuration
        """
        super().__init__(config)
        self.timeout = self.config.get("timeout", 5)
    
    async def initialize(self) -> bool:
        """Initialize the plugin.
        
        Returns:
            bool: True (always succeeds)
        """
        return True
    
    async def execute(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute Python code safely.
        
        Args:
            code: The Python code to execute
            timeout: Execution timeout in seconds (overrides default)
            
        Returns:
            Dict: Execution result with stdout, stderr, and success flag
        """
        timeout_seconds = timeout or self.timeout
        result = {
            "success": False,
            "output": "",
            "error": "",
            "execution_time": 0
        }
        
        # Run in a separate thread to facilitate timeout
        thread_result = {}
        
        def run_code():
            start_time = time.time()
            
            # Capture stdout and stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = stdout_capture = StringIO()
            sys.stderr = stderr_capture = StringIO()
            
            try:
                # Create a restricted globals dictionary
                restricted_globals = {
                    "__builtins__": {
                        k: v for k, v in __builtins__.items()
                        if k not in ["open", "eval", "exec", "__import__"]
                    }
                }
                
                # Add safe imports
                safe_modules = ["math", "random", "datetime", "json", "re"]
                for module_name in safe_modules:
                    try:
                        module = __import__(module_name)
                        restricted_globals[module_name] = module
                    except ImportError:
                        pass
                
                # Execute code with restricted globals
                exec(code, restricted_globals, {})
                thread_result["success"] = True
                
            except Exception as e:
                thread_result["success"] = False
                sys.stderr.write(f"{type(e).__name__}: {e}")
                
            finally:
                # Get captured output
                stdout_content = stdout_capture.getvalue()
                stderr_content = stderr_capture.getvalue()
                
                # Restore stdout and stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                # Set result
                thread_result["output"] = stdout_content
                thread_result["error"] = stderr_content
                thread_result["execution_time"] = time.time() - start_time
        
        # Create and start thread
        thread = threading.Thread(target=run_code)
        thread.daemon = True
        thread.start()
        
        # Wait for thread to complete or timeout
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            # Timeout occurred
            result["success"] = False
            result["error"] = f"Execution timed out after {timeout_seconds} seconds"
            result["execution_time"] = timeout_seconds
        else:
            # Thread completed
            result = {
                "success": thread_result.get("success", False),
                "output": thread_result.get("output", ""),
                "error": thread_result.get("error", ""),
                "execution_time": thread_result.get("execution_time", 0)
            }
        
        return result