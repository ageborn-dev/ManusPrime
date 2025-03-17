import logging
import time
import os
from typing import Dict, Any, Tuple
import re

from plugins.registry import registry

logger = logging.getLogger("manusprime.core.sandbox_manager")

class SandboxManager:
    """Manages sandbox execution operations."""
    
    def __init__(self):
        """Initialize SandboxManager."""
        self.file_manager = None
        self.sandbox_plugin = None
        
    async def initialize(self) -> bool:
        """Initialize required plugins.
        
        Returns:
            bool: True if initialization was successful
        """
        self.file_manager = registry.get_plugin("file_manager")
        self.sandbox_plugin = registry.get_plugin("selenium_sandbox")
        
        return bool(self.file_manager and self.sandbox_plugin)
    
    def _extract_code_from_content(self, content: str) -> Tuple[Dict[str, str], str, str]:
        """Extract code blocks from the content.
        
        Args:
            content: The content to parse
            
        Returns:
            Tuple[Dict[str, str], str, str]: Dictionary of extracted code files, main file type, main file content
        """
        logger.debug("Extracting code from content")
        try:
            extracted_files = {}
            main_file_type = ""
            main_file_content = ""
            
            # Extract HTML code blocks
            html_match = re.search(r"```(?:html|HTML)\s*([\s\S]*?)```", content, re.MULTILINE)
            if html_match:
                logger.debug("HTML code block found")
                html_code = html_match.group(1).strip()
                extracted_files["index.html"] = html_code
                main_file_type = "html"
                main_file_content = html_code
            
            # Extract JavaScript code blocks
            js_matches = re.finditer(r"```(?:javascript|js|JS)\s*([\s\S]*?)```", content, re.MULTILINE)
            for i, match in enumerate(js_matches):
                js_code = match.group(1).strip()
                filename_match = re.search(r"([\w\-]+\.js)", content[max(0, match.start()-100):match.start()])
                js_filename = filename_match.group(1) if filename_match else f"script{i if i > 0 else ''}.js"
                
                extracted_files[js_filename] = js_code
                logger.debug(f"JavaScript code block found: {js_filename}")
                
                if not main_file_type:
                    main_file_type = "javascript"
                    main_file_content = js_code
            
            # Extract CSS code blocks
            css_matches = re.finditer(r"```(?:css|CSS)\s*([\s\S]*?)```", content, re.MULTILINE)
            for i, match in enumerate(css_matches):
                css_code = match.group(1).strip()
                filename_match = re.search(r"([\w\-]+\.css)", content[max(0, match.start()-100):match.start()])
                css_filename = filename_match.group(1) if filename_match else f"style{i if i > 0 else ''}.css"
                
                extracted_files[css_filename] = css_code
                logger.debug(f"CSS code block found: {css_filename}")
            
            return extracted_files, main_file_type, main_file_content
            
        except Exception as e:
            logger.error(f"Error extracting code: {e}")
            logger.error("Traceback:", exc_info=True)
            return {}, "", ""
    
    async def execute(self, content: str, **kwargs) -> Dict[str, Any]:
        """Execute code in the sandbox.
        
        Args:
            content: The content containing code
            **kwargs: Additional execution parameters
            
        Returns:
            Dict[str, Any]: Execution result
        """
        logger.info("Starting sandbox execution")
        
        if not await self.initialize():
            return {
                "success": False,
                "error": "Required plugins not available",
                "enhanced_content": content
            }
        
        try:
            # Extract code
            extracted_files, main_file_type, main_file_content = self._extract_code_from_content(content)
            
            if not extracted_files:
                return {
                    "success": False,
                    "error": "No executable code found in response",
                    "enhanced_content": content
                }
            
            # Create temporary directory
            timestamp = int(time.time())
            temp_dir = f"sandbox_simulations/{timestamp}"
            dir_result = await self.file_manager.execute(
                operation="mkdir",
                path=temp_dir
            )
            
            if not dir_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Failed to create temp directory: {dir_result.get('error')}",
                    "enhanced_content": content
                }
            
            # Create files
            created_files = {}
            for filename, file_content in extracted_files.items():
                file_path = f"{temp_dir}/{filename}"
                file_result = await self.file_manager.execute(
                    operation="write",
                    path=file_path,
                    content=file_content
                )
                
                if file_result.get("success", False):
                    created_files[filename] = file_path
            
            if not created_files:
                return {
                    "success": False,
                    "error": "Failed to create any files",
                    "enhanced_content": content
                }
            
            # Execute in sandbox
            file_url = f"file://{os.path.abspath(next(iter(created_files.values())))}"
            
            # Execute with visible browser (always in sandbox mode)
            execute_params = {
                'file_url': file_url,
                'timeout': kwargs.get('timeout', 30),
                'width': kwargs.get('width', 1024),
                'height': kwargs.get('height', 768),
                'task_type': 'sandbox',  # Always use sandbox mode for UI
                'maintain_session': True,  # Always maintain session
                'task_id': kwargs.get('task_id')  # Pass through task ID if provided
            }
            
            # Execute with prepared parameters
            sandbox_result = await self.sandbox_plugin.execute(**execute_params)
            
            if not sandbox_result.get("success", False):
                return {
                    "success": False,
                    "error": sandbox_result.get("error", "Unknown sandbox error"),
                    "enhanced_content": content
                }
            
            # Enhance response
            enhanced_content = content + "\n\n## Execution Results\n\n"
            
            if console_logs := sandbox_result.get("console_logs", []):
                enhanced_content += "### Console Output:\n```\n"
                enhanced_content += "\n".join(console_logs)
                enhanced_content += "\n```\n\n"
            
            enhanced_content += f"Application is now running. Files are saved in: `{temp_dir}`\n"
            
            return {
                "success": True,
                "enhanced_content": enhanced_content,
                "console_logs": sandbox_result.get("console_logs", []),
                "temp_dir": temp_dir
            }
            
        except Exception as e:
            logger.error(f"Error during sandbox execution: {e}")
            logger.error("Traceback:", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "enhanced_content": content
            }
