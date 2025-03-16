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
            
            # Extract React/JSX code blocks
            react_matches = re.finditer(r"```(?:jsx|JSX|react|React)\s*([\s\S]*?)```", content, re.MULTILINE)
            for i, match in enumerate(react_matches):
                react_code = match.group(1).strip()
                filename_match = re.search(r"([\w\-]+\.jsx)", content[max(0, match.start()-100):match.start()])
                react_filename = filename_match.group(1) if filename_match else f"component{i if i > 0 else ''}.jsx"
                
                extracted_files[react_filename] = react_code
                logger.debug(f"React/JSX code block found: {react_filename}")
                
                if not main_file_type:
                    main_file_type = "react"
                    main_file_content = react_code
            
            if not extracted_files:
                any_match = re.search(r"```([\s\S]*?)```", content, re.MULTILINE)
                if any_match:
                    code = any_match.group(1).strip()
                    logger.debug("Generic code block found, attempting to determine type")
                    
                    if "<html" in code or "<!DOCTYPE" in code or ("<script" in code and "<style" in code):
                        extracted_files["index.html"] = code
                        main_file_type = "html"
                        main_file_content = code
                    elif "function" in code or "const" in code or "var" in code or "let" in code:
                        extracted_files["script.js"] = code
                        main_file_type = "javascript"
                        main_file_content = code
                    else:
                        extracted_files["index.html"] = code
                        main_file_type = "html"
                        main_file_content = code
            
            return extracted_files, main_file_type, main_file_content
            
        except Exception as e:
            logger.error(f"Error extracting code: {e}")
            logger.error("Traceback:", exc_info=True)
            return {}, "", ""
    
    async def _create_file_wrappers(self, created_files: Dict[str, str], main_file_type: str, temp_dir: str) -> str:
        """Create necessary wrapper files for JavaScript or React code.
        
        Args:
            created_files: Dictionary of created files
            main_file_type: The main file type
            temp_dir: Temporary directory path
            
        Returns:
            str: Path to the main file to execute
        """
        main_file = None
        
        try:
            if "index.html" in created_files:
                main_file = created_files["index.html"]
                
            elif main_file_type == "javascript" and any(f.endswith('.js') for f in created_files):
                js_wrapper = self._get_js_wrapper_template(created_files)
                wrapper_path = f"{temp_dir}/index.html"
                
                wrapper_result = await self.file_manager.execute(
                    operation="write",
                    path=wrapper_path,
                    content=js_wrapper
                )
                
                if wrapper_result.get("success", False):
                    main_file = wrapper_path
                    logger.info(f"Created JavaScript wrapper HTML: {wrapper_path}")
                
            elif main_file_type == "react" and any(f.endswith('.jsx') for f in created_files):
                react_wrapper = self._get_react_wrapper_template(created_files)
                wrapper_path = f"{temp_dir}/index.html"
                
                wrapper_result = await self.file_manager.execute(
                    operation="write",
                    path=wrapper_path,
                    content=react_wrapper
                )
                
                if wrapper_result.get("success", False):
                    main_file = wrapper_path
                    logger.info(f"Created React wrapper HTML: {wrapper_path}")
            
            # If still no main file, use the first created file
            if not main_file and created_files:
                main_file = next(iter(created_files.values()))
                
            return main_file
            
        except Exception as e:
            logger.error(f"Error creating file wrappers: {e}")
            logger.error("Traceback:", exc_info=True)
            return None
    
    def _get_js_wrapper_template(self, files: Dict[str, str]) -> str:
        """Get the HTML wrapper template for JavaScript files."""
        js_files = [f for f in files if f.endswith('.js')]
        css_files = [f for f in files if f.endswith('.css')]
        
        js_links = "\n".join(f'    <script src="{js_file}"></script>' for js_file in js_files)
        css_links = "\n".join(f'    <link rel="stylesheet" href="{css_file}">' for css_file in css_files)
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JavaScript Execution</title>
    {css_links}
</head>
<body>
    <div id="output"></div>
    <script>
    // Console output capture
    (function() {{
        const output = document.getElementById('output');
        const originalConsole = {{}};
        
        ['log', 'info', 'warn', 'error'].forEach(function(method) {{
            originalConsole[method] = console[method];
            console[method] = function() {{
                const args = Array.from(arguments);
                const message = args.map(arg => 
                    typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
                ).join(' ');
                
                const element = document.createElement('div');
                element.className = 'console-' + method;
                element.textContent = message;
                output.appendChild(element);
                
                originalConsole[method].apply(console, arguments);
            }};
        }});
    }})();
    </script>
    {js_links}
    <style>
        #output {{
            font-family: monospace;
            white-space: pre-wrap;
            border: 1px solid #ccc;
            padding: 10px;
            margin-top: 20px;
            background: #f5f5f5;
        }}
        .console-error {{ color: red; }}
        .console-warn {{ color: orange; }}
        .console-info {{ color: blue; }}
        .console-log {{ color: black; }}
    </style>
</body>
</html>
"""
    
    def _get_react_wrapper_template(self, files: Dict[str, str]) -> str:
        """Get the HTML wrapper template for React files."""
        jsx_files = [f for f in files if f.endswith('.jsx')]
        js_files = [f for f in files if f.endswith('.js') and not f.endswith('.jsx')]
        css_files = [f for f in files if f.endswith('.css')]
        
        jsx_links = "\n".join(f'    <script type="text/babel" src="{jsx_file}"></script>' for jsx_file in jsx_files)
        js_links = "\n".join(f'    <script src="{js_file}"></script>' for js_file in js_files)
        css_links = "\n".join(f'    <link rel="stylesheet" href="{css_file}">' for css_file in css_files)
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>React Application</title>
    <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    {css_links}
</head>
<body>
    <div id="root"></div>
    <div id="output"></div>
    <script>
    // Console output capture
    (function() {{
        const output = document.getElementById('output');
        const originalConsole = {{}};
        
        ['log', 'info', 'warn', 'error'].forEach(function(method) {{
            originalConsole[method] = console[method];
            console[method] = function() {{
                const args = Array.from(arguments);
                const message = args.map(arg => 
                    typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
                ).join(' ');
                
                const element = document.createElement('div');
                element.className = 'console-' + method;
                element.textContent = message;
                output.appendChild(element);
                
                originalConsole[method].apply(console, arguments);
            }};
        }});
    }})();
    </script>
    {jsx_links}
    {js_links}
    <script type="text/babel">
        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
    <style>
        #output {{
            font-family: monospace;
            white-space: pre-wrap;
            border: 1px solid #ccc;
            padding: 10px;
            margin-top: 20px;
            background: #f5f5f5;
        }}
        .console-error {{ color: red; }}
        .console-warn {{ color: orange; }}
        .console-info {{ color: blue; }}
        .console-log {{ color: black; }}
    </style>
</body>
</html>
"""
    
    async def execute(self, content: str) -> Dict[str, Any]:
        """Execute code in the sandbox.
        
        Args:
            content: The content containing code
            
        Returns:
            Dict[str, Any]: Execution result
        """
        logger.info(f"Handling sandbox execution for content length: {len(content)}")
        
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
            
            # Create wrapper and get main file
            main_file = await self._create_file_wrappers(created_files, main_file_type, temp_dir)
            
            if not main_file:
                return {
                    "success": False,
                    "error": "Failed to determine main file for execution",
                    "enhanced_content": content
                }
            
            # Execute in sandbox
            file_url = f"file://{os.path.abspath(main_file)}"
            sandbox_result = await self.sandbox_plugin.execute(
                file_url=file_url,
                timeout=30,
                width=1024,
                height=768,
                take_screenshot=False,
                task_type='sandbox'  # Explicitly set task type for sandbox execution
            )
            
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
