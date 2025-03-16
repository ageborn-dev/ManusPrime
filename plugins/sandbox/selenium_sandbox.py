# plugins/sandbox/selenium_sandbox.py
import os
import time
import logging
import asyncio
import tempfile
import base64
import uuid
import json
from typing import Dict, List, ClassVar, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    import chromedriver_autoinstaller
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("Selenium not installed. Install with 'pip install selenium chromedriver-autoinstaller'")

from plugins.base import Plugin, PluginCategory

logger = logging.getLogger("manusprime.plugins.sandbox")

class SeleniumSandboxPlugin(Plugin):
    """Plugin for executing and rendering web content in a Selenium sandbox."""
    
    name: ClassVar[str] = "selenium_sandbox"
    description: ClassVar[str] = "Executes and renders HTML, CSS, and JavaScript code in a Selenium-powered sandbox"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.CODE_EXECUTION
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Selenium sandbox plugin.
        
        Args:
            config: Plugin configuration
        """
        super().__init__(config)
        self.drivers_pool = []
        self.max_pool_size = self.config.get("max_pool_size", 3)
        self.timeout = self.config.get("timeout", 30)
        self.headless = True
        self.screenshot_dir = Path(self.config.get("screenshot_dir", "sandbox_output"))
        self.save_artifacts = self.config.get("save_artifacts", False)
        self.allowed_domains = self.config.get("allowed_domains", ["localhost", "127.0.0.1"])
        self.max_memory = self.config.get("max_memory", 1024)
        self.pool_lock = asyncio.Lock()
        self.active_sessions = {}
        
        # Create directories
        if self.save_artifacts:
            os.makedirs(self.screenshot_dir, exist_ok=True)
        
    async def initialize(self) -> bool:
        """Initialize the Selenium sandbox plugin.
        
        Returns:
            bool: True if initialization was successful
        """
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium is not available. Please install required packages.")
            return False
            
        try:
            # Only install chromedriver, don't pre-initialize browser
            chromedriver_autoinstaller.install()
            logger.info("Selenium sandbox plugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Selenium sandbox plugin: {e}")
            return False
    
    async def execute(self, 
                code: Optional[str] = None,
                code_type: str = "html",
                file_url: Optional[str] = None,
                timeout: Optional[int] = None,
                width: int = 1024,
                height: int = 768,
                take_screenshot: bool = False,  # Disable screenshots by default
                **kwargs) -> Dict[str, Any]:
        """Execute and render code in the Selenium sandbox.
        
        Args:
            code: The code to execute (HTML, JavaScript, etc.)
            code_type: The type of code ("html", "javascript", "react")
            file_url: Optional URL to a file to load (instead of code)
            timeout: Execution timeout in seconds
            width: Viewport width
            height: Viewport height
            take_screenshot: Whether to take a screenshot of the result
            **kwargs: Additional execution parameters
            
        Returns:
            Dict: Execution result
        """
        execution_timeout = timeout or self.timeout
        session_id = str(uuid.uuid4())
        
        # Track timing
        start_time = datetime.now()
        
        try:
            # Check if this is a sandbox task
            task_type = kwargs.get('task_type')
            if task_type not in ['sandbox', 'code', 'visualization']:
                return {
                    "success": True,
                    "content": "Task does not require sandbox execution",
                    "execution_time": 0.0
                }
            
            # Get a driver from the pool for sandbox tasks
            driver = await self._get_driver(width, height, task_type)
            
            try:
                temp_file_path = None
                temp_file_url = None
                
                # Use file_url if provided, otherwise use code
                if file_url:
                    temp_file_url = file_url
                    logger.debug(f"Using provided file URL: {temp_file_url}")
                elif code:
                    # Prepare code based on type
                    html_content = await self._prepare_code(code, code_type, **kwargs)
                    
                    # Create a temporary HTML file
                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
                        temp_file_path = f.name
                        f.write(html_content)
                    
                    # Create URL from file path
                    temp_file_url = f"file://{temp_file_path}"
                    logger.debug(f"Created temporary file: {temp_file_url}")
                else:
                    return {
                        "success": False,
                        "error": "Either code or file_url must be provided"
                    }
                
                # Navigate to the temp file
                driver.get(temp_file_url)
                
                # Wait for page to load
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                except TimeoutException:
                    logger.warning(f"Timeout waiting for page load in session {session_id}")
                
                # Execute any additional JavaScript if needed
                if kwargs.get("execute_script"):
                    driver.execute_script(kwargs.get("execute_script"))
                
                # Wait for any specified selectors to be present
                if wait_for := kwargs.get("wait_for_selector"):
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, wait_for))
                        )
                    except TimeoutException:
                        logger.warning(f"Timeout waiting for selector '{wait_for}' in session {session_id}")
                
                # Take a screenshot if requested
                screenshot_path = None
                screenshot_data = None
                if take_screenshot:
                    screenshot_data = driver.get_screenshot_as_base64()
                    if self.save_artifacts:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = self.screenshot_dir / f"sandbox_{session_id}_{timestamp}.png"
                        with open(screenshot_path, "wb") as f:
                            f.write(base64.b64decode(screenshot_data))
                
                # Get console logs if available
                console_logs = []
                try:
                    logs = driver.get_log('browser')
                    console_logs = [log.get('message', '') for log in logs]
                except:
                    logger.warning("Could not retrieve console logs")
                
                # Get HTML content after any JS modifications
                final_html = driver.page_source
                
                # Store driver in active sessions if task_id provided, otherwise return to pool
                if 'task_id' in kwargs:
                    self.active_sessions[kwargs['task_id']] = driver
                    logger.info(f"Stored active session for task {kwargs['task_id']}")
                else:
                    await self._return_driver(driver)
                
                # Cleanup temp file if we created one
                if temp_file_path and not file_url:
                    try:
                        os.unlink(temp_file_path)
                    except Exception as e:
                        logger.warning(f"Error removing temporary file: {e}")
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "html": final_html,
                    "screenshot": screenshot_data,
                    "screenshot_path": str(screenshot_path) if screenshot_path else None,
                    "console_logs": console_logs,
                    "execution_time": execution_time,
                    "file_url": temp_file_url
                }
                
            except Exception as e:
                # Make sure to return the driver to the pool even if an error occurs
                await self._return_driver(driver)
                raise e
                
        except Exception as e:
            logger.error(f"Error executing code in Selenium sandbox: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _prepare_code(self, code: str, code_type: str, **kwargs) -> str:
        """Prepare code for execution based on type.
        
        Args:
            code: The code to prepare
            code_type: The type of code
            
        Returns:
            str: Prepared HTML content
        """
        if code_type == "html":
            # Use the HTML code directly, ensuring it has proper structure
            if not code.strip().startswith("<!DOCTYPE html>") and not code.strip().startswith("<html"):
                return f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Sandbox Execution</title>
                </head>
                <body>
                    {code}
                </body>
                </html>
                """
            return code
            
        elif code_type == "javascript":
            # Wrap JavaScript code in HTML
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>JavaScript Execution</title>
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
                            
                            // Call original method
                            originalConsole[method].apply(console, arguments);
                        }};
                    }});
                }})();
                
                // User code execution
                (function() {{
                    try {{
                        {code}
                    }} catch (error) {{
                        console.error('Execution error:', error.message);
                    }}
                }})();
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
            
        elif code_type == "react":
            # Create a React application environment
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>React Application</title>
                <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
                <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
                <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
            </head>
            <body>
                <div id="root"></div>
                <script type="text/babel">
                {code}
                
                // Assume the last component is the one to render if not specified
                {kwargs.get("render_component", "ReactDOM.render(<App />, document.getElementById('root'));")}
                </script>
            </body>
            </html>
            """
            
        else:
            # Default fallback - wrap in HTML
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Code Execution</title>
            </head>
            <body>
                <pre>{code}</pre>
            </body>
            </html>
            """
    
    async def _add_driver_to_pool(self, task_type: Optional[str] = None) -> webdriver.Chrome:
        """Add a new driver to the pool.
        
        Args:
            task_type: Type of task being executed
            
        Returns:
            webdriver.Chrome: The created driver
        """
        chrome_options = Options()

        # Only show browser window for sandbox/visualization tasks
        use_headless = task_type not in ['sandbox', 'visualization']
        if use_headless:
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
        else:
            # For visible windows, ensure proper display
            chrome_options.add_argument("--start-maximized")
            chrome_options.add_argument("--window-type=normal")
        
        # Security and performance options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument(f"--js-flags=--max-old-space-size={self.max_memory}")
        
        # Disable navigation to external sites for security
        chrome_options.add_experimental_option("prefs", {
            "profile.default_content_setting_values.notifications": 2,
            "profile.managed_default_content_settings.images": 2,
        })
        
        # Create driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(self.timeout)
        driver.set_script_timeout(self.timeout)
        
        # Add to pool
        async with self.pool_lock:
            self.drivers_pool.append(driver)
            
        return driver
    
    async def _get_driver(self, width: int, height: int, task_type: Optional[str] = None) -> webdriver.Chrome:
        """Get a driver from the pool or create a new one.
        
        Args:
            width: Viewport width
            height: Viewport height
            task_type: Type of task being executed
            
        Returns:
            webdriver.Chrome: A Chrome webdriver instance
        """
        async with self.pool_lock:
            # Try to get a driver from the pool first
            if self.drivers_pool:
                driver = self.drivers_pool.pop()
            else:
                # Create a new driver only when needed
                driver = await self._add_driver_to_pool(task_type)
                driver = self.drivers_pool.pop()
        
        # Set viewport size
        driver.set_window_size(width, height)
        
        logger.debug("Created new browser window for sandbox execution")
        return driver
    
    async def _return_driver(self, driver: webdriver.Chrome) -> None:
        """Return a driver to the pool or close it if pool is full.
        
        Args:
            driver: The Chrome webdriver to return
        """
        try:
            # Clear browser data
            driver.delete_all_cookies()
            
            # Execute JavaScript to clear storage
            try:
                driver.execute_script("""
                    localStorage.clear();
                    sessionStorage.clear();
                """)
            except:
                pass
            
            # Return to pool or close based on pool size
            async with self.pool_lock:
                if len(self.drivers_pool) < self.max_pool_size:
                    self.drivers_pool.append(driver)
                else:
                    driver.quit()
                    
        except Exception as e:
            logger.warning(f"Error returning driver to pool: {e}")
            try:
                driver.quit()
            except:
                pass
    
    async def cleanup(self, task_id: Optional[str] = None) -> None:
        """Clean up resources used by the plugin.
        
        Args:
            task_id: Optional task ID to cleanup specific session
        """
        async with self.pool_lock:
            if task_id:
                # Cleanup specific session
                if task_id in self.active_sessions:
                    try:
                        driver = self.active_sessions[task_id]
                        driver.quit()
                        del self.active_sessions[task_id]
                        logger.info(f"Cleaned up session for task {task_id}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up session for task {task_id}: {e}")
            else:
                # Cleanup all sessions
                logger.info("Cleaning up all Selenium sandbox resources")
                for driver in self.drivers_pool:
                    try:
                        driver.quit()
                    except Exception as e:
                        logger.warning(f"Error closing driver: {e}")
                
                for task_id, driver in self.active_sessions.items():
                    try:
                        driver.quit()
                    except Exception as e:
                        logger.warning(f"Error closing session for task {task_id}: {e}")
                
                # Clear pools
                self.drivers_pool = []
                self.active_sessions = {}
