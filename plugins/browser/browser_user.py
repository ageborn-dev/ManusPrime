# plugins/browser/browser_user.py
import os
import logging
import asyncio
from typing import Dict, List, ClassVar, Optional, Any
from pathlib import Path

# Try to import browser-use
try:
    from browser_use import Browser
    from browser_use import BrowserConfig
    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False
    logging.warning("browser-use not installed. Install with 'pip install browser-use'")

from plugins.base import Plugin, PluginCategory

logger = logging.getLogger("manusprime.plugins.browser_user")

class BrowserUserPlugin(Plugin):
    """Plugin for browser automation using browser-use library."""
    
    name: ClassVar[str] = "browser_user"
    description: ClassVar[str] = "Automates browser actions like navigation, screenshots, and interaction"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.BROWSER
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the browser automation plugin.
        
        Args:
            config: Plugin configuration
        """
        super().__init__(config)
        self.browser = None
        self.context = None
        self.headless = self.config.get("headless", True)
        self.screenshot_dir = Path(self.config.get("screenshot_dir", "screenshots"))
        self.lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """Initialize the browser.
        
        Returns:
            bool: True if initialization was successful
        """
        if not BROWSER_AVAILABLE:
            logger.error("browser-use library not available")
            return False
            
        try:
            # Ensure screenshot directory exists
            os.makedirs(self.screenshot_dir, exist_ok=True)
            
            # Initialize browser
            browser_config = BrowserConfig(headless=self.headless)
            self.browser = Browser(browser_config)
            
            logger.info("Browser automation plugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing browser plugin: {e}")
            return False
    
    async def execute(self, 
                   action: str,
                   url: Optional[str] = None,
                   element_index: Optional[int] = None,
                   text: Optional[str] = None,
                   script: Optional[str] = None,
                   **kwargs) -> Dict[str, Any]:
        """Execute a browser action.
        
        Args:
            action: The action to perform (navigate, click, input_text, screenshot, etc.)
            url: URL for navigation
            element_index: Element index for interactions
            text: Text to input
            script: JavaScript to execute
            **kwargs: Additional action-specific parameters
            
        Returns:
            Dict: Action result
        """
        if not BROWSER_AVAILABLE:
            return {
                "success": False,
                "error": "Browser automation not available. Install browser-use."
            }
            
        if not self.browser:
            return {
                "success": False,
                "error": "Browser not initialized"
            }
            
        # Use lock to ensure browser actions are sequential
        async with self.lock:
            try:
                # Ensure context is created if not exists
                if not self.context:
                    self.context = await self.browser.new_context()
                
                # Execute the requested action
                if action == "navigate":
                    return await self._navigate(url)
                elif action == "click":
                    return await self._click(element_index)
                elif action == "input_text":
                    return await self._input_text(element_index, text)
                elif action == "screenshot":
                    return await self._screenshot()
                elif action == "get_html":
                    return await self._get_html()
                elif action == "get_text":
                    return await self._get_text()
                elif action == "execute_js":
                    return await self._execute_js(script)
                elif action == "get_elements":
                    return await self._get_elements()
                elif action == "close":
                    return await self._close_browser()
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported action: {action}"
                    }
                    
            except Exception as e:
                logger.error(f"Error in browser action '{action}': {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
    
    async def _navigate(self, url: Optional[str]) -> Dict[str, Any]:
        """Navigate to a URL.
        
        Args:
            url: The URL to navigate to
            
        Returns:
            Dict: Navigation result
        """
        if not url:
            return {
                "success": False,
                "error": "URL is required for navigation"
            }
            
        try:
            await self.context.navigate_to(url)
            current_url = await self.context.get_current_url()
            
            return {
                "success": True,
                "url": current_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Navigation error: {e}"
            }
    
    async def _click(self, element_index: Optional[int]) -> Dict[str, Any]:
        """Click an element by index.
        
        Args:
            element_index: The index of the element to click
            
        Returns:
            Dict: Click result
        """
        if element_index is None:
            return {
                "success": False,
                "error": "Element index is required for click action"
            }
            
        try:
            element = await self.context.get_dom_element_by_index(element_index)
            if not element:
                return {
                    "success": False,
                    "error": f"Element with index {element_index} not found"
                }
                
            download_path = await self.context._click_element_node(element)
            
            result = {
                "success": True,
                "element_index": element_index
            }
            
            if download_path:
                result["download_path"] = download_path
                
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Click error: {e}"
            }
    
    async def _input_text(self, element_index: Optional[int], text: Optional[str]) -> Dict[str, Any]:
        """Input text into an element.
        
        Args:
            element_index: The index of the element
            text: The text to input
            
        Returns:
            Dict: Input result
        """
        if element_index is None:
            return {
                "success": False,
                "error": "Element index is required for input_text action"
            }
            
        if text is None:
            return {
                "success": False,
                "error": "Text is required for input_text action"
            }
            
        try:
            element = await self.context.get_dom_element_by_index(element_index)
            if not element:
                return {
                    "success": False,
                    "error": f"Element with index {element_index} not found"
                }
                
            await self.context._input_text_element_node(element, text)
            
            return {
                "success": True,
                "element_index": element_index,
                "text": text
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Input text error: {e}"
            }
    
    async def _screenshot(self) -> Dict[str, Any]:
        """Take a screenshot of the current page.
        
        Returns:
            Dict: Screenshot result with path
        """
        try:
            timestamp = int(asyncio.get_event_loop().time())
            filename = f"screenshot_{timestamp}.png"
            filepath = self.screenshot_dir / filename
            
            screenshot_data = await self.context.take_screenshot(full_page=True)
            
            # Save screenshot to file
            with open(filepath, "wb") as f:
                f.write(screenshot_data)
                
            return {
                "success": True,
                "filepath": str(filepath),
                "filename": filename
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Screenshot error: {e}"
            }
    
    async def _get_html(self) -> Dict[str, Any]:
        """Get the HTML content of the current page.
        
        Returns:
            Dict: HTML content
        """
        try:
            html = await self.context.get_page_html()
            
            return {
                "success": True,
                "html": html
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Get HTML error: {e}"
            }
    
    async def _get_text(self) -> Dict[str, Any]:
        """Get the text content of the current page.
        
        Returns:
            Dict: Text content
        """
        try:
            text = await self.context.execute_javascript("document.body.innerText")
            
            return {
                "success": True,
                "text": text
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Get text error: {e}"
            }
    
    async def _execute_js(self, script: Optional[str]) -> Dict[str, Any]:
        """Execute JavaScript on the current page.
        
        Args:
            script: The JavaScript code to execute
            
        Returns:
            Dict: Execution result
        """
        if not script:
            return {
                "success": False,
                "error": "Script is required for execute_js action"
            }
            
        try:
            result = await self.context.execute_javascript(script)
            
            return {
                "success": True,
                "result": str(result)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"JavaScript execution error: {e}"
            }
    
    async def _get_elements(self) -> Dict[str, Any]:
        """Get interactive elements on the current page.
        
        Returns:
            Dict: List of elements
        """
        try:
            state = await self.context.get_state()
            elements = state.element_tree.clickable_elements_to_string()
            
            return {
                "success": True,
                "elements": elements,
                "count": len(elements) if isinstance(elements, list) else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Get elements error: {e}"
            }
    
    async def _close_browser(self) -> Dict[str, Any]:
        """Close the browser.
        
        Returns:
            Dict: Close result
        """
        try:
            if self.context:
                await self.context.close()
                self.context = None
                
            if self.browser:
                await self.browser.close()
                self.browser = None
                
            return {
                "success": True,
                "message": "Browser closed successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Close browser error: {e}"
            }
    
    async def cleanup(self) -> None:
        """Clean up browser resources."""
        try:
            if self.context:
                await self.context.close()
                self.context = None
                
            if self.browser:
                await self.browser.close()
                self.browser = None
                
        except Exception as e:
            logger.error(f"Error cleaning up browser: {e}")