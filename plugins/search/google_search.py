# plugins/search/google_search.py
import logging
import asyncio
from typing import Dict, List, ClassVar, Optional, Any

from plugins.base import Plugin, PluginCategory

# Try to import googlesearch-python
try:
    from googlesearch import search as gsearch
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    logging.warning("googlesearch-python not installed. Install with 'pip install googlesearch-python'")

logger = logging.getLogger("manusprime.plugins.google_search")

class GoogleSearchPlugin(Plugin):
    """Plugin for performing Google searches."""
    
    name: ClassVar[str] = "google_search"
    description: ClassVar[str] = "Performs searches using Google and returns results"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.SEARCH
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Google search plugin.
        
        Args:
            config: Plugin configuration
        """
        super().__init__(config)
        self.max_results = self.config.get("max_results", 10)
        self.timeout = self.config.get("timeout", 5)
        self.safe_search = self.config.get("safe_search", True)
    
    async def initialize(self) -> bool:
        """Initialize the plugin.
        
        Returns:
            bool: True if the required library is available
        """
        return SEARCH_AVAILABLE
    
    async def execute(self, 
                   query: str, 
                   num_results: Optional[int] = None, 
                   language: str = "en",
                   **kwargs) -> Dict[str, Any]:
        """Perform a Google search.
        
        Args:
            query: The search query
            num_results: Number of results to return (overrides default)
            language: Language for search results
            **kwargs: Additional search parameters
            
        Returns:
            Dict: Search results with URLs and success flag
        """
        if not SEARCH_AVAILABLE:
            return {
                "success": False,
                "error": "Google search functionality not available. Install googlesearch-python.",
                "results": []
            }
        
        # Set number of results
        count = min(num_results or self.max_results, 20)  # Cap at 20 for safety
        
        try:
            # Run search in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: list(gsearch(
                    query,
                    num_results=count,
                    lang=language,
                    timeout=self.timeout,
                    safe=self.safe_search,
                    **kwargs
                ))
            )
            
            # Format results
            results = []
            for i, url in enumerate(search_results):
                results.append({
                    "position": i + 1,
                    "url": url
                })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error performing Google search: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }