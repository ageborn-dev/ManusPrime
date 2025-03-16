"""Base classes for web crawler plugins."""

import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Any, ClassVar, Union
from enum import Enum

from plugins.base import Plugin, PluginCategory

logger = logging.getLogger("manusprime.plugins.web_crawler")

class CrawlStrategy(str, Enum):
    """Crawling strategies."""
    BASIC = "basic"
    BFS = "bfs"
    DFS = "dfs"
    BEST_FIRST = "best_first"

class WebCrawlerPlugin(Plugin):
    """Base class for web crawler plugins."""

    category: ClassVar[PluginCategory] = PluginCategory.WEB_CRAWLER

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the web crawler plugin."""
        super().__init__(config)

    @abstractmethod
    async def arun(self, url: str, **kwargs) -> Dict[str, Any]:
        """Execute a web crawling operation.
        
        Args:
            url: The URL to crawl
            **kwargs: Additional crawling parameters
            
        Returns:
            Dict: Crawling result with standardized format
        """
        pass

    async def get_markdown(self, url: str, **kwargs) -> str:
        """Get markdown from a webpage."""
        return (await self.arun(url, output_format="markdown", **kwargs))["content"]
    
    async def get_structured_data(self, url: str, schema: Dict, **kwargs) -> Dict:
        """Extract structured data from a webpage."""
        return (await self.arun(url, schema=schema, output_format="json", **kwargs))["content"]

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the crawler's primary function (arun).
        
        Args:
            **kwargs: Arguments for the arun method
            
        Returns:
            Dict: The crawling result
        """
        return await self.arun(**kwargs)
