import asyncio
from typing import List

from googlesearch import search

from app.tool.base import BaseTool
from app.utils.cache import cached
from app.utils.monitor import resource_monitor


class GoogleSearch(BaseTool):
    name: str = "google_search"
    description: str = """Perform a Google search and return a list of relevant links.
Use this tool when you need to find information on the web, get up-to-date data, or research specific topics.
The tool returns a list of URLs that match the search query.
"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to Google.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    @cached(expiration_time=3600)  # Cache results for 1 hour
    async def execute(self, query: str, num_results: int = 10) -> List[str]:
        """
        Execute a Google search and return a list of URLs.

        Args:
            query (str): The search query to submit to Google.
            num_results (int, optional): The number of search results to return. Default is 10.

        Returns:
            List[str]: A list of URLs matching the search query.
        """
        # Track tool usage
        resource_monitor.track_tool_usage("google_search")
        
        # Start timer
        resource_monitor.start_timer("google_search")
        
        try:
            # Run the search in a thread pool to prevent blocking
            loop = asyncio.get_event_loop()
            links = await loop.run_in_executor(
                None, lambda: list(search(query, num_results=num_results))
            )
            
            return links
        finally:
            # End timer
            elapsed = resource_monitor.end_timer("google_search")
            if elapsed:
                print(f"Google search completed in {elapsed:.2f} seconds")
