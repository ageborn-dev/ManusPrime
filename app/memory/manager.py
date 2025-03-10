import time
from typing import List, Optional, Union

from app.logger import logger
from app.memory.embeddings import EmbeddingClient
from app.memory.vector_store import VectorMemory
from app.schema import Memory, Message


class MemoryManager:
    """
    Memory manager that integrates vector memory with the existing Memory class.
    
    This class provides a bridge between the existing Message-based memory
    and the vector database for semantic search.
    """
    
    def __init__(
        self,
        base_memory: Optional[Memory] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        vector_memory: Optional[VectorMemory] = None,
        max_context_messages: int = 10,
        max_context_tokens: int = 4000
    ):
        """
        Initialize the memory manager.
        
        Args:
            base_memory: The existing Memory instance to integrate with
            embedding_client: Client for generating embeddings
            vector_memory: Vector memory store for semantic search
            max_context_messages: Maximum number of messages to keep in context
            max_context_tokens: Maximum number of tokens to keep in context
        """
        self.base_memory = base_memory or Memory()
        self.embedding_client = embedding_client or EmbeddingClient()
        self.vector_memory = vector_memory or VectorMemory()
        
        self.max_context_messages = max_context_messages
        self.max_context_tokens = max_context_tokens
        
        # Tokenizer for counting tokens (simple approximation)
        self.tokens_per_word = 1.33  # Approximation
    
    async def add_message(self, message: Message) -> None:
        """
        Add a message to both the base memory and vector memory.
        
        Args:
            message: The message to add
        """
        # Add to base memory
        self.base_memory.add_message(message)
        
        # Skip empty messages or those without content
        if not hasattr(message, 'content') or not message.content:
            return
            
        # Add to vector memory
        await self._add_to_vector_memory(message)
    
    async def add_messages(self, messages: List[Message]) -> None:
        """
        Add multiple messages to memory.
        
        Args:
            messages: List of messages to add
        """
        # Add to base memory
        self.base_memory.add_messages(messages)
        
        # Add to vector memory
        for message in messages:
            # Skip empty messages
            if not hasattr(message, 'content') or not message.content:
                continue
                
            await self._add_to_vector_memory(message)
    
    async def search_memory(self, query: str, limit: int = 5) -> List[Message]:
        """
        Search memory for relevant messages.
        
        Args:
            query: The search query
            limit: Maximum number of results
            
        Returns:
            List of relevant messages
        """
        # Get embedding for query
        embeddings = await self.embedding_client.get_embedding(query)
        query_embedding = embeddings[0]
        
        # Search vector memory
        results = await self.vector_memory.search(
            query=query,
            embedding=query_embedding,
            limit=limit
        )
        
        # Convert results to Message objects
        messages = []
        for entry, _ in results:
            role = entry.message_type or "system"
            messages.append(Message(role=role, content=entry.content))
        
        return messages
    
    async def get_context(
        self, 
        query: Optional[str] = None, 
        include_recent: bool = True,
        include_relevant: bool = True,
        recent_count: int = 5,
        relevant_count: int = 3
    ) -> List[Message]:
        """
        Get context for the current conversation.
        
        Args:
            query: Optional query to find relevant context
            include_recent: Whether to include recent messages
            include_relevant: Whether to include relevant messages
            recent_count: Number of recent messages to include
            relevant_count: Number of relevant messages to include
            
        Returns:
            List of context messages
        """
        context_messages = []
        
        # Add recent messages from base memory
        if include_recent:
            recent_messages = self.base_memory.get_recent_messages(recent_count)
            context_messages.extend(recent_messages)
        
        # Add relevant messages from vector memory if query provided
        if include_relevant and query:
            relevant_messages = await self.search_memory(query, limit=relevant_count)
            
            # Filter out duplicates
            existing_contents = {msg.content for msg in context_messages if hasattr(msg, 'content')}
            unique_relevant = [
                msg for msg in relevant_messages 
                if hasattr(msg, 'content') and msg.content not in existing_contents
            ]
            
            context_messages.extend(unique_relevant)
        
        # Ensure we don't exceed token limits
        return self._trim_context_to_fit(context_messages)
    
    async def summarize_memory(self, query: Optional[str] = None) -> str:
        """
        Generate a summary of memory relevant to the query.
        
        Args:
            query: Optional query to focus the summary
            
        Returns:
            Summary string
        """
        # Get messages to summarize
        if query:
            messages = await self.search_memory(query, limit=10)
        else:
            messages = self.base_memory.get_recent_messages(10)
        
        # Simple summary by concatenation (in production, use an LLM to generate a proper summary)
        summary_parts = []
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                role = getattr(msg, 'role', 'unknown')
                content = msg.content[:100] + ("..." if len(msg.content) > 100 else "")
                summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
    
    async def _add_to_vector_memory(self, message: Message) -> None:
        """Add a message to vector memory with embedding."""
        content = message.content or ""
        if not content:
            return
        
        # Get embedding
        embeddings = await self.embedding_client.get_embedding(content)
        embedding = embeddings[0]
        
        # Add to vector memory
        await self.vector_memory.add_entry(
            content=content,
            embedding=embedding,
            message_type=message.role,
            timestamp=time.time(),
            metadata={"role": message.role}
        )
    
    def _trim_context_to_fit(self, messages: List[Message]) -> List[Message]:
        """Trim context messages to fit within token limits."""
        if not messages:
            return []
        
        # Estimate token count (simple approximation)
        def estimate_tokens(text: str) -> int:
            if not text:
                return 0
            words = len(text.split())
            return int(words * self.tokens_per_word)
        
        # Sort messages by priority (recent messages have higher priority)
        sorted_messages = sorted(
            enumerate(messages),
            key=lambda x: x[0],  # Sort by original position
            reverse=True  # Most recent first
        )
        
        # Include messages until we hit token limit
        included_messages = []
        total_tokens = 0
        
        for _, message in sorted_messages:
            content = message.content or ""
            tokens = estimate_tokens(content)
            
            if total_tokens + tokens <= self.max_context_tokens:
                included_messages.append(message)
                total_tokens += tokens
            
            # Stop if we've included enough messages
            if len(included_messages) >= self.max_context_messages:
                break
        
        # Re-sort included messages to original order
        included_messages.sort(key=lambda x: messages.index(x))
        
        return included_messages
