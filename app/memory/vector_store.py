import time
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
from pydantic import BaseModel, Field

from app.logger import logger
from app.memory.embeddings import EmbeddingClient
from app.schema import Message


class MemoryEntry(BaseModel):
    """A single entry in the memory system."""
    
    content: str = Field(..., description="The actual content of the memory entry")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the content")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata about the entry")
    
    # For tracking conversations
    timestamp: Optional[float] = Field(None, description="When the entry was created")
    message_type: Optional[str] = Field(None, description="Type of message (user, assistant, etc.)")
    
    class Config:
        arbitrary_types_allowed = True


class VectorMemory:
    """
    A vector-based memory system using FAISS for efficient similarity search.
    
    This class provides methods to store, retrieve, and search through memory entries
    using vector embeddings for semantic similarity.
    """
    
    def __init__(
        self, 
        embedding_dimension: int = 1536,
        embedding_client: Optional[EmbeddingClient] = None,
        index_type: str = "L2",
        similarity_threshold: float = 0.7,
        max_entries: int = 1000
    ):
        """
        Initialize the vector memory system.
        
        Args:
            embedding_dimension: Dimension of the embedding vectors
            embedding_client: Client for generating embeddings
            index_type: Type of FAISS index ('L2' or 'IP' for inner product)
            similarity_threshold: Threshold for considering entries similar (0-1)
            max_entries: Maximum number of entries to store
        """
        self.embedding_dimension = embedding_dimension
        self.embedding_client = embedding_client or EmbeddingClient(embedding_dimension=embedding_dimension)
        self.index_type = index_type
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        
        # Initialize FAISS index
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(embedding_dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(embedding_dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Storage for memory entries
        self.entries: List[MemoryEntry] = []
        
        # For keeping track of current position
        self.next_id = 0
        
        logger.info(f"Initialized VectorMemory with dimension {embedding_dimension}")
    
    async def add_entry(
        self, 
        content: str, 
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        message_type: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> int:
        """
        Add a new entry to the memory.
        
        Args:
            content: The text content to store
            embedding: Optional pre-computed embedding
            metadata: Optional metadata about the entry
            message_type: Type of message (user, assistant, etc.)
            timestamp: When the entry was created
            
        Returns:
            int: ID of the added entry
        """
        if not content:
            logger.warning("Attempted to add empty content to memory")
            return -1
            
        # Get embedding if not provided
        if embedding is None:
            embeddings = await self.embedding_client.get_embedding(content)
            embedding = embeddings[0]
        
        # Create entry
        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            message_type=message_type,
            timestamp=timestamp or time.time()
        )
        
        # Add to storage
        self.entries.append(entry)
        
        # Add to FAISS index
        embedding_array = np.array([embedding], dtype=np.float32)
        self.index.add(embedding_array)
        
        # Increment ID counter
        current_id = self.next_id
        self.next_id += 1
        
        # Check if we need to remove old entries
        if len(self.entries) > self.max_entries:
            self._prune_oldest_entries()
        
        return current_id
    
    async def search(
        self, 
        query: str, 
        embedding: Optional[List[float]] = None,
        limit: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for similar entries in memory.
        
        Args:
            query: The text query to search for
            embedding: Optional pre-computed query embedding
            limit: Maximum number of results to return
            filter_metadata: Optional metadata filter criteria
            
        Returns:
            List of tuples (entry, similarity_score)
        """
        if not self.entries:
            logger.info("Memory is empty, nothing to search")
            return []
        
        # Get embedding if not provided
        if embedding is None:
            embeddings = await self.embedding_client.get_embedding(query)
            embedding = embeddings[0]
        
        # Prepare query vector
        query_vector = np.array([embedding], dtype=np.float32)
        
        # Search in FAISS index
        k = min(limit * 2, len(self.entries))  # Get extra results for filtering
        distances, indices = self.index.search(query_vector, k)
        
        # Convert to similarity scores (1 = most similar, 0 = least similar)
        if self.index_type == "L2":
            # Convert L2 distance to similarity (smaller distance = higher similarity)
            max_dist = np.max(distances[0]) + 1e-6  # Avoid division by zero
            similarities = 1 - (distances[0] / max_dist)
        else:
            # IP similarity is already a similarity score
            similarities = distances[0]
        
        # Collect and filter results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < 0 or idx >= len(self.entries) or similarity < self.similarity_threshold:
                continue
                
            entry = self.entries[idx]
            
            # Apply metadata filter if provided
            if filter_metadata and not self._matches_filter(entry, filter_metadata):
                continue
                
            results.append((entry, float(similarity)))
            
            if len(results) >= limit:
                break
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    async def update_from_messages(self, messages: List[Message]) -> None:
        """
        Update memory from a list of conversation messages.
        
        Args:
            messages: List of messages to add to memory
        """
        added_count = 0
        for message in messages:
            if not hasattr(message, 'content') or not message.content:
                continue
                
            # Add entry with message type
            await self.add_entry(
                content=message.content,
                message_type=getattr(message, 'role', 'unknown'),
                metadata={"role": getattr(message, 'role', 'unknown')}
            )
            added_count += 1
            
        if added_count > 0:
            logger.info(f"Added {added_count} messages to memory")
    
    async def get_relevant_context(
        self, 
        query: str, 
        limit: int = 5, 
        message_types: Optional[List[str]] = None
    ) -> str:
        """
        Get relevant context from memory for a query.
        
        Args:
            query: The query to find relevant context for
            limit: Maximum number of entries to include
            message_types: Optional filter for message types
            
        Returns:
            str: Combined relevant context
        """
        filter_metadata = None
        if message_types:
            filter_metadata = {"role": message_types}
        
        # Search for relevant entries
        results = await self.search(query, limit=limit, filter_metadata=filter_metadata)
        
        if not results:
            return ""
        
        # Format results
        contexts = []
        for entry, similarity in results:
            role = entry.message_type or "memory"
            contexts.append(f"{role}: {entry.content} [Relevance: {similarity:.2f}]")
        
        return "\n\n".join(contexts)
    
    def _matches_filter(self, entry: MemoryEntry, filter_criteria: Dict) -> bool:
        """Check if an entry matches the filter criteria."""
        for key, value in filter_criteria.items():
            if key not in entry.metadata:
                return False
                
            if isinstance(value, list):
                if entry.metadata[key] not in value:
                    return False
            elif entry.metadata[key] != value:
                return False
                
        return True
    
    def _prune_oldest_entries(self, keep_count: Optional[int] = None) -> None:
        """Remove oldest entries to keep memory size manageable."""
        if keep_count is None:
            keep_count = self.max_entries // 2
            
        # Sort by timestamp if available, otherwise use position
        if all(e.timestamp for e in self.entries):
            entries_with_idx = [(i, e) for i, e in enumerate(self.entries)]
            sorted_entries = sorted(
                entries_with_idx, 
                key=lambda x: x[1].timestamp or 0,
                reverse=True
            )
        else:
            # Keep newest entries (at the end of the list)
            entries_with_idx = [(i, e) for i, e in enumerate(self.entries)]
            sorted_entries = sorted(entries_with_idx, key=lambda x: x[0], reverse=True)
        
        # Keep only the specified number of entries
        keep_entries = sorted_entries[:keep_count]
        keep_indices = [idx for idx, _ in keep_entries]
        
        # Create new entries list
        new_entries = [entry for _, entry in keep_entries]
        
        # Reset FAISS index
        if self.index_type == "L2":
            new_index = faiss.IndexFlatL2(self.embedding_dimension)
        else:
            new_index = faiss.IndexFlatIP(self.embedding_dimension)
        
        # Add kept entries to new index
        embeddings = [entry.embedding for entry in new_entries if entry.embedding]
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            new_index.add(embeddings_array)
        
        # Update instance variables
        self.entries = new_entries
        self.index = new_index
        self.next_id = len(new_entries)
        
        logger.info(f"Pruned memory to {len(new_entries)} entries")
    
    async def add_message(self, message: Message) -> int:
        """
        Add a message to memory.
        
        Args:
            message: The message to add
            
        Returns:
            int: ID of the added entry
        """
        if not hasattr(message, 'content') or not message.content:
            return -1
            
        return await self.add_entry(
            content=message.content,
            message_type=getattr(message, 'role', 'unknown'),
            metadata={"role": getattr(message, 'role', 'unknown')}
        )
    
    def clear(self) -> None:
        """Clear all entries from memory."""
        self.entries = []
        
        # Reset FAISS index
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            
        self.next_id = 0
        logger.info("Memory cleared")
