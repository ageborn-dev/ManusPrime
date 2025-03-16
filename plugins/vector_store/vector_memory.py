# plugins/vector_store/vector_memory.py
import os
import logging
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, ClassVar, Optional, Any, Tuple, Union

# Try to import vector database libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not installed. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")

# Try to import embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not installed. Install with 'pip install sentence-transformers'")

from plugins.base import Plugin, PluginCategory

logger = logging.getLogger("manusprime.plugins.vector_memory")

class MemoryEntry:
    """A single entry in the vector memory store."""
    
    def __init__(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        id: Optional[str] = None
    ):
        """Initialize a memory entry.
        
        Args:
            content: The text content
            embedding: Vector embedding of the content
            metadata: Additional metadata about the entry
            timestamp: Creation time
            id: Unique identifier
        """
        self.content = content
        self.embedding = embedding
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
        self.id = id or hashlib.md5(f"{content}:{self.timestamp}".encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entry to a dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[List[float]] = None) -> "MemoryEntry":
        """Create an entry from a dictionary."""
        return cls(
            content=data["content"],
            embedding=embedding,
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
            id=data.get("id")
        )


class VectorMemoryPlugin(Plugin):
    """Plugin for vector storage and retrieval."""
    
    name: ClassVar[str] = "vector_memory"
    description: ClassVar[str] = "Vector-based memory storage for semantic search and retrieval"
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.VECTOR_STORE
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the vector memory plugin.
        
        Args:
            config: Plugin configuration
        """
        super().__init__(config)
        self.storage_dir = Path(self.config.get("storage_dir", "data/vector_memory"))
        self.embedding_model = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_dimension = self.config.get("embedding_dimension", 384)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.75)
        self.max_entries = self.config.get("max_entries", 10000)
        self.index_type = self.config.get("index_type", "L2")
        
        # Storage
        self.entries: Dict[str, MemoryEntry] = {}
        self.index = None
        self.embedder = None
        self.last_id = 0
    
    async def initialize(self) -> bool:
        """Initialize the vector memory plugin.
        
        Returns:
            bool: True if initialization was successful
        """
        # Check dependencies
        if not FAISS_AVAILABLE:
            logger.error("FAISS is required for vector memory")
            return False
            
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers is required for vector memory")
            return False
        
        try:
            # Create storage directory
            os.makedirs(self.storage_dir, exist_ok=True)
            
            # Initialize embedder
            self.embedder = SentenceTransformer(self.embedding_model)
            self.embedding_dimension = self.embedder.get_sentence_embedding_dimension()
            
            # Initialize FAISS index
            if self.index_type == "L2":
                self.index = faiss.IndexFlatL2(self.embedding_dimension)
            elif self.index_type == "IP":
                self.index = faiss.IndexFlatIP(self.embedding_dimension)
            else:
                logger.error(f"Unsupported index type: {self.index_type}")
                return False
            
            # Load existing data if available
            index_file = self.storage_dir / "index.faiss"
            entries_file = self.storage_dir / "entries.json"
            
            if index_file.exists() and entries_file.exists():
                # Load index
                try:
                    self.index = faiss.read_index(str(index_file))
                    
                    # Load entries
                    with open(entries_file, "r") as f:
                        entries_data = json.load(f)
                        
                    # Rebuild entries dictionary
                    for entry_data in entries_data:
                        entry = MemoryEntry.from_dict(entry_data)
                        self.entries[entry.id] = entry
                        
                    self.last_id = len(self.entries)
                    logger.info(f"Loaded {len(self.entries)} entries from storage")
                    
                except Exception as e:
                    logger.error(f"Error loading existing data: {e}")
                    # Initialize new index
                    if self.index_type == "L2":
                        self.index = faiss.IndexFlatL2(self.embedding_dimension)
                    else:
                        self.index = faiss.IndexFlatIP(self.embedding_dimension)
            
            logger.info(f"Vector memory initialized with embedding dimension {self.embedding_dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing vector memory: {e}")
            return False
    
    async def execute(self, 
                   operation: str,
                   content: Optional[str] = None,
                   query: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   limit: int = 5,
                   filter_metadata: Optional[Dict[str, Any]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Execute a vector memory operation.
        
        Args:
            operation: Operation to perform ("add", "search", "get", "clear")
            content: Content to add
            query: Query for search
            metadata: Metadata for content
            limit: Maximum number of results for search
            filter_metadata: Filter for search results
            **kwargs: Additional operation-specific parameters
            
        Returns:
            Dict: Operation result
        """
        if not self.embedder or not self.index:
            return {
                "success": False,
                "error": "Vector memory not initialized"
            }
        
        try:
            # Execute the requested operation
            if operation == "add":
                return await self._add_entry(content, metadata)
            elif operation == "search":
                return await self._search(query, limit, filter_metadata)
            elif operation == "get":
                return await self._get_entries(limit)
            elif operation == "clear":
                return await self._clear()
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {operation}"
                }
                
        except Exception as e:
            logger.error(f"Error in vector memory operation '{operation}': {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _add_entry(self, content: Optional[str], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Add an entry to the vector memory.
        
        Args:
            content: The content to add
            metadata: Additional metadata
            
        Returns:
            Dict: Operation result
        """
        if not content:
            return {
                "success": False,
                "error": "Content is required for add operation"
            }
            
        # Generate embedding
        embedding = self.embedder.encode([content])[0]
        
        # Create entry
        entry = MemoryEntry(
            content=content,
            embedding=embedding.tolist(),
            metadata=metadata or {}
        )
        
        # Add to entries dictionary
        self.entries[entry.id] = entry
        self.last_id += 1
        
        # Add to index
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # Save to disk
        await self._save_data()
        
        # Check if we need to prune
        if len(self.entries) > self.max_entries:
            await self._prune_entries()
        
        return {
            "success": True,
            "id": entry.id,
            "entry": entry.to_dict()
        }
    
    async def _search(
        self, 
        query: Optional[str], 
        limit: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for entries similar to the query.
        
        Args:
            query: The search query
            limit: Maximum number of results
            filter_metadata: Filter for results
            
        Returns:
            Dict: Search results
        """
        if not query:
            return {
                "success": False,
                "error": "Query is required for search operation"
            }
            
        if not self.entries:
            return {
                "success": True,
                "results": [],
                "count": 0
            }
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0]
        
        # Search in FAISS index
        k = min(limit * 2, len(self.entries))  # Get extra results for filtering
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        
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
                
            # Get entry by index (id is position in the original add order)
            entry_id = list(self.entries.keys())[idx]
            entry = self.entries[entry_id]
            
            # Apply metadata filter if provided
            if filter_metadata and not self._matches_filter(entry, filter_metadata):
                continue
                
            results.append({
                "id": entry.id,
                "content": entry.content,
                "metadata": entry.metadata,
                "similarity": float(similarity),
                "timestamp": entry.timestamp
            })
            
            if len(results) >= limit:
                break
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    
    async def _get_entries(self, limit: int = 100) -> Dict[str, Any]:
        """Get a list of entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            Dict: List of entries
        """
        # Get entries sorted by timestamp (most recent first)
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.timestamp,
            reverse=True
        )
        
        # Limit the number of entries
        entries = sorted_entries[:limit]
        
        return {
            "success": True,
            "entries": [entry.to_dict() for entry in entries],
            "count": len(entries),
            "total": len(self.entries)
        }
    
    async def _clear(self) -> Dict[str, Any]:
        """Clear all entries.
        
        Returns:
            Dict: Operation result
        """
        # Clear in-memory data
        self.entries = {}
        
        # Reset index
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            
        self.last_id = 0
        
        # Clear files
        index_file = self.storage_dir / "index.faiss"
        entries_file = self.storage_dir / "entries.json"
        
        if index_file.exists():
            os.remove(index_file)
            
        if entries_file.exists():
            os.remove(entries_file)
        
        return {
            "success": True,
            "message": "Vector memory cleared successfully"
        }
    
    async def _save_data(self) -> None:
        """Save index and entries to disk."""
        try:
            # Save FAISS index
            index_file = self.storage_dir / "index.faiss"
            faiss.write_index(self.index, str(index_file))
            
            # Save entries
            entries_file = self.storage_dir / "entries.json"
            entries_data = [entry.to_dict() for entry in self.entries.values()]
            
            with open(entries_file, "w") as f:
                json.dump(entries_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving vector memory data: {e}")
    
    async def _prune_entries(self) -> None:
        """Prune old entries to maintain maximum size."""
        if len(self.entries) <= self.max_entries:
            return
            
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(
            list(self.entries.items()),
            key=lambda item: item[1].timestamp
        )
        
        # Determine how many to remove
        to_remove = len(self.entries) - self.max_entries
        entries_to_remove = sorted_entries[:to_remove]
        
        # Create new set of entries
        new_entries = {}
        for entry_id, entry in sorted_entries[to_remove:]:
            new_entries[entry_id] = entry
            
        # Rebuild index
        if self.index_type == "L2":
            new_index = faiss.IndexFlatL2(self.embedding_dimension)
        else:
            new_index = faiss.IndexFlatIP(self.embedding_dimension)
            
        # Add embeddings to new index
        if new_entries:
            embeddings = [np.array(entry.embedding, dtype=np.float32) for entry in new_entries.values()]
            embeddings_array = np.vstack(embeddings)
            new_index.add(embeddings_array)
        
        # Update instance variables
        self.entries = new_entries
        self.index = new_index
        
        # Save updated data
        await self._save_data()
        
        logger.info(f"Pruned vector memory to {len(self.entries)} entries")
    
    def _matches_filter(self, entry: MemoryEntry, filter_criteria: Dict[str, Any]) -> bool:
        """Check if an entry matches the filter criteria.
        
        Args:
            entry: Memory entry
            filter_criteria: Filter criteria
            
        Returns:
            bool: True if entry matches filter criteria
        """
        for key, value in filter_criteria.items():
            if key not in entry.metadata:
                return False
                
            if isinstance(value, list):
                if entry.metadata[key] not in value:
                    return False
            elif entry.metadata[key] != value:
                return False
                
        return True
    
    async def cleanup(self) -> None:
        """Clean up resources used by the vector memory plugin."""
        # Save data before cleanup
        if self.entries and self.index:
            await self._save_data()