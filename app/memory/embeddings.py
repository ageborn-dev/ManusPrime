import hashlib
from typing import List, Optional, Union

import numpy as np

from app.logger import logger


class EmbeddingClient:
    """Client for generating embeddings from text using local methods."""
    
    def __init__(
        self,
        embedding_dimension: int = 1536,
    ):
        """
        Initialize the embedding client.
        
        Args:
            embedding_dimension: Dimension of the embedding vectors
        """
        self.embedding_dimension = embedding_dimension
        logger.info(f"Initialized local embedding client with dimension {embedding_dimension}")
    
    async def get_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for one or more text strings using a local method.
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            List of embedding vectors (list of floats)
        """
        # Handle empty input
        if not text:
            return [[0.0] * self.embedding_dimension]
            
        # Convert single string to list
        if isinstance(text, str):
            text = [text]
        
        # Generate embeddings using local method
        embeddings = []
        for item in text:
            embedding = self._generate_local_embedding(item)
            embeddings.append(embedding)
            
        return embeddings
    
    def _generate_local_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a text string using a deterministic local method.
        
        This uses a combination of text hashing and random number generation to create
        consistent embeddings for the same text inputs without requiring API calls.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text:
            return [0.0] * self.embedding_dimension
        
        # Use hash of text as random seed for reproducibility
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash, 16) % (2**32)
        
        # Use tokenization-like approach for more meaningful embeddings
        words = text.lower().split()
        
        # Initialize embedding vector
        embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
        
        # Add word contributions to the embedding
        for i, word in enumerate(words):
            # Hash each word to get a deterministic vector
            word_hash = hashlib.md5(f"{word}_{i}".encode()).hexdigest()
            word_seed = int(word_hash, 16) % (2**32)
            
            # Use word seed to generate a component vector
            np.random.seed(word_seed)
            word_vector = np.random.normal(0, 1, self.embedding_dimension).astype(np.float32)
            
            # Add to embedding with position-based weighting
            position_weight = 1.0 / (1 + i * 0.1)  # Words earlier in text have slightly higher weight
            embedding += word_vector * position_weight
        
        # If no words, use the text hash
        if not words:
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, self.embedding_dimension).astype(np.float32)
        
        # Normalize the vector to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
