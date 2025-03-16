import logging
from typing import Dict, List, Optional

logger = logging.getLogger("manusprime.core.memory_manager")

class MemoryManager:
    """Manages vector memory operations."""
    
    def __init__(self, vector_memory=None):
        """Initialize MemoryManager.
        
        Args:
            vector_memory: The vector memory plugin instance
        """
        self.vector_memory = vector_memory
    
    async def get_similar_experiences(self, prompt: str, task_type: str) -> List[Dict]:
        """Get similar past experiences from vector memory.
        
        Args:
            prompt: The prompt to search for
            task_type: The type of task for filtering
            
        Returns:
            List[Dict]: List of similar experiences
        """
        logger.debug(f"Getting similar experiences for prompt: {prompt[:100]}...")
        if not self.vector_memory:
            logger.debug("No vector memory plugin available")
            return []
            
        try:
            search_result = await self.vector_memory.execute(
                operation="search",
                query=prompt,
                limit=3,
                filter_metadata={"task_type": task_type} if task_type != "default" else None
            )
            logger.debug(f"Vector memory search success: {search_result.get('success', False)}")
            return search_result["results"] if search_result["success"] else []
        except Exception as e:
            logger.warning(f"Failed to retrieve similar experiences: {e}")
            logger.warning("Traceback:", exc_info=True)
            return []
    
    async def store_result(self, prompt: str, result: Dict, task_type: str) -> bool:
        """Store successful result in vector memory.
        
        Args:
            prompt: The original prompt
            result: The execution result
            task_type: The type of task
            
        Returns:
            bool: True if storage was successful
        """
        logger.debug(f"Storing result in vector memory for prompt: {prompt[:100]}...")
        if not self.vector_memory or not result["success"]:
            if not self.vector_memory:
                logger.debug("No vector memory plugin available")
            else:
                logger.debug("Not storing unsuccessful result in vector memory")
            return False
            
        try:
            await self.vector_memory.execute(
                operation="add",
                content=result["content"],
                metadata={
                    "task": prompt,
                    "task_type": task_type,
                    "model": result["model"],
                    "execution_time": result["execution_time"],
                    "tokens": result["tokens"],
                    "cost": result["cost"]
                }
            )
            logger.debug("Successfully stored in vector memory")
            return True
        except Exception as e:
            logger.warning(f"Failed to store in vector memory: {e}")
            logger.warning("Traceback:", exc_info=True)
            return False
    
    def enhance_prompt_with_context(self, task: str, similar_experiences: List[Dict], task_type: str) -> str:
        """Enhance a task prompt with context from similar past experiences.
        
        Args:
            task: The original task prompt
            similar_experiences: List of similar past experiences
            task_type: The type of task
            
        Returns:
            str: Enhanced prompt with context
        """
        logger.debug(f"Enhancing prompt with {len(similar_experiences)} similar experiences")
        try:
            if not similar_experiences:
                return task
            
            # Customize context format based on task type
            if task_type == "code":
                context = "Previous similar coding tasks and solutions:\n"
            elif task_type == "creative":
                context = "Previous similar creative tasks and samples:\n"
            else:
                context = "Previous similar tasks and solutions:\n"
                
            for exp in similar_experiences[:3]:  # Limit to top 3 experiences
                context += f"Task: {exp['metadata']['task']}\n"
                context += f"Solution: {exp['content']}\n\n"
            
            enhanced_prompt = f"{context}\nCurrent task: {task}\n"
            logger.debug(f"Enhanced prompt created (original length: {len(task)}, enhanced length: {len(enhanced_prompt)})")
            return enhanced_prompt
        except Exception as e:
            logger.error(f"Error enhancing prompt with context: {e}")
            logger.error("Traceback:", exc_info=True)
            return task
