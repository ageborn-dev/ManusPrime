# plugins/file_system/file_manager.py
import os
import logging
import aiofiles
from pathlib import Path
from typing import Dict, List, ClassVar, Optional, Any, Union

from plugins.base import Plugin, PluginCategory

logger = logging.getLogger("manusprime.plugins.file_manager")

class FileManagerPlugin(Plugin):
    """Plugin for file system operations."""
    
    name: ClassVar[str] = "file_manager"
    description: ClassVar[str] = "Manages file system operations like read, write, list, etc."
    version: ClassVar[str] = "0.1.0"
    category: ClassVar[PluginCategory] = PluginCategory.FILE_SYSTEM
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the file manager plugin.
        
        Args:
            config: Plugin configuration
        """
        super().__init__(config)
        self.base_dir = Path(self.config.get("base_dir", "."))
        self.allow_absolute_paths = self.config.get("allow_absolute_paths", False)
        self.restricted_dirs = self.config.get("restricted_dirs", [])
    
    async def initialize(self) -> bool:
        """Initialize the plugin.
        
        Returns:
            bool: True if initialization was successful
        """
        # Ensure base directory exists
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error initializing file manager: {e}")
            return False
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve and validate a file path.
        
        Args:
            path: The path to resolve
            
        Returns:
            Path: The resolved path
            
        Raises:
            ValueError: If the path is not allowed
        """
        path_obj = Path(path)
        
        # Handle absolute paths
        if path_obj.is_absolute():
            if not self.allow_absolute_paths:
                raise ValueError("Absolute paths are not allowed")
        else:
            # Resolve relative to base directory
            path_obj = self.base_dir / path_obj
        
        # Check for path traversal attacks
        try:
            resolved = path_obj.resolve()
            
            # Check if path is outside base directory when using relative paths
            if not self.allow_absolute_paths and not resolved.is_relative_to(self.base_dir.resolve()):
                raise ValueError("Path is outside the allowed directory")
                
            # Check restricted directories
            for restricted in self.restricted_dirs:
                restricted_path = Path(restricted).resolve()
                if resolved == restricted_path or resolved.is_relative_to(restricted_path):
                    raise ValueError(f"Access to {restricted} is restricted")
                    
            return resolved
            
        except (ValueError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {e}")
    
    async def execute(self, 
                   operation: str, 
                   path: str, 
                   content: Optional[str] = None,
                   **kwargs) -> Dict[str, Any]:
        """Execute a file system operation.
        
        Args:
            operation: The operation to perform ("read", "write", "append", "list", "exists", "delete", "mkdir")
            path: The target file or directory path
            content: File content for write/append operations
            **kwargs: Additional operation-specific parameters
            
        Returns:
            Dict: Operation result
        """
        try:
            # Resolve and validate path
            resolved_path = self._resolve_path(path)
            
            # Execute the requested operation
            if operation == "read":
                return await self._read_file(resolved_path)
            elif operation == "write":
                return await self._write_file(resolved_path, content, mode="w")
            elif operation == "append":
                return await self._write_file(resolved_path, content, mode="a")
            elif operation == "list":
                return await self._list_directory(resolved_path)
            elif operation == "exists":
                return await self._check_exists(resolved_path)
            elif operation == "delete":
                return await self._delete_item(resolved_path)
            elif operation == "mkdir":
                return await self._make_directory(resolved_path)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {operation}"
                }
                
        except Exception as e:
            logger.error(f"Error in file operation '{operation}': {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _read_file(self, path: Path) -> Dict[str, Any]:
        """Read a file.
        
        Args:
            path: The file path
            
        Returns:
            Dict: The file content
        """
        if not path.is_file():
            return {
                "success": False,
                "error": f"File not found: {path}"
            }
            
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as file:
                content = await file.read()
                
            return {
                "success": True,
                "content": content,
                "path": str(path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error reading file: {e}"
            }
    
    async def _write_file(self, path: Path, content: str, mode: str = "w") -> Dict[str, Any]:
        """Write or append to a file.
        
        Args:
            path: The file path
            content: The content to write
            mode: Write mode ("w" for write, "a" for append)
            
        Returns:
            Dict: Operation result
        """
        if content is None:
            return {
                "success": False,
                "error": "No content provided for write operation"
            }
            
        # Create parent directories if they don't exist
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(path, mode, encoding="utf-8") as file:
                await file.write(content)
                
            return {
                "success": True,
                "path": str(path),
                "operation": "append" if mode == "a" else "write"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error writing to file: {e}"
            }
    
    async def _list_directory(self, path: Path) -> Dict[str, Any]:
        """List directory contents.
        
        Args:
            path: The directory path
            
        Returns:
            Dict: List of files and directories
        """
        if not path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {path}"
            }
            
        if not path.is_dir():
            return {
                "success": False,
                "error": f"Not a directory: {path}"
            }
            
        try:
            items = []
            for item in path.iterdir():
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
                
            return {
                "success": True,
                "path": str(path),
                "items": items,
                "count": len(items)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error listing directory: {e}"
            }
    
    async def _check_exists(self, path: Path) -> Dict[str, Any]:
        """Check if a file or directory exists.
        
        Args:
            path: The path to check
            
        Returns:
            Dict: Existence check result
        """
        exists = path.exists()
        return {
            "success": True,
            "exists": exists,
            "path": str(path),
            "type": "directory" if exists and path.is_dir() else "file" if exists else None
        }
    
    async def _delete_item(self, path: Path) -> Dict[str, Any]:
        """Delete a file or directory.
        
        Args:
            path: The path to delete
            
        Returns:
            Dict: Deletion result
        """
        if not path.exists():
            return {
                "success": False,
                "error": f"Item not found: {path}"
            }
            
        try:
            if path.is_file():
                path.unlink()
                item_type = "file"
            else:
                import shutil
                shutil.rmtree(path)
                item_type = "directory"
                
            return {
                "success": True,
                "path": str(path),
                "type": item_type
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error deleting item: {e}"
            }
    
    async def _make_directory(self, path: Path) -> Dict[str, Any]:
        """Create a directory.
        
        Args:
            path: The directory path
            
        Returns:
            Dict: Directory creation result
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            return {
                "success": True,
                "path": str(path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error creating directory: {e}"
            }