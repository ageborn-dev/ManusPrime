# utils/logger.py
import os
import logging
import sys
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
def setup_logger(name: str = "manusprime", level: int = logging.INFO):
    """Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handlers if they don't exist
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create file handler with timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

# Create the main logger
logger = setup_logger()