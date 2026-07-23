"""
Utility functions including project-wide logging.
"""
import logging
import sys
import os

def setup_logger(name: str, log_file: str = 'logs/tracker.log', level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger with both console and file handlers.
    
    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    
    # Create file handler
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except IOError:
        pass # Handle gracefully if directory doesn't exist yet
        
    logger.addHandler(ch)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

logger = setup_logger('football_tracker')
