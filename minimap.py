"""
Minimap projection module.
"""
from utils import logger

class Minimap:
    """
    Handles projection from camera view to 2D pitch minimap.
    """
    def __init__(self):
        """Initializes the minimap projector."""
        logger.info("Initializing Minimap")
        
    def project(self, tracks):
        """
        Projects 3D/camera coordinates to 2D pitch coordinates.
        
        Args:
            tracks: Tracked objects in camera view.
            
        Returns:
            list: Tracked objects in minimap view.
        """
        pass
