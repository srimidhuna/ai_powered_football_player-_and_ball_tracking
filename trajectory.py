"""
Trajectory analysis module.
"""
import cv2
from collections import defaultdict, deque
from config import Config
from utils import logger

class Trajectory:
    """
    Maintains and renders object trajectories over time.
    """
    def __init__(self):
        """Initializes trajectory analysis."""
        logger.info("Initializing Trajectory Rendering")
        self.trail_length = Config.TRAJECTORY_LENGTH
        self.trails = defaultdict(lambda: deque(maxlen=self.trail_length))
        
        self.colors = {
            'Team A': Config.TEAM_A_COLOR,
            'Team B': Config.TEAM_B_COLOR,
            None: Config.UNKNOWN_COLOR
        }
        
    def draw(self, frame, tracks):
        """
        Maintains positions and draws fading trails on the frame.
        
        Args:
            frame: The current image/frame.
            tracks: List of tracked objects.
            
        Returns:
            object: Frame with drawn trajectories.
        """
        for track in tracks:
            # We only draw trajectories for players in this specific feature
            if track.get('class') != 'player':
                continue
                
            track_id = track['track_id']
            bbox = track['bbox']
            
            # Foot position: x_center, y_bottom
            x_center = int((bbox[0] + bbox[2]) / 2.0)
            y_bottom = int(bbox[3])
            foot_pos = (x_center, y_bottom)
            
            self.trails[track_id].append(foot_pos)
            
            team = track.get('team')
            base_color = self.colors.get(team, Config.UNKNOWN_COLOR)
            
            pts = list(self.trails[track_id])
            for i in range(1, len(pts)):
                # Calculate fading effect
                fraction = i / len(pts)
                
                # Fade color towards black to simulate transparency/fading out
                color = (
                    int(base_color[0] * fraction),
                    int(base_color[1] * fraction),
                    int(base_color[2] * fraction)
                )
                
                # Tapering thickness
                thickness = int(max(1, 4 * fraction))
                
                # Draw directly on frame to avoid heavy full-frame overlay copies
                cv2.line(frame, pts[i-1], pts[i], color, thickness, cv2.LINE_AA)
                
        return frame
