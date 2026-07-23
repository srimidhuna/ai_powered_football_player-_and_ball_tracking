"""
Ball possession analysis module.
"""
import numpy as np
from config import Config
from utils import logger

class Possession:
    """
    Computes ball possession statistics.
    """
    def __init__(self):
        """Initializes possession tracking."""
        logger.info("Initializing Possession tracking")
        self.team_a_frames = 0
        self.team_b_frames = 0
        self.max_distance = Config.POSSESSION_MAX_DISTANCE
        
    def _get_foot_position(self, bbox):
        return (bbox[0] + bbox[2]) / 2.0, float(bbox[3])
        
    def _get_center(self, bbox):
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    def update(self, player_tracks, ball_track):
        """
        Updates possession based on player and ball positions.
        
        Args:
            player_tracks: Current player positions (should include 'team' attribute).
            ball_track: Current ball position.
            
        Returns:
            dict: Current possession percentages.
        """
        if ball_track and ball_track.get('bbox') is not None:
            ball_bbox = ball_track['bbox']
            ball_pos = self._get_center(ball_bbox)
            
            min_dist = float('inf')
            nearest_team = None
            
            for track in player_tracks:
                if track.get('class') != 'player':
                    continue
                    
                team = track.get('team')
                if not team:
                    continue
                    
                foot_pos = self._get_foot_position(track['bbox'])
                dist = np.hypot(foot_pos[0] - ball_pos[0], foot_pos[1] - ball_pos[1])
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_team = team
                    
            # Assign possession if within threshold
            if nearest_team and min_dist <= self.max_distance:
                if nearest_team == 'Team A':
                    self.team_a_frames += 1
                elif nearest_team == 'Team B':
                    self.team_b_frames += 1

        return self.get_percentages()

    def get_percentages(self):
        """
        Returns possession percentages.
        
        Returns:
            dict: {'Team A': float, 'Team B': float}
        """
        total = self.team_a_frames + self.team_b_frames
        if total == 0:
            return {'Team A': 50.0, 'Team B': 50.0}
            
        return {
            'Team A': round((self.team_a_frames / total) * 100, 1),
            'Team B': round((self.team_b_frames / total) * 100, 1)
        }
