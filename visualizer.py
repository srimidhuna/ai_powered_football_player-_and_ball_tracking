"""
Visualization module.
"""
import cv2
import numpy as np
from config import Config
from utils import logger

class Visualizer:
    """
    Handles drawing bounding boxes, trails, and statistics on frames.
    """
    def __init__(self):
        """Initializes the visualizer."""
        logger.info("Initializing Visualizer")
        self.team_colors = {
            'Team A': Config.TEAM_A_COLOR,
            'Team B': Config.TEAM_B_COLOR,
            'Referee': (0, 215, 255),   # Yellow for referees
            None: Config.UNKNOWN_COLOR
        }
        
    def draw(self, frame, tracks, ball_track, possession_stats, frame_idx, fps, team_a_name="", team_b_name=""):
        """
        Draws annotations on the frame.
        
        Args:
            frame: The original image/frame.
            tracks: Tracked players.
            ball_track: Tracked ball.
            possession_stats: Current possession statistics.
            frame_idx: Current frame number.
            fps: Current processing FPS.
            
        Returns:
            object: Annotated frame.
        """
        player_count = 0
        referee_count = 0
        team_a_count = 0
        team_b_count = 0
        
        # 1. Bounding boxes, Track IDs, Team Colors
        for track in tracks:
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            track_id = track.get('track_id', '?')
            cls = track.get('class', 'unknown')
            team = track.get('team')
            
            if cls == 'player':
                player_count += 1
                if team == 'Team A':
                    team_a_count += 1
                elif team == 'Team B':
                    team_b_count += 1
            elif cls == 'referee' or team == 'Referee':
                referee_count += 1
                
            color = self.team_colors.get(team, Config.UNKNOWN_COLOR)
            
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with outline for better visibility
            label = f"ID: {track_id}"
            if cls == 'referee':
                label = "REF " + label
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # 2. Ball
        if ball_track and ball_track.get('bbox'):
            bbox = ball_track['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            radius = max(3, int((x2 - x1) / 2))
            
            # Yellow ball
            cv2.circle(frame, (cx, cy), radius, (0, 255, 255), -1)
            
            # Outline for predicted/interpolated vs detected
            status = ball_track.get('status', 'detected')
            if status != 'detected':
                cv2.circle(frame, (cx, cy), radius + 2, (0, 0, 255), 2) # Red outline if guessed
            else:
                cv2.circle(frame, (cx, cy), radius + 2, (0, 0, 0), 2) # Black outline if detected
                
        # 3. HUD (FPS, Frame, Counts, Possession)
        hud_lines = [
            f"Frame: {frame_idx}",
            f"FPS: {fps:.1f}",
            f"Players: {player_count}",
            f"Referees: {referee_count}",
        ]
        
        # Draw semi-transparent background for HUD using ROI to save memory/compute
        x_start, y_start, x_end, y_end = 10, 10, 300, 310
        
        # Ensure we don't go out of bounds
        if y_end <= frame.shape[0] and x_end <= frame.shape[1]:
            roi = frame[y_start:y_end, x_start:x_end]
            black_rect = np.zeros(roi.shape, dtype=np.uint8)
            cv2.addWeighted(roi, 0.5, black_rect, 0.5, 0, roi)
        
        y_offset = 40
        for line in hud_lines:
            # Draw black outline
            cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            # Draw white text
            cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            
        y_offset += 10
        cv2.putText(frame, "Possession & Count:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, "Possession & Count:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        team_a_pct = possession_stats.get('Team A', 50.0)
        team_b_pct = possession_stats.get('Team B', 50.0)
        
        team_a_label = f"Team A ({team_a_name})" if team_a_name else "Team A"
        team_b_label = f"Team B ({team_b_name})" if team_b_name else "Team B"

        text_a   = f"{team_a_label}: {team_a_pct}% ({team_a_count} players)"
        text_b   = f"{team_b_label}: {team_b_pct}% ({team_b_count} players)"
        text_ref = f"Referee: {referee_count}"

        # Draw outlines
        cv2.putText(frame, text_a, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text_a, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

        cv2.putText(frame, text_b, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text_b, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

        # Referee counter in yellow so it stands out
        cv2.putText(frame, text_ref, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text_ref, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        
        return frame
