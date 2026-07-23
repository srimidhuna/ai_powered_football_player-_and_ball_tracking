import numpy as np
import cv2
from trajectory import Trajectory
from utils import logger

def test_trajectory():
    logger.info("Initializing trajectory test...")
    trajectory_renderer = Trajectory()
    
    # Create a dummy blank frame (black background)
    base_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Simulate a track moving right and down over 10 frames
    # Track 1 (Team A)
    for i in range(10):
        # We must clone base_frame so we accumulate drawings visually if we wanted to
        # actually Trajectory module expects the fresh frame and returns it drawn with ALL trails.
        frame = base_frame.copy()
        
        # Player moves from x=100 to 200, y=100 to 200
        x = 100 + i * 10
        y = 100 + i * 10
        # Bbox format: x1, y1, x2, y2
        bbox = [x, y, x + 20, y + 50]
        
        tracks = [
            {'track_id': 1, 'class': 'player', 'bbox': bbox, 'team': 'Team A'}
        ]
        
        frame = trajectory_renderer.draw(frame, tracks)
        
    # Validate the state
    assert len(trajectory_renderer.trails[1]) == 10, "Deque did not store 10 positions!"
    
    # Verify the last position stored is the foot position of the last frame
    # Bbox was: x1=190, y1=190, x2=210, y2=240
    # Foot position: x_center = (190+210)/2 = 200. y_bottom = 240
    last_pos = trajectory_renderer.trails[1][-1]
    assert last_pos == (200, 240), f"Expected foot position (200, 240), got {last_pos}"
    
    logger.info("Trajectory Rendering Validation Passed! Deque state and foot positions calculated correctly.")

if __name__ == "__main__":
    test_trajectory()
