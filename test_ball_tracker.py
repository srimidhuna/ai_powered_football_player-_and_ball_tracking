import numpy as np
from ball_tracker import BallTracker
from utils import logger
from config import Config

def test_ball_tracker():
    logger.info("Initializing ball tracker test...")
    tracker = BallTracker()
    
    # Let's mock a ball moving linearly
    # Frame 0: ball detected at [100, 100] (bbox: 95, 95, 105, 105)
    # Frame 1: ball detected at [110, 100]
    # Frame 2: missing (should be predicted near [120, 100])
    # Frame 3: missing (should be predicted near [130, 100])
    # Frame 4: ball detected at [140, 100] (should trigger interpolation for Frame 2, 3)
    # Frame 5: bad detection jump [500, 500] (should be ignored, predicted near [150, 100])
    
    detections_sequence = [
        [{'class': 'ball', 'bbox': [95, 95, 105, 105], 'confidence': 0.9}],
        [{'class': 'ball', 'bbox': [105, 95, 115, 105], 'confidence': 0.9}],
        [],  # missing
        [],  # missing
        [{'class': 'ball', 'bbox': [135, 95, 145, 105], 'confidence': 0.9}],
        [{'class': 'ball', 'bbox': [500, 500, 510, 510], 'confidence': 0.9}], # unrealistic jump
    ]
    
    dummy_frame = None
    
    for i, dets in enumerate(detections_sequence):
        logger.info(f"--- Frame {i} ---")
        track = tracker.update(dets, dummy_frame)
        if track:
            logger.info(f"  -> Bbox: {np.round(track['bbox'], 1)}, Status: {track['status']}")
        else:
            logger.info("  -> No track")
            
    # Validate retroactive interpolation
    assert tracker.history[2]['status'] == 'interpolated', "Frame 2 was not interpolated!"
    assert tracker.history[3]['status'] == 'interpolated', "Frame 3 was not interpolated!"
    
    # Validate jump ignore
    assert tracker.history[5]['status'] == 'predicted', "Frame 5 unrealistic jump was not ignored!"
    
    logger.info("Ball Tracker Validation Passed! Kalman filtering, interpolation, and outlier rejection worked correctly.")

if __name__ == "__main__":
    test_ball_tracker()
