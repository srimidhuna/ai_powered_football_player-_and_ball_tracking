import numpy as np
from tracker import Tracker
from utils import logger

def test_tracking_consistency():
    logger.info("Initializing tracker for consistency test...")
    tracker = Tracker()
    
    # Mock a player moving across 5 frames
    # [x1, y1, x2, y2]
    # Frame 1: player is at [100, 100, 200, 200]
    # Frame 2: player moves to [110, 110, 210, 210]
    # Frame 3: player moves to [120, 120, 220, 220] (brief occlusion added)
    # Frame 4: player occluded (no detection)
    # Frame 5: player reappears at [140, 140, 240, 240]
    
    mock_frames_detections = [
        [{'class': 'player', 'bbox': [100, 100, 200, 200], 'confidence': 0.9}],
        [{'class': 'player', 'bbox': [110, 110, 210, 210], 'confidence': 0.9}],
        [{'class': 'player', 'bbox': [120, 120, 220, 220], 'confidence': 0.9}],
        [],  # Occlusion
        [{'class': 'player', 'bbox': [140, 140, 240, 240], 'confidence': 0.9}],
    ]
    
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    expected_id = None
    
    for i, dets in enumerate(mock_frames_detections):
        active_tracks = tracker.update(dets, dummy_frame)
        
        logger.info(f"Frame {i+1}: Active tracks: {len(active_tracks)}")
        if active_tracks:
            for t in active_tracks:
                logger.info(f"  -> ID: {t['track_id']}, Class: {t['class']}, Frames Seen: {t['frames_seen']}, Hist Len: {len(t['history'])}")
                
                # Check consistency
                if expected_id is None:
                    expected_id = t['track_id']
                elif i != 3: # Frame 4 is occlusion, shouldn't reach here for frame 4 if no tracks returned, but ByteTrack might keep it alive!
                    assert t['track_id'] == expected_id, f"ID Switch detected! Expected {expected_id}, got {t['track_id']}"
                    
        else:
            logger.info("  -> No active tracks returned by tracker.")
            
    logger.info("Tracking consistency validation passed!")

if __name__ == "__main__":
    test_tracking_consistency()
