import numpy as np
import cv2
from team_classifier import TeamClassifier
from utils import logger
from config import Config

def test_team_classification():
    logger.info("Initializing team classifier test...")
    
    # Override training frames for quick test
    classifier = TeamClassifier()
    classifier.training_frames = 5
    
    # Create a dummy frame with "grass"
    # HSV for grass in config is 35-85. We use Green: BGR(0, 255, 0) -> HSV(60, 255, 255)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:] = (0, 255, 0) # Green grass
    
    # Let's draw some players on the grass
    # Player 1 (Red jersey): bbox [10, 10, 50, 90]
    frame[10:90, 10:50] = (0, 0, 255) # Red in BGR
    # Player 2 (Blue jersey): bbox [110, 10, 150, 90]
    frame[10:90, 110:150] = (255, 0, 0) # Blue in BGR
    
    # Note: Upper 40% will be from y=10 to y=42. Both are filled with solid colors.
    
    tracks = [
        {'track_id': 1, 'class': 'player', 'bbox': [10, 10, 50, 90]},
        {'track_id': 2, 'class': 'player', 'bbox': [110, 10, 150, 90]},
        {'track_id': 3, 'class': 'referee', 'bbox': [60, 10, 100, 90]} # Should be ignored
    ]
    
    # Run for 6 frames (5 training + 1 post-training)
    for i in range(1, 7):
        logger.info(f"--- Frame {i} ---")
        out_tracks = classifier.classify(frame, tracks)
        
        for t in out_tracks:
            team = t.get('team')
            logger.info(f"  Track {t['track_id']} ({t['class']}): Team = {team}")
            
    # Validate final assignments
    p1_team = out_tracks[0]['team']
    p2_team = out_tracks[1]['team']
    ref_team = out_tracks[2]['team']
    
    assert p1_team is not None and p2_team is not None, "Players were not assigned a team!"
    assert p1_team != p2_team, "Players with completely different colors were assigned the SAME team!"
    assert ref_team is None, "Referee was incorrectly assigned a team!"
    
    logger.info("Team Classifier Validation Passed! Teams were correctly identified and separated.")

if __name__ == "__main__":
    test_team_classification()
