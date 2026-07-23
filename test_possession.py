from possession import Possession
from config import Config
from utils import logger

def test_possession():
    logger.info("Initializing possession test...")
    possession_tracker = Possession()
    
    # Simulate Frame 1: Team A is close to the ball
    # Ball is at center [100, 100]
    ball_track = {'bbox': [90, 90, 110, 110], 'status': 'detected'}
    
    # Team A player foot: center=100, y_bottom=100 -> dist = 0
    # Team B player foot: center=200, y_bottom=200 -> dist = 141
    player_tracks_1 = [
        {'track_id': 1, 'class': 'player', 'team': 'Team A', 'bbox': [80, 50, 120, 100]},
        {'track_id': 2, 'class': 'player', 'team': 'Team B', 'bbox': [180, 150, 220, 200]}
    ]
    
    # Simulate Frame 2: Team A is still close
    player_tracks_2 = player_tracks_1
    
    # Simulate Frame 3: Team B is close, Team A is far
    # Team B foot: center=105, y_bottom=105 -> dist ~ 7
    # Team A foot: center=200, y_bottom=200 -> dist = 141
    player_tracks_3 = [
        {'track_id': 1, 'class': 'player', 'team': 'Team A', 'bbox': [180, 150, 220, 200]},
        {'track_id': 2, 'class': 'player', 'team': 'Team B', 'bbox': [85, 55, 125, 105]}
    ]
    
    # Simulate Frame 4: No one is close (both > POSSESSION_MAX_DISTANCE=50)
    # Ball at [500, 500]
    ball_track_far = {'bbox': [490, 490, 510, 510], 'status': 'detected'}
    
    # Update and print
    logger.info("Frame 1: Team A has ball")
    stats = possession_tracker.update(player_tracks_1, ball_track)
    assert stats == {'Team A': 100.0, 'Team B': 0.0}, f"Expected 100/0, got {stats}"
    
    logger.info("Frame 2: Team A has ball")
    stats = possession_tracker.update(player_tracks_2, ball_track)
    assert stats == {'Team A': 100.0, 'Team B': 0.0}, f"Expected 100/0, got {stats}"
    
    logger.info("Frame 3: Team B has ball")
    stats = possession_tracker.update(player_tracks_3, ball_track)
    # Total frames: 2 for A, 1 for B. A=66.7%, B=33.3%
    assert stats == {'Team A': 66.7, 'Team B': 33.3}, f"Expected 66.7/33.3, got {stats}"
    
    logger.info("Frame 4: Nobody has ball (too far)")
    stats = possession_tracker.update(player_tracks_3, ball_track_far)
    # Should not increment any frames. Stats remain same.
    assert stats == {'Team A': 66.7, 'Team B': 33.3}, f"Expected unchanged stats, got {stats}"
    
    logger.info("Possession Validation Passed! Calculation, assignment, and thresholds work perfectly.")

if __name__ == "__main__":
    test_possession()
