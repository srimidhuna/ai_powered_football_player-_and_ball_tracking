"""
Configuration parameters for the football tracker project.
"""
import torch

class Config:
    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model configuration
    YOLO_MODEL_PATH = "weights/football_yolo.pt"
    YOLO_FALLBACK_MODEL = "yolov8m.pt"
    
    # Confidence thresholds
    CONF_PLAYER = 0.3
    CONF_REFEREE = 0.3
    CONF_GOALKEEPER = 0.3
    CONF_BALL = 0.2
    
    # Class mapping for COCO (fallback)
    COCO_PERSON_CLASS = 0
    COCO_BALL_CLASS = 32
    
    # Tracker configuration
    TRACKER_TYPE = "bytetrack" # Options: 'bytetrack', 'botsort'
    TRACK_HISTORY_LIMIT = 30 # Number of frames to keep in history
    
    # Player Validation configuration
    PLAYER_MIN_AREA = 100
    PLAYER_MAX_AREA = 25000
    PLAYER_MIN_ASPECT_RATIO = 0.8
    PLAYER_MAX_ASPECT_RATIO = 4.0
    BROADCAST_TOP_REGION = 0.1
    PLAYER_PERSISTENCE_FRAMES = 3
    PLAYER_STATIC_FRAMES = 15
    PLAYER_MIN_MOTION_PX = 5.0
    
    # Team Classifier configuration
    TEAM_CLASSIFICATION_FRAMES = 30
    GRASS_HSV_LOWER = [35, 40, 40]
    GRASS_HSV_UPPER = [85, 255, 255]

    # Referee Detection configuration
    # Fraction of upper-body pixels that must be dark/black to classify as Referee.
    # Increase if real players are being misclassified; decrease if referees are missed.
    REFEREE_BLACK_THRESHOLD = 0.45
    # HSV Value upper bound — pixels below this are considered "dark"
    REFEREE_BLACK_V_MAX = 80
    # HSV Saturation upper bound — prevents dark-green pitch reflections being counted
    REFEREE_BLACK_S_MAX = 80
    
    # Ball Tracker configuration
    BALL_MAX_GAP = 5
    BALL_MAX_JUMP = 150
    BALL_KF_MEASUREMENT_NOISE = 1.0
    BALL_KF_PROCESS_NOISE = 0.1
    
    # Trajectory configuration
    TRAJECTORY_LENGTH = 30
    TEAM_A_COLOR = (0, 0, 255) # Red in BGR
    TEAM_B_COLOR = (255, 0, 0) # Blue in BGR
    UNKNOWN_COLOR = (200, 200, 200) # Gray
    
    # Possession configuration
    POSSESSION_MAX_DISTANCE = 50 # Max pixels between foot and ball
    
    # Visualization configuration
    # (To be filled in later phases)

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"Using device: {cls.DEVICE}")
