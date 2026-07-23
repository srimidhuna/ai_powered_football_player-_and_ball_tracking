"""
Main execution script for the football tracker.
"""
import argparse
import sys
import cv2
import time
import os
from rfdetr import RFDETRLarge
from config import Config
from utils import logger
from detector import Detector
from tracker import Tracker
from team_classifier import TeamClassifier
from referee_detector import RefereeDetector
from ball_tracker import BallTracker
from trajectory import Trajectory
from possession import Possession
from visualizer import Visualizer

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Football Analytics System")
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output/output.mp4', help='Path to output video')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting football tracker with input: {args.input}")
    Config.print_config()
    
    if not os.path.exists(args.input):
        logger.error(f"Input video not found: {args.input}")
        sys.exit(1)
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Initialize RF-DETR Large once at startup (do NOT move inside frame loop)
    ball_model = RFDETRLarge()
    print("[INFO] RF-DETR Large initialized successfully.")
    
    # Initialize all modules
    detector = Detector()
    tracker = Tracker()
    team_classifier = TeamClassifier()
    referee_detector = RefereeDetector()
    ball_tracker = BallTracker()
    trajectory = Trajectory()
    possession = Possession()
    visualizer = Visualizer()
    
    # Open video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {args.input}")
        sys.exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    frame_idx = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        t0 = time.time()
        
        # 1. Detection
        detections = detector.detect(frame)
        
        # 1b. RF-DETR ball detection (sports ball only)
        # Convert BGR frame to RGB as required by RF-DETR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run RF-DETR inference — returns a supervision Detections object
        rfdetr_detections = ball_model.predict(rgb, threshold=0.3)
        
        # Immediately discard every class except 'sports ball'
        # class_name is stored as a numpy array in detections.data["class_name"]
        rfdetr_ball_detections = []
        if rfdetr_detections is not None and len(rfdetr_detections) > 0:
            class_names = rfdetr_detections.data.get("class_name", [])
            for i, cls_name in enumerate(class_names):
                if cls_name == "sports ball":
                    x1, y1, x2, y2 = rfdetr_detections.xyxy[i].tolist()
                    conf = float(rfdetr_detections.confidence[i])
                    rfdetr_ball_detections.append({
                        'class': 'ball',
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf
                    })
        
        # Log RF-DETR ball detections at debug level (avoids stdout flood in production)
        if rfdetr_ball_detections:
            logger.debug(f"[RF-DETR] Frame {frame_idx}: {len(rfdetr_ball_detections)} ball(s) detected")
        
        # Split YOLO detections — keep players/goalkeeper/referee ONLY, discard YOLO ball
        player_ref_detections = [d for d in detections if d['class'] in ['player', 'referee', 'goalkeeper']]
        # Ball class from YOLO is intentionally excluded here
        
        # Merge: YOLO (player/goalkeeper/referee) + RF-DETR (ball only)
        merged_detections = player_ref_detections + rfdetr_ball_detections
        logger.debug(
            f"Frame {frame_idx}: YOLO players={len(player_ref_detections)}, "
            f"RF-DETR balls={len(rfdetr_ball_detections)}, "
            f"merged={len(merged_detections)}"
        )
        
        # 2. Tracking — ByteTrack receives merged detections (YOLO players + RF-DETR ball)
        # ByteTrack assigns tracking IDs to both players and the football.
        # ByteTrack itself is NOT modified.
        tracks = tracker.update(merged_detections, frame)
        
        # Separate ball tracks from player tracks after ByteTrack assignment.
        # Ball tracks must NOT be passed to TeamClassifier (corrupts jersey color model)
        # or the player-drawing loop in Visualizer (would draw a duplicate box over the ball circle).
        player_tracks = [t for t in tracks if t['class'] in ['player', 'referee', 'goalkeeper']]

        # 3a. Referee Detection — runs BEFORE K-Means (black jersey check).
        # Any track whose upper-body crop is predominantly black is forced to
        # class='referee' / team='Referee' and excluded from TeamClassifier.
        # YOLO-labelled referees are also captured here for consistency.
        for track in player_tracks:
            if referee_detector.is_referee(frame, track):
                track['class'] = 'referee'
                track['team']  = 'Referee'

        # 3b. Team Classification — receives ONLY non-referee tracks.
        # K-Means still uses exactly 2 clusters; the referee is simply never fed in.
        non_referee_tracks = [t for t in player_tracks if t['class'] != 'referee' and t.get('team') != 'Referee']
        team_assignments = team_classifier.classify(frame, non_referee_tracks)
        
        # 4. Ball Tracking — uses RF-DETR ball detections exclusively (no YOLO ball)
        ball_track = ball_tracker.update(rfdetr_ball_detections, frame)
        
        # 5. Possession Estimation — player_tracks only; ball_track is the RF-DETR result
        possession.update(player_tracks, ball_track)
        possession_stats = possession.get_percentages()
        
        # 6. Render Trajectories (Trails) — player tracks only; ball has its own renderer
        frame = trajectory.draw(frame, player_tracks)
        
        # 7. Render Final Visuals — pass player_tracks so the player bbox loop in
        # Visualizer.draw() does not draw a redundant rectangle over the ball circle.
        proc_fps = 1.0 / (time.time() - t0)
        team_a_name = team_classifier.get_team_color_name("Team A")
        team_b_name = team_classifier.get_team_color_name("Team B")
        frame = visualizer.draw(frame, player_tracks, ball_track, possession_stats, frame_idx, proc_fps, team_a_name, team_b_name)
        
        # Write frame
        writer.write(frame)
        
        if frame_idx % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Processed {frame_idx}/{total_frames} frames. Avg FPS: {frame_idx/elapsed:.1f}")
            
    cap.release()
    writer.release()
    
    # Complete Performance Summary
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    
    logger.info("="*50)
    logger.info("--- PERFORMANCE SUMMARY ---")
    logger.info(f"Total Frames Processed : {frame_idx}")
    logger.info(f"Total Processing Time  : {total_time:.2f} seconds")
    logger.info(f"Average Pipeline FPS   : {avg_fps:.1f} FPS")
    logger.info(f"Resolution             : {width}x{height}")
    logger.info(f"Output Video           : {args.output}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
