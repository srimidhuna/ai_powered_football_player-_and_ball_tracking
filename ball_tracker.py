"""
Ball tracking module.
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from config import Config
from utils import logger

class BallTracker:
    """
    Specialized tracking and trajectory prediction for the football.
    """
    def __init__(self):
        """Initializes the ball tracker."""
        logger.info("Initializing BallTracker")
        
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # State: [x, y, dx, dy]
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        
        self.kf.R *= Config.BALL_KF_MEASUREMENT_NOISE
        self.kf.Q *= Config.BALL_KF_PROCESS_NOISE
        
        self.is_initialized = False
        self.history = [] # Stores dicts: {'bbox': [x1,y1,x2,y2], 'status': 'detected'|'predicted'|'interpolated'}
        self.last_detection = None # (cx, cy)
        self.last_detection_frame_idx = -1
        self.current_frame_idx = -1
        
        # New: counter to track consecutive missing frames
        self.missing_frames = 0
        
    def _get_center(self, bbox):
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        
    def _get_bbox(self, cx, cy, w, h):
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    def update(self, ball_detections, frame):
        """
        Updates the ball position.
        
        Args:
            ball_detections: Detections for the ball.
            frame: The current image/frame.
            
        Returns:
            dict or None: Updated ball track with keys 'bbox' and 'status'.
        """
        self.current_frame_idx += 1
        
        best_detection = None
        
        # 1. Detect ball and associate across frames (Rule 1 is handled in main loop)
        if ball_detections:
            if not self.is_initialized:
                best_detection = max(ball_detections, key=lambda d: d['confidence'])
            else:
                # Pick the one closest to Kalman prediction
                pred_x, pred_y = self.kf.x[0, 0], self.kf.x[1, 0]
                min_dist = float('inf')
                for d in ball_detections:
                    cx, cy = self._get_center(d['bbox'])
                    dist = np.hypot(cx - pred_x, cy - pred_y)
                    if dist < min_dist:
                        min_dist = dist
                        best_detection = d
                
                # Rule 2: If a valid ball detection exists, ALWAYS use it.
                # (Removed jump > Config.BALL_MAX_JUMP rejection logic)
                        
        # 2. Process Detection or Predict Missing
        if best_detection is not None:
            bbox = best_detection['bbox']
            cx, cy = self._get_center(bbox)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Rule 7: If ball is detected again, immediately discard the old prediction
            # and continue from the new detection. (Re-initialize KF state if there were missing frames)
            if not self.is_initialized or self.missing_frames > 0:
                self.kf.x = np.array([[cx], [cy], [0], [0]])
                self.is_initialized = True
            else:
                self.kf.predict()
                # Rule 3: Update the Kalman Filter with the detected position.
                self.kf.update([cx, cy])
                
            # Rule 4: Reset the missing frame counter.
            self.missing_frames = 0
            
            # Fill short gaps using linear interpolation
            gap = self.current_frame_idx - self.last_detection_frame_idx - 1
            if 0 < gap <= Config.BALL_MAX_GAP and self.last_detection_frame_idx != -1:
                start_hist_idx = self.last_detection_frame_idx
                end_hist_idx = self.current_frame_idx
                
                start_bbox = self.history[start_hist_idx]['bbox']
                end_bbox = bbox
                
                for i in range(1, gap + 1):
                    alpha = i / (gap + 1)
                    interp_bbox = [
                        start_bbox[0] + alpha * (end_bbox[0] - start_bbox[0]),
                        start_bbox[1] + alpha * (end_bbox[1] - start_bbox[1]),
                        start_bbox[2] + alpha * (end_bbox[2] - start_bbox[2]),
                        start_bbox[3] + alpha * (end_bbox[3] - start_bbox[3])
                    ]
                    self.history[start_hist_idx + i] = {'bbox': interp_bbox, 'status': 'interpolated'}
                    
            self.last_detection = (cx, cy)
            self.last_detection_frame_idx = self.current_frame_idx
            
            # Store in history
            # Rule 8: Do not draw predicted positions as detected balls. 
            # (Use raw detection bounding box here, not KF smoothed state).
            current_track = {'bbox': bbox, 'status': 'detected'}
            self.history.append(current_track)
            
        else:
            # Missing detection
            if self.is_initialized:
                self.missing_frames += 1
                
                # Rule 6: Stop predicting after 5 consecutive missed frames.
                if self.missing_frames <= 5:
                    self.kf.predict()
                    pred_cx, pred_cy = self.kf.x[0, 0], self.kf.x[1, 0]
                    
                    if self.last_detection_frame_idx != -1:
                        last_bbox = self.history[self.last_detection_frame_idx]['bbox']
                        w, h = last_bbox[2] - last_bbox[0], last_bbox[3] - last_bbox[1]
                    else:
                        w, h = 10, 10
                        
                    pred_bbox = self._get_bbox(pred_cx, pred_cy, w, h)
                    # Rule 5: Use Kalman prediction ONLY when the ball is not detected.
                    current_track = {'bbox': pred_bbox, 'status': 'predicted'}
                    self.history.append(current_track)
                else:
                    self.is_initialized = False # Give up tracking until a new detection occurs
                    current_track = None
                    self.history.append(current_track)
            else:
                current_track = None
                self.history.append(current_track)
                
        return current_track
