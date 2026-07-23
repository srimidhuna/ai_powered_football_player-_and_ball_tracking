"""
Object detection module.
"""
import os
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from config import Config
from utils import logger

def compute_iou(box1, box2):
    """Computes IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

class Detector:
    """
    Handles multi-object detection (e.g., players, referees, ball).
    """
    def __init__(self):
        """Initializes the detector."""
        logger.info("Initializing Detector")
        
        # Determine which weights to use
        if os.path.exists(Config.YOLO_MODEL_PATH):
            model_path = Config.YOLO_MODEL_PATH
            logger.info(f"Using football-specific weights: {model_path}")
            self.is_custom = True
        else:
            model_path = Config.YOLO_FALLBACK_MODEL
            logger.warning(f"Football-specific weights not found. Falling back to {model_path}")
            self.is_custom = False
            
        # Load model and move to configured device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = True if self.device == 'cuda' else False
        logger.info(f"Detector using device: {self.device}, FP16: {self.half}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Configure thresholds
        self.conf_player = Config.CONF_PLAYER
        self.conf_referee = Config.CONF_REFEREE
        self.conf_goalkeeper = Config.CONF_GOALKEEPER
        self.conf_ball = Config.CONF_BALL
        
        # Player validation state
        self.candidates = {} # {cand_id: {'bbox': list, 'frames': int, 'history': deque}}
        self.next_cand_id = 0
        self.grass_lower = np.array(Config.GRASS_HSV_LOWER)
        self.grass_upper = np.array(Config.GRASS_HSV_UPPER)

    def _get_center(self, bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        
    def detect(self, frame):
        """
        Detects objects in a single frame.
        
        Args:
            frame: The input image/frame (numpy array).
            
        Returns:
            list: List of dictionary detection objects:
                  [{'class': str, 'bbox': [x1, y1, x2, y2], 'confidence': float}]
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to detector.")
            return []
            
        # Generate pitch mask (Rule 4 & 5)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pitch_mask = cv2.inRange(hsv, self.grass_lower, self.grass_upper)
            
        # Run inference (disable tracking, standard detection)
        with torch.no_grad():
            results = self.model.predict(
                frame, 
                conf=0.1, 
                verbose=False, 
                device=self.device, 
                half=self.half
            )[0]
        
        raw_detections = []
        
        # Parse results
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            class_name = None
            if self.is_custom:
                if cls_id == 0 and conf >= self.conf_player:
                    class_name = 'player'
                elif cls_id == 1 and conf >= self.conf_referee:
                    class_name = 'referee'
                elif cls_id == 2 and conf >= self.conf_goalkeeper:
                    class_name = 'goalkeeper'
                elif cls_id == 3 and conf >= self.conf_ball:
                    class_name = 'ball'
            else:
                if cls_id == Config.COCO_PERSON_CLASS and conf >= self.conf_player:
                    class_name = 'player'
                elif cls_id == Config.COCO_BALL_CLASS and conf >= self.conf_ball:
                    class_name = 'ball'
                    
            if class_name:
                raw_detections.append({
                    'class': class_name,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
                
        # Validate detections
        valid_detections = []
        current_candidates = []
        img_h, img_w = frame.shape[:2]
        
        for det in raw_detections:
            if det['class'] != 'player':
                valid_detections.append(det)
                continue
                
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            area = w * h
            aspect_ratio = h / w if w > 0 else 0
            
            # Rule 2: Area
            if not (Config.PLAYER_MIN_AREA <= area <= Config.PLAYER_MAX_AREA):
                continue
                
            # Rule 3: Aspect ratio
            if not (Config.PLAYER_MIN_ASPECT_RATIO <= aspect_ratio <= Config.PLAYER_MAX_ASPECT_RATIO):
                continue
                
            # Rule 7: Reject near top broadcast region
            if y2 < img_h * Config.BROADCAST_TOP_REGION:
                continue
                
            # Rule 4 & 5: Bottom lies on the pitch (overlaps pitch mask)
            # Check a small region at the bottom center of the bounding box
            bx = int((x1 + x2) / 2)
            by = int(y2)
            # Clamp coordinates
            bx = max(0, min(img_w - 1, bx))
            by = max(0, min(img_h - 1, by))
            
            # Check a small 5x5 window around the bottom center
            y_start = max(0, by - 5)
            y_end = min(img_h, by + 5)
            x_start = max(0, bx - 5)
            x_end = min(img_w, bx + 5)
            
            mask_roi = pitch_mask[y_start:y_end, x_start:x_end]
            if np.count_nonzero(mask_roi) == 0:
                continue
                
            current_candidates.append(det)
            
        # Match current candidates to previous candidates using IoU
        matched_cands = {} # new_cand_id -> old_cand_id
        unmatched_current = list(range(len(current_candidates)))
        
        if self.candidates:
            # Create distance matrix (1 - IoU)
            iou_matrix = np.zeros((len(current_candidates), len(self.candidates)))
            old_ids = list(self.candidates.keys())
            
            for i, cand in enumerate(current_candidates):
                for j, old_id in enumerate(old_ids):
                    iou_matrix[i, j] = compute_iou(cand['bbox'], self.candidates[old_id]['bbox'])
                    
            # Greedy matching
            while True:
                if iou_matrix.size == 0 or np.max(iou_matrix) < 0.3:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                old_id = old_ids[j]
                matched_cands[i] = old_id
                
                # Zero out row and col
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1
                if i in unmatched_current:
                    unmatched_current.remove(i)
                    
        # Update state
        new_state = {}
        for i, cand in enumerate(current_candidates):
            cx, cy = self._get_center(cand['bbox'])
            
            if i in matched_cands:
                cand_id = matched_cands[i]
                state = self.candidates[cand_id]
                state['frames'] += 1
                state['bbox'] = cand['bbox']
                state['history'].append((cx, cy))
                new_state[cand_id] = state
            else:
                cand_id = self.next_cand_id
                self.next_cand_id += 1
                new_state[cand_id] = {
                    'bbox': cand['bbox'],
                    'frames': 1,
                    'history': deque([(cx, cy)], maxlen=Config.PLAYER_STATIC_FRAMES)
                }
                
            state = new_state[cand_id]
            
            # Rule 6: Persistence
            if state['frames'] >= Config.PLAYER_PERSISTENCE_FRAMES:
                # Rule 8: Reject static objects
                if len(state['history']) >= Config.PLAYER_STATIC_FRAMES:
                    pts = np.array(state['history'])
                    max_dist = np.max(np.sqrt(np.sum((pts - pts[0])**2, axis=1)))
                    if max_dist < Config.PLAYER_MIN_MOTION_PX:
                        continue # Static object, skip
                
                # If passed, add to valid detections
                valid_detections.append(cand)
                
        self.candidates = new_state
        return valid_detections
