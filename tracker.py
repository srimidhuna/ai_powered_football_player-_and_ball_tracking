"""
Multi-object tracking module.
"""
import numpy as np
import supervision as sv
from utils import logger
from config import Config

class Tracker:
    """
    Handles multi-object tracking across frames.
    """
    def __init__(self):
        """Initializes the tracker."""
        logger.info(f"Initializing Tracker with {Config.TRACKER_TYPE}")
        
        self.tracker_type = Config.TRACKER_TYPE
        
        if self.tracker_type == "botsort":
            logger.warning("BoT-SORT not natively exposed cleanly in supervision yet. Falling back to ByteTrack.")
        
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
            minimum_consecutive_frames=1
        )
        
        # State: track_id -> dict
        # { 'class': str, 'bbox': list, 'confidence': float, 'history': list, 'frames_seen': int }
        self.tracks = {}
        self.history_limit = Config.TRACK_HISTORY_LIMIT
        
    def update(self, detections, frame):
        """
        Updates tracks based on new detections.
        
        Args:
            detections: List of dictionary detection objects:
                        [{'class': str, 'bbox': [x1, y1, x2, y2], 'confidence': float}]
            frame: The current image/frame.
            
        Returns:
            list: Updated tracks:
                  [{'track_id': int, 'class': str, 'bbox': [x1, y1, x2, y2], 'confidence': float, 'history': list, 'frames_seen': int}]
        """
        if not detections:
            # Just tick the tracker with empty detections to age out lost tracks
            # supervision expects a Detections object, even if empty
            sv_detections = sv.Detections.empty()
        else:
            # Convert our detections to supervision Detections object
            xyxy = np.array([d['bbox'] for d in detections])
            confidence = np.array([d['confidence'] for d in detections])
            
            # We map class strings to ints temporarily for supervision to process them
            # Let's just create a quick mapping for current classes
            class_map = {'player': 0, 'referee': 1, 'goalkeeper': 2, 'ball': 3}
            # Fallback for unexpected classes just to 0
            class_ids = np.array([class_map.get(d['class'], 0) for d in detections])
            
            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_ids
            )
            
            # Keep original class string in data so we can map back
            sv_detections.data['class_name'] = np.array([d['class'] for d in detections])

        # Update tracker
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        # We need to manage our persistent state (history, frames_seen, etc.)
        current_frame_track_ids = set()
        
        for i in range(len(tracked_detections)):
            track_id = tracked_detections.tracker_id[i]
            bbox = tracked_detections.xyxy[i].tolist()
            conf = tracked_detections.confidence[i]
            # Retrieve the original class name we passed in, if available.
            # If not (e.g. tracker interpolates), we might have to use class_id
            if 'class_name' in tracked_detections.data:
                class_name = tracked_detections.data['class_name'][i]
            else:
                # Reverse lookup just in case
                reverse_map = {0: 'player', 1: 'referee', 2: 'goalkeeper', 3: 'ball'}
                cls_id = tracked_detections.class_id[i]
                class_name = reverse_map.get(cls_id, 'player')
                
            current_frame_track_ids.add(track_id)
            
            # Compute center of bbox for history
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            center = (cx, cy)
            
            if track_id not in self.tracks:
                # New track
                self.tracks[track_id] = {
                    'track_id': track_id,
                    'class': class_name,
                    'bbox': bbox,
                    'confidence': conf,
                    'history': [center],
                    'frames_seen': 1
                }
            else:
                # Update existing track
                state = self.tracks[track_id]
                state['bbox'] = bbox
                state['confidence'] = conf
                state['frames_seen'] += 1
                state['history'].append(center)
                
                # Truncate history if needed
                if len(state['history']) > self.history_limit:
                    state['history'].pop(0)
                    
        # Return only the active tracks for this frame
        active_tracks = []
        for tid in current_frame_track_ids:
            active_tracks.append(self.tracks[tid])
            
        return active_tracks
