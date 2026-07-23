"""
Team classification module (Hybrid approach).
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from collections import defaultdict, deque
from utils import logger
from config import Config

class TeamClassifier:
    """
    Classifies players into teams using hybrid color grouping and K-Means fallback.
    """
    def __init__(self):
        logger.info("Initializing Improved TeamClassifier")
        self.grass_lower = np.array(Config.GRASS_HSV_LOWER)
        self.grass_upper = np.array(Config.GRASS_HSV_UPPER)
        
        self.training_frames = Config.TEAM_CLASSIFICATION_FRAMES
        self.frame_count = 0
        self.is_trained = False
        
        self.dark_threshold = 30 # Value channel
        self.color_dist_threshold = 0.3 # Configurable HSV distance
        self.confidence_threshold = 0.8
        
        # Cache per track_id
        # {track_id: {'colors': deque, 'dominant_color': HSV, 'team': str, 'confidence': float, 'history': deque}}
        self.track_data = {}
        
        self.team_a_color = None
        self.team_b_color = None

    def _get_dominant_color(self, crop):
        """Extracts dominant jersey color ignoring grass and dark pixels."""
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None
            
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        grass_mask = cv2.inRange(hsv_crop, self.grass_lower, self.grass_upper)
        dark_mask = (hsv_crop[:, :, 2] < self.dark_threshold).astype(np.uint8) * 255
        
        combined_mask = cv2.bitwise_or(grass_mask, dark_mask)
        non_bg_mask = cv2.bitwise_not(combined_mask)
        
        pixels = hsv_crop[non_bg_mask == 255]
        
        if len(pixels) < 10:
            pixels = hsv_crop.reshape(-1, 3)
            
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=1)
        kmeans.fit(pixels)
        
        labels = kmeans.labels_
        counts = np.bincount(labels)
        dominant_label = np.argmax(counts)
        dominant_color = kmeans.cluster_centers_[dominant_label]
        
        return dominant_color

    def _color_distance(self, c1, c2):
        """Calculates distance between two HSV colors."""
        dh = min(abs(c1[0] - c2[0]), 180 - abs(c1[0] - c2[0])) / 180.0
        ds = abs(c1[1] - c2[1]) / 255.0
        dv = abs(c1[2] - c2[2]) / 255.0
        return np.sqrt(dh**2 + ds**2 + dv**2)

    def _train_assignments(self):
        """Runs the hybrid rule-based + KMeans fallback clustering."""
        logger.info("Running hybrid Team Classification logic...")
        
        valid_tracks = {tid: data for tid, data in self.track_data.items() if data['dominant_color'] is not None}
        if len(valid_tracks) < 4:
            logger.warning("Not enough tracks to classify teams. Delaying...")
            self.training_frames += 10
            return
            
        colors = np.array([data['dominant_color'] for data in valid_tracks.values()])
        track_ids = list(valid_tracks.keys())
        
        # Step 3: Group Similar Colors using DBSCAN with precomputed distance matrix
        n = len(colors)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = self._color_distance(colors[i], colors[j])
                
        clustering = DBSCAN(eps=self.color_dist_threshold, min_samples=1, metric='precomputed')
        labels = clustering.fit_predict(dist_matrix)
        
        # Step 4: Rule-Based Assignment
        groups = defaultdict(list)
        for idx, label in enumerate(labels):
            groups[label].append(track_ids[idx])
            
        teams = []
        referee_candidates = []
        
        for label, tids in groups.items():
            if len(tids) > 2:
                teams.append(tids)
            elif len(tids) == 1:
                referee_candidates.extend(tids)
            else:
                # Group of size 2, maybe a small team or referees?
                # We will treat them as referee candidates for now if we already have 2 teams
                if len(teams) < 2:
                    teams.append(tids)
                else:
                    referee_candidates.extend(tids)
                
        # Mark referee candidates
        for tid in referee_candidates:
            self.track_data[tid]['team'] = 'Referee'
            self.track_data[tid]['confidence'] = 1.0
            
        # Step 5: K-Means Fallback
        non_ref_tids = [tid for tid in track_ids if self.track_data[tid]['team'] != 'Referee' and self.track_data[tid]['class'] != 'goalkeeper']
        non_ref_colors = np.array([self.track_data[tid]['dominant_color'] for tid in non_ref_tids])
        
        if len(teams) >= 2:
            # Pick the two largest teams
            teams.sort(key=len, reverse=True)
            team_a_tids = teams[0]
            team_b_tids = teams[1]
            self.team_a_color = np.mean([self.track_data[t]['dominant_color'] for t in team_a_tids], axis=0)
            self.team_b_color = np.mean([self.track_data[t]['dominant_color'] for t in team_b_tids], axis=0)
            
            for t in team_a_tids:
                self.track_data[t]['team'] = 'Team A'
                self.track_data[t]['confidence'] = 0.9
            for t in team_b_tids:
                self.track_data[t]['team'] = 'Team B'
                self.track_data[t]['confidence'] = 0.9
            logger.info("Rule-based team classification successful.")
        elif len(non_ref_colors) >= 2:
            logger.info("Rule-based classification failed to find exactly 2 teams. Using KMeans fallback.")
            kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
            kmeans.fit(non_ref_colors)
            self.team_a_color = kmeans.cluster_centers_[0]
            self.team_b_color = kmeans.cluster_centers_[1]
            
            for idx, tid in enumerate(non_ref_tids):
                label = kmeans.labels_[idx]
                team = "Team A" if label == 0 else "Team B"
                self.track_data[tid]['team'] = team
                self.track_data[tid]['confidence'] = 0.8
        else:
            logger.warning("Not enough non-referee tracks for KMeans fallback.")
            self.training_frames += 10
            return
            
        self.is_trained = True
        logger.info(f"Team A Color (HSV): {self.team_a_color}, Team B Color (HSV): {self.team_b_color}")

    def classify(self, frame, tracks):
        """
        Classifies the team for each tracked player.
        """
        self.frame_count += 1
        output = {}
        
        # 1 & 2: Extract color and maintain rolling average
        for track in tracks:
            track_class = track['class']
            track_id = track['track_id']
            
            if track_id not in self.track_data:
                self.track_data[track_id] = {
                    'colors': deque(maxlen=40),
                    'dominant_color': None,
                    'team': None,
                    'confidence': 0.0,
                    'history': deque(maxlen=10),
                    'class': track_class
                }
                
            x1, y1, x2, y2 = map(int, track['bbox'])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
            h = y2 - y1
            
            if h > 0 and (x2 - x1) > 0:
                crop = frame[y1:int(y1 + 0.4 * h), x1:x2]
                color = self._get_dominant_color(crop)
                if color is not None:
                    self.track_data[track_id]['colors'].append(color)
                    # Rolling average
                    avg_color = np.mean(self.track_data[track_id]['colors'], axis=0)
                    self.track_data[track_id]['dominant_color'] = avg_color
                    
            # Placeholder before assignment
            track['team'] = self.track_data[track_id]['team']

        # Train assignments after initial frames
        if not self.is_trained and self.frame_count >= self.training_frames:
            self._train_assignments()
            
        # Post-training: assign teams dynamically
        if self.is_trained:
            for track in tracks:
                track_id = track['track_id']
                t_data = self.track_data.get(track_id)
                if not t_data or t_data['dominant_color'] is None:
                    continue
                    
                # If YOLO detected as referee or previously assigned referee
                if track['class'] == 'referee' or t_data['team'] == 'Referee':
                    t_data['team'] = 'Referee'
                    t_data['confidence'] = 1.0
                    output[track_id] = {'team': 'Referee', 'dominant_color': t_data['dominant_color'], 'confidence': 1.0}
                    track['team'] = 'Referee'
                    continue
                    
                if track['class'] == 'goalkeeper':
                    d_a = self._color_distance(t_data['dominant_color'], self.team_a_color)
                    d_b = self._color_distance(t_data['dominant_color'], self.team_b_color)
                    assigned_team = "Team A" if d_a < d_b else "Team B"
                    t_data['team'] = assigned_team
                    t_data['confidence'] = 0.9
                    output[track_id] = {'team': assigned_team, 'dominant_color': t_data['dominant_color'], 'confidence': 0.9}
                    track['team'] = assigned_team
                    continue

                # Player assignment
                d_a = self._color_distance(t_data['dominant_color'], self.team_a_color)
                d_b = self._color_distance(t_data['dominant_color'], self.team_b_color)
                
                # Softmax confidence
                conf_a = 1.0 / (d_a + 1e-5)
                conf_b = 1.0 / (d_b + 1e-5)
                confidence = max(conf_a, conf_b) / (conf_a + conf_b)
                
                current_team = "Team A" if d_a < d_b else "Team B"
                
                # Temporal consistency
                t_data['history'].append(current_team)
                history_list = list(t_data['history'])
                most_common = max(set(history_list), key=history_list.count)
                
                if t_data['team'] is None or confidence > self.confidence_threshold or history_list.count(most_common) > 7:
                    t_data['team'] = most_common
                    t_data['confidence'] = confidence
                    
                output[track_id] = {
                    'team': t_data['team'],
                    'dominant_color': t_data['dominant_color'],
                    'confidence': t_data['confidence']
                }
                track['team'] = t_data['team']
                
        # To satisfy requirements, we return the output dictionary.
        # Ensure that `main.py` handles this.
        return output

    def get_team_color_name(self, team_name):
        """
        Returns a human-readable color name for the specified team based on its average HSV color.
        Thresholds are tuned for real football jersey colors under typical field lighting.
        """
        if not self.is_trained:
            return ""
            
        color = self.team_a_color if team_name == 'Team A' else self.team_b_color
        if color is None:
            return ""
            
        h, s, v = color
        
        # Black: very low brightness
        if v < 70: return "Black"
        # White: low saturation AND decent brightness
        # Relaxed thresholds because white jerseys appear off-white/grey under field lighting
        if s < 90 and v > 150: return "White"
        # Grey: low saturation but not bright enough for white
        if s < 90 and v <= 150: return "Grey"
        
        # Chromatic colors — use hue
        if h < 10 or h > 160: return "Red"
        if h < 25: return "Orange"
        if h < 35: return "Yellow"
        if h < 85: return "Green"
        if h < 130: return "Blue"
        if h < 160: return "Purple"
        return "Unknown"
