"""
Referee Detection module.

Runs BEFORE the K-Means team classifier. Checks every player track's
upper-body crop for a predominantly black jersey. If the fraction of
dark/black pixels exceeds Config.REFEREE_BLACK_THRESHOLD the track is
immediately labelled "Referee" and excluded from team clustering.

No changes to TeamClassifier, Tracker, or any other module are needed.
"""
import cv2
import numpy as np
from config import Config
from utils import logger


class RefereeDetector:
    """
    Lightweight black-jersey referee filter.

    Usage (in main.py, BEFORE TeamClassifier.classify):
        ref_detector = RefereeDetector()
        ...
        for track in player_tracks:
            if ref_detector.is_referee(frame, track):
                track['class'] = 'referee'
                track['team']  = 'Referee'
    """

    def __init__(self):
        # Grass HSV mask — reuse existing config values so we strip grass the
        # same way TeamClassifier does, keeping colour analysis consistent.
        self.grass_lower = np.array(Config.GRASS_HSV_LOWER)
        self.grass_upper = np.array(Config.GRASS_HSV_UPPER)

        # Black-pixel thresholds (all configurable in Config)
        self.black_v_max = Config.REFEREE_BLACK_V_MAX   # max Value for "black"
        self.black_s_max = Config.REFEREE_BLACK_S_MAX   # max Saturation for "black"
        self.black_threshold = Config.REFEREE_BLACK_THRESHOLD  # min dark fraction

        logger.info(
            f"RefereeDetector initialised — "
            f"black_threshold={self.black_threshold}, "
            f"V_max={self.black_v_max}, S_max={self.black_s_max}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_referee(self, frame: np.ndarray, track: dict) -> bool:
        """
        Returns True if the track's upper-body is predominantly black.

        Args:
            frame : Full BGR video frame.
            track : Track dict with keys 'bbox', 'class', 'track_id'.

        Returns:
            bool — True → classify as Referee, False → send to K-Means.
        """
        # Referees identified by YOLO model label are always referees
        if track.get('class') == 'referee':
            return True

        dark_fraction = self._black_fraction(frame, track['bbox'])
        if dark_fraction is None:
            return False

        return dark_fraction >= self.black_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _black_fraction(self, frame: np.ndarray, bbox) -> float | None:
        """
        Computes the fraction of dark (black) pixels in the upper-body crop.

        Returns:
            float in [0, 1], or None if the crop is too small to analyse.
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Safety clamp to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        h = y2 - y1
        w = x2 - x1

        if h < 10 or w < 5:
            return None

        # Crop the upper 40 % of the bounding box (jersey region only).
        # This matches the exact same crop used in TeamClassifier._get_dominant_color,
        # so both modules look at the same jersey area.
        jersey_y2 = y1 + int(0.4 * h)
        crop = frame[y1:jersey_y2, x1:x2]

        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Remove grass pixels (same mask as TeamClassifier)
        grass_mask = cv2.inRange(hsv, self.grass_lower, self.grass_upper)
        non_grass = grass_mask == 0

        total_pixels = int(np.sum(non_grass))
        if total_pixels < 10:
            # Too few non-grass pixels — skip, not enough information
            return None

        # Black pixel: low Value AND low Saturation.
        # The Saturation guard prevents dark-green pitch reflections from
        # being counted as black jersey pixels.
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        black_mask = (
            (v_channel < self.black_v_max) &
            (s_channel < self.black_s_max) &
            non_grass
        )

        dark_pixels = int(np.sum(black_mask))
        return dark_pixels / total_pixels
