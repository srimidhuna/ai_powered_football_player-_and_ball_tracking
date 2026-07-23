import cv2
import urllib.request
import os
from detector import Detector
from utils import logger

def main():
    video_path = "test_vid.mp4"
    if not os.path.exists(video_path):
        logger.info("Downloading test video...")
        url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
        try:
            urllib.request.urlretrieve(url, video_path)
        except Exception as e:
            logger.error(f"Failed to download test video: {e}")
            return
            
    logger.info("Initializing detector...")
    detector = Detector()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video")
        return
        
    logger.info("Running detection on 5 frames...")
    
    frame_count = 0
    while frame_count < 5:
        ret, frame = cap.read()
        if not ret:
            break
            
        detections = detector.detect(frame)
        logger.info(f"Frame {frame_count}: Found {len(detections)} detections")
        for d in detections[:3]: # Log up to 3 detections per frame to avoid spam
            logger.info(f"  -> {d['class']} ({d['confidence']:.2f}) at {d['bbox']}")
            
        frame_count += 1
        
    cap.release()
    logger.info("Validation complete.")

if __name__ == "__main__":
    main()
