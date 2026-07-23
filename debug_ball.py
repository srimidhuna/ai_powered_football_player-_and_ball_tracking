import cv2
from detector import Detector
from ball_tracker import BallTracker

def main():
    detector = Detector()
    ball_tracker = BallTracker()
    
    cap = cv2.VideoCapture("test_vid.mp4")
    if not cap.isOpened():
        print("Failed to open test_vid.mp4")
        return
        
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        results = detector.model.predict(frame, conf=0.01, verbose=False)[0]
        raw_det_details = ", ".join([f"{int(box.cls[0].item())}({float(box.conf[0].item()):.2f})" for box in results.boxes])
        if not raw_det_details:
            raw_det_details = "None"
            
        detections = detector.detect(frame)
        ball_detections = [d for d in detections if d['class'] == 'ball']
        ball_track = ball_tracker.update(ball_detections, frame)
        
        num_detections = len(ball_detections)
        det_details = ", ".join([f"[Conf: {d['confidence']:.2f}, BBox: {[int(x) for x in d['bbox']]}]" for d in ball_detections])
        if not det_details:
            det_details = "None"
            
        used_status = ball_track['status'] if ball_track else "None"
        
        if num_detections > 0 or "32(" in raw_det_details:
            print(f"Frame {frame_idx:03d} | Detections: {num_detections} | Details: {det_details} | Tracker: {used_status} | Raw: {raw_det_details}")
        elif frame_idx % 100 == 0:
            print(f"Frame {frame_idx:03d} | No ball detected")

if __name__ == "__main__":
    main()
