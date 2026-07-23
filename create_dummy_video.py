import cv2
import numpy as np

def create_dummy_video():
    width, height = 640, 480
    fps = 30
    duration = 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('dummy_match.mp4', fourcc, fps, (width, height))
    
    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (0, 150, 0) # Green pitch
        
        # A mock player
        x = 100 + i * 2
        y = 200
        cv2.rectangle(frame, (x, y), (x+20, y+50), (0,0,255), 2)
        
        # A mock ball
        bx = 120 + i * 4
        by = 240
        cv2.circle(frame, (bx, by), 5, (255,255,255), -1)
        
        out.write(frame)
        
    out.release()
    print("Created dummy_match.mp4")

if __name__ == "__main__":
    create_dummy_video()
