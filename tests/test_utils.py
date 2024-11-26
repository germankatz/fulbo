import sys
import os

# Agrega la raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils import read_video
import cv2

def test_read_video(path):
    frames = read_video(path)

    print(f"Total frames: {len(frames)}")
    
    # Show video
    for frame in frames:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break