import sys
import os

# Agrega la raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils import read_video
from src.functions.YOLO_custom import YOLO_custom

def detect_with_YOLO(frame):
    model = YOLO_custom("yolo11n.pt", True)
    # model.to('cuda')

    detection = model(frame)
    return detection

if __name__ == "__main__":
    video_path = "C:/Users/germa/Documents/Facultad/PFC/desarrollo/fulbo/data/temp/izquierda_sync.mp4"
    video = read_video(video_path)

    frames_detected = []
    # Show video
    for frame in video:
    #     # cv2.imshow("Frame", frame)
        detected_frame = detect_with_YOLO(frame)
        frames_detected.append(detected_frame)
        # cv2.imshow("Frame", detected_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # cv2.destroyAllWindows()

    # Save frames as video


    # for i, fd in enumerate(frames_detected):
        
    #     cv2.imshow("Frame", fd)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    
    # detected_frame = detect_with_YOLO("C:/Users/germa/Documents/Facultad/PFC/desarrollo/fulbo/data/temp/result.jpg")
    # frames_detected.append(detected_frame)

    # # Show detected video
    # for frame in frames_detected:
    #     cv2.imshow("Frame", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break