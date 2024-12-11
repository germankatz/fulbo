import sys
import os
import matplotlib.pyplot as plt
import cv2

# Agrega la raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils import read_video
from src.functions.YOLO_custom import YOLO_custom
from src.functions.ROI_definer import ROIDefiner
from src.functions.Process import Process

def detect_with_YOLO(frame):
    model = YOLO_custom("yolo11n.pt", True)
    # model.to('cuda')

    detection = model(frame)
    return detection

if __name__ == "__main__":
    video_path = "C:/Users/germa/Documents/Facultad/PFC/desarrollo/fulbo/data/temp/david.MOV"

    
    # Select ROI
    points = ROIDefiner.define_roi_from_video(video_path)
    print("Puntos seleccionados:", points)

    # Detect and track with YOLO
    model = YOLO_custom("yolo11n.pt", True)
    tracked_data = model.track(video_path, points, show_plot=False)

    # Modelo de datos de tracked_data

    # tracked_data = {
    #     track_id: [
    #         {
    #         "frame": frame_idx,
    #         "x1": x1,
    #         "y1": y1,
    #         "x2": x2,
    #         "y2": y2,
    #         "class_id": class_id,
    #         "class_name": class_name
    #         },
    #         ...
    #     ],
    #     ...
    # }

    # Process tracked data
    process = Process()

    # Find id of player with most detections
    player_id = max(tracked_data, key=lambda k: len(tracked_data[k]))

    heatmap, transformed_positions, H = process.process_tracked_data(tracked_data, points, player_id)

    plt.figure(figsize=(8, 12))  # Ajusta el tamaño de la figura (ancho x alto)
    plt.imshow(heatmap, origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(label='Densidad de Presencia')
    plt.title("Mapa de Calor del Jugador en la Cancha")
    plt.xlabel("Eje X (ancho, en metros)")
    plt.ylabel("Eje Y (largo, en metros)")
    plt.show()


    # video = read_video(video_path)

    # frames_detected = []
    # # Show video
    # for frame in video:
    # #     # cv2.imshow("Frame", frame)
    #     detected_frame = detect_with_YOLO(frame)
    #     frames_detected.append(detected_frame)
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