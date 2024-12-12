import sys
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Agrega la raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils import read_video
from src.functions.YOLO_custom import YOLO_custom
from src.functions.ROI_definer import ROIDefiner
from src.functions.Process import Process
from src.functions.modules.select_teams import SelectTeams

def detect_with_YOLO(frame):
    model = YOLO_custom("yolo11n.pt", True)
    # model.to('cuda')

    detection = model(frame)
    return detection

if __name__ == "__main__":
    video_path = "C:/Users/germa/Documents/Facultad/PFC/desarrollo/fulbo/data/temp/izquierda_sync_trim.mp4"

    
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

    # Find id of second player with most detections
    # player_id = max(tracked_data, key=lambda k: len(tracked_data[k]) if k != player_id else 0)

    heatmap, transformed_positions, H = process.process_tracked_data(tracked_data, points, player_id)
    
    # Print quantity of players
    print(f"Se han detectado {len(tracked_data)} jugadores en el video.")

    # Plot heatmap
    # process.plot_heatmap(heatmap)

    # Draw selected player box
    # process.draw_player_box(video_path, tracked_data, player_id)

    # Classify players into teams
    select_teams = SelectTeams(video_path, tracked_data)
    player_groups = select_teams.classify_players()
    print("Player groups:", player_groups)

    # Plot the first bounding box of each player in their respective groups using the new function
    process.plot_bounding_boxes(video_path, tracked_data, player_groups)
