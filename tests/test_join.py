import sys
import os
from vidstab import VidStab

# Agrega la raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.functions.join_videos import JoinVideos
import cv2
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    path1 = "data/temp/sync/opc1/izquierda_sync.mp4"
    path2 = "data/temp/sync/opc1/derecha_sync.mp4"
    # path1 = "data/temp/sync/opc2/left_sync_e.mp4"
    # path2 = "data/temp/sync/opc2/right_sync.mp4"
    output_path = "C:/Users/germa/Documents/Facultad/PFC/desarrollo/fulbo/data/temp/23/output3.mp4"

    # Ejecutar el proceso
    joiner = JoinVideos(path1, path2, output_path)
    joiner.stitch_videos()
    print("Video stitching completed successfully!")