import sys
import os

# Agrega la raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.functions.join_videos import JoinVideos


if __name__ == "__main__":

    path1 = "data/temp/23/left_short.mp4"
    path2 = "data/temp/23/right_short.mp4"

    joiner = JoinVideos(path1, path2, True) #True si es video, False si queremos una foto
    joined_frames = joiner.join(threshold=0.1)
    joiner.reproduce(joined_frames)
    print(len(joined_frames))
    joiner.save_video(joined_frames, "C:/Users/agusr/OneDrive/Escritorio/Íconos/Ordenado/Formación/Ing. en Informática/PFC/Informe final/fulbo/data/temp/23/output2.mp4")