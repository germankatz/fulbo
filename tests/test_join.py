import sys
import os

# Agrega la raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.functions.join_videos import JoinVideos


if __name__ == "__main__":
    path1 = "C:/Users/germa/Documents/Facultad/PFC/desarrollo/fulbo/data/temp/23/left.mp4"
    path2 = "C:/Users/germa/Documents/Facultad/PFC/desarrollo/fulbo/data/temp/23/right.mp4"

    joiner = JoinVideos(path1, path2, False)
    joiner.join(threshold=0.1)