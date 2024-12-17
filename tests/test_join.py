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
    # joined_frames = joiner.join(output_path) #threshold=0.1
    # joiner.reproduce(joined_frames)
    # print(len(joined_frames))
    # joiner.save_video(joined_frames, output_path)

    #Pruebo estabilización

    

    # Verifica si el archivo se creó correctamente
    # input_video = "data/temp/23/output2.mp4"
    # converted_video = "data/temp/23/output3.mp4"

    # convert_video_to_h264(input_video, converted_video, fps=30)


    # Estabiliza el video
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # stabilizer = VidStab(kp_method='ORB')
    # stabilizer.stabilize(input_path=converted_video, output_path="data/temp/23/stable_video.avi")
    # print("Stabilization complete.")

    # stabilizer.plot_trajectory()
    # plt.show()

    # stabilizer.plot_transforms()
    # plt.show()