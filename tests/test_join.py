import sys
import os
from vidstab import VidStab

# Agrega la raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.functions.join_videos import JoinVideos
import cv2
import os
import matplotlib.pyplot as plt


def convert_video_to_h264(input_path, output_path, fps=30):
    """
    Converts a video to the H264 format to ensure compatibility with stabilization.

    Parameters
    ----------
    input_path : str
        Path to the input video file.
    output_path : str
        Path to save the converted video.
    fps : int, optional
        Frames per second for the output video. Default is 30.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist.")

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open the input video: {input_path}")

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video dimensions: {width}x{height}")

    # Define the codec as H264 and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

    print(f"Video successfully converted to H264 and saved to: {output_path}")


if __name__ == "__main__":

    path1 = "data/temp/23/left_short.mp4"
    path2 = "data/temp/23/right_short.mp4"

    output_path="C:/Users/agusr/OneDrive/Escritorio/Íconos/Ordenado/Formación/Ing. en Informática/PFC/Informe final/fulbo/data/temp/23/output3.mp4"

    joiner = JoinVideos(path1, path2, True) #True si es video, False si queremos una foto
    joined_frames = joiner.join(output_path) #threshold=0.1
    joiner.reproduce(joined_frames)
    print(len(joined_frames))
    joiner.save_video(joined_frames, output_path)

    #Pruebo estabilización

    

    # Verifica si el archivo se creó correctamente
    # input_video = "data/temp/23/output2.mp4"
    converted_video = "data/temp/23/converted_output2.mp4"

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