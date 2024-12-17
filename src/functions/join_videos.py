import sys
import os
import cv2
import time  # Add this import at the top

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from stitching.images import Images  # Importing Images class
from stitching import Stitcher
from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher  # Add this line
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.cropper import Cropper
from stitching.blender import Blender
from stitching.subsetter import Subsetter
from functions.VideoStitcher import VideoStitcher
import cv2

from stitching.feature_matcher import FeatureMatcher  # Add this line
import matplotlib.pyplot as plt


class JoinVideos:
    
    def __init__(self, path_video_1, path_video_2, output_path, confidence_threshold=0.35):
        self.path_video_1 = path_video_1
        self.path_video_2 = path_video_2
        self.output_path = output_path
        
        # Inicializamos capturadores de video
        self.cap1 = cv2.VideoCapture(self.path_video_1)
        self.cap2 = cv2.VideoCapture(self.path_video_2)
        
        # Obtener propiedades del video
        self.frame_width = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap1.get(cv2.CAP_PROP_FPS)
        
        # Crear un escritor de video para guardar el resultado
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para mp4
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width*2, self.frame_height))  # Output size es el doble del ancho
        
        # Inicializar Stitcher
        self.stitcher = VideoStitcher(warper_type='transverseMercator',crop=False,confidence_threshold=confidence_threshold, detector='sift')
        self.cameras_initialized = False

    def stitch_videos(self):
        total_frames = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        times = {
            'read_frames': 0,
            'stitch': 0,
            'write': 0
        }

        first_write = False

        cont = 0
        while cont < 500:
            cont += 1
            # Medir tiempo de lectura
            t_start = time.time()
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            times['read_frames'] += time.time() - t_start
            
            if not ret1 or not ret2:
                break
            
            current_frame += 1
            progress = (current_frame / total_frames) * 100
            
            # Medir tiempo de stitching
            t_start = time.time()
            if not self.cameras_initialized:
                self.stitcher.stitch([frame1, frame2])
                self.cameras_initialized = True
            else:
                result = self.stitcher.stitch([frame1, frame2])
                times['stitch'] += time.time() - t_start
                
                if not first_write:
                    first_write = True
                    w,h = result.shape[1], result.shape[0]
                    self.out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w,h))  # Output size es el doble del ancho
                
                # Medir tiempo de escritura
                t_start = time.time()
                self.out.write(result)
                times['write'] += time.time() - t_start

            # Mostrar progreso y tiempos promedio
            avg_read = times['read_frames'] / current_frame if current_frame > 0 else 0
            avg_stitch = times['stitch'] / current_frame if current_frame > 0 else 0
            avg_write = times['write'] / current_frame if current_frame > 0 else 0
            
            print(f"\rProgress: {current_frame}/{total_frames} ({progress:.1f}%) | "
                  f"Avg times - Read: {avg_read:.3f}s, "
                  f"Stitch: {avg_stitch:.3f}s, "
                  f"Write: {avg_write:.3f}s", end="")

        print("\n\nFinal average times per frame:")
        print(f"Read frames: {times['read_frames']/total_frames:.3f}s")
        print(f"Stitching: {times['stitch']/total_frames:.3f}s")
        print(f"Writing: {times['write']/total_frames:.3f}s")
        
        self.cap1.release()
        self.cap2.release()
        self.out.release()
        cv2.destroyAllWindows()



