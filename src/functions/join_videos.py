import sys
import os
import cv2

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


from stitching.images import Images  # Importing Images class
from src.utils import read_video, get_first_frame, save_video
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
import cv2

from stitching.feature_matcher import FeatureMatcher  # Add this line
import matplotlib.pyplot as plt


class JoinVideos:
    def __init__(self, path_video_1, path_video_2, is_video=False):
        """
        Parameters
        ----------
        path_video_1 : str
            Path to the first video.
        path_video_2 : str
            Path to the second video.
        is_video : bool, optional
            If True, the input is a video; if False, the input is a frame. Default is False.
        """
        self.path_video_1 = path_video_1
        self.path_video_2 = path_video_2
        self.is_video = is_video
        
        if self.is_video:
            self.video_1 = read_video(self.path_video_1)
            self.video_2 = read_video(self.path_video_2)
            
            # Ensure videos have the same length by trimming the longer one
            if len(self.video_1) > len(self.video_2):
                self.video_1 = self.video_1[:len(self.video_2)]
            elif len(self.video_2) > len(self.video_1):
                self.video_2 = self.video_2[:len(self.video_1)]
        
        self.frame_1 = get_first_frame(self.path_video_1)
        self.frame_2 = get_first_frame(self.path_video_2)
            
    def join(self, threshold=0.1, detector="sift"):
        """
        Stitch all frames from two videos with a fixed transformation.

        Parameters
        ----------
        threshold : float
            Threshold for feature matching (default: 0.1).
        detector : str
            Feature detector to use (default: "sift").

        Returns
        -------
        list
            List of stitched frames.
        """
        if not self.is_video:
            raise ValueError("Stitching requires video input. Set `is_video=True` when initializing.")

        stitched_frames = []

        # Pre-compute features, cameras, and transformations using the first two frames
        print("Precomputing transformations using the first two frames...")
        frame_1 = self.video_1[0]
        frame_2 = self.video_2[0]
        initial_images = [frame_1, frame_2]

        # Convert images for stitching bajamos resoluciones
        images = Images.of(initial_images, medium_megapix=0.6, low_megapix=0.1, final_megapix=-1)

        # Resize images traemos las imagenes
        low_imgs = list(images.resize(Images.Resolution.LOW))
        medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
        final_imgs = list(images.resize(Images.Resolution.FINAL))
        self._plot_images(low_imgs, (10,10))

        # Feature detection
        finder = FeatureDetector()
        features = [finder.detect_features(img) for img in medium_imgs]
        keypoints_center_img = finder.draw_keypoints(medium_imgs[1], features[1])

        # Match features
        matcher = FeatureMatcher()
        matches = matcher.match_features(features)
        all_relevant_matches = matcher.draw_matches_matrix(medium_imgs, features, matches, conf_thresh=1, inliers=True, matchColor=(0, 255, 0))
        
        # # Plot matches
        # for idx1, idx2, img in all_relevant_matches:
        #     print(f"Matches Image {idx1+1} to Image {idx2+1}")
        #     self._plot_images(img, (20,10))

        #Subsetter
        
        # subsetter = Subsetter()
        # dot_notation = subsetter.get_matches_graph(images_obj.names, matches)

        # indices = subsetter.get_indices_to_keep(features, matches)

        # medium_imgs = subsetter.subset_list(medium_imgs, indices)
        # low_imgs = subsetter.subset_list(low_imgs, indices)
        # final_imgs = subsetter.subset_list(final_imgs, indices)
        # features = subsetter.subset_list(features, indices)
        # matches = subsetter.subset_matches(matches, indices)


        # Camera estimation
        camera_estimator = CameraEstimator()
        camera_adjuster = CameraAdjuster()
        wave_corrector = WaveCorrector()

        cameras = camera_estimator.estimate(features, matches)
        cameras = camera_adjuster.adjust(features, matches, cameras)
        cameras = wave_corrector.correct(cameras)

        # ## Warp images
        warper = Warper(warper_type='transverseMercator') # Usamos este tipo de warper porque es el que mejor se ajusta a la proyección de la cámara
        warper.set_scale(cameras)


        low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
        camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)  # Como se ajusta la cámara de baja resolución a la de media resolución

        warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
        warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
        low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

        final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
        camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)

        warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
        warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
        final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

        self._plot_images(warped_low_imgs, (10,10))
        self._plot_images(warped_low_masks, (10,10))
        # Use the fixed transformations for all frames
        
        # ## Mask

        cropper = Cropper()
        mask = cropper.estimate_panorama_mask(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

        # Mostramos la máscara final que se forma
        self.plot_image(mask, (5,5))

        # # Join images

        blender = Blender()
        blender.prepare(final_corners, final_sizes)
        for img, mask, corner in zip(warped_final_imgs, warped_final_masks, final_corners):
            blender.feed(img, mask, corner)
        panorama, _ = blender.blend()

        # Imagen final
        self.plot_image(panorama, (20,20))

        for i in range(len(self.video_1)):
            print(f"Processing frame {i + 1}/{len(self.video_1)}")
            frame_1 = self.video_1[i]
            frame_2 = self.video_2[i]

            current_images = [frame_1, frame_2]
            current_images_obj = Images.of(current_images, final_megapix=-1)
            current_final_imgs = list(current_images_obj.resize(Images.Resolution.FINAL))

            # Apply the fixed warp transformation
            warped_final_imgs = list(warper.warp_images(current_final_imgs, cameras, camera_aspect))
            # self._plot_images(warped_final_imgs)
            final_sizes = current_images_obj.get_scaled_img_sizes(Images.Resolution.FINAL)
            warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
            final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)
    
            # Join images
            blender = Blender()
            blender.prepare(final_corners, final_sizes)
            for img, mask, corner in zip(warped_final_imgs, warped_final_masks, final_corners):
                blender.feed(img, mask, corner)

            panorama, _ = blender.blend()
            # self.plot_image(panorama)
            stitched_frames.append(panorama)

        print("Stitching completed for all frames.")
        return stitched_frames



    
    def reproduce(self, frames, fps=30):
        """
        Play a video from a list of frames.

        Parameters
        ----------
        frames : list
            List of stitched frames.
        fps : int, optional
            Frames per second for video playback. Default is 30.
        """
        for frame in frames:
            cv2.imshow("Stitched Video", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def save_video(self, frames, output_path, fps=30):
        """
        Save a video from a list of frames.

        Parameters
        ----------
        frames : list
            List of frames to save as a video.
        output_path : str
            Path to save the video file (e.g., "output.mp4").
        fps : int, optional
            Frames per second for the video. Default is 30.
        """
        if not frames or len(frames) == 0:
            raise ValueError("No frames provided to save the video.")
        
        # Get dimensions from the first frame
        height, width, _ = frames[0].shape

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i, frame in enumerate(frames):
            if frame.shape != (height, width, 3):
                print(f"Resizing frame {i} from {frame.shape} to {(height, width, 3)}")
                frame = cv2.resize(frame, (width, height))  # Resize frame to match dimensions
            out.write(frame)

        out.release()
        print(f"Video successfully saved to {output_path}")

    def swap(self):
        """
        Swap the videos and their first frames.
        """
        self.video_1, self.video_2 = self.video_2, self.video_1
        self.frame_1, self.frame_2 = self.frame_2, self.frame_1

    def extract_features(self):
        """
        Extract features from the videos' middle frames using FeatureDetector.
        """
        finder = FeatureDetector()
        
        # Use middle frames for feature extraction
        middle_idx = len(self.video_1) // 2
        frame_1 = self.video_1[middle_idx]
        frame_2 = self.video_2[middle_idx]

        # Detect features
        features_1 = finder.detect_features(frame_1)
        features_2 = finder.detect_features(frame_2)
        
        # Draw keypoints on frames
        keypoints_frame_1 = finder.draw_keypoints(frame_1, features_1)
        keypoints_frame_2 = finder.draw_keypoints(frame_2, features_2)

        # Display keypoints
        self._plot_images([keypoints_frame_1, keypoints_frame_2], (15, 10))
        
        return features_1, features_2

    def plot_image(self, img, figsize_in_inches=(5, 5)):
        """
        Display a single image using Matplotlib.

        Parameters
        ----------
        img : np.ndarray
            The image to display.
        figsize_in_inches : tuple
            Figure size in inches (default: (5, 5)).
        """
        fig, ax = plt.subplots(figsize=figsize_in_inches)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    def _plot_images(self, images, figsize_in_inches=(5, 5)):
        """
        Helper function to plot images side by side.
        
        Parameters
        ----------
        images : list
            List of images to plot.
        figsize_in_inches : tuple
            Size of the figure (default: (5, 5)).
        """
        fig, axs = plt.subplots(1, len(images), figsize=figsize_in_inches)
        for col, img in enumerate(images):
            axs[col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    def initialize_transform(self):
        
        frame_1 = self.video_1[0]
        frame_2 = self.video_2[0]
        initial_images = [frame_1, frame_2]

        # 1. Convertir frames iniciales a la clase Images
        images = Images.of(initial_images, final_megapix=-1)  # sin downsampling final si gustas

        # Redimensionar a resolución media para detección de características
        medium_imgs = list(images.resize(Images.Resolution.MEDIUM))

        # 2. Detectar características
        finder = FeatureDetector(detector='orb', nfeatures=500)
        features = [finder.detect_features(img) for img in medium_imgs]

        # 3. Emparejar características
        matcher = FeatureMatcher(matcher_type='homography', range_width=-1)
        matches = matcher.match_features(features)

        # 4. Subset (si fuera necesario, con solo 2 imágenes no lo es, pero mostramos el flujo)
        subsetter = Subsetter(confidence_threshold=0.5)
        indices = subsetter.get_indices_to_keep(features, matches)
        medium_imgs = subsetter.subset_list(medium_imgs, indices)
        features = subsetter.subset_list(features, indices)
        matches = subsetter.subset_matches(matches, indices)
        images.subset(indices)

        # 5. Estimar y ajustar cámaras
        camera_estimator = CameraEstimator()
        cameras = camera_estimator.estimate(features, matches)
        camera_adjuster = CameraAdjuster()
        cameras = camera_adjuster.adjust(features, matches, cameras)

        # 6. Corrección de onda
        # wave_corrector = WaveCorrector(wave_correct_kind='horiz')
        # cameras = wave_corrector.correct(cameras)
        camera_aspect = images.get_ratio(Images.Resolution.FINAL, Images.Resolution.MEDIUM)  # Como se ajusta la cámara de baja resolución a la de media resolución


        # 7. Crear warper
        warper = Warper(warper_type='transverseMercator')
        warper.set_scale(cameras)

        # Si crop=False no se realiza recorte, pero no importa, ya tenemos las cámaras y el warper
        return images, cameras, warper, camera_aspect

    def apply_transform_to_frames(self, cameras, warper, camera_aspect):
        # Definir imágenes iniciales para trabajar
        images = Images.of([self.frame_1, self.frame_2], final_megapix=-1)
        
        # Calcular los tamaños y aspectos de las cámaras
        camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)  # Relación entre resolución final y media
        final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)

        # Crear y ajustar máscaras y áreas de interés (ROI)
        warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
        final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

        # Mostrar imágenes intermedias
        self._plot_images(warped_final_masks, (10, 10))

        # Crear la máscara final para recorte
        cropper = Cropper()
        mask = cropper.estimate_panorama_mask(images, warped_final_masks, final_corners, final_sizes)
        self.plot_image(mask, (5, 5))  # Mostrar la máscara final

        # Iterar sobre todos los frames de los videos
        for i in range(len(self.video_1)):
            # Cargar las imágenes actuales de los videos
            current_images = [self.video_1[i], self.video_2[i]]
            current_images_obj = Images.of(current_images, final_megapix=-1)
            
            # Redimensionar las imágenes a resolución final
            current_final_imgs = list(current_images_obj.resize(Images.Resolution.FINAL))

            # Aplicar la transformación de warp
            warped_final_imgs = list(warper.warp_images(current_final_imgs, cameras, camera_aspect))

            # Calcular los tamaños y máscaras finales para las imágenes actuales
            final_sizes = current_images_obj.get_scaled_img_sizes(Images.Resolution.FINAL)
            warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
            final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)
            
            # Realizar la unión de las imágenes
            blender = Blender()
            blender.prepare(final_corners, final_sizes)
            for img, mask, corner in zip(warped_final_imgs, warped_final_masks, final_corners):
                blender.feed(img, mask, corner)

            # Generar el panorama final
            panorama, _ = blender.blend()

            # Mostrar el panorama generado
            cv2.imshow("Panorama", panorama)
            if cv2.waitKey(1) & 0xFF == 27:  # Salir con la tecla 'Esc'
                break

        cv2.destroyAllWindows()


    def join_simple(self):
        """
        Stitch the first frames of the videos using the specified warper type.

        Parameters
        ----------
        warper_type : str, optional
            The type of warper to use (default: 'transverseMercator').
        """
        frame_1 = self.video_1[0]
        frame_2 = self.video_2[0]

        print(type(frame_1), frame_1.dtype, frame_1.shape)
        print(type(frame_2), frame_2.dtype, frame_2.shape)

        print("Stitching frames...")
        stitcher = Stitcher(warper_type='transverseMercator',crop=False, confidence_threshold=0.3)
        # stitched_frame = stitcher.stitch([frame_1, frame_2])
        # print("Frames stitched.")
        # cv2.imshow("Stitched Frame", stitched_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        for i in range(len(self.video_1)):
            print(f"Processing frame {i + 1}/{len(self.video_1)}")
            frame_1 = self.video_1[i]
            frame_2 = self.video_2[i]

            stitched_frame = stitcher.stitch([frame_1, frame_2])
            cv2.imshow("Stitched Frame", stitched_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Salir con la tecla 'Esc'
                break

        cv2.destroyAllWindows()

        # images, cameras, warper, camera_aspect = self.initialize_transform()

        # print("Applying transformation to frames...")

        # # Ahora aplicamos la transformación a todos los frames
        # self.apply_transform_to_frames(cameras, warper, camera_aspect)





