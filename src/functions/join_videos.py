import sys
import os
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
        Stitch all frames from two videos.

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
        stitcher = Stitcher()

        for i in range(len(self.video_1)):
            print(f"Processing frame {i + 1}/{len(self.video_1)}")
            frame_1 = self.video_1[i]
            frame_2 = self.video_2[i]

            images = [frame_1, frame_2]

            # Convert images for stitching
            images = Images.of(images)

            # Resize images
            medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
            low_imgs = list(images.resize(Images.Resolution.LOW))
            final_imgs = list(images.resize(Images.Resolution.FINAL))

            # Feature detection
            finder = FeatureDetector()
            features = [finder.detect_features(img) for img in medium_imgs]

            # Match features
            matcher = FeatureMatcher()
            matches = matcher.match_features(features)

            # Camera estimation
            camera_estimator = CameraEstimator()
            camera_adjuster = CameraAdjuster()
            wave_corrector = WaveCorrector()

            cameras = camera_estimator.estimate(features, matches)
            cameras = camera_adjuster.adjust(features, matches, cameras)
            cameras = wave_corrector.correct(cameras)

            # Warp images
            warper = Warper(warper_type='transverseMercator')
            warper.set_scale(cameras)

            final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
            camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)
            warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
            warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
            final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

            # Join images
            blender = Blender()
            blender.prepare(final_corners, final_sizes)
            for img, mask, corner in zip(warped_final_imgs, warped_final_masks, final_corners):
                blender.feed(img, mask, corner)

            panorama, _ = blender.blend()
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
