import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils import read_video, get_first_frame, save_video	
from stitching import Stitcher
from stitching.feature_detector import FeatureDetector
import cv2
import matplotlib.pyplot as plt


class JoinVideos:
    def __init__(self, video1, video2, is_video=False):
        """
        Parameters
        ----------
        video1 : str
            Path to the first video.
        video2 : str
            Path to the second video.
        is_video : bool, optional
            If True, the input is a video, if False, the input is a frame. The default is False.
        """

        self.video1 = video1
        self.video2 = video2
        self.is_video = is_video
        
        if self.is_video:
            video1 = read_video(self.video1)
            video2 = read_video(self.video2)
            
            # Check if the videos have the same length, if not trim the longest one
            if len(video1) > len(video2):
                video1 = video1[:len(video2)]
            elif len(video2) > len(video1):
                video2 = video2[:len(video1)]
        
        self.frame1 = get_first_frame(self.video1)
        self.frame2 = get_first_frame(self.video2)

    
    def join(self, threshold=0.1, detector="sift"):
        """
        Join two videos.
        """
        images = []
        images.append(self.frame1)
        images.append(self.frame2)

        # plot images in one figure
        fig, axs = plt.subplots(1, len(images),figsize=(5,5))
        for col, img in enumerate(images):
            axs[col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))        
        plt.show()
        cv2.waitKey(0)



        stitcher = Stitcher()
        panorama = stitcher.stitch(images)

        # Plot panorama
        cv2.imshow("Panorama", panorama)
        cv2.waitKey(0)

        return panorama
    
    def swap(self):
        """
        Swap the videos and the frames.
        """
        self.video1, self.video2 = self.video2, self.video1
        self.frame1, self.frame2 = self.frame2, self.frame1
    
    # def extract_features(self):
    #     """
    #     Extract features from the videos.
    #     """

    #     finder = FeatureDetector()
        

    #     # Search for features only in middle part of the image
    #     height, width, _ = self.frame1.shape
    #     width_cutoff = width // 2
    #     features_img = []
    #     for index, img in enumerate(final_imgs):
    #         # select middle part of the image left in x for the first and right for the second
    #         if index == 0:
    #             img_trim = img[:, :width_cutoff]
    #         else:
    #             img_trim = img[:, width_cutoff:]
    #         features_img.append(finder.detect_features(img_trim))

    #     # features = [finder.detect_features(img) for img in final_imgs]
    #     # For every keypoint add width_cutoff to x coordinate
    #     for index, feature in enumerate(features_img):
    #         kpoints = feature.keypoints
    #         for kpoint in kpoints:
    #             kpoint.pt = (kpoint.pt[0] + width_cutoff, kpoint.pt[1])
    #         features_img[index].keypoints = kpoints

    #     keypoints_center_img = finder.draw_keypoints(final_imgs[1], features_img[1])
