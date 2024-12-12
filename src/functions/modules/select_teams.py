import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

class SelectTeams:
    def __init__(self, video_path, tracked_data):
        self.video_path = video_path
        self.tracked_data = tracked_data

    def process_frame(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        player_region = frame[y1:y2, x1:x2]
        height = y2 - y1
        region_height = height // 6
        regions = [player_region[i*region_height:(i+1)*region_height, :] for i in range(6)]
        region_means = [np.mean(region, axis=(0, 1)) for region in regions]
        return np.concatenate(region_means[1:4])  # Flatten the selected regions into a single 1D array

    def extract_features(self):
        cap = cv2.VideoCapture(self.video_path)
        features = []
        labels = []
        for track_id, detections in self.tracked_data.items():
            first_detection = detections[0]
            frame_idx = first_detection["frame"]
            bbox = (first_detection["x1"], first_detection["y1"], first_detection["x2"], first_detection["y2"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                feature = self.process_frame(frame, bbox)
                features.append(feature)
                labels.append(track_id)
        cap.release()
        return np.array(features), np.array(labels)

    def classify_players(self):
        features, labels = self.extract_features()
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(features)
        centroids = kmeans.cluster_centers_

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(centroids, [0, 1, 2])  # Assign arbitrary labels to centroids

        player_groups = knn.predict(features)
        return {labels[i]: player_groups[i] for i in range(len(labels))}
