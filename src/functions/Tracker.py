from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        # Deshabilitar cuando no haya placa de video
        self.model.to('cuda')

        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # Detecto objectos en los frames 
            # 0.1 detecta bien sin falsos positivos
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        print("Cantidad de frames: ", len(frames))


        detections = self.detect_frames(frames)

        tracks = {
            "players": [], 
            "referees": [],
            "ball": []
        }
        
        for frame_num, detection in enumerate(detections):
            class_name = detection.names

            # Convertimos a supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Trackeamos objectos
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]
                print("Detecto al jugador ",track_id, " en el frame ", frame_num)	

                # {0: 'player', 1: 'referee', 2: 'ball'}
                if class_id == 0:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif class_id == 1:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id):
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(frame, center, axes, 0, 0, 360, color, 2)
        cv2.putText(frame, f"{track_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, player in player_dict.items():                
                frame = self.draw_ellipse(frame, player["bbox"], (0,0,255), track_id)
            
            output_video_frames.append(frame)

        return output_video_frames
    