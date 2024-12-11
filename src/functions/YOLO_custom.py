from ultralytics import YOLO
from PIL import Image
import torch
import cv2
from collections import defaultdict
import numpy as np

class YOLO_custom:
    def __init__(self, model_path, cuda=False):

        # Check for cuda
        # if cuda: 
        #    self.print_cuda()

        self.model = YOLO(model_path)

        if cuda:
            self.model.to('cuda')
        
    def __call__(self, frame):

        results = self.model.predict(frame, imgsz=(1920,1088), conf=0.2)
        annotated_frames = []
        for i, r in enumerate(results):
            # Visualize the results on the frame
            annotated_frame = r.plot(line_width=1, font_size=2, conf=True)
            annotated_frames.append(annotated_frame)

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)
            # # save img
            # cv2.imwrite("C:/Users/germa/Documents/Facultad/PFC/desarrollo/fulbo/data/temp/result_yolo.jpg", annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return annotated_frames
    
    def track(self, video_path, mask=None, show_plot=True):
        # Store the track history
        track_history = defaultdict(lambda: [])

        cap = cv2.VideoCapture(video_path)

        if show_plot:
            cv2.namedWindow("Muestra de tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Muestra de tracking", 1600, 900)

        # Definir colores y estilos
        box_color = (0, 0, 255) 
        box_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 2
        text_color = (255, 255, 255)

        # Convertir máscara si existe
        if mask is not None:
            mask_contour = np.array(mask, dtype=np.int32).reshape((-1,1,2))
        else:
            mask_contour = None

        frame_idx = 0
        tracked_data = defaultdict(list)  # Para almacenar info por track_id

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Ejecutar el tracking
            results = self.model.track(
                frame, 
                persist=True, 
                tracker="bytetrack.yaml", 
                imgsz=(1088,1920), 
                conf=0.5, 
                iou=0.5, 
                device='cuda'
            )

            # Extraer info de las detecciones
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Si el modelo soporta cls
            if hasattr(results[0].boxes, 'cls'):
                class_ids = results[0].boxes.cls.int().cpu().tolist()
            else:
                class_ids = [None] * len(track_ids)

            names = results[0].names  # Mapeo de id de clase a nombre

            annotated_frame = frame.copy()

            for (box, track_id, class_id) in zip(boxes, track_ids, class_ids):

                # Obtener nombre de clase si está disponible
                class_name = names[class_id] if class_id is not None and class_id < len(names) else "unknown"

                # Si no detecta a una persona lo saltea
                if class_name != "Person":
                    continue

                x, y, w, h = box
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)

                # Verificar si el centro de la detección está dentro de la máscara
                if mask_contour is not None:
                    inside = cv2.pointPolygonTest(mask_contour, (x, y), False)
                    if inside < 0:
                        continue

                # Almacenar información del track
                tracked_data[track_id].append({
                    "frame": frame_idx,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_id": class_id,
                    "class_name": class_name
                })

                # Ejemplo
                # Primer fotograma (frame_idx=0):
                # Se encuentra un objeto 
                # Objeto 1 con track_id=1 y coordenadas (x1=10, y1=20, x2=50, y2=80) de la clase "person".

                # tracked_data[1].append({
                #     "frame": 0,
                #     "x1": 10, "y1": 20, "x2": 50, "y2": 80,
                #     "class_id": 0,
                #     "class_name": "person"
                # })

                # Dibujar si se requiere plot
                if show_plot:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, box_thickness)
                    label = f"{class_name} {track_id}"
                    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                    cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), box_color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), font, font_scale, text_color, font_thickness)

                    # Dibujar las trayectorias
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            # Dibujar la máscara como referencia
            if show_plot and mask_contour is not None:
                cv2.polylines(annotated_frame, [mask_contour], True, (0,255,0), 2)

            if show_plot:
                cv2.imshow("Muestra de tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        if show_plot:
            cv2.destroyAllWindows()

        # Retornar la información
        return tracked_data
    
    def print_cuda(self):
        print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Storing ID of current CUDA device
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")
            
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")