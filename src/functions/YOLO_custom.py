from ultralytics import YOLO
from PIL import Image
import torch
import cv2

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

    def print_cuda(self):
        print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Storing ID of current CUDA device
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")
            
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")