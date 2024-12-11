import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video en la ruta: {video_path}")

    frames = []

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        count += 1
        if count % 50 == 0:
            print(f"Reading frame {count}")
        if count > 500:
            break
    
    cap.release()
    return frames

def get_first_frame(video_path):
    # video_path = "C:/Users/agusr/OneDrive/Escritorio/Íconos/Ordenado/Formación/Ing. en Informática/PFC/Informe final/fulbo/data/temp/23/left.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video en la ruta: {video_path}")

    ret, frame = cap.read()
    if not ret:
        raise ValueError("No se pudo leer el primer frame del video.")
    
    cap.release()
    return frame

def save_video(ouput_video_frames,output_video_path):
    if not ouput_video_frames:
        raise ValueError("La lista de frames está vacía, no se puede guardar el video.")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()