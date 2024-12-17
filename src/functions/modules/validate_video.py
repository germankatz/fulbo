import cv2

# Calidad mínima para el análisis: full HD (en cualquier orientación) y 25 cuadros por segundo
MIN_LONG_SIDE = 1920
MIN_SHORT_SIDE = 1080
MIN_FPS = 25

def validate_video(video_path):
    """
    Check that a video can be read and has enough quality for the analysis.

    Args:
        video_path: Path to the video file

    Returns:
        (True, "") if the video is valid, (False, reason) otherwise
    """
    if not video_path:
        return False, "No se seleccionó ningún archivo."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return False, "El archivo no se puede abrir como video."

    ret, _ = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not ret:
        return False, "No se pudo leer el primer cuadro del video."

    if max(width, height) < MIN_LONG_SIDE or min(width, height) < MIN_SHORT_SIDE:
        return False, (f"La resolución del video ({width}x{height}) es menor a la mínima "
                       f"requerida ({MIN_LONG_SIDE}x{MIN_SHORT_SIDE}).")

    if fps < MIN_FPS:
        return False, (f"La tasa del video ({fps:.0f} cuadros por segundo) es menor a la mínima "
                       f"requerida ({MIN_FPS}).")

    return True, ""
