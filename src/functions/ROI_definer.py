import cv2
import numpy as np

class ROIDefiner:
    def __init__(self, image, max_points=6):
        self.image = image
        self.points = []
        self.max_points = max_points
        self.radius = 5
        self.window_name = "Define ROI"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            clicked_point = (x, y)
            # Verificar si se hizo clic cerca de un punto existente
            remove_index = None
            for i, p in enumerate(self.points):
                if np.hypot(p[0]-x, p[1]-y) < self.radius*2:
                    remove_index = i
                    break

            if remove_index is not None:
                # Eliminar el punto clickeado
                self.points.pop(remove_index)
            else:
                # Agregar punto si no excede el máximo
                if len(self.points) < self.max_points:
                    self.points.append(clicked_point)

    def run(self):
        while True:
            display_img = self.draw_points_and_polygon()
            cv2.imshow(self.window_name, display_img)

            key = cv2.waitKey(50) & 0xFF

            # Presionar 'q' para salir sin confirmar
            if key == ord('q'):
                self.points = []
                break
            
            # Presionar 'c' para confirmar solo si hay 6 puntos
            if key == ord('c'):
                if len(self.points) == self.max_points:
                    break
            
            # Presionar 'd' para eliminar todos los puntos (usar esto como placeholder de SUPR)
            if key == ord('d'):
                self.points = []

        cv2.destroyWindow(self.window_name)
        return self.points

    def draw_points_and_polygon(self):
        img_copy = self.image.copy()
        h, w = img_copy.shape[:2]

        # Dibujar puntos
        for p in self.points:
            cv2.circle(img_copy, p, self.radius, (0,0,255), -1)

        if len(self.points) == self.max_points:
            # Crear la máscara para el polígono
            mask = np.zeros(img_copy.shape[:2], dtype=np.uint8)
            contour = np.array(self.points, dtype=np.int32).reshape((-1,1,2))
            cv2.fillPoly(mask, [contour], 255)

            # Fondo gris
            gray_background = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            gray_background = cv2.cvtColor(gray_background, cv2.COLOR_GRAY2BGR)

            # Combinar interior original y exterior gris
            img_copy = np.where(mask[:,:,None]==255, img_copy, (gray_background*0.5).astype(np.uint8))
            cv2.polylines(img_copy, [contour], True, (0,255,0), 2)

        # Instrucciones en la esquina superior derecha
        # Ajustar posición según el texto y tamaño de la ventana
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 1
        
        texto1 = "C: Confirmar (solo con 6 pts)"
        texto2 = "SUPR: Eliminar todos (aqui 'd')"
        puntos_restantes = self.max_points - len(self.points)
        texto3 = f"Puntos restantes: {puntos_restantes}"

        # Calcular ancho de los textos para posicionarlos en la derecha
        (t1_w, t1_h), _ = cv2.getTextSize(texto1, font, font_scale, font_thickness)
        (t2_w, t2_h), _ = cv2.getTextSize(texto2, font, font_scale, font_thickness)
        (t3_w, t3_h), _ = cv2.getTextSize(texto3, font, font_scale, font_thickness)

        # Colocar el texto en la esquina superior derecha
        # Dejamos un margen de 10 pixeles del borde derecho
        x_pos = w - max(t1_w, t2_w, t3_w) - 10
        y_pos = 20  # Primera linea
        cv2.putText(img_copy, texto1, (x_pos, y_pos), font, font_scale, (255,255,255), font_thickness)

        y_pos += t1_h + 5
        cv2.putText(img_copy, texto2, (x_pos, y_pos), font, font_scale, (255,255,255), font_thickness)

        y_pos += t2_h + 5
        cv2.putText(img_copy, texto3, (x_pos, y_pos), font, font_scale, (255,255,255), font_thickness)

        return img_copy


    def define_roi_from_video(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("No se pudo leer el primer frame del video.")
            return [], None

        roi_definer = ROIDefiner(frame, max_points=6)
        points = roi_definer.run()
        marked_frame = roi_definer.draw_points_and_polygon()
        return points, marked_frame