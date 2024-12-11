import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Process:
    def __init__(self, field_width=68, field_length=105):
        self.field_width = field_width
        self.field_length = field_length

    def compute_homography(self, roi_points):
        """
        roi_points: lista de 6 puntos obtenidos del ROIDefiner en el orden:
            [top_left, top_center, top_right, bottom_left, bottom_center, bottom_right]
        Cada punto es (x, y) en coordenadas de la imagen.

        Retorna:
            H: Matriz de homografía (3x3)
        """

        # Puntos del mundo (coordenadas reales de la cancha)
        world_points = np.array([
            [0, 0], 
            [self.field_width/2, 0], 
            [self.field_width, 0],
            [0, self.field_length],
            [self.field_width/2, self.field_length],
            [self.field_width, self.field_length]
        ], dtype=np.float32)

        image_points = np.array(roi_points, dtype=np.float32)

        # Calcular la homografía usando RANSAC para mayor robustez
        H, mask = cv2.findHomography(image_points, world_points, cv2.RANSAC, 5.0)
        return H

    def transform_points(self, points, H):
        """
        Transforma una lista de puntos (x, y) en la imagen a 
        coordenadas del campo usando la homografía H.

        points: lista de (x, y) en pixeles (imagen)
        H: homografía (3x3)

        Retorna:
            transformed_points: lista de (X, Y) en coordenadas del campo (m)
        """

        if len(points) == 0:
            return []

        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H)
        transformed_points = transformed.reshape(-1, 2)
        return transformed_points.tolist()

    def generate_heatmap(self, player_positions, bin_size=1):
        """
        Genera un mapa de calor sencillo a partir de las posiciones del jugador en la cancha.

        player_positions: lista de (X, Y) en metros (coordenadas de la cancha)
        bin_size: tamaño de la celda en metros para el histograma

        Retorna:
            heatmap: arreglo 2D con la frecuencia de presencia en cada celda
        """
        # Discretizamos la cancha en una rejilla
        x_bins = int(self.field_width / bin_size)
        y_bins = int(self.field_length / bin_size)

        heatmap = np.zeros((y_bins, x_bins), dtype=np.float32)

        for (X, Y) in player_positions:
            # Verificar si el punto está dentro de la cancha
            if 0 <= X <= self.field_width and 0 <= Y <= self.field_length:
                # Calcular la celda en el histograma
                x_idx = int(X / bin_size)
                y_idx = int(Y / bin_size)
                # Incrementar la celda
                heatmap[y_idx, x_idx] += 1

        return heatmap

    def process_tracked_data(self, tracked_data, roi_points, player_id):
        """
        1. A partir de roi_points calcular la homografía.
        2. Usar tracked_data para obtener las posiciones del player_id en la imagen.
        3. Transformarlas a coordenadas de la cancha.
        4. Generar un mapa de calor a partir de las posiciones transformadas.

        tracked_data: diccionario {track_id: [ {frame, x1, y1, x2, y2, class_id, class_name}, ... ]}
        roi_points: los 6 puntos seleccionados con ROIDefiner:
                    orden esperable: 
                    top_left, top_center, top_right, bottom_left, bottom_center, bottom_right
        player_id: el track_id del jugador a analizar

        Retorna:
            heatmap (2D array), 
            transformed_positions (lista de posiciones (X,Y) en metros),
            H (homografía)
        """

        # Calcular homografía
        H = self.compute_homography(roi_points)
        if H is None:
            print("No se pudo calcular la homografía con los puntos proporcionados.")
            return None, None, None

        # Extraer las posiciones del jugador solicitado
        # Usaremos el centro de la caja para representarlo
        if player_id not in tracked_data:
            print("No existe el jugador con ese track_id")
            return None, None, None

        player_data = tracked_data[player_id]

        # Obtener el centro de cada caja
        # Nota: la caja está en (x1,y1,x2,y2)
        # El centro: X = (x1+x2)/2, Y = (y1+y2)/2
        image_positions = []
        for entry in player_data:
            x_center = (entry['x1'] + entry['x2']) / 2.0
            y_center = (entry['y1'] + entry['y2']) / 2.0
            image_positions.append((x_center, y_center))

        # Transformar a coordenadas del campo
        transformed_positions = self.transform_points(image_positions, H)

        # Generar el mapa de calor
        heatmap = self.generate_heatmap(transformed_positions, bin_size=1)

        return heatmap, transformed_positions, H

    def plot_heatmap(heatmap, field_width=68, field_length=105):
        # Crear un colormap personalizado: verde (mínimo), amarillo (medio), rojo (máximo)
        cmap = mcolors.LinearSegmentedColormap.from_list('field_cmap', ['green','yellow','red'])

        plt.figure(figsize=(8, 12))  
        # extent = [xmin, xmax, ymin, ymax]
        # Queremos que el eje X vaya de 0 a field_width y el Y de 0 a field_length
        plt.imshow(heatmap, 
                origin='lower', 
                cmap=cmap, 
                aspect='equal', 
                extent=[0, field_width, 0, field_length])

        plt.colorbar(label='Densidad de Presencia')
        plt.title("Mapa de Calor del Jugador en la Cancha")
        plt.xlabel("Ancho de la cancha (m)")
        plt.ylabel("Largo de la cancha (m)")
        plt.show()