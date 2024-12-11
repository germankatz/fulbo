import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Process:
    def __init__(self, field_width=68, field_length=105):
        self.field_width = field_width
        self.field_length = field_length

    def compute_homography(self, roi_points):
        """
        roi_points: lista de 6 puntos obtenidos del ROIDefiner en el orden:
            [top_left, top_center, top_right, bottom_right, bottom_center, bottom_left]
        Cada punto es (x, y) en coordenadas de la imagen.

        Retorna:
            H: Matriz de homografía (3x3)
        """
        # Validar que los puntos estén en el orden correcto
        if len(roi_points) != 6:
            raise ValueError("Se necesitan exactamente 6 puntos para calcular la homografía.")

        # Puntos del mundo (coordenadas reales de la cancha) ajustados al nuevo orden
        world_points = np.array([
            [0, 0],  # top_left
            [self.field_width / 2, 0],  # top_center
            [self.field_width, 0],  # top_right
            [self.field_width, self.field_length],  # bottom_right
            [self.field_width / 2, self.field_length],  # bottom_center
            [0, self.field_length]  # bottom_left
        ], dtype=np.float32)

        image_points = np.array(roi_points, dtype=np.float32)

        # Calcular la homografía usando RANSAC
        H, mask = cv2.findHomography(image_points, world_points, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("No se pudo calcular la homografía con los puntos proporcionados.")
        
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
            if 0 <= X <= self.field_width and 0 <= Y <= self.field_length:
                x_idx = int(X / bin_size)
                y_idx = int(Y / bin_size)
                heatmap[y_idx, x_idx] += 1

        return heatmap

    def process_tracked_data(self, tracked_data, roi_points, player_id):
        """
        Procesa datos de tracking y genera un mapa de calor del jugador.

        tracked_data: diccionario {track_id: [{x1, y1, x2, y2}, ...]}
        roi_points: lista de 6 puntos para calcular la homografía.
        player_id: id del jugador a analizar.

        Retorna:
            heatmap (2D array),
            transformed_positions (lista de (X, Y) en metros),
            H (homografía)
        """
        # Calcular homografía
        H = self.compute_homography(roi_points)

        # Obtener datos del jugador
        if player_id not in tracked_data:
            raise ValueError(f"El jugador con track_id {player_id} no está en los datos.")

        player_data = tracked_data[player_id]

        # Obtener el centro de cada caja
        image_positions = [
            ((entry['x1'] + entry['x2']) / 2.0, (entry['y1'] + entry['y2']) / 2.0)
            for entry in player_data
        ]

        # Transformar puntos a coordenadas del campo
        transformed_positions = self.transform_points(image_positions, H)

        # Generar mapa de calor
        heatmap = self.generate_heatmap(transformed_positions, bin_size=1)

        return heatmap, transformed_positions, H

    def plot_heatmap(self, heatmap):
        """
        Plotea un mapa de calor sobre un esquema de cancha.

        heatmap: arreglo 2D del mapa de calor.
        """
        cmap = mcolors.LinearSegmentedColormap.from_list('field_cmap', ['green', 'yellow', 'red'])

        plt.figure(figsize=(8, 12))
        plt.imshow(
            heatmap,
            origin='lower',
            cmap=cmap,
            aspect='equal',
            extent=[0, self.field_width, 0, self.field_length]
        )
        plt.colorbar(label='Densidad de Presencia')
        plt.title("Mapa de Calor del Jugador en la Cancha")
        plt.xlabel("Ancho de la cancha (m)")
        plt.ylabel("Largo de la cancha (m)")
        plt.show()
