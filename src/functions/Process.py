import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy.ndimage import zoom


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

    
    
    
    def plot_heatmap(self, heatmap, title="Mapa de Calor del Jugador en la Cancha"):
        """
        Plotea un mapa de calor sobre un esquema de cancha con orientación vertical.

        heatmap: arreglo 2D del mapa de calor.
        title: título del mapa de calor.
        """
        cmap = mcolors.LinearSegmentedColormap.from_list('field_cmap', ['green', 'yellow', 'red'])

        # Expandir el heatmap
        scale_factor = 3  # Ajusta este valor para cambiar el tamaño del heatmap
        heatmap = zoom(heatmap, scale_factor, order=3)  # Interpolación cúbica para suavizado

        # Recortar el heatmap a las dimensiones de la cancha
        heatmap = heatmap[:int(self.field_length * scale_factor), :int(self.field_width * scale_factor)]


        # Espejar el heatmap en el eje X si es necesario
        heatmap = np.fliplr(heatmap)

        plt.figure(figsize=(8, 12))
        plt.imshow(
            heatmap,
            origin='lower',
            cmap=cmap,
            aspect='equal',
            extent=[0, self.field_width, 0, self.field_length],  # Ancho es X, largo es Y
            interpolation='bilinear'  # Suavizar el heatmap
        )

        # Dibujar líneas de la cancha
        field_color = 'white'
        line_width = 2

        # Bordes del campo
        plt.plot([0, self.field_width], [0, 0], color=field_color, lw=line_width)  # Línea inferior
        plt.plot([0, self.field_width], [self.field_length, self.field_length], color=field_color, lw=line_width)  # Línea superior
        plt.plot([0, 0], [0, self.field_length], color=field_color, lw=line_width)  # Línea izquierda
        plt.plot([self.field_width, self.field_width], [0, self.field_length], color=field_color, lw=line_width)  # Línea derecha

        # Línea de mitad de campo
        plt.plot([0, self.field_width], [self.field_length / 2, self.field_length / 2], color=field_color, lw=line_width)

        # Círculo central
        center_circle_radius = 9.15  # Radio estándar en metros
        center = (self.field_width / 2, self.field_length / 2)
        circle = plt.Circle(center, center_circle_radius, color=field_color, fill=False, lw=line_width)
        plt.gca().add_artist(circle)

        # Punto central
        plt.plot(center[0], center[1], 'o', color=field_color)

        # Áreas grandes (16.5 m desde la línea de gol)
        area_width = 40.3  # Ancho del área grande (en el eje X)
        area_height = 16.5  # Profundidad del área grande (en el eje Y)

        # Área grande superior
        plt.plot(
            [self.field_width / 2 - area_width / 2, self.field_width / 2 + area_width / 2],
            [self.field_length - area_height, self.field_length - area_height],
            color=field_color, lw=line_width
        )  # Línea horizontal inferior del área superior
        plt.plot(
            [self.field_width / 2 - area_width / 2, self.field_width / 2 - area_width / 2],
            [self.field_length, self.field_length - area_height],
            color=field_color, lw=line_width
        )  # Línea vertical izquierda del área superior
        plt.plot(
            [self.field_width / 2 + area_width / 2, self.field_width / 2 + area_width / 2],
            [self.field_length, self.field_length - area_height],
            color=field_color, lw=line_width
        )  # Línea vertical derecha del área superior

        # Área grande inferior
        plt.plot(
            [self.field_width / 2 - area_width / 2, self.field_width / 2 + area_width / 2],
            [area_height, area_height],
            color=field_color, lw=line_width
        )  # Línea horizontal superior del área inferior
        plt.plot(
            [self.field_width / 2 - area_width / 2, self.field_width / 2 - area_width / 2],
            [0, area_height],
            color=field_color, lw=line_width
        )  # Línea vertical izquierda del área inferior
        plt.plot(
            [self.field_width / 2 + area_width / 2, self.field_width / 2 + area_width / 2],
            [0, area_height],
            color=field_color, lw=line_width
        )  # Línea vertical derecha del área inferior

        # Arcos (puntos de penalti y semicírculos)
        penalty_spot_distance = 11  # Distancia del punto de penalti a la línea de gol
        arc_radius = 9.15  # Radio del semicírculo fuera del área grande

        # Puntos de penalti
        plt.plot(self.field_width / 2, penalty_spot_distance, 'o', color=field_color)
        plt.plot(self.field_width / 2, self.field_length - penalty_spot_distance, 'o', color=field_color)

        
        # Semicírculo superior
        top_arc = patches.Arc(
            (self.field_width / 2, self.field_length - area_height),
            arc_radius * 2,  # Ancho del arco
            arc_radius * 2,  # Alto del arco
            angle=0,  # Sin rotación
            theta1=180,  # Comienza desde la izquierda
            theta2=360,  # Termina en la derecha
            color=field_color,
            lw=line_width
        )
        plt.gca().add_patch(top_arc)

        # Semicírculo inferior
        bottom_arc = patches.Arc(
            (self.field_width / 2, area_height),
            arc_radius * 2,  # Ancho del arco
            arc_radius * 2,  # Alto del arco
            angle=0,  # Sin rotación
            theta1=0,  # Comienza desde la derecha
            theta2=180,  # Termina en la izquierda
            color=field_color,
            lw=line_width
        )
        plt.gca().add_patch(bottom_arc)

        # Mostrar mapa de calor con las líneas de la cancha
        plt.colorbar(label='Densidad de Presencia')
        plt.title(title)
        plt.xlabel("Ancho de la cancha (m)")
        plt.ylabel("Largo de la cancha (m)")
        plt.axis('off')  # Opcional: Quita los ejes si no quieres números en los bordes
        plt.show()

    def draw_player_box(self, video_path, tracked_data, player_id):
        cap = cv2.VideoCapture(video_path)

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reinicia el video al inicio
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Salir del bucle interno si se termina el video

                # Buscar datos para el frame actual
                player_data = tracked_data[player_id]
                for entry in player_data:
                    if entry['frame'] == frame_idx:
                        x1, y1, x2, y2 = int(entry['x1']), int(entry['y1']), int(entry['x2']), int(entry['y2'])
                        # Dibujar el rectángulo y texto
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Track ID: {player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        break  # Salir del bucle si encontramos el box para este frame

                # Mostrar el frame
                cv2.imshow("Player Tracking", frame)

                # Salir si se presiona la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                frame_idx += 1

    def get_first_bounding_box(self, video_path, tracked_data, player_id):
        """
        Retorna la primera bounding box para un jugador específico en el video.

        video_path: ruta del video
        tracked_data: diccionario con datos de tracking
        player_id: id del jugador a buscar

        Retorna:
            bbox: (x1, y1, x2, y2) o None si no se encuentra
        """
        if player_id not in tracked_data:
            raise ValueError(f"El jugador con track_id {player_id} no está en los datos.")

        player_data = tracked_data[player_id]
        if not player_data:
            return None

        first_detection = player_data[0]
        bbox = (first_detection["x1"], first_detection["y1"], first_detection["x2"], first_detection["y2"])
        return bbox
    

    def get_first_bounding_box_image(self, video_path, tracked_data, player_id):
        """
        Retorna la primera bounding box como una imagen para un jugador específico en el video.

        video_path: ruta del video
        tracked_data: diccionario con datos de tracking
        player_id: id del jugador a buscar

        Retorna:
            bbox_image: imagen de la bounding box o None si no se encuentra
        """
        if player_id not in tracked_data:
            raise ValueError(f"El jugador con track_id {player_id} no está en los datos.")

        player_data = tracked_data[player_id]
        if not player_data:
            return None

        first_detection = player_data[0]
        bbox = (first_detection["x1"], first_detection["y1"], first_detection["x2"], first_detection["y2"])
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_detection["frame"])
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        x1, y1, x2, y2 = bbox
        bbox_image = frame[y1:y2, x1:x2]
        return bbox_image
    
    
    def get_first_bounding_box_image_coord(self, video_path, tracked_points):
        """
        Retorna la primera bounding box como una imagen para un jugador específico en el video.

        video_path: ruta del video
        tracked_points: lista de diccionarios con datos de tracking

        Retorna:
            bbox_image: imagen de la bounding box o None si no se encuentra
        """
        if not tracked_points:
            return None

        first_detection = tracked_points[0]
        bbox = (first_detection["x1"], first_detection["y1"], first_detection["x2"], first_detection["y2"])
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_detection["frame"])
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        x1, y1, x2, y2 = bbox
        bbox_image = frame[y1:y2, x1:x2]
        return bbox_image

    def plot_bounding_boxes(self, video_path, tracked_data, player_groups):
        """
        Plots the first bounding box of each player in their respective groups.

        video_path: path to the video
        tracked_data: dictionary with tracking data
        player_groups: dictionary with player groups
        """
        group_titles = ["Group 0", "Group 1", "Group 2"]
        group_images = {0: [], 1: [], 2: []}
        target_height = 100  # Set a target height for all images

        for group in range(3):
            group_players = [player_id for player_id, grp in player_groups.items() if grp == group]
            for player_id in group_players:
                bbox_image = self.get_first_bounding_box_image(video_path, tracked_data, player_id)
                if bbox_image is not None:
                    height, width = bbox_image.shape[:2]
                    aspect_ratio = width / height
                    new_width = int(target_height * aspect_ratio)
                    resized_image = cv2.resize(bbox_image, (new_width, target_height))
                    group_images[group].append((resized_image, player_id))

        combined_rows = []
        for group in range(3):
            if group_images[group]:
                combined_image = cv2.hconcat([img for img, _ in group_images[group]])
                for idx, (_, player_id) in enumerate(group_images[group]):
                    cv2.putText(combined_image, f"Player ID: {player_id}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                combined_rows.append(combined_image)

        if combined_rows:
            max_width = max(img.shape[1] for img in combined_rows)
            for i in range(len(combined_rows)):
                if combined_rows[i].shape[1] < max_width:
                    padding = np.zeros((target_height, max_width - combined_rows[i].shape[1], 3), dtype=np.uint8)
                    combined_rows[i] = cv2.hconcat([combined_rows[i], padding])
            final_image = cv2.vconcat(combined_rows)
            cv2.imshow("Player Groups", final_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_player_heatmap(self, video_path, players_data, roi_points, player_id):
        """
        Plots the heatmap for a specific player using players_data and ROI points.

        video_path: path to the video
        players_data: list of player data dictionaries
        roi_points: list of 6 points for calculating the homography
        player_id: id of the player to plot the heatmap for
        """
        # Find the player data for the given player_id
        player_data = next((player for player in players_data if player["player_id"] == player_id), None)
        if not player_data:
            raise ValueError(f"Player with ID {player_id} not found in players_data.")

        # Extract tracked points for the player
        tracked_points = player_data["tracked_points"]

        # Process tracked data to generate heatmap
        heatmap, transformed_positions, H = self.process_tracked_data({player_id: tracked_points}, roi_points, player_id)

        # Plot the heatmap
        self.plot_heatmap(heatmap)

    def plot_team_heatmap(self, video_path, combined_tracked_points, roi_points, team_id):
        """
        Plots the heatmap for a specific team using combined tracked points and ROI points.

        video_path: path to the video
        combined_tracked_points: list of tracked points for the team
        roi_points: list of 6 points for calculating the homography
        team_id: id of the team to plot the heatmap for
        """
        # Process tracked data to generate heatmap
        heatmap, transformed_positions, H = self.process_tracked_data({team_id: combined_tracked_points}, roi_points, team_id)

        # Plot the heatmap
        self.plot_heatmap(heatmap, title=f"Mapa de Calor del Equipo {team_id}")

    def calculate_real_world_distance(self, tracked_points, roi_points, field_dimensions):
        """Calculate total distance in meters using perspective transform"""
        if not tracked_points or len(tracked_points) < 2:
            return 0

        field_width, field_height = field_dimensions
        
        # Extract only the corner points (removing center points)
        corner_points = np.float32([
            roi_points[0],  # top_left
            roi_points[2],  # top_right
            roi_points[3],  # bottom_right
            roi_points[5]   # bottom_left
        ])
        
        # Define destination points for the corners
        dst_points = np.float32([
            [0, 0],                    # top_left
            [field_width, 0],          # top_right
            [field_width, field_height], # bottom_right
            [0, field_height]           # bottom_left
        ])

        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(corner_points, dst_points)
        
        # Convert tracked points to real-world coordinates
        real_world_points = []
        for point in tracked_points:
            # Extract x,y from the middle of the bounding box
            x = (point['x1'] + point['x2']) / 2
            y = (point['y1'] + point['y2']) / 2
            
            point_array = np.float32([[x, y]])
            transformed = cv2.perspectiveTransform(point_array.reshape(-1, 1, 2), matrix)
            real_world_points.append(transformed.reshape(2))
        
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(real_world_points)):
            dx = real_world_points[i][0] - real_world_points[i-1][0]
            dy = real_world_points[i][1] - real_world_points[i-1][1]
            distance = np.sqrt(dx*dx + dy*dy)
            total_distance += distance
        
        return round(total_distance, 2)

