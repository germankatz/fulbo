import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
from functions.Process import Process
from functions.modules.calculate_distance import calculate_distance_traveled
import cv2
import time

ctk.set_default_color_theme("green") 

class ResultsWindow(ctk.CTkToplevel):
    def __init__(self, parent, players_data, joinedPath, ROIpoints):
        super().__init__(parent)
        self.title("Resultados")
        self.geometry("800x600")
        self.configure(fg_color="#344E41")
        self.joinedPath = joinedPath
        self.process = Process()
        self.players_data = players_data
        self.name_update_time = 0
        self.ROIpoints = ROIpoints

        # Add a title label
        title_label = ctk.CTkLabel(self, text="Resultados de Tracking", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Create a frame for the list of players
        players_frame = ctk.CTkFrame(self)
        players_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Add a canvas for scrolling
        canvas = tk.Canvas(players_frame, bg="#344E41")
        canvas.pack(side="left", fill="both", expand=True)

        # Add a scrollbar
        scrollbar = ctk.CTkScrollbar(players_frame, orientation="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        # Configure the canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Create a frame inside the canvas
        self.scrollable_frame = ctk.CTkFrame(canvas, fg_color="#344E41")
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Populate the scrollable frame with player items
        self.player_frames = {}
        self.populate_scrollable_frame(players_data)

        # Add a close button
        close_button = ctk.CTkButton(self, text="Cerrar", command=self.destroy)
        close_button.pack(side="right", pady=10, padx=5)

        # Add buttons to view team heatmaps
        team0_heatmap_button = ctk.CTkButton(self, text="Ver mapa calor equipo 0", command=lambda: self.show_team_heatmap(0))
        team0_heatmap_button.pack(side="left", pady=10, padx=5)

        team1_heatmap_button = ctk.CTkButton(self, text="Ver mapa calor equipo 1", command=lambda: self.show_team_heatmap(1))
        team1_heatmap_button.pack(side="left", pady=10, padx=5)

    def populate_scrollable_frame(self, players_data):
        for player in players_data:
            self.add_player_item(player)

    def add_player_item(self, player):
        # Create a frame for the player item
        player_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="#344E41")
        player_frame.pack(fill="x", padx=5, pady=2)
        self.player_frames[player["player_id"]] = player_frame

        # Add the bounding box image
        bbox_image = self.process.get_first_bounding_box_image_coord(self.joinedPath, player["tracked_points"])
        if bbox_image is not None:
            bbox_image_rgb = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)
            bbox_image_pil = Image.fromarray(bbox_image_rgb)
            aspect_ratio = bbox_image_pil.width / bbox_image_pil.height
            new_width = int(100 * aspect_ratio)
            bbox_image_pil = bbox_image_pil.resize((new_width, 100), Image.Resampling.LANCZOS)
            bbox_image_tk = ImageTk.PhotoImage(bbox_image_pil)
            image_label = ctk.CTkLabel(player_frame, image=bbox_image_tk, text="", fg_color="#344E41")
            image_label.image = bbox_image_tk  # Keep a reference to avoid garbage collection
            image_label.grid(row=0, column=0, rowspan=3, padx=5, pady=5)

        # Add player details in the first column
        details_frame = ctk.CTkFrame(player_frame, fg_color="#344E41")
        details_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5)

        id_label = ctk.CTkLabel(details_frame, text=f"ID: {player['player_id']}", font=("Arial", 12), fg_color="#344E41")
        id_label.pack(anchor="w")

        name_entry = ctk.CTkEntry(details_frame, font=("Arial", 12))
        name_entry.insert(0, player["name"])
        name_entry.pack(anchor="w", pady=5)
        name_entry.bind("<KeyRelease>", lambda e, p=player: self.debounce_update_name(p, name_entry.get()))

        team_selector = ctk.CTkOptionMenu(details_frame, values=["Equipo 1", "Equipo 2", "Equipo 3"], font=("Arial", 12))
        team_selector.set(f"Equipo {player['team']}")
        team_selector.pack(anchor="w", pady=5)
        team_selector.bind("<<ComboboxSelected>>", lambda e, p=player: self.update_team(p, team_selector.get()))

        # Add merge player option in the second column
        merge_frame = ctk.CTkFrame(player_frame, fg_color="#344E41")
        merge_frame.grid(row=0, column=2, rowspan=3, padx=5, pady=5)

        merge_label = ctk.CTkLabel(merge_frame, text="Unir con jugador", font=("Arial", 12), fg_color="#344E41")
        merge_label.pack(anchor="w")

        merge_selector = ctk.CTkOptionMenu(merge_frame, values=["Ninguno"], font=("Arial", 12))
        merge_selector.pack(anchor="w", pady=5)
        merge_selector.bind("<Button-1>", lambda e, p=player: self.populate_merge_selector(merge_selector, player))
        merge_selector.bind("<<ComboboxSelected>>", lambda e, p=player: self.merge_players(p, merge_selector.get()))

        # Add distance label
        distance_label = ctk.CTkLabel(player_frame, 
                                    text=f"Distancia recorrida: {player['distance']} metros",
                                    font=("Arial", 12),
                                    fg_color="#344E41")
        distance_label.grid(row=1, column=3, rowspan=1, padx=5, pady=5)

        # Add heatmap button in the third column
        heatmap_button = ctk.CTkButton(player_frame, text="Ver mapa calor", command=lambda p=player: self.show_heatmap(p, player["player_id"]))
        heatmap_button.grid(row=0, column=3, rowspan=1, padx=5, pady=5)  # Adjust row

        # Add delete button in the fourth column
        delete_button = ctk.CTkButton(player_frame, text="Eliminar", command=lambda p=player: self.delete_player(p))
        delete_button.grid(row=0, column=4, rowspan=3, padx=5, pady=5)

    def debounce_update_name(self, player, new_name):
        current_time = time.time()
        if current_time - self.name_update_time > 1.5:  # 1500ms debounce
            self.update_name(player, new_name)
            self.name_update_time = current_time

    def update_name(self, player, new_name):
        player["name"] = new_name
        self.update_player_item(player)

    def update_team(self, player, new_team):
        player["team"] = int(new_team.split(" ")[1])
        self.update_player_item(player)

    def merge_players(self, player, selected_player_info):
        if selected_player_info == "Ninguno":
            return
        selected_player_id = int(selected_player_info.split(" - ")[0])
        selected_player = next(p for p in self.players_data if p["player_id"] == selected_player_id)
        player["tracked_points"].extend(selected_player["tracked_points"])
        self.players_data.remove(selected_player)
        self.update_player_item(player)
        self.remove_player_item(selected_player)

    def show_heatmap(self, player, player_id):
        self.process.plot_player_heatmap(self.joinedPath, self.players_data, self.ROIpoints, player_id)

    def show_team_heatmap(self, team_id):
        team_players = [player for player in self.players_data if player["team"] == team_id]
        if not team_players:
            return

        combined_tracked_points = []
        for player in team_players:
            combined_tracked_points.extend(player["tracked_points"])

        self.process.plot_team_heatmap(self.joinedPath, combined_tracked_points, self.ROIpoints, team_id)

    def delete_player(self, player):
        self.players_data.remove(player)
        self.remove_player_item(player)

    def update_player_item(self, player):
        player_frame = self.player_frames[player["player_id"]]
        for widget in player_frame.winfo_children():
            widget.destroy()
        self.populate_player_item(player, player_frame)

    def remove_player_item(self, player):
        player_frame = self.player_frames.pop(player["player_id"])
        player_frame.destroy()

    def populate_player_item(self, player, player_frame):
        # Add the bounding box image
        bbox_image = self.process.get_first_bounding_box_image_coord(self.joinedPath, player["tracked_points"])
        if bbox_image is not None:
            bbox_image_rgb = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)
            bbox_image_pil = Image.fromarray(bbox_image_rgb)
            aspect_ratio = bbox_image_pil.width / bbox_image_pil.height
            new_width = int(100 * aspect_ratio)
            bbox_image_pil = bbox_image_pil.resize((new_width, 100), Image.Resampling.LANCZOS)
            bbox_image_tk = ImageTk.PhotoImage(bbox_image_pil)
            image_label = ctk.CTkLabel(player_frame, image=bbox_image_tk, text="", fg_color="#344E41")
            image_label.image = bbox_image_tk  # Keep a reference to avoid garbage collection
            image_label.grid(row=0, column=0, rowspan=3, padx=5, pady=5)

        # Add player details in the first column
        details_frame = ctk.CTkFrame(player_frame, fg_color="#344E41")
        details_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5)

        id_label = ctk.CTkLabel(details_frame, text=f"ID: {player['player_id']}", font=("Arial", 12), fg_color="#344E41")
        id_label.pack(anchor="w")

        name_entry = ctk.CTkEntry(details_frame, font=("Arial", 12))
        name_entry.insert(0, player["name"])
        name_entry.pack(anchor="w", pady=5)
        name_entry.bind("<KeyRelease>", lambda e, p=player: self.debounce_update_name(p, name_entry.get()))

        team_selector = ctk.CTkOptionMenu(details_frame, values=["Equipo 1", "Equipo 2", "Equipo 3"], font=("Arial", 12))
        team_selector.set(f"Equipo {player['team']}")
        team_selector.pack(anchor="w", pady=5)
        team_selector.bind("<<ComboboxSelected>>", lambda e, p=player: self.update_team(p, team_selector.get()))

        # Add merge player option in the second column
        merge_frame = ctk.CTkFrame(player_frame, fg_color="#344E41")
        merge_frame.grid(row=0, column=2, rowspan=3, padx=5, pady=5)

        merge_label = ctk.CTkLabel(merge_frame, text="Unir con jugador", font=("Arial", 12), fg_color="#344E41")
        merge_label.pack(anchor="w")

        merge_selector = ctk.CTkOptionMenu(merge_frame, values=["Ninguno"], font=("Arial", 12))
        merge_selector.pack(anchor="w", pady=5)
        merge_selector.bind("<Button-1>", lambda e, p=player: self.populate_merge_selector(merge_selector, player))
        merge_selector.bind("<<ComboboxSelected>>", lambda e, p=player: self.merge_players(p, merge_selector.get()))

        # Add distance label
        distance_label = ctk.CTkLabel(player_frame, 
                                    text=f"Distancia recorrida: {player['distance']} metros",
                                    font=("Arial", 12),
                                    fg_color="#344E41")
        distance_label.grid(row=1, column=3, rowspan=1, padx=5, pady=5)

        # Add heatmap button in the third column
        heatmap_button = ctk.CTkButton(player_frame, text="Ver mapa calor", command=lambda p=player: self.show_heatmap(p, player["player_id"]))
        heatmap_button.grid(row=0, column=3, rowspan=1, padx=5, pady=5)  # Adjust row

        # Add delete button in the fourth column
        delete_button = ctk.CTkButton(player_frame, text="Eliminar", command=lambda p=player: self.delete_player(p))
        delete_button.grid(row=0, column=4, rowspan=3, padx=5, pady=5)

    def populate_merge_selector(self, merge_selector, player):
        other_players = ["Ninguno"] + [f"{p['player_id']} - {p['name']}" for p in self.players_data if p["player_id"] != player["player_id"]]
        merge_selector.configure(values=other_players)
