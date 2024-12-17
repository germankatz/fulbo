# Python program to create a basic form 
# GUI application using the customtkinter module
from CTkMessagebox import CTkMessagebox
import customtkinter as ctk
import tkinter as tk
import time
import cv2
from PIL import Image, ImageTk
from functions.ROI_definer import ROIDefiner
from functions.YOLO_custom import YOLO_custom
from functions.Process import Process
from functions.modules.select_teams import SelectTeams
from functions.results_window import ResultsWindow

# Sets the appearance of the window
# Supported modes : Light, Dark, System
# "System" sets the appearance mode to 
# the appearance mode of the system
ctk.set_appearance_mode("System") 

# Sets the color of the widgets in the window
# Supported themes : green, dark-blue, blue 
ctk.set_default_color_theme("green") 

# Dimensions of the window
appWidth, appHeight = 600, 700


# App Class
class App(ctk.CTk):
	path1 = None
	path2 = None
	joinedPath = None
	ROIpoints = None

	# The layout of the window will be written
	# in the init function itself
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# Set the background color to green
		self.configure(fg_color="#344E41")

		# Sets the title of the window to "App"
		self.title("PROFUL") 
		# Sets the dimensions of the window to 600x700
		self.geometry(f"{appWidth}x{appHeight}") 

		# Title Label
		self.titleLabel = ctk.CTkLabel(self, text="PROFUL", font=("Times new roman", 24, "bold"))
		self.titleLabel.grid(row=0, column=0, columnspan=4, pady=(20, 0), sticky="n")

		# Subtitle Label
		self.subtitleLabel = ctk.CTkLabel(self, text="Procesamiento de videos deportivos", font=("Times new roman", 16, "bold"))
		self.subtitleLabel.grid(row=1, column=0, columnspan=4, pady=(0, 20), sticky="n")

		# Section Title Label
		self.sectionTitleLabel = ctk.CTkLabel(self, text="1. Carga de videos", font=("Arial", 16, "bold"))
		self.sectionTitleLabel.grid(row=2, column=0, columnspan=4, padx=20, pady=(0, 5), sticky="w")

		# Video 1 Label 
		self.video1Label = ctk.CTkLabel(self, text="Video 1 (izquierda)")
		self.video1Label.grid(row=3, column=0, padx=20, pady=0, sticky="w")
		
		# Video 1 Button
		self.video1Button = ctk.CTkButton(self, text="Upload Video 1", command=self.upload_video1)
		self.video1Button.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
		
		# Video 1 path 
		self.video1Path = ctk.CTkLabel(self, text="")
		self.video1Path.grid(row=5, column=0, padx=20, pady=0, sticky="w")

		# Video 2 Label
		self.video2Label = ctk.CTkLabel(self, text="Video 2 (derecha)")
		self.video2Label.grid(row=3, column=1, padx=20, pady=0, sticky="w")

		# Video 2 Button
		self.video2Button = ctk.CTkButton(self, text="Upload Video 2", command=self.upload_video2)
		self.video2Button.grid(row=4, column=1, padx=20, pady=10, sticky="ew")
		
		# Video 2 path 
		self.video2Path = ctk.CTkLabel(self, text="")
		self.video2Path.grid(row=5, column=1, padx=20, pady=0, sticky="w")
        
		# Dimension Title Label
		self.sectionTitleLabel = ctk.CTkLabel(self, text="2. Dimensión de la cancha", font=("Arial", 16, "bold"))
		self.sectionTitleLabel.grid(row=6, column=0, columnspan=4, padx=20, pady=(0, 5), sticky="w")


		# Dimension alto Label aligned left
		self.altoLabel = ctk.CTkLabel(self, text="Alto", anchor="w")
		self.altoLabel.grid(row=7, column=0,
						padx=20, pady=0,
						sticky="ew")

		# Dimension alto Field
		self.altoEntry = ctk.CTkEntry(self,
							placeholder_text="Alto en metros",
							width=30)
		self.altoEntry.insert(0, "108")
		self.altoEntry.grid(row=8, column=0, padx=20,
						pady=(0,20), sticky="ew")
		
        
		# Dimension ancho Label
		self.anchoLabel = ctk.CTkLabel(self, text="Ancho", anchor="w")
		self.anchoLabel.grid(row=7, column=1,
						padx=20, pady=0,
						sticky="ew")

		# Dimension ancho Field
		self.anchoEntry = ctk.CTkEntry(self,
							placeholder_text="Ancho en metros")
		self.anchoEntry.insert(0, "68")
		self.anchoEntry.grid(row=8, column=1, padx=20,
						pady=(0,20), sticky="ew")

		# Generate Button
		self.generateResultsButton = ctk.CTkButton(self,
										text="Unir videos",
										command=self.joinVideos)
		self.generateResultsButton.grid(row=9, column=0,
										columnspan=2, padx=20, 
										pady=20, sticky="ew")


	def joinVideos(self):
		print("Uniendo videos")
		"""
		Take the two video paths, show a loading spinner for 3 seconds,
		and display a frame from the first video.
		"""
		# Check if both videos have been uploaded
		if not self.path1 or not self.path2:
			CTkMessagebox(title="Error", message="Please upload both videos first.", icon="cancel")  # Use a valid icon
			return

		# Show loading spinner
		loading_label = ctk.CTkLabel(self, text="Loading...", font=("Arial", 16, "bold"))
		loading_label.grid(row=10, column=0, columnspan=2, pady=20)
		self.update()

		# Simulate loading time
		time.sleep(3)

		# Remove loading spinner
		loading_label.grid_forget()

		# Display a frame from the first video
		first_frame = self.getFirstFrame(self.path1)
		if first_frame is not None:
			# TODO Combined frames
			self.joinedPath = self.path1 
			self.displayFrame(first_frame)
		else:
			CTkMessagebox(title="Error", message="Failed to read frame", icon="cancel")

	def getFirstFrame(self, video_path):
		cap = cv2.VideoCapture(video_path)
		if cap.isOpened():
			ret, frame = cap.read()
			if ret:
				return frame
			else:
				print("Failed to read frame from Video 1.")
				cap.release()
				return None
		else:
			print("Failed to open Video 1.")
			cap.release()
			return None

	def displayFrame(self, frame):
		# Convert the frame to a format suitable for Tkinter
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_image = Image.fromarray(frame_rgb)
		frame_image.thumbnail((200, 200))  # Resize to fit in the UI
		frame_photo = ImageTk.PhotoImage(frame_image)

		# Display the frame in the UI
		self.resultLabel = ctk.CTkLabel(self, text="3. Marcar cancha", font=("Arial", 16, "bold"))
		self.resultLabel.grid(row=2, column=3, padx=20, pady=(20, 0), sticky="w")

		self.frameLabel = ctk.CTkLabel(self, text="", image=frame_photo, cursor="hand2")
		self.frameLabel.image = frame_photo  # Keep a reference to avoid garbage collection
		self.frameLabel.grid(row=3, column=3, padx=20, pady=10, sticky="n")
		self.frameLabel.bind("<Button-1>", lambda e: self.showFullImage(self.joinedPath))

		self.processButton = ctk.CTkButton(self, text="Procesar Video", command=self.processVideo)
		self.processButton.grid(row=4, column=3, padx=20, pady=10, sticky="ew")

	def showFullImage(self, path):
		points, marked_frame = ROIDefiner.define_roi_from_video(path)
		self.ROIpoints = points
		if marked_frame is not None:
			# Convert the marked frame to a format suitable for Tkinter
			marked_frame_rgb = cv2.cvtColor(marked_frame, cv2.COLOR_BGR2RGB)
			marked_frame_image = Image.fromarray(marked_frame_rgb)
			marked_frame_image.thumbnail((200, 200))  # Resize to fit in the UI
			marked_frame_photo = ImageTk.PhotoImage(marked_frame_image)

			# Update the frameLabel with the marked frame
			self.frameLabel.configure(image=marked_frame_photo)
			self.frameLabel.image = marked_frame_photo  # Keep a reference to avoid garbage collection

	def processVideo(self):
		print("Processing video...")
		model = YOLO_custom("yolo11n.pt", True)

		def progress_callback(current_frame, total_frames):
			if current_frame == -1:
				CTkMessagebox(title="Error", message="Tracking failed.", icon="cancel")
			else:
				progress = (current_frame / total_frames) * 100
				# print(f"Progress: {current_frame}/{total_frames} ({progress:.2f}%)")
				self.progressLabel.configure(text=f"Progreso: {current_frame}/{total_frames} ({progress:.2f}%)")
				self.update_idletasks()  # Ensure the label is updated

		# Add a progress label
		self.progressLabel = ctk.CTkLabel(self, text="Progreso: 0%", font=("Arial", 12))
		self.progressLabel.grid(row=5, column=3, padx=20, pady=10, sticky="ew")

		tracked_data = model.track(self.joinedPath, self.ROIpoints, show_plot=False, progress_callback=progress_callback)

		# Classify players into teams
		# Update the progress label
		self.progressLabel.configure(text=f"Clasificando jugadores en equipos...")
		self.update_idletasks()  # Ensure the label is updated
		# Select teams
		select_teams = SelectTeams(self.joinedPath, tracked_data)
		player_groups = select_teams.classify_players()
		print("Player groups:", player_groups)

		# Create player data structure
		players_data = []
		for track_id, track_info in tracked_data.items():
			player = {
				"name": "",
				"player_id": track_id,
				"team": player_groups.get(track_id, "Unknown"),
				"tracked_points": track_info
			}
			players_data.append(player)

		# Remove progress label after processing
		self.progressLabel.grid_forget()

		# Add your video processing code here
		CTkMessagebox(title="Proceso completado", message="El video ha sido procesado exitosamente.", icon="check")

		# Change the button text
		self.processButton.configure(text="Ver resultados", command=lambda: ResultsWindow(self, players_data, self.joinedPath, self.ROIpoints))

	def createText(self):
		checkboxValue = ""

		# .get() is used to get the value of the checkboxes and entryfields

		if self.choice1._check_state and self.choice2._check_state:
			checkboxValue += self.choice1.get() + " and " + self.choice2.get()
		elif self.choice1._check_state:
			checkboxValue += self.choice1.get()
		elif self.choice2._check_state:
			checkboxValue += self.choice2.get()
		else:
			checkboxValue = "none of the available options"

		# Constructing the text variable
		text = f"{self.nameEntry.get()} : \n{self.genderVar.get()} {self.ageEntry.get()} years old and prefers {checkboxValue}\n"
		text += f"{self.genderVar.get()} currently a {self.occupationOptionMenu.get()}"

		return text

	# Function to handle video 1 upload
	def upload_video1(self):
		filepath = tk.filedialog.askopenfilename()
		self.path1 = filepath
		print(f"Video 1 path: {filepath}")
		self.video1Path.configure(text=("...",filepath[-22:]))

	# Function to handle video 2 upload
	def upload_video2(self):
		filepath = tk.filedialog.askopenfilename()
		self.path2 = filepath
		print(f"Video 2 path: {filepath}")
		self.video2Path.configure(text=("...",filepath[-22:]))

if __name__ == "__main__":
	app = App()
	# Used to run the application
	app.mainloop()
