# Python program to create a basic form 
# GUI application using the customtkinter module
import customtkinter as ctk
import tkinter as tk

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
		self.anchoEntry.grid(row=8, column=1, padx=20,
						pady=(0,20), sticky="ew")

		# Generate Button
		self.generateResultsButton = ctk.CTkButton(self,
										text="Unir videos",
										command=self.generateResults)
		self.generateResultsButton.grid(row=9, column=0,
										columnspan=2, padx=20, 
										pady=20, sticky="ew")



	# This function is used to insert the 
	# details entered by users into the textbox
	def generateResults(self):
		self.displayBox.delete("0.0", "200.0")
		text = self.createText()
		self.displayBox.insert("0.0", text)

	# This function is used to get the selected 
	# options and text from the available entry
	# fields and boxes and then generates 
	# a prompt using them
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
		print(f"Video 1 path: {filepath}")
		self.video1Path.configure(text=("...",filepath[-22:]))

	# Function to handle video 2 upload
	def upload_video2(self):
		filepath = tk.filedialog.askopenfilename()
		print(f"Video 2 path: {filepath}")
		self.video2Path.configure(text=("...",filepath[-22:]))
	

if __name__ == "__main__":
	app = App()
	# Used to run the application
	app.mainloop()
