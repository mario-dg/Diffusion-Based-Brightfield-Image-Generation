import tkinter as tk
from tkinter import messagebox, Label
from PIL import Image, ImageTk, ImageOps
import jsonlines
import os


base_path = "datasets/scc_v1/train"
# Load metadata
metadata_file = os.path.join(base_path, 'metadata.jsonl')
metadata = []
with jsonlines.open(metadata_file) as reader:
    for obj in reader:
        metadata.append(obj)

# Initialize some variables
current_index = 0
total_images = len(metadata)

def save_metadata():
    with jsonlines.open(metadata_file, mode='w') as writer:
        writer.write_all(metadata)

def update_well_edge(value):
    global current_index
    if isinstance(value, int):
        current_index += value
        if current_index < 0:
            current_index = 0
            messagebox.showwarning("First Image", "You are at the start of the image stack.")
        if current_index >= total_images:
            current_index = total_images - 1
            messagebox.showinfo("Complete", "All images have been reviewed.")
            root.destroy()
        display_image()
    elif value is None:
        metadata[current_index]['well_edge'] = not metadata[current_index]['well_edge']
        save_metadata()
        display_image()

def display_image():
    global photo_image
    global current_index

    file_name = metadata[current_index]['file_name']
    well_edge_status = metadata[current_index]['well_edge']
    
    # Update labels
    file_name_label.config(text=file_name)
    progress_label.config(text=f"{current_index + 1}/{total_images}")
    well_edge_label.config(text="Well Edge" if well_edge_status else "No Well Edge",
                           fg="green" if well_edge_status else "red")

    # Load and display imag

    image_path = metadata[current_index]['file_name']
    img = Image.open(os.path.join(base_path, image_path))
    
    border_color = "green" if well_edge_status else "red"
    border_width = 5  # Thickness of the border
    img_with_border = ImageOps.expand(img, border=border_width, fill=border_color)
    img_with_border = img_with_border.resize((512 + border_width * 2, 512 + border_width * 2))
    
    photo_image = ImageTk.PhotoImage(img_with_border)
    image_label.config(image=photo_image)
    image_label.photo = photo_image  # Keep a reference

def on_key_press(event):
    if event.keysym == 'Left':
        update_well_edge(-1)
    elif event.keysym == 'Right':
        update_well_edge(1)
    elif event.keysym == 'Return':
        update_well_edge(None)

# Setup GUI
root = tk.Tk()
root.title("Image Review")

# Create labels
file_name_label = Label(root, font=('Arial', 12))
file_name_label.pack()
progress_label = Label(root, font=('Arial', 12))
progress_label.pack()

image_label = Label(root)
image_label.pack()

well_edge_label = Label(root, font=('Arial', 12))
well_edge_label.pack()

root.bind('<Left>', on_key_press)
root.bind('<Right>', on_key_press)
root.bind('<Return>', on_key_press)

# On close event
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to save before quitting?"):
        save_metadata()
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Display the first image
display_image()

root.mainloop()
