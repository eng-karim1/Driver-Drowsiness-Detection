import os
import cv2
import numpy as np

# Folder paths for Closed and Open images
closed_dir = 'Closed'  # Path for Closed images
open_dir = 'Open'      # Path for Open images

# Initialize lists to store images and labels
X = []
Y = []

# Function to load images from a folder and resize them
def load_images_from_folder(folder, label):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)  # Get image file path
        img = cv2.imread(img_path)  # Read the image
        if img is not None:
            # Resize image to (32, 32)
            img_resized = cv2.resize(img, (32, 32))
            X.append(img_resized)  # Add image to X list
            Y.append(label)        # Add label to Y list

# Load images from both folders
load_images_from_folder(closed_dir, label=0)  # 0 for Closed eye class
load_images_from_folder(open_dir, label=1)    # 1 for Open eye class

# Convert lists to NumPy arrays for easy processing
X = np.array(X)
Y = np.array(Y)

# Save the data as a .npz file
np.savez('data.npz', X, Y)
print("Data saved successfully in 'data.npz'")  # Confirm data is saved
