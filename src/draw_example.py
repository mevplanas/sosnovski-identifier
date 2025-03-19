import cv2
import numpy as np
import os 

# Infering the current directory the file is in 
current_directory = os.path.dirname(__file__)
img_path = os.path.join(current_directory, '../docs/DJI_0066_W.JPG')
txt_path = os.path.join(current_directory, '../docs/DJI_0066_W.txt')
polygon_img_path = os.path.join(current_directory, '../docs/DJI_0066_W_with_polygon.JPG')

# Load the image
img = cv2.imread(img_path)
height, width = img.shape[:2]

# Read polygon coordinates from the text file.
# Assume each line is formatted as: x,y
points = []
with open(txt_path, 'r') as f:
    for line in f:
        # Split the line into parts (first element is the class id)
        parts = line.strip().split()
        if not parts:
            continue
        
        # Remove the class id (first element) and convert coordinates to float
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            print("Invalid number of coordinates in line:", line)
            continue

        # Convert normalized coordinates to pixel coordinates
        points = []
        for i in range(0, len(coords), 2):
            # Normalized x and y
            x_norm = coords[i]
            y_norm = coords[i+1]
            # Convert to pixel coordinates
            x_pixel = int(x_norm * width)
            y_pixel = int(y_norm * height)
            points.append([x_pixel, y_pixel])
        
        # Convert list to numpy array (required by cv2.polylines)
        points = np.array(points, dtype=np.int32)

        # Draw the polygon on the image;
        cv2.polylines(img, [points], isClosed=True, color=(0, 0, 255), thickness=10)

# Saving the image
cv2.imwrite(polygon_img_path, img)