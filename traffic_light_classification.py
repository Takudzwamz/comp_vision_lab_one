
import cv2
import numpy as np
import os
from zipfile import ZipFile
import shutil

# Step 1: Extract ZIP file
zip_dir = 'tlights.zip'  # Specify the path to your ZIP file here
extract_dir = 'output'  # Specify the path to the extraction directory here

with ZipFile(zip_dir, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Step 2: Get the list of files
tlights_dir = os.path.join(extract_dir, 'tlights')
file_list = os.listdir(tlights_dir)
file_list.sort()

# Create directories for each category
output_dirs = {
    "red": os.path.join(extract_dir, 'red_signals'),
    "green": os.path.join(extract_dir, 'green_signals'),
    "yellow": os.path.join(extract_dir, 'yellow_signals'),
    "unclassifiable": os.path.join(extract_dir, 'unclassifiable_signals')
}
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Function to classify traffic light image
def classify_traffic_light(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Apply contrast enhancement (histogram equalization)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Create a circular mask to focus on the central region
    h, w = img.shape[0], img.shape[1] # height and width of the image
    center = (w//2, h//2) # center of the image
    radius = min(w, h)//2 # radius of the circle
    mask = np.zeros((h, w), dtype=np.uint8) # create a black mask
    cv2.circle(mask, center, radius, (1, 1, 1), thickness=-1) # draw a white circle on the mask
    
    # Apply the mask
    img = cv2.bitwise_and(img, img, mask=mask)
    
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define the color ranges for red, yellow, and green in the HSV space
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    
    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([90, 255, 255])
    
    # Create masks for each color
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.add(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    
    # Count the number of pixels for each color
    count_red = cv2.countNonZero(mask_red)
    count_yellow = cv2.countNonZero(mask_yellow)
    count_green = cv2.countNonZero(mask_green)
    
    
    
    # Determine the dominant color
    max_count = max(count_red, count_yellow, count_green)
    if max_count < 50:  # Threshold value for the minimum number of pixels
        return "unclassifiable"
    elif max_count == count_red:
        return "red"
    elif max_count == count_yellow:
        return "yellow"
    elif max_count == count_green:
        return "green"

# Initialize lists to store classified file names
red_signals = []
green_signals = []
yellow_signals = []
unclassifiable_signals = []

# Loop through all files and classify them
for file_name in file_list:
    file_path = os.path.join(tlights_dir, file_name)
    classification = classify_traffic_light(file_path)
    
    # Move the file to the corresponding directory
    output_dir = output_dirs[classification]
    shutil.move(file_path, os.path.join(output_dir, file_name))

    # Add the file name to the corresponding list
    if classification == "red":
        red_signals.append(file_name)
    elif classification == "green":
        green_signals.append(file_name)
    elif classification == "yellow":
        yellow_signals.append(file_name)
    else:
        unclassifiable_signals.append(file_name)

# Print the number of images in each category
print("Red signals:", len(red_signals))
print("Green signals:", len(green_signals))
print("Yellow signals:", len(yellow_signals))
print("Unclassifiable signals:", len(unclassifiable_signals))
