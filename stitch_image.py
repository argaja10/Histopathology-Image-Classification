
from PIL import Image
import os
import re
import numpy as np

def stitch_images_from_folder(folder_path, patch_size=(50, 50)):
    # Get list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # Regex to extract coordinates from filenames
    coord_pattern = re.compile(r'_x(\d+)_y(\d+)_')
    
    # Determine the size of the final stitched image
    max_x = 0
    max_y = 0
    max_pixel_values = [0, 0, 0]  # To store max pixel values for R, G, B channels

    for image_file in image_files:
        # Extract x and y coordinates using regex
        match = coord_pattern.search(image_file)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
            # Open the image patch
            patch = Image.open(os.path.join(folder_path, image_file))
            patch_array = np.array(patch)
            
            # Update max pixel values for each channel
            for i in range(3):
                max_pixel_values[i] = max(max_pixel_values[i], patch_array[:, :, i].max())
                
    # Calculate the size of the final image
    stitched_width = (max_x // patch_size[0] + 1) * patch_size[0]
    stitched_height = (max_y // patch_size[1] + 1) * patch_size[1]
    
    # Create a blank canvas for the final image
    stitched_image = Image.new('RGB', (stitched_width, stitched_height), (max_pixel_values[0],max_pixel_values[1],max_pixel_values[2]))
    
    # Paste each patch onto the canvas
    for image_file in image_files:
        # Open the image patch
        patch = Image.open(os.path.join(folder_path, image_file))
        
        # Extract x and y coordinates using regex
        match = coord_pattern.search(image_file)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            
            # Calculate the position to paste the patch
            position = (x, y)
            
            # Paste the patch onto the canvas
            stitched_image.paste(patch, position)
    
    return stitched_image

