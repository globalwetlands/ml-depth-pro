import os
from PIL import Image
import torch
import depth_pro
import numpy as np
import matplotlib.pyplot as plt

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Use Argparse after
# Directory containing the input images
input_dir = "data/images/"

# Directory to save the output depth maps
output_dir = "data/depth-maps/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all image files in the input directory
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])

# Process each image
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)

    # Load and preprocess the image
    image, _, f_px = depth_pro.load_rgb(image_path)
    image_input = transform(image)

    # Run inference
    with torch.no_grad():
        prediction = model.infer(image_input, f_px=f_px)
        depth = prediction["depth"]  # Depth in meters

    # Convert depth to NumPy array
    depth_map = depth.squeeze().cpu().numpy()

    # Normalize the depth map for visualization
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
    print(f"Min: {depth_min} | Max: {depth_max} | Norm: {depth_norm} | Depth: {depth}")

    # Save the depth map visualization
    output_path = os.path.join(output_dir, image_file)
    plt.imsave(output_path, depth_norm, cmap="plasma")

    print(f"Processed {image_file}")
