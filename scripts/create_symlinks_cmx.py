import os
import json
from pathlib import Path

# Paths to your current dataset
dataset_path = "data/lindenthal_camera_traps/lindenthal_coco/"
output_path = "data/lindenthal_camera_traps_cmx/"

# Create necessary directories for the new structure
Path(os.path.join(output_path, "RGBFolder")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_path, "ModalXFolder")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_path, "LabelFolder")).mkdir(parents=True, exist_ok=True)

# Load the train/test splits
with open(os.path.join(dataset_path, 'train.json')) as f:
    train_split = json.load(f)
with open(os.path.join(dataset_path, 'test.json')) as f:
    test_split = json.load(f)

# Symlink images
for video_id in os.listdir(os.path.join(dataset_path, "images")):
    for image_id in os.listdir(os.path.join(dataset_path, "images", video_id, "color")):
        rgb_image = os.path.join(dataset_path, "images", video_id, "color", image_id)
        depth_image = os.path.join(dataset_path, "images", video_id, "depth", image_id)
        
        # Symlink RGB and ModalX (depth) images
        rgb_link = os.path.join(output_path, "RGBFolder", image_id)
        depth_link = os.path.join(output_path, "ModalXFolder", image_id)
        os.symlink(rgb_image, rgb_link)
        os.symlink(depth_image, depth_link)

# Write train.txt and test.txt
with open(os.path.join(output_path, "train.txt"), 'w') as train_file:
    for train_item in train_split:
        train_file.write(f"{train_item}\n")

with open(os.path.join(output_path, "test.txt"), 'w') as test_file:
    for test_item in test_split:
        test_file.write(f"{test_item}\n")
