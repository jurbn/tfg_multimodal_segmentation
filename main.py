"""
This file contains the main code for the project.
"""
import os
import rosbag
import pyrealsense2 as rs
from src.utils.video_player import play_video_from_bag

BAG_FOLDER = "data/lindenthal-camera-traps/bagfiles"

if __name__ == "__main__":
    # iterate over all files in the bag folder
    for file_name in os.listdir(BAG_FOLDER):
        if file_name.endswith(".bag"):
            # generate the full path to the file
            file_path = f"{BAG_FOLDER}/{file_name}"
            play_video_from_bag(file_path)
