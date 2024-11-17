import imageio
import time
import json
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import pyrealsense2 as rs
import numpy as np
import cv2 as cv

DATASET_PATH = 'data/lindenthal-camera-traps/bagfiles'
OUTPUT_PATH = 'data/lindenthal-camera-traps/lindenthal_coco/images'

def create_dir(bagfile_path, video_path):
    """
    Creates the corresponding directory for the bagfile if it doesn't exist yet.
    It also generates an intrinsics.json file with the intrinsics of the video.
    """
    dir_processed = True
    if not os.path.exists(video_path):
        print(f'Creating directory for {video_name}')
        os.makedirs(video_path)
        dir_processed = False

    color_path = os.path.join(video_path, 'color')
    if not os.path.exists(color_path):
        os.makedirs(color_path)
        dir_processed = False
    depth_path = os.path.join(video_path, 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
        dir_processed = False
    
    # Generate intrinsics.json if it doesn't exist
    intrinsics_path = os.path.join(video_path, 'intrinsics.json')
    if not os.path.exists(intrinsics_path):
        intrinsics_dict = {}
        # load the video
        config = rs.config()
        config.enable_device_from_file(bagfile_path)
        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        stream = profile.get_streams()[0]
        intrinsics = profile.get_stream(stream.stream_type()).as_video_stream_profile().get_intrinsics()
        intrinsics_dict['coeffs'] = intrinsics.coeffs
        intrinsics_dict['fx'] = intrinsics.fx
        intrinsics_dict['fy'] = intrinsics.fy
        intrinsics_dict['ppx'] = intrinsics.ppx
        intrinsics_dict['ppy'] = intrinsics.ppy
        intrinsics_dict['model'] = intrinsics.model.__str__()
        intrinsics_dict['width'] = intrinsics.width
        intrinsics_dict['height'] = intrinsics.height
        json.dump(intrinsics_dict, open(intrinsics_path, 'w'))
        pipeline.stop()
        dir_processed = False
    return dir_processed

def process_video(file, video_path):
    """
    Processes the video and saves the frames in the corresponding directory
    """
    # Set up a dedicated pipeline for each file
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(os.path.join(DATASET_PATH, file), repeat_playback=False)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    available_streams = {stream.stream_type(): stream for stream in profile.get_streams()}

    try:
        # Process frames in the video
        frame_count = 1
        while True:
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame() if rs.stream.color in available_streams else frameset.get_infrared_frame()
            depth_frame = frameset.get_depth_frame() if rs.stream.depth in available_streams else None
            if not color_frame and not depth_frame:
                print(f"End of video or missing frame at {frame_count}.")
                break

            # Convert and save color frame
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                color_image_path = os.path.join(os.path.join(video_path, 'color'), f'%06d.png' % frame_count)
                imageio.imwrite(color_image_path, color_image)

            # Convert and save depth frame
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
                depth_image_path = os.path.join(os.path.join(video_path, 'depth'), f'%06d.png' % frame_count)
                imageio.imwrite(depth_image_path, depth_image)

            frame_count += 1

    except RuntimeError as e:
        print(f"Runtime error encountered: {e}. Stopping video processing at frame {frame_count}.")
    
    finally:
        # Ensure pipeline is stopped and resources are released after each file
        pipeline.stop()

if __name__ == "__main__":   
    rs.log_to_console(rs.log_severity.error)

    for file in os.listdir(DATASET_PATH):
        if not file.endswith('.bag'):
            print(f'Skipping {file}')
            continue
        print(f'Processing {file}')
        video_name = file.split('.')[0]
        video_path = os.path.join(OUTPUT_PATH, video_name)
        dir_processed = create_dir(os.path.join(DATASET_PATH, file), video_path)
        if not dir_processed:
            # Process video frames
            process_video(file, video_path)
        else:
            print(f'Already processed {file}')
