"""
This file contains the code necessary to play video files from the RealSense camera bag files.
"""
import cv2 as cv
import numpy as np
import pyrealsense2 as rs

def generate_full_frame(frames):
    """Will generate the full frame to display from the given frames.
    :param frames: the frame object from the RealSense camera.
    :return: the full frame to display.
    """
    # generate some blank frames first (480, 848)
    blank_frame = np.zeros((480, 848, 3), np.uint8)

    color_frame = frames.get_color_frame()
    if color_frame:
        color_frame = np.asanyarray(color_frame.get_data())
        color_frame = cv.resize(color_frame, (848, 480))
    else:
        color_frame = blank_frame
    
    infrared_frame = frames.get_infrared_frame()
    if infrared_frame:
        infrared_frame = np.asanyarray(infrared_frame.get_data())
        infrared_frame = cv.cvtColor(infrared_frame, cv.COLOR_GRAY2BGR)
        infrared_frame = cv.resize(infrared_frame, (848, 480))
    else:
        infrared_frame = blank_frame

    depth_frame = frames.get_depth_frame()
    if depth_frame:
        depth_frame = np.asanyarray(depth_frame.get_data())
        depth_frame = cv.applyColorMap(cv.convertScaleAbs(depth_frame, alpha=0.03), cv.COLORMAP_JET)
    else:
        depth_frame = blank_frame

    full_frame = np.hstack((color_frame, infrared_frame, depth_frame))
    return full_frame
    

def play_video_from_bag(bag_file_path):
    """
    Will automatically play the video from the RealSense camera bag file.
    Controls:
    - Space: Pause/Play
    - Esc: Exit
    :param bag_file_path: Path to the bag file.
    :return: None
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file_path)
    pipeline.start(config)

    # Control playback to run in non-real-time mode
    device = pipeline.get_active_profile().get_device()
    playback = device.as_playback()
    playback.set_real_time(True)  # Disable real-time mode to avoid frame drops

    # get the type of video
    has_color = False
    has_infrared = False
    profile = pipeline.get_active_profile()
    for stream in profile.get_streams():
        if stream.stream_type() == rs.stream.color:
            has_color = True
        if stream.stream_type() == rs.stream.infrared:
            has_infrared = True
    if has_color and has_infrared:
        raise ValueError("Cannot have both color and infrared streams in the same video!!!")
    
    try:
        while True:
            # TODO: implement the video player controls
            frame_pair = pipeline.wait_for_frames()
            full_frame = generate_full_frame(frame_pair)
            
            cv.imshow(bag_file_path, full_frame)
            key = cv.waitKey(1)
            if key == 27:  # Esc key to exit
                break
            elif key == 32:  # Spacebar to pause/resume playback
                if playback.pause():
                    playback.resume()
    finally:
        pipeline.stop()
        cv.destroyAllWindows()

def get_n_frame(bag_file_path, n, return_color=False, return_infrared=False, return_depth=False):
    """
    Will return the nth frame from the bag file.
    :param bag_file_path: Path to the bag file.
    :param n: The frame number to return.
    :param return_color: Return the color frame.
    :param return_infrared: Return the infrared frame.
    :param return_depth: Return the depth frame.
    :return: The nth frame from the bag file.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file_path)
    pipeline.start(config)
    
    try:
        for i in range(n):
            frame_pair = pipeline.wait_for_frames()
        
        if return_color:
            frame = frame_pair.get_color_frame()
        elif return_infrared:
            frame = frame_pair.get_infrared_frame()
        elif return_depth:
            frame = frame_pair.get_depth_frame()
        else:
            raise ValueError("Must specify a frame type to return!!!")
        return np.asanyarray(frame.get_data())
    finally:
        pipeline.stop()

