"""
The code in this file is used to generate the metadata for the dataset.
"""
import os
import json
import pyrealsense2 as rs

BAG_FOLDER = "data/lindenthal-camera-traps/bagfiles"

def return_data_type(bag_file_path):
    """
    Will return the type of data stored in the bag file.
    :param bag_file_path: Path to the bag file.
    :return is_color, is_infrared, is_depth: Boolean values for the data type.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file_path)
    pipeline.start(config)

    # get the type of video
    has_color = False
    has_infrared = False
    has_depth = False
    profile = pipeline.get_active_profile()
    for stream in profile.get_streams():
        if stream.stream_type() == rs.stream.color:
            has_color = True
        if stream.stream_type() == rs.stream.infrared:
            has_infrared = True
        if stream.stream_type() == rs.stream.depth:
            has_depth = True
    
    return has_color, has_infrared, has_depth

if __name__ == "__main__":
    file_names = sorted(os.listdir(BAG_FOLDER))
    # get the last file registered in the json
    # if the metadata file does not exist, create it with an empty dictionary
    if not os.path.exists('data/lindenthal-camera-traps/metadata.json'):
        with open('data/lindenthal-camera-traps/metadata.json', 'w') as f:
            json.dump({'videos': {}}, f)
    with open('data/lindenthal-camera-traps/metadata.json', 'r') as f:
        metadata = json.load(f)
    if len(metadata['videos']) == 0:
        last_index = -1
    else:
        last_file = list(metadata['videos'].keys())[-1]
        last_index = file_names.index(last_file + '.bag')
    # iterate over all files in the bag folder starting from the last file
    for file_name in file_names[last_index + 1:]:
        print(f"Processing {file_name}")
        # generate the full path to the file
        file_path = f"{BAG_FOLDER}/{file_name}"
        has_color, has_infrared, has_depth = return_data_type(file_path)
        # append the metadata to the file
        with open('data/lindenthal-camera-traps/metadata.json', 'r') as f:
            metadata = json.load(f)
        metadata['videos'][file_name[:-4]] = {  # remove the .bag extension
            'types': {
                'color': has_color,
                'infrared': has_infrared,
                'depth': has_depth
            }
        }
        with open('data/lindenthal-camera-traps/metadata.json', 'w') as f:
            json.dump(metadata, f)
    print('Done')