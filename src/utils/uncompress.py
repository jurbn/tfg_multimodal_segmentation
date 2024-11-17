"""
This module contains all functions related to uncompressing files.
"""

import os
import lz4.frame

def uncompress_lz4(input_file, output_file):
    """
    Uncompresses a file using the lz4 algorithm.

    :param input_file: Path to the compressed file.
    :param output_file: Path to the uncompressed file.
    """
    # open the input lz4 compressed file
    with open(input_file, 'rb') as f_in:
        # create the output file where we will store the uncompressed data
        with open(output_file, 'wb') as f_out:
            # uncompress the data
            decompressed_data = lz4.frame.decompress(f_in.read())
            # write the uncompressed data to the output file
            f_out.write(decompressed_data)
