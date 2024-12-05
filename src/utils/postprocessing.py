import cv2 as cv
import numpy as np

# Define the JET colormap manually in float64 precision
JET_COLORMAP = np.array([
        [0.5, 0, 0],     # Dark red
        [1, 0, 0],       # Red
        [1, 0.5, 0],     # Orange
        [1, 1, 0],       # Yellow
        [0.5, 1, 0.5],   # Green
        [0, 1, 1],       # Light cyan
        [0, 0.5, 1],     # Cyan
        [0, 0, 1],       # Blue
        [0, 0, 0.5],     # Dark blue
], dtype=np.float64)

def depth_to_normal(depth_map):
    """
    Uses the depth map to generate a normal map
    :param depth_map: The input depth map
    :out: The computed normal map
    """
    h, w = depth_map.shape
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # now compute the partial derivatives of depth respect to x and y
    dx = cv.Sobel(depth_map, cv.CV_32F, 1, 0)
    dy = cv.Sobel(depth_map, cv.CV_32F, 0, 1)

    # compute the normal vector per each pixel
    normal = np.dstack((-dx, -dy, np.ones((h, w))))
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm!=0)

    # map these normal vectors to 65535
    normal = (normal + 1) * 32767.5
    normal = normal.clip(0, 65535).astype(np.uint16)

    # return the map
    return normal

def equalize_histogram(image):
    """
    Equalizes the histogram of the input image
    :param image: The uint16 image to equalize
    :out: The equalized image
    """
    hist, bins = np.histogram(image.flatten(), bins=65536, range=[1, 65535])
    # Calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    # Use the CDF to map the original image values to the equalized values
    equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized * 65535)
    # Reshape back to the original image shape and cast to uint16
    equalized_histogram = equalized.reshape(image.shape).astype(np.uint16)
    return equalized_histogram

def depth_to_jet(depth_map):
    """
    Uses the depth map to generate a jet map
    :param depth_map: The input depth map
    :out: The computed jet map
    """
    # Equalize the histogram of the depth image
    equalized_map = equalize_histogram(depth_map)
    # Normalize the image to [0, 1]
    normalized_image = equalized_map.astype(np.float64) / 65535.0
    # Generate interpolated colormap values
    indices = np.linspace(0, 1, len(JET_COLORMAP))
    red_interp = np.interp(normalized_image.flat, indices, JET_COLORMAP[:, 0]).reshape(depth_map.shape)
    green_interp = np.interp(normalized_image.flat, indices, JET_COLORMAP[:, 1]).reshape(depth_map.shape)
    blue_interp = np.interp(normalized_image.flat, indices, JET_COLORMAP[:, 2]).reshape(depth_map.shape)
    # Combine channels and scale back to uint16 range [0, 65535]
    colormapped_image = np.stack([
        (red_interp * 65535).astype(np.uint16),
        (green_interp * 65535).astype(np.uint16),
        (blue_interp * 65535).astype(np.uint16)
    ], axis=-1)
    return colormapped_image
