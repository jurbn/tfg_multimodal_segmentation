import cv2 as cv
import numpy as np

# Define the JET colormap manually in float64 precision
JET_COLORMAP = np.array([
        [0.5, 0, 0],     # Dark red     # further away
        [1, 0, 0],       # Red
        [1, 0.5, 0],     # Orange
        [1, 1, 0],       # Yellow
        [0.5, 1, 0.5],   # Green
        [0, 1, 1],       # Light cyan
        [0, 0.5, 1],     # Cyan
        [0, 0, 1],       # Blue
        [0, 0, 0.5],     # Dark blue    # closer
], dtype=np.float64)

# Distance colormap is meant to encode distance information in a more
# coherent way than the JET colormap
DISTANCE_COLORMAP = np.array([
    [1.0, 0.0, 0.0],  # further away
    [0.5, 0.5, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.0, 0.0, 1.0],  # closer
], dtype=np.float64)


def depth_to_normal(depth_map, equalize=True):
    """
    Uses the depth map to generate a normal map
    :param depth_map: The input depth map
    :out: The computed normal map
    """
    if len(depth_map.shape) != 2:
        raise ValueError("The input depth map must be a single channel image")
    if equalize:
        depth_map = equalize_histogram(depth_map)

    if len(depth_map.shape) != 2:
        depth_map = depth_map[:, :, 0]
    
    # set every 0 value to null value
    depth_map = depth_map.astype(np.float32)
    depth_map[depth_map == 0] = np.nan
    
    # now compute the partial derivatives of depth respect to x and y
    dx = cv.Sobel(depth_map, cv.CV_32F, 1, 0)
    dy = cv.Sobel(depth_map, cv.CV_32F, 0, 1)

    # compute the normal vector per each pixel
    normal = np.dstack((-dx, -dy, np.ones(depth_map.shape)))
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm!=0)

    # map these normal vectors to 65535
    normal = (normal + 1) * 32767.5
    # set every nan value to 0
    normal[np.isnan(normal)] = 0
    normal = normal.clip(0, 65535).astype(np.uint16)

    
    # return the map
    return normal

def equalize_histogram(image):
    """
    Equalizes the histogram of the input image
    :param image: The uint16 image to equalize
    :out: The equalized image
    """
    hist, bins = np.histogram(image.flatten(), bins=65536, range=[1, 65534])
    # Calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    # Use the CDF to map the original image values to the equalized values
    equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized * 65535)
    # Reshape back to the original image shape and cast to uint16
    equalized_histogram = equalized.reshape(image.shape).astype(np.uint16)
    return equalized_histogram

def depth_to_colormap(depth_map, colormap='jet', equalize=True):
    """
    Uses the depth map to generate a jet map
    :param depth_map: The input depth map
    :out: The computed jet map
    """
    # Determine the colormap to use
    if colormap.lower() == 'jet':
        colormap = JET_COLORMAP
    elif colormap.lower() == 'distance':
        colormap = DISTANCE_COLORMAP
    else:
        raise ValueError(f"Unsupported colormap: {colormap}")
    # Equalize the histogram of the depth image
    if equalize:
        depth_map = equalize_histogram(depth_map)
    # Normalize the image to [0, 1]
    normalized_image = depth_map.astype(np.float64) / 65535.0
    # 0 values to Null
    normalized_image[normalized_image == 0] = np.nan
    # Generate interpolated colormap values
    indices = np.linspace(0, 1, len(colormap))
    red_interp = np.interp(normalized_image.flat, indices, colormap[:, 0]).reshape(depth_map.shape)
    red_interp = np.nan_to_num(red_interp, 0)
    green_interp = np.interp(normalized_image.flat, indices, colormap[:, 1]).reshape(depth_map.shape)
    green_interp = np.nan_to_num(green_interp, 0)
    blue_interp = np.interp(normalized_image.flat, indices, colormap[:, 2]).reshape(depth_map.shape)
    blue_interp = np.nan_to_num(blue_interp, 0)
    # Combine channels and scale back to uint16 range [0, 65535]
    colormapped_image = np.stack([
        (red_interp * 65535).astype(np.uint16),
        (green_interp * 65535).astype(np.uint16),
        (blue_interp * 65535).astype(np.uint16)
    ], axis=-1)
    return colormapped_image
