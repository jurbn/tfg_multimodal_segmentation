import cv2 as cv
import numpy as np

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
