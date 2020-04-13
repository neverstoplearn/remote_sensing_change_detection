import cv2
import numpy as np
from PIL import Image

def denoise(mask, eps):
    """Removes noise from a mask.

    Args:
      mask: the mask to remove noise from.
      eps: the morphological operation's kernel size for noise removal, in pixel.

    Returns:
      The mask after applying denoising.
    """

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct)


def grow(mask, eps):
    """Grows a mask to fill in small holes, e.g. to establish connectivity.

    Args:
      mask: the mask to grow.
      eps: the morphological operation's kernel size for growing, in pixel.

    Returns:
      The mask after filling in small holes.
    """

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct)

def erode_and_dilate(temp,iterations = 1,ks = 5):
    ### 先腐蚀再膨胀
    kernel = np.ones((ks,ks),np.uint8)  
    erosion = cv2.erode(temp,kernel,iterations )
    dilation = cv2.dilate(erosion,kernel,iterations )
    return dilation
