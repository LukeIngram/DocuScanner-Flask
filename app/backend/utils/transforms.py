#   align.py

import numpy as np 
import cv2
from typing import Dict, Any, Tuple, List

import torchvision.transforms as vt

from .boundingQuad import find_bounding_quad


def boundingQuad(points: np.ndarray) -> np.ndarray:

    """
    TODO DOCSTRING 
    """

    hull_points = cv2.convexHull(points)
    bounding_quad = find_bounding_quad(hull_points.squeeze())

    return bounding_quad



def detectContours(img: np.ndarray, img_shape: Tuple[int, ...], tol: float = 0.3) -> List[np.ndarray]: 

    """
    TODO DOCSTRING
    """

    img = cv2.blur(img, (3,3))
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)

    return cnt 


def approximateCorners(contour: np.ndarray, tol: float = 0.02) -> np.ndarray: 

    """
    TODO DOCSTRING
    """

    ep = tol * cv2.arcLength(contour, True) 

    appx_corners = cv2.approxPolyDP(contour, ep, True)
    appx_corners = np.concatenate(appx_corners).tolist()
    appx_corners = np.array(appx_corners, dtype="float32")

    return appx_corners




def detectCorners(contours: List[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:

    """
    TODO DOCSTRING
    """

    points = []

    cnt = max(contours, key=cv2.contourArea)
    appx_corners = approximateCorners(cnt, 0.02)
   
    # TODO HANDLE CASE WHERE CORNERS LIE AT EXTREME POINTS (IMAGE BOUNDARIES)

    # Approximate bounding quadrilateral if needed
    if 4 < appx_corners.shape[0] <= 11:
        appx_corners = boundingQuad(appx_corners)

    #re-order the corners for 4-point transform algorithm
    if appx_corners.shape[0] == 4:
        rect = np.zeros((4, 2), dtype="float32") 
        s = appx_corners.sum(axis=1)
        rect[0] = appx_corners[np.argmin(s)]
        rect[2] = appx_corners[np.argmax(s)]
        diff = np.diff(appx_corners, axis=1)
        rect[1] = appx_corners[np.argmin(diff)]
        rect[3] = appx_corners[np.argmax(diff)] 
        points = rect

    return points
    

# Based off four-point-transform:  
# https://github.com/meizhoubao/pyimagesearch/tree/master/getperspectivetransform
def destinationPoints(corners: np.ndarray, buffer: int = 1) -> np.ndarray: 
    """
    TODO DOCSTRING
    """
    (tl, tr, br, bl) = corners.astype(np.float32)
    x1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    x2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    x = max(int(x1), int(x2))

    y1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    y2 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    y = max(int(y1), int(y2))

    dest_corners = np.array([[0, 0], [x - buffer, 0], [x - buffer, y - buffer],[0, y - buffer]], dtype="float32")

    return dest_corners, x, y


def homography(img: np.ndarray, src: np.ndarray, dest: np.ndarray) -> np.ndarray: 
    """
    TODO DOCSTRING 
    """
    h,w = img.shape[:2]
    H, _ = cv2.findHomography(src, dest, method=cv2.RANSAC, ransacReprojThreshold=0.1)
    return cv2.warpPerspective(img, H, (w,h), flags=cv2.INTER_LINEAR)
 

def get_optimal_font_scale(text: str, width: int) -> float:
    """
    TODO DOCSTRING
    """
    for scale in reversed(range(0, 60, 1)):
      textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
      new_width = textSize[0][0]

      if (new_width <= width):
          return scale/10
      
    return 1.0


# TODO EVALUATE NECESSITY
def scalePoints(points: np.ndarray, sfacts: np.ndarray) -> np.ndarray:
    """
    TODO DOCSTRING
    """
    tf = np.diag(sfacts)
    scaled = points @ tf 
    return scaled


#TODO 
def scaleImg(img: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]: 
    """
    TODO DOCSTRING
    """
    h,w = img.shape[:2] 
    h_sfact, w_sfact = shape[0]/h, shape[1]/w
    resized = cv2.resize(img.copy(), shape, cv2.INTER_LANCZOS4)
    return resized, np.array([w_sfact, h_sfact])


def preprocessTransform(
        mean: Tuple[float, ...] = (0.4611, 0.4359, 0.3905), 
        std: Tuple[float, ...] = (0.2193, 0.2150, 0.2109)
    ) -> vt.Compose: 
    """
    TODO DOCSTRING
    """
    transforms = vt.Compose([
        vt.ToTensor(),
        vt.Normalize(mean, std)
    ])
    return transforms
