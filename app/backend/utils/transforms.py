#   align.py

import numpy as np 
import cv2
from typing import Dict, Any, Tuple, List

import torchvision.transforms as vt


def detectContours(img: np.ndarray, img_shape: Tuple[int, ...], tol: float = 0.3) -> List[np.ndarray]: 
    """
    TODO DOCSTRING
    """
    img = cv2.blur(img, (3,3))
    canvas = np.zeros(img_shape, np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)

    # Remove small contours (less than 30% of image area) #TODO 
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 3)
    cnt = [c for c in cnt if cv2.contourArea(c) >=  (tol * (canvas.shape[0] * canvas.shape[1]))]

    return cnt 


def detectCorners(contours: List[np.ndarray]) -> np.ndarray:
    # Utilizes the Douglas-Peuckert Algorithm
    """
    TODO DOCSTRING
    """
    points = []
    for cnt in contours:
        ep = 0.01 * cv2.arcLength(cnt, True) 
        appx_corners = cv2.approxPolyDP(cnt, ep, True)
       
        if len(appx_corners) == 4: 
            appx_corners = np.concatenate(appx_corners).tolist()
            appx_corners = np.array(appx_corners, dtype="float32")
        
            #re-order the corners for 4-point transform algorithm
            rect = np.zeros((4, 2), dtype="float32") 
            s = appx_corners.sum(axis=1)
            rect[0] = appx_corners[np.argmin(s)]
            rect[2] = appx_corners[np.argmax(s)]
            diff = np.diff(appx_corners, axis=1)
            rect[1] = appx_corners[np.argmin(diff)]
            rect[3] = appx_corners[np.argmax(diff)] 
            points = rect
            break # break at first(largest) quadrilateral contour 
    
    """
    # DEBUG 

    import matplotlib.pyplot as plt 
    canvas = np.zeros((480, 480),np.uint8)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 3)
    for point in points:
        cv2.circle(canvas, tuple(point.astype(np.uint32)), 10, (255, 255, 255), -1)  # Mark corners in red
    plt.imshow(canvas)
    plt.show() 
    """

    return points
    

# Based off four-point-transform:  
# https://github.com/meizhoubao/pyimagesearch/tree/master/getperspectivetransform
def destinationPoints(corners: np.ndarray) -> np.ndarray: 
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

    dest_corners = np.array([[0, 0], [x-1, 0], [x-1, y-1],[0, y-1]], dtype="float32")

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


def preprocess_transform(
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
