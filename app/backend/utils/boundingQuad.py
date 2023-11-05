# boundingQuad.py

import numpy as np
import cv2


# Utilizes Stingy Algorithm Described here: 
# REFERENCE: https://stackoverflow.com/questions/2048024/minimum-area-quadrilateral-algorithm


def order_points_clockwise(points: np.ndarray) -> np.ndarray: 

    """
    TODO DOCSTRING
    """

    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_points = points[np.argsort(-angles)]
    
    return sorted_points



def line_equation(p1, p2): 

    """
    TODO DOCSTRING
    """

    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]

    return m, b

def consolidate(points: np.ndarray, i: int, j: int) -> np.ndarray: 

    """
    TODO DOCSTRING
    """

    # Fine Line equations for segments (A, i) & (j, B)
    A, B = points[i-1], points[(j+1) % len(points)]
    m1, c1 = line_equation(A, points[i])
    m2, c2 = line_equation(points[j], B)

    # Intersection
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    
    return np.array([x, y])


def area_diff(points, i, j, consolidated): 
    
    """
    TODO DOCSTRING
    """
  
    orig = np.array([points[i-1], points[i], points[j]])
    new = np.array([points[i-1], consolidated, points[(j+1) % len(points)]])
    orig_area = cv2.contourArea(orig)
    new_area = cv2.contourArea(new)

    return new_area


def find_bounding_quad(points: np.ndarray) -> np.ndarray: 
   
    """
    TODO DOCSTRING
    """    
    points = order_points_clockwise(points)

    while len(points) > 4: 
        min_area = float('inf')
        min_index = (-1, -1)
        min_consolidated_point = None
        for i in range(len(points)):
            j = (i + 1) % len(points)
            consolidated_point = consolidate(points.copy(), i, j)
            area = area_diff(points.copy(), i, j, consolidated_point)
            if area < min_area: 
                min_area = area
                min_index = (i, j)
                min_consolidated_point = consolidated_point
        
        new_points = np.vstack([points[:min_index[0]], min_consolidated_point, points[min_index[1]+1:]])
  
        if len(new_points) > len(points): 
            raise(ValueError('Approximation Failed: are all corners visible?'))
        
        points = new_points

    return points

