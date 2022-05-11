# EditContour.py
#
#
#
#---------------------UNUSED----------------------------------------
#

import numpy as np 
import cv2
import sys 

# The following code are extrapolated from code cited here:
# https://stackoverflow.com/questions/35226993/how-to-crop-away-convexity-defects

def ed2(lhs, rhs):
    return(lhs[0] - rhs[0])*(lhs[0] - rhs[0]) + (lhs[1] - rhs[1])*(lhs[1] - rhs[1])

#TODO DEBUG THIS

def remove_from_contour(contour, defectsIdx):
    minDist = sys.maxsize
    startIdx, endIdx = 0, 0

    for i in range(0,len(defectsIdx)):
        for j in range(i+1, len(defectsIdx)):
            dist = ed2(contour[defectsIdx[i]][0], contour[defectsIdx[j]][0])
            if minDist > dist:
                minDist = dist
                startIdx = defectsIdx[i]
                endIdx = defectsIdx[j]

    if startIdx <= endIdx:
        inside = contour[startIdx:endIdx]
        len1 = 0 if inside.size == 0 else cv2.arcLength(inside, False)
        outside1 = contour[0:startIdx]
        outside2 = contour[endIdx:len(contour)]
        len2 = (0 if outside1.size == 0 else cv2.arcLength(outside1, False)) + (0 if outside2.size == 0 else cv2.arcLength(outside2, False))
        if len2 < len1:
            startIdx,endIdx = endIdx,startIdx     
    else:
        inside = contour[endIdx:startIdx]
        len1 = 0 if inside.size == 0 else cv2.arcLength(inside, False)
        outside1 = contour[0:endIdx]
        outside2 = contour[startIdx:len(contour)]
        len2 = (0 if outside1.size == 0 else cv2.arcLength(outside1, False)) + (0 if outside2.size == 0 else cv2.arcLength(outside2, False))
        if len1 < len2:
            startIdx,endIdx = endIdx,startIdx

    if startIdx <= endIdx:
        out = np.concatenate((contour[0:startIdx], contour[endIdx:len(contour)]), axis=0)
    else:
        out = contour[endIdx:startIdx]
    return out


def remove_defects(canvas,contour):
    hull = cv2.convexHull(contour,returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    temp = np.zeros(canvas.shape,np.uint8)
    cv2.drawContours(temp,contour,-1,(0,0,255),10)
    while True:
        defectsIdx = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            depth = d / 256
            if depth > 2:
                defectsIdx.append(f)
        if len(defectsIdx) < 2:
            break
        contour = remove_from_contour(contour, defectsIdx)
        hull = cv2.convexHull(contour, returnPoints=False)
        #cv2.drawContours(temp,hull,-1,(0,255,0),10)
        defects = cv2.convexityDefects(contour, hull)
    cv2.drawContours(canvas,contour,-1,(255,255,255),10)
    cv2.drawContours(temp,contour,-1,(0,255,0),10)
    cv2.imwrite('debug.jpg',temp)
    return canvas,contour


#TODO
# Alternate method, Hough Transform and extract promient lines

def clean_contour(canva,cnt): 
    pass 
