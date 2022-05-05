#   align.py

import numpy as np 
import cv2
import matplotlib.pyplot as plt


def detectContour(img,img_shape): 
    canvas = np.zeros(img_shape,np.uint8)
    contours,hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours,key=cv2.contourArea,reverse=True)[0]
    cv2.drawContours(canvas,cnt,-1,(255,255,255),3)
    #plt.imshow(canvas) 
    #plt.savefig("contour.png")
    return canvas,cnt 

#TODO add "draw the lightest rectangular contour that takes up atelast 20% of the total area checking"
# to the detect contour function
 

def detectCorners(canvas,cnt): #Utilizes the Douglas-Peuckert Algorithm
    ep = 0.01 * cv2.arcLength(cnt,True) 
    appx_corners = cv2.approxPolyDP(cnt,ep,True)
    cv2.drawContours(canvas,appx_corners,-1,(255,255,255),10)
    #plt.imshow(canvas) 
    #plt.savefig("corners.png")
    appx_corners = sorted(np.concatenate(appx_corners).tolist())
    appx_corners = np.array(appx_corners,dtype="float32")

    #re-order the corners for 4-point transform algorithm
    rect = np.zeros((4, 2), dtype="float32")
    s = appx_corners.sum(axis=1)
    rect[0] = appx_corners[np.argmin(s)]
    rect[2] = appx_corners[np.argmax(s)]
    diff = np.diff(appx_corners, axis=1)
    rect[1] = appx_corners[np.argmin(diff)]
    rect[3] = appx_corners[np.argmax(diff)]
    #plt.imshow(canvas) 
    #plt.savefig("rect.png")
    return rect


# Based off four-point-transform:  
# https://github.com/meizhoubao/pyimagesearch/tree/master/getperspectivetransform
def destinationPoints(corners): 
    (tl, tr, br, bl) = corners

    x1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    x2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    x = max(int(x1),int(x2))

    y1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    y2 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    y = max(int(y1),int(y2))
    
    dest_corners = np.array([[0,0],[x-1,0],[x-1,y-1],[0,y-1]],dtype="float32")
    return dest_corners,x,y

def homography(img,src,dest): 
    h,w = img.shape[:2]
    H, _ = cv2.findHomography(src,dest,method=cv2.RANSAC,ransacReprojThreshold=0.1)
    return cv2.warpPerspective(img,H,(w,h),flags=cv2.INTER_LINEAR)

