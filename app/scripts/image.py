#   Image Class: Image.py

import numpy as np 
import cv2
import pytesseract as tess
from scripts.align import *


class Img: 
    def __init__(self,name,data): 
        self._raw = data
        self._name = name
        self._gray = self.erode(self.sharpen(self.set_grayscale(self._raw.copy())))
        self._thresh = self.threshImg(self._gray)
        self._contours = detectContour(self._thresh,self._thresh.shape)
        self._dewarped = self.alignImg()
        self._annotated = self.identifyDocument()


    #-----BEGIN PREPROCESSING------

    def sharpen(self,data): 
        ker = np.ones((5,5),np.float32)/90
        return cv2.filter2D(data,-1,ker)

    def erode(self,data): 
        ker = np.ones((5,5),np.float32)/90
        return cv2.erode(data,ker,iterations=1)

    def set_grayscale(self,data): 
        return cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)

    def threshImg(self,data): 
        temp = cv2.GaussianBlur(data.copy(),(5,5),0)
        return cv2.threshold(temp,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def updateScales(self,data): 
        self._gray = self.set_grayscale(data)
        self._thresh = self.threshImg(self._gray)

    def alignImg(self):
        corners = detectCorners(self._contours[0],self._contours[1][0]) 
        dest,w,h = destinationPoints(corners)
        R = homography(self._raw,np.float32(corners),dest)

        self.updateScales(R)
        return R[0:h,0:w]
        #TODO add cropping utility to alignImg method 


    #-----END PREPROCESSING---------

    #-----BEGIN DISPLAY METHODS---------

    #draw boxes around pre-processed image
    def identifyDocument(self): #TODO draw bounding rect around rectangular document
        scale = 1 
        fontScale = min(self._raw.shape[0],self._raw.shape[1])/(25/scale)
        temp = self._raw.copy()
        for cnt in self._contours[1]: 
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            if len(approx) == 4: 
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(temp,(x,y),(x+w,y+h),(0,255,0),10)
                cv2.drawContours(temp,cnt,-1,(255,255,255),3)
                cv2.putText(temp,'Document',(x+30,y+h-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                break
        return temp 
            # Note this might annotate everything remotely square. & might be a problem in the future. 

    #-----END DISPLAY METHODS------------

    def saveAll(self,dir):
        cv2.imwrite(dir + '/contour' + '.jpeg',self._contours[0])
        cv2.imwrite(dir + '/gray' + '.jpeg',self._gray)
        cv2.imwrite(dir + '/thresh' + '.jpeg',self._thresh)
        cv2.imwrite(dir + '/annotated' + '.jpeg',self._annotated)
        cv2.imwrite(dir + '/dewarped' + '.jpeg',self._dewarped)
        


    def getPdf(self): 
        temp = cv2.cvtColor(self._dewarped,cv2.COLOR_BGR2RGB)
        pdf = tess.image_to_pdf_or_hocr(temp,extension='pdf')
        return bytearray(pdf)