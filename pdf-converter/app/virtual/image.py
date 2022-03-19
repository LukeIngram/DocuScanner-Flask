#   Image Class: Image.py

import numpy as np 
import cv2
import matplotlib.pyplot as plt
import pytesseract as tess
from align import *


class Img: 
    def __init__(self,name,data): 
        self._raw = data
        self._name = name
        self._gray = self.erode(self.sharpen(self.set_grayscale(self._raw.copy())))
        self._thresh = self.threshImg()
        self._imgdata = self.alignImg()


    #-----BEGIN PREPROCESSING------

    def sharpen(self,data): 
        ker = np.ones((5,5),np.float32)/90
        return cv2.filter2D(data,-1,ker)

    def erode(self,data): 
        ker = np.ones((5,5),np.float32)/90
        return cv2.erode(data,ker,iterations=1)

    def set_grayscale(self,data): 
        return cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)

    def threshImg(self): 
        temp = cv2.GaussianBlur(self._gray.copy(),(5,5),0)
        return cv2.threshold(temp,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def alignImg(self):
        canvas,cnt = detectContour(self._thresh,self._thresh.shape)
        corners = detectCorners(canvas,cnt)
        dest,w,h= destinationPoints(corners)
        R = homography(self._raw,np.float32(corners),dest)
        self.updateScales(R)
        return R[0:h,0:w]
        #TODO add cropping utility to alignImg method 

    def updateScales(self,data): 
        self._gray = self.set_grayscale(data)
        self._thresh = self.threshImg()

    #-----END PREPROCESSING---------

    #-----BEGIN DISPLAY METHODS---------

    #draw boxes around pre-processed image
    def displayBoundingText(self): 
        temp = self._imgdata.copy()
        imgH,imgW, _ = temp.shape
        boxes = tess.image_to_boxes(temp)

        for box in boxes.splitlines(): 
            box = box.split(' ')
            x, y, w, h = int(box[1]),int(box[2]),int(box[3]),int(box[4])
            cv2.rectangle(temp,(x,imgH-y),(w,imgH-h),(50,50,255),1)
            #cv2.putText(temp,box[0],(x,imgH-y+20),cv2.FONT_HERSHEY_SIMPLEX,13,(50,205,50),4)

        plt.imshow(cv2.cvtColor(temp,cv2.COLOR_BGR2RGB))
        plt.title("Characters found in: \'%s\'"%(self._name))
        plt.show()


    def display(self):
        temp = cv2.cvtColor(self._imgdata,cv2.COLOR_BGR2RGB)
        plt.imshow(temp)
        plt.title(self._name)
        plt.show()

    #-----END DISPLAY METHODS------------

    def getString(self):
        self.updateScales(self._imgdata)
        return tess.image_to_string(self._imgdata) 
    
    def getPdf(self): 
        temp = cv2.cvtColor(self._imgdata,cv2.COLOR_BGR2RGB)
        pdf = tess.image_to_pdf_or_hocr(temp,extension='pdf')
        return pdf