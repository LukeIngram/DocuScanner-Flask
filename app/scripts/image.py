#   Image Class: Image.py

from ctypes.wintypes import LPWIN32_FIND_DATAA
import numpy as np 
import cv2
from pyparsing import identbodychars
import pytesseract as tess
from scripts.align import *


class Img: 
    def __init__(self,name,data): 
        self._raw = data
        self._name = name
        self._gray = self.erode(self.sharpen(self.set_grayscale(self._raw.copy())))
        self._thresh = self.threshImg(self._gray)
        self._contours = detectContour(self._thresh,self._thresh.shape)
        self._corners = detectCorners(self._contours[0],self._contours[1])
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
        corners = self._corners
        if len(corners) == 0: 
            return self._raw.copy()

        dest,w,h = destinationPoints(corners)
        R = homography(self._raw,np.float32(corners),dest)
        self.updateScales(R)
        return R[0:h,0:w]
        #TODO add cropping utility to alignImg method 


    #-----END PREPROCESSING---------

    #-----BEGIN DISPLAY METHODS---------

    #draw boxes around pre-processed image
    def identifyDocument(self): #TODO draw bounding rect around rectangular document
        temp = self._raw.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        linWeight = 3
        if (temp.shape[0]*temp.shape[1]) < 1000**2: 
            linWeight = 1

        if len(self._corners) == 0: 
            text = 'NO DOCUMENT DETECTED'
            fontScale = get_optimal_font_scale(text,(temp.shape[1]//2))
            textSize = cv2.getTextSize(text,font,fontScale,linWeight+2)[0]
            X = (temp.shape[1] - textSize[0])//2
            Y = (temp.shape[0] + textSize[1])//2
            r1 = (X,Y-(textSize[1]))
            r2 = (r1[0]+textSize[0],r1[1]+textSize[1])
            buffer = 20*linWeight
            r1 = (r1[0]-buffer,r1[1]-buffer)
            r2 = (r2[0]+buffer,r2[1]+buffer)
            cv2.rectangle(temp,r1,r2,(255,0,0),-1)
            cv2.rectangle(self._dewarped,r1,r2,(255,255,255),-1)
            cv2.putText(temp,text,(X,Y),font,fontScale,(0,0,255),linWeight+2)
            cv2.putText(self._dewarped,text,(X,Y),font,fontScale,(0,0,255),linWeight+2)

        else: 
            corners = np.int_(self._corners)
            text = 'Document'
            fontScale = get_optimal_font_scale(text,(corners[1][0]-corners[0][0])//4)
            cv2.line(temp,tuple(corners[0]),tuple(corners[1]),(0,255,0),linWeight*3)
            cv2.line(temp,tuple(corners[0]),tuple(corners[3]),(0,255,0),linWeight*3)
            cv2.line(temp,tuple(corners[2]),tuple(corners[3]),(0,255,0),linWeight*3)
            cv2.line(temp,tuple(corners[2]),tuple(corners[1]),(0,255,0),linWeight*3)
            cv2.putText(temp,text,(corners[3][0]+(10*linWeight),corners[3][1]+(20*linWeight))\
                                                ,font,fontScale,(0,255,0),linWeight)
        return temp 


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