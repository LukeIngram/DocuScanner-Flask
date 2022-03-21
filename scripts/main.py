#   MIT License
#
#   Copyright (c) 2022 Luke Ingram
#   
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#   
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#   
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#   
#   main.py

import os,sys 
from scripts.image import Img
import cv2


def preprocessingDebug(img,imgpath):
    image = Img(imgpath,img)
    image.display()
    image.displayBoundingText()

def getTextdebug(img,imgpath): 
    image = Img(imgpath,img)
    print("\nimg text:")
    print('\n'+ image.getString())

def imgToPdf(img,imgpath,dest): 
    try: 
        if dest[-1] != '/': 
            dest += '/'
        image = Img(imgpath,img)
        f = open(dest+os.path.splitext(os.path.basename(imgpath))[0]+'.pdf',"w+b")
        f.write(bytearray(image.getPdf()))
        f.close()
        return 0
    except IOError: 
        return -1
    

def main(imgpath,dest): 
    #   Basic security and image file checks, more robust version required for deployment
    #   TODO os.path checks are quite slow, maybe change to stat for better performance 
    (stauts,msg) = (-1,"unknown error")
    if os.path.splitext(imgpath)[1] not in {".jpeg",".png",".jpg",".tiff",".tif"}:
        (status,msg) = (-1,"unsupported file format")
    elif not os.path.isdir(dest): 
        (status,msg) = (-1,"destination directory not found")
    elif not os.path.isfile(imgpath): 
        (status,msg) = (-1,"specified image not found")
    else:
        img = cv2.imread(imgpath)
        if img.size == 0: 
            (status,msg) = (-1,"unable to open specified file")
        else: 
            if imgToPdf(img,imgpath,dest) < 0:
                (status,msg) = (-1,"unable to create pdf")
            else:
                (status,msg) = (0,"conversion successful")
    return (status,msg)
