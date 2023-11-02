import os, sys 
from scripts.image import Img
import cv2
import threading


#TODO investigate pixel loss in pdf output

def convert(img,imgpath,dest,save): 
    try: 
        if dest[-1] != '/': 
            dest += '/'
        basename = os.path.splitext(os.path.basename(imgpath))[0]
        image = Img(basename,img)
        if save: 
            savepath = "img/dewarp/" + basename
            if not os.path.isdir(savepath):
                os.mkdir(savepath)
            worker = threading.Thread(target=image.saveAll,args=(savepath,))
            worker.start()
            worker.join()
        f = open(dest+basename+'.pdf',"wb+")
        f.write(image.getPdf())
        f.close()
        return 0
    except IOError as e: 
        return -1
    

def main(imgpath,dest,save): 
    (stauts,msg) = (-1,"unknown error")
    if os.path.splitext(imgpath)[1] not in {".jpeg",".png",".jpg",".tiff",".tif",".JPG"}:
        (status,msg) = (-1,"unsupported file format")
    elif not os.path.isdir(dest): 
        (status,msg) = (-1,"destination directory not found")
    elif not os.path.isfile(imgpath): 
        (status,msg) = (-1,"specified image not found")
    else:
        img = cv2.imread(imgpath)
        if img.size == 0: 
            (status,msg) = (1,"unable to open specified file")
        else: 
            if convert(img,imgpath,dest,save) < 0:
                (status,msg) = (1,"unable to create pdf")
            else:
                (status,msg) = (0,"conversion successful")
    return (status,msg)
