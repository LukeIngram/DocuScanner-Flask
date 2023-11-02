# test.py

import cv2 
import matplotlib.pyplot as plt

from backend.Scanner import Scanner

"""
img = cv2.imread("/Users/luke/Documents/GitHub/DocuSegement-Pytorch/samples/imgs/IMG_5344_resized_rect.jpeg")


out = Scanner.scan(img, True)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))

axes[0].imshow(out['original'])
axes[2].imshow(out['dewarped'])
axes[1].imshow(out['annotated'])

plt.show()
"""
Scanner = Scanner("/Users/luke/Documents/GitHub/img-to-pdf-converter-webapp/app/backend/models/saves/unet_32.pth", 'mps')

img = cv2.imread("/Users/luke/Documents/GitHub/DocuSegement-Pytorch/samples/imgs/grocery.jpeg")
out = Scanner.scan(img, True)

#report = Scanner.build_report(out)
#cv2.imshow('Image Window', report)
#cv2.waitKey(0)  # Waits indefinitely for a key stroke
#cv2.destroyAllWindows()


img = cv2.imread("/Users/luke/Documents/GitHub/DocuSegement-Pytorch/samples/imgs/IMG_5432.jpg")
#img = cv2.imread("/Users/luke/Desktop/alt.jpg")
out = Scanner.scan(img, True)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))

axes[0].imshow(out['original'])
#axes[2].imshow(out['dewarped'])
axes[1].imshow(out['annotated'])

plt.show()