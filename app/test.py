# test.py

import cv2 
import matplotlib.pyplot as plt

from backend.Scanner import Scanner


img = cv2.imread("/Users/luke/Documents/GitHub/DocuSegement-Pytorch/samples/imgs/IMG_5344_resized_rect.jpeg")
Scanner = Scanner("/Users/luke/Documents/GitHub/img-to-pdf-converter-webapp/app/backend/models/saves/unet_32.pth", 'mps')

out = Scanner.scan(img, True)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))

axes[0].imshow(out['original'])
axes[2].imshow(out['dewarped'])
axes[1].imshow(out['annotated'])

plt.show()

img = cv2.imread("/Users/luke/Documents/GitHub/DocuSegement-Pytorch/samples/grocery.jpeg")
out = Scanner.scan(img, True)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))

axes[0].imshow(out['original'])
axes[2].imshow(out['dewarped'])
axes[1].imshow(out['annotated'])

plt.show()


img = cv2.imread("/Users/luke/Desktop/Desktop Images/nasa-JkaKy_77wF8-unsplash.jpg")
out = Scanner.scan(img, True)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))

axes[0].imshow(out['original'])
axes[2].imshow(out['dewarped'])
axes[1].imshow(out['annotated'])

plt.show()