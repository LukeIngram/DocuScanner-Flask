

# **DocuScanner Flask**

A web app designed for online document scanning. Featuring built-in ML-powered perspective correction. 

![sample](media/report_fingers.jpg)

## **Contents**

1. [Introduction](#introduction)
2. [Usage](#usage)
   - [Installation](#installation)
   - [Running Locally](#running-locally)
   - [Garbage Collection](#garbage-collection)
3. [How it All Works](#how-it-all-works)
   - [Image Segmentation](#image-segmentation)
   - [Bounding Quadrilateral Approximation](#bounding-quadrilateral-approximation)
   - [Perspective Correction](#perspective-correction)
4. [Contact](#contact)
5. [Future Work](#future-work)

## **Usage**

Install all dependencies before proceeding. 
```bash
pip install -r requirements.txt
```

### **Running Locally**
1. Navigate to the app directory
``` bash
cd app
```
2. Run app with flask, changing the host and port to your needs.
```bash 
flask run --host=0.0.0.0 --port=5001
```

3. Navigate to the server's location over http. Further instructions are located in the UI.  


## **How it All Works** 

The general workflow is as follows 

![workflow](media/workflow.jpg)

The complex components of the workflow are explained below

### **Image Segmentation** 

We utilize deep learning to simplify a crucial step in the perspective correction process: contour detection. 
on 
The model is custom UNet for binary segmentation of rectangular documents.

 Implementation details on this model can be found in this repo: https://github.com/LukeIngram/DocuSegement-Pytorch

Example inference:

![sample_segmentation](media/segment_example.png)

These masks remove all other subjects from the image, which greatly improves the accuracy of the contour detection algorithms.

### **Bounding Quadrilateral Approximation**

Four corner points are required for accurate perspective correction, and in the case where the input isn't a quadrilateral, a minimum bounding one is computed around the subject. 

Approximation example: 

![approximation_example](media/contour_repair.png)

The implementation details can be found at [*app/backend/utils/boundingQuad.py*](https://github.com/LukeIngram/DocuScanner-Flask/blob/main/app/backend/utils/boundingQuad.py)


### **Perspective Correction**

The core feature of this app is the *perspective transform*. 

The first step is finding the new *corrected* corners of the quadrilateral. This is done by computing a four point linear transform from the quadrilateral approximated in the previous step. (More details can be found in [*app/backend/utils/transforms.py*](https://github.com/LukeIngram/DocuScanner-Flask/blob/main/app/backend/utils/transforms.py))

Then the perspective is corrected using a process called a [homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) transformation. 

The results are as follows: 

![sample](media/sample_correction.png)


## **Contact** 

Luke Ingram - lukeingram01@gmail.com


## **Future Work**

* Containerization
* Device Camera Support
* event logging
* file cleanup (both uploads and outbox)

