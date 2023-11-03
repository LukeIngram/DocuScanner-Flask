# Scanner.py
# Class Definition for scanner object 

import os 
import io 
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from .models import *
from .utils import transforms, ModelError


""" TODO: 
-> DOCSTRINGS

"""


class Scanner(): 

    """TODO DOCSTRING"""

    def __init__(self, 
                model_fpath: str, 
                device: str = 'cpu', 
                supported_size: Tuple[int, int] = (480, 480),
                cropBuffer: int = 1  
        ) -> None:

        self.supported_size = supported_size
        self.cropBuffer = cropBuffer

        try:
            self.device = torch.device(device) 
        except RuntimeError: 
            self.device = torch.device('cpu')

        self.model = self.__build_model(model_fpath) 
      
    
    def __build_model(self, model_fpath: str) -> nn.Module: 

        """
        TODO DOCSTRING
        """
        try: 
            load = torch.load(model_fpath, map_location=self.device)
            state_dict = load['state_dict']

            model = UNet(n_channels=3, 
                        n_classes=load['n_classes'], 
                        n_blocks=load['n_blocks'], 
                        start=load['start_channels']
                    ) 
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
        
        except FileNotFoundError:
            raise ModelError('the model file was not found')

        except KeyError:
            raise ModelError("the loaded state_dict does not match the model architecture.")
            
        except Exception as e: 
            raise ModelError(f"Unexpected error: {e}")
    
        else: 
            return model
        

    def __preprocess(self, input: np.ndarray) -> Tuple[np.ndarray, float]: 
        
        """
        TODO DOCSTRING
        """
         # Strip Alpha channel
        input = input[:, :, :3]
        
        resized, sfactrs = transforms.scaleImg(input, self.supported_size)
        data = resized.astype(np.float32)
        data = transforms.preprocessTransform()(data)

        return data, sfactrs
        

    def __segment(self, input: torch.Tensor) -> np.ndarray: 
       
        """
        TODO DOCSTRING
        """
        # Create 4D tensor of dimm (1, C, H, W)
        input = input.unsqueeze(0) 
        input = input.to(self.device)

        with torch.no_grad(): 
            pred = self.model(input).cpu()
            logits = F.sigmoid(pred).float()
            logits  = logits.argmax(dim=1).long().squeeze().numpy()

        # Create mask of [0, 255] from [0, 1]
        out = (logits.astype(np.uint8) * 255)
        return out
    

    def __getCorners(self, input: np.ndarray, sfacts: float) -> np.ndarray: 

        """
        TODO DOCSTRING 
        """

        try:
            # Find Corners & Scale TODO MEASURE TIME OF THIS
            contours = transforms.detectContours(input, input.shape, tol=0.1)
            corners = transforms.detectCorners(contours, input.shape)

            # Scale corners back to locations in original image
            corners_scaled = transforms.scalePoints(corners, tuple(1/s for s in sfacts))
            corners_scaled = corners_scaled.astype(np.uint32)
        
        except ValueError: 
            corners_scaled = []

        return corners_scaled
    


    def __dewarp(self, image: np.ndarray, src_points: np.ndarray) -> np.ndarray:
        """
        TODO DOCSTRING      
        """
        try:
            dest_points, crop_h, crop_w = transforms.destinationPoints(src_points, self.cropBuffer)
            out = transforms.homography(image, src_points.astype(np.float32), (dest_points + self.cropBuffer))
            out = out[0:(crop_w + self.cropBuffer), 0:(crop_h + self.cropBuffer)]

        except (ValueError, AttributeError):
            out = None 
        
        return out
        
    
    def __annotate(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray: 
        """
        TODO DOCSTRING
        """
        canvas = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        linWeight = int(np.ceil(sum(image.shape[:2])/2 * 0.003))

        if len(corners) > 0:
            cv2.line(canvas, (corners[0]), (corners[1]), (0, 255, 0), linWeight*3)
            cv2.line(canvas, (corners[0]), (corners[3]), (0, 255, 0), linWeight*3)
            cv2.line(canvas, (corners[2]), (corners[3]), (0, 255, 0), linWeight*3)
            cv2.line(canvas, (corners[2]), (corners[1]), (0, 255, 0), linWeight*3)
            

        else: 
            text = 'NO DOCUMENT DETECTED'
            fontScale = 2 * transforms.get_optimal_font_scale(text, (canvas.shape[1]//2))
            textSize = cv2.getTextSize(text, font, fontScale, linWeight+2)[0]

            X = (canvas.shape[1] - textSize[0])//2
            Y = (canvas.shape[0] + textSize[1])//2

            r1 = (X, Y-(textSize[1]))
            r2 = (r1[0]+textSize[0], r1[1]+textSize[1])

            buffer = 20*linWeight
            r1 = (r1[0]-buffer, r1[1]-buffer)
            r2 = (r2[0]+buffer, r2[1]+buffer)

            cv2.rectangle(canvas, r1, r2, (255, 255, 255), -1)
            cv2.putText(canvas, text, (X, Y), font, fontScale, (255, 0, 0), linWeight)

        return canvas 


    def scan(self, input: np.ndarray, verbose: bool = False, tol: int = 50) -> Dict[str, np.ndarray]:
       
        """
        TODO DOCSTRING
        """

        input_pr, sfact = self.__preprocess(input.copy())
        mask = self.__segment(input_pr)

        
        import matplotlib.pyplot as plt 
        plt.imshow(mask)
        plt.show()
        
    
        src_points = self.__getCorners(mask, sfact)
        
        dewarped = self.__dewarp(input, src_points)

        if verbose:
            annotated = self.__annotate(input, src_points)
        else: 
            annotated = None
    
        return { 
            'original': input,
            'dewarped': dewarped,
            'annotated': annotated
        }


    def build_report(self, scan_data: Dict[str, np.ndarray]) -> np.ndarray: 
        
        """
        TODO DOCSTRING 
        """

        # Build sample
        fig, axes = plt.subplots(nrows=1, ncols=3)
        axes[0].imshow(scan_data['original'])
        axes[1].imshow(scan_data['annotated'])
        axes[2].imshow(scan_data['dewarped'])
        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')
        fig.tight_layout()
        
        # Convert to Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        array = np.asarray(bytearray(buf.read()), dtype=np.uint8)
        out = cv2.cvtColor(cv2.imdecode(array, cv2.IMREAD_ANYCOLOR), cv2.COLOR_BGR2RGB)
       
        fig.clf()
        buf.close()

        return out

        
        

