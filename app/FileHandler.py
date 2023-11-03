# FileHandler.py 
# Class Definition for I/O manager 

import os

from PIL import Image
import numpy as np

from werkzeug.utils import secure_filename 
from werkzeug.datastructures.file_storage import FileStorage

from backend.utils.verify import *
from backend.utils.errors import FileError


#TODO FileHandler 

class FileHandler(): 
    
    """
    TODO DOCSTRING
    """

    def __init__(self, uploads_path: str, outbox_path: str) -> None:
        self.uploads_path = uploads_path
        self.outbox_path = outbox_path

    
    def upload_image(self, file: FileStorage) -> str:

        """
        TODO DOCSTRING
        """

        fname = secure_filename(file.filename)

        if fname != '':
            file_ext = os.path.splitext(fname)[1]

            if file_ext != validate_image(file.stream, file_ext): 
                raise(FileError(f"Unsupported Filetype, '{file_ext}"))
            
            fpath = os.path.join(self.uploads_path, fname)

            file.seek(0) # Reset cursor
            file.save(fpath)
            sterilize_img(fpath)    

            return fname
        
        

    def fetch_img(self, fname: str) -> np.ndarray:
        
        """
        TODO DOCSTRING
        """

        try: 
            img = Image.open(os.path.join(self.uploads_path, fname))
        except FileNotFoundError as e: 
            raise(FileError(f"Unable to fetch {fname}: {e}"))
        
        img_arr = np.array(img)

        if 3 > img_arr.shape[2] > 4: 
            raise(FileError(f"Unsupported channel count '{img_arr.shape[2]}'"))

        return img_arr


    def fetch_download(self, fname: str) -> str: 

        """
        TODO DOCSTRING
        """
        try: 
            fpath = os.path.join(self.outbox_path, fname)
        except FileNotFoundError: 
            fpath = None

        return fpath 




    def save_img(self, data: np.ndarray, fname: str) -> None: 

        """
        TODO DOCSTRING
        """

        fpath = os.path.join(self.outbox_path, fname)

        img = Image.fromarray(data)
        img.save(fpath)




    
    def clear_uploads(self) -> None: 
        
        """
        TODO DOCSTRING
        """

    def clear_outbox(self) -> None:

        """
        TODO DOCSTRING
        """ 





        
        