# verify.py

import imghdr
from PIL import Image


def validate_image(stream, exten: str) -> str: 

    """
    TODO DOCSTRING
    """
    header = stream.read(512)
    format = imghdr.what(None,header)
    if not format: 
        return None
    if format in ['jpg', 'jpeg', 'tiff', 'png']:
        return exten


def sterilize_img(fpath: str) -> None: 

    """
    TODO DOCSTRING
    """

    #Strip EXIF Data from image
    img = Image.open(fpath)
    data = list(img.getdata())
    image_without_exif = Image.new(img.mode, img.size)
    image_without_exif.putdata(data)
    image_without_exif.save(fpath)

    
