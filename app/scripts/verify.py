from flask import Flask 
import imghdr
import os 


def validate_image(stream): 
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None,header)
    if not format: 
        return None
    if format == 'png': 
        return '.' + format
    return '.' + (format if format != 'jpeg' else 'jpg')

