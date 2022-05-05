import imghdr
from PIL import Image



#TODO build magic number identifaction tool or find api? 
def validate_image(stream,exten): 
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None,header)
    if not format: 
        return None
    if format in ['jpg','jpeg','tiff','png']:
        return exten


def sterilize_img(fpath): 
    #Strip EXIF Data from image
    img = Image.open(fpath)
    data = list(img.getdata())
    image_without_exif = Image.new(img.mode, img.size)
    image_without_exif.putdata(data)
    image_without_exif.save(fpath)

    #TODO Further sterilization (as program currently only stops exif attacks)
