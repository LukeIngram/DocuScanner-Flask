from curses import keyname
from flask import Flask 
import imghdr
import os 
import json 
import hashlib
import time
from datetime import datetime,timedelta
from dateutil import parser
import virustotal_python
from virus_total_apis import PublicApi as VirusTotalPublicApi
from PIL import Image
import keys.vt_key


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
    

def virustotal_scan(fpath): #TODO virustotal api scanning 

    if os.path.exists(fpath): 
        
        
        #Uploading file 
        hash = ""
        files = {"file": (os.path.basename(fpath), open(os.path.abspath(fpath), "rb"))}
        with virustotal_python.Virustotal(keys.vt_key.api_key) as vtotal:
            resp = vtotal.request("files", files=files, method="POST")
            if resp != None: 
                print(resp.json()) #-------------------------------------------------------REMOVE 
        time.sleep(3)
        # Evaluating Results
        sha256 = hashlib.sha256()
        with open(fpath,'rb') as f:
            for byte_block in iter(lambda: f.read(4096),b""):
                    sha256.update(byte_block)
        fHash = sha256.hexdigest()
        vt = VirusTotalPublicApi(keys.vt_key.api_key)
        data = json.loads(json.dumps(vt.get_file_report(fHash)))
        print(data)
        try:
            while data['results']['response_code'] == -2: 
                time.sleep(60)
                data = json.loads(json.dumps(vt.get_file_report(fHash)))
                print(data['results']['response_code'])

            if data['results']['response_code'] == 1:
                positives = data['results']['positives']
                scan_date = data['results']['scan_date']
            else: 
                print("Virustotal error")
                return True
        except KeyError:
            print("keyerror")
            print(data)
            return True
       
        scan_date = parser.parse(scan_date)
        
        if (positives > 5) or (scan_date < datetime.now()-timedelta(days=365)):
            return True
        
        else: 
            return False
        
    else: 
        return True
