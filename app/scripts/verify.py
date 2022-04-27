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

import keys.vt_key



def validate_image(stream): 
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None,header)
    if not format: 
        return None
    if format == 'png': 
        return '.' + format
    return '.' + (format if format != 'jpeg' else 'jpg')



def virustotal_scan(fpath): #TODO virustotal api scanning 

    if os.path.exists(fpath): 
        
        
        #Uploading file 
        hash = ""
        files = {"file": (os.path.basename(fpath), open(os.path.abspath(fpath), "rb"))}
        with virustotal_python.Virustotal(keys.vt_key.api_key) as vtotal:
            resp = vtotal.request("files", files=files, method="POST")
            if resp != None: 
                print(resp.json()) #-------------------------------------------------------REMOVE 

        # Evaluating Results
        sha256 = hashlib.sha256()
        with open(fpath,'rb') as f:
            for byte_block in iter(lambda: f.read(4096),b""):
                    sha256.update(byte_block)
        fHash = sha256.hexdigest()

        vt = VirusTotalPublicApi(keys.vt_key.api_key)
        data = json.loads(json.dumps(vt.get_file_report(fHash)))
        try:
            attempts = 0
            while (data['results']['response_code'] == -2) or (attempts < 12): 
                time.sleep(5)
                data = json.loads(json.dumps(vt.get_file_report(fHash)))
                attempts += 1 

            if data['results']['response_code'] == 1:

                positives = data['results']['positives']
                scan_date = data['results']['scan_date']
        except KeyError:
            print(data)
            return True
       
        scan_date = parser.parse(scan_date)
        
        if (positives > 5) or (scan_date < datetime.now()-timedelta(days=365)):
            return True
        
        else: 
            return False
        
    else: 
        return True
