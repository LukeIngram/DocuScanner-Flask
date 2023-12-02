# app.py 

import os
from typing import Dict, Any, Tuple

from flask import Flask, render_template, request, redirect, url_for, abort, send_file, flash, session

from backend.Scanner import Scanner
from backend.utils.errors import FileError
from FileHandler import FileHandler



# Define Persistent
app = Flask(__name__)
app.config.from_pyfile('keys/config.py')

Scanner = Scanner(
    app.config['MODEL_WEIGHTS_PATH'],
    app.config['MODEL_DEVICE'],
    app.config['SUPPORTED_IMG_SIZE'],
    app.config['CROP_BUFFER']
    )

FileHandler = FileHandler(app.config['UPLOAD_PATH'], app.config['OUTBOX_PATH'])

EXT_2_TYPE = {
    '.jpg': 'image',
    '.jpeg': 'image',
    '.png': 'image',
    '.gif': 'image',
    '.tiff': 'image',
    '.pdf': 'pdf'
}



@app.route("/",methods=['GET'])
def index(): 
    return render_template("index.html")


@app.route("/css/<filename>",methods=['GET'])
def styles(filename):
    return url_for('static', filename=filename)
    

@app.route("/",methods=['POST'])
def scan_input():
    if 'file' not in request.files:
        abort(400)
    file = request.files['file']

    try: 
        fname = FileHandler.upload_image(file) #TODO TEMPFILES FOR FASTER PROCESSING
    except FileError as e: 
        flash(f"\nConversion Unsuccessful: {e}")
        return redirect(url_for('index'))


    scan_data = Scanner.scan(
        input=FileHandler.fetch_img(fname), 
        verbose=('verbose' in request.form),
        tol = 50 
    )

    if scan_data.get('dewarped') is None: 
        flash(f"\nConversion Unsuccessful. Please try a different Image")
        return redirect(url_for('index'))
   
    else: 
        out_fname = fname
        if ('pdf' in request.form): 
            out_fname = os.path.splitext(fname)[0] + '.pdf'
        else: 
            out_fname = fname

        FileHandler.save_img(scan_data.get('dewarped'), out_fname)
        output_url = url_for('serve_image', filename=out_fname)
        session['output_url'] = output_url

        if ('verbose' in request.form): 
            report = Scanner.build_report(scan_data)
            FileHandler.save_img(report, 'report_' + fname)
            report_url = url_for('serve_image', filename=('report_' + fname))
        else:
            report_url = request.args.get('serve_image', None)

        output_type = EXT_2_TYPE.get(os.path.splitext(out_fname)[1])
        return render_template('index.html', report_url=report_url, output_url=output_url, content_type=output_type)
   


@app.route('/serve_image/<filename>')
def serve_image(filename): 
    ext = os.path.splitext(filename)[1]
    fpath = FileHandler.fetch_download(filename)
    mime_type = 'application/pdf' if ext == '.pdf' else f'image/{ext}'
    return send_file(fpath, mimetype=mime_type, as_attachment=False)


if __name__ == "__main__": 
    app.run(debug=True,port=5001)