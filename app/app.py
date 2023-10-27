# app.py 

import os
from werkzeug.utils import secure_filename 
from flask import Flask, render_template, request,redirect, url_for, abort, send_from_directory, flash

from app.backend.utils.verify import *
import scripts.main as converter
import backend


# Define Persistent
app = Flask(__name__)
app.config.from_pyfile('keys/config.py')

Scanner = backend.Scanner(app.config[''])


@app.route("/",methods=['GET'])
def index(): 
    return render_template("index.html")


@app.route("/",methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        abort(400)
    file = request.files['file']
    fname = secure_filename(file.filename)
    if fname != '': 
        file_ext = os.path.splitext(fname)[1]
        if file_ext != validate_image(file.stream,file_ext):
            flash("Unsupported File. Supported Types: (png, jpg, jpeg)")
            return redirect(url_for('index'))
        fpath = os.path.join(app.config["UPLOAD_PATH"],fname)
        file.save(fpath)
        flash("File Successfully Uploaded. Evaluating your submission...")
        sterilize_img(fpath)
        flash("\nConverting you image now. Your Download will Begin Shortly.")
        outgoing = convertImg(fpath)
        if outgoing == None: 
            flash("\nConversion Unsuccessful. Please Try a Different File.")
            return redirect(url_for('index'))
    return redirect(url_for('download',filename=outgoing))

def convertImg(filename): 
    if os.path.exists(filename):
        if converter.main(filename,app.config["OUTBOX_PATH"], True)[0] == 0:
            return os.path.splitext(os.path.basename(filename))[0] + '.pdf'
    return None

@app.route("/outbox/<filename>")
def download(filename): 
    try:
        return send_from_directory(app.config['OUTBOX_PATH'],filename,as_attachment=True)
    except FileNotFoundError:
        abort(500)

@app.route("/display/<filename>")
def display(filename): 
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__": 
    app.run(debug=True,port=5001)