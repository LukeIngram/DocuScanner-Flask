from flask import Flask, render_template,request,redirect,url_for,abort,send_from_directory,flash
from werkzeug.utils import secure_filename 
import os 
from scripts.verify import *
import scripts.main as converter
from pathlib import Path


app = Flask(__name__)
app.config.from_object('config')
app.config["MAX_CONTENT_LENGTH"] = 10**7
app.config['UPLOAD_EXTENSIONS'] = ['.jpg','.png','.jpeg']
app.config['UPLOAD_PATH'] = './uploads'
app.config['OUTBOX_PATH'] = './outbox'


@app.route("/")
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
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(file.stream):
            flash("Unsupported File. Supported Types: (png, jpg, jpeg)")
            return redirect(url_for('index'))
        fpath = os.path.join(app.config["UPLOAD_PATH"],fname)
        file.save(fpath)
        flash("File Successfully Uploaded. Your Download will Begin Shortly.")
        outgoing = convertImg(fpath)
        if outgoing == None: 
            flash("Conversion Unsuccessful. Please Try a Different File.")
            return redirect(url_for('index'))
    return redirect(url_for('download',filename=outgoing))

def convertImg(filename): 
    if os.path.exists(filename): 
        if converter.main(filename,app.config["OUTBOX_PATH"])[0] == 0:
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
    try:
       return send_from_directory(app.config['UPLOAD_PATH'],filename)
    except FileNotFoundError:
        abort(500)


if __name__ == "__main__": 
    app.run(debug=True,port=5001)