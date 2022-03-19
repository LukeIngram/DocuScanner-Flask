from flask import Flask, render_template,request,redirect,url_for,abort,send_from_directory
from werkzeug.utils import secure_filename 
import os 
from validate import *


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10**7
app.config['UPLOAD_EXTENSIONS'] = ['.jpg','.png','.jpeg']
app.config['UPLOAD_PATH'] = './uploads'

@app.route("/")
def index(): 
    return render_template("index.html")


@app.route("/",methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        abort(400)
    file = request.files['file']
    print(file)
    fname = secure_filename(file.filename)
    if fname != '': 
        file_ext = os.path.splitext(fname)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(file.stream):
            abort(400)
        file.save(os.path.join(app.config["UPLOAD_PATH"],fname))
    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

if __name__ == "__main__": 
    app.run(debug=True)