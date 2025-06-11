# coding:utf-8
 
from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
 
app = Flask(__name__)
 
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/uploads', f.filename)
        f.save(upload_path)
        print('uploading ...')
    return render_template('upload.html')
 
@app.route('/download')
def download():
    dir_path = "static/uploads"
    filename = "7ab20500f360d543612221e3f74a6c08296909317.png"
    print('downloading ...')
    return send_from_directory(dir_path, filename, as_attachment=True)

@app.route('/upload/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
 
 
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)