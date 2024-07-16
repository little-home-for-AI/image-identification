from flask import render_template, request, redirect, url_for
from app import app
from app.utils import load_image, predict_image, class_names
import os

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', 'uploads', f.filename)
        f.save(file_path)
        return redirect(url_for('predict', file_path=file_path))
    return render_template('index.html')

@app.route('/predict/<file_path>', methods=['GET'])
def predict(file_path):
    img_array = load_image(file_path)
    prediction = predict_image(img_array)
    result = class_names[prediction]
    return render_template('result.html', result=result, image_path=file_path)
