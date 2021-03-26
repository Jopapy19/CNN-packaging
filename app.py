from __future__ import division, print_function

# coding=utf-8
import os
import sys
import glob
import re
import numpy as np

# keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# Flask utils
from flask import Flask, flash, render_template, request, jsonify, make_response, url_for, redirect
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import time
import base64
import io
from PIL import Image


import warnings
from tensorboard.summary.v1 import image
from tensorflow.python.data.experimental.ops.optimization import model
warnings.filterwarnings("ignore")

# define global paths for Image
# IMG_FOLDER = os.path.join('static', 'images')
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define a flask app
app = Flask(__name__)

# Config environment variables
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["UPLOADED_IMG_URL"] = ""


def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Save & Load Model
def get_model():
    model_path = 'static/models/VGG16_model_at_20210319_235059.h5'
    model = load_model(model_path)
    print("* Model loaded!")

    # Load your trained model
    model = load_model(model_path)
    model._make_predict_function()          # Necessary

# Preprocessing function
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize.convert("RGB")
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

    print(" *Loading Keras model...")
    get_model()

# Predict model
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


# Upload file
@app.route("/upload/<filename>")
def uploaded_file(filename):
    fileURL = os.path.join("uploads", filename)
    return render_template("index.html", fileURL=fileURL)


@app.route("/predict", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file path
        if 'file' not in request.files:
            flash('No file path')
            return redirect(request.url)
        # Get the file from post request
        f = request.files['file']
        # if user does't select file, browser also
        # submit an empty part without filename
        if f.filename == '':
            flash('No selected file')
            time.sleep(10)
            return redirect(request.url)
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            file_path = f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))

         # Make model prediction
        pred = preprocess_image(file_path, model)

        #Read prediction
    #     pred = model_predict(filename, model)
    #     pred_class = decode_predictions(pred, top=1)
    #     result = str(pred_class[0][0][1])

    #     return result
    # return None
    

   
# # Read Prediction
@app.route("/", methods=["POST"])
@cross_origin()
def predict():
    # print("request", request)
    message = request.get_json()
    # print("### message", message)
    encoded = message['image'].split(",")[-1]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    input_image_size = (224, 224)
    processed_image = preprocess_image(image, target_size=input_image_size)
    print("######### processed_img size", processed_image.shape)
    prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'men': men-prediction[0][0],
            'women': women-prediction[0][1]
        }
    }
    return jsonify(response)



# Main
@app.route("/", methods=['GET'])
@cross_origin()
def index():
    # Main page
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
