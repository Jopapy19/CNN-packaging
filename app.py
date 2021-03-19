from flask import Flask, render_template, request, jsonify, make_response
import os
import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask_cors import CORS, cross_origin
import warnings
warnings.filterwarnings("ignore")

# define global paths for Image and csv folders
IMG_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)

# config environment variables
app.config['IMG_FOLDER'] = IMG_FOLDER

def get_model():
    global model
    model = load_model('static/models/VGG16_model_at_20210317_083528.h5')
    print("* Model loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize.convert("RGB")
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

    print(" *Loading Keras model...")
    get_model()


# TODO: Implement predict route for POST Method
@app.route("/predict", methods=["POST"])
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
            'men': prediction[0][0],
            'women': prediction[0][1]
        }
    }
    return jsonify(response)


@app.route("/", methods=["GET"])
@cross_origin()
def index():

    # if request.method == "POST":
    #     result = request.form['name']
    #     print(result)
    # return render_template("index.html", name={"firstName":"Jopapy19"}, workat="Plexes")
    return render_template("index.html")
    response = make_response('The page named %s doesnt exists.'
                             % index.html, 404)
    return response


if __name__ == "__main__":
    app.run(debug=True)
