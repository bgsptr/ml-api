from flask import Flask, Response, jsonify, send_from_directory
import os
import time
from flask_cors import CORS
import json

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app, resources={r"/events": {"origins": "http://localhost:5173"}})

model = load_model('ecopc_b3.h5')

IMAGE_DIR = 'D:\\pkb\\foto-botol'

@app.route('/images/<filename>')
def serve_image(filename):
    # return send_from_directory('D:\\pkb\\foto-botol\\le-1.jpg')
    return send_from_directory(IMAGE_DIR, filename)

def list_images():
    files = os.listdir(IMAGE_DIR)
    image_files = []
    image_urls = []

    for file in files:
        print(file)
        file_path = os.path.join(IMAGE_DIR, file)
        if os.path.isfile(file_path):
            image_files.append(file)

    for image in image_files:
        print(image)
        image_urls.append(f"http://localhost:5000/images/{image}")

    image_map = dict(zip(image_files, image_urls))
    # return image_urls
    return image_map

def prediction(image_file):
    print(image_file)
    img = Image.open(image_file)
    img = img.resize((32, 32)) 
    # img = load_img(image_file, target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    cleo_test = model.predict(x)

    classes_x = np.argmax(cleo_test, axis=1)
    if classes_x == 0:
        merk = "aqua"
    elif classes_x == 1:
        merk = "cleo"
    elif classes_x == 2:
        merk = "lemineral"
    else:
        merk = ""

    return merk

def image_from_os():
    image_map = list_images()
    # for image_url in image_urls:
    for image_file, image_url in image_map.items():
        predict_val = prediction(f"foto-botol/{image_file}")
        data = {"image_url": image_url, "prediction": predict_val}
        yield f"data: {json.dumps(data)}\n\n"
        time.sleep(2)
    yield f"END"

@app.route("/events")
def sse():
    return Response(image_from_os(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
