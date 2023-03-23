#flask, flask-cors, tensorflow, numpy, pillow

import numpy as np
import tensorflow as tf

from flask import Flask, request
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/one-piece", methods = ['POST'])
def one_piece_classifier ():

    # Ambil gambar yang dikirim pas request
    image_request = request.files['image']

    # konversi gambar menjadi array
    image_pil = Image.open(image_request)

    # ngeresize gambarnya
    expected_size = (90, 90)
    resized_image_pil = image_pil.resize(expected_size)

    # generate array dengan numpy
    image_array = np.array(resized_image_pil)
    rescaled_image_array = image_array/255.
    batched_rescaled_image_array = np.array([rescaled_image_array])
    # print(batched_rescaled_image_array.shape)

    # load model
    loaded_model = tf.keras.models.load_model("one-piece-classifier.h5")
    # print(loaded_model.get_config())
    result = loaded_model.predict(batched_rescaled_image_array)

    return get_formated_predict_result(result)

def get_formated_predict_result(predict_result) :
    class_indices = {'Ace': 0, 'Akainu': 1, 'Brook': 2, 'Chopper': 3, 'Crocodile': 4, 'Franky': 5, 'Jinbei': 6, 'Kurohige': 7, 'Law': 8,
     'Luffy': 9, 'Mihawk': 10, 'Nami': 11, 'Rayleigh': 12, 'Robin': 13, 'Sanji': 14, 'Shanks': 15, 'Usopp': 16,
     'Zoro': 17}
    inverted_class_indices = {}

    for key in class_indices:
        class_indices_key = key
        class_indices_value = class_indices[key]

        inverted_class_indices[class_indices_value] = class_indices_key

    processed_predict_result = predict_result[0]
    maxIndex = 0
    maxValue = 0

    for index in range(len(processed_predict_result)):
        if processed_predict_result[index] > maxValue:
            maxValue = processed_predict_result[index]
            maxIndex = index

    return inverted_class_indices[maxIndex]


if __name__ == "__main__":
    app.run()