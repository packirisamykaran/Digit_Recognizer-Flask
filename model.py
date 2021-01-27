import numpy as np
import pandas as pd
import pickle
import requests
import json
from keras.models import load_model
from flask import Flask, jsonify, render_template, request, send_from_directory
from keras.preprocessing.image import load_img, img_to_array
import cv2
count = 0

app = Flask(__name__)
model = load_model('static/digitrecognizer2.h5')


@app.route('/')
def man():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def home():
    global count
    img = request.files['image']
    img.save('static/{}.jpg'.format(count))
    model = load_model('static/digitrecognizer2.h5')
    img = cv2.imread('static/{}.jpg'.format(count))

    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_to_array(img) / 255.0

    print(img.shape)
    img = np.expand_dims(img, -1)
    img = np.array([img])
    predict = model.predict(img)
    predict = int(predict.argmax())
    print(predict)

    count+=1
    return render_template('prediction.html', data=predict)

@app.route('/load_img')
def load_img():
    global count
    return send_from_directory('static', "{}.jpg".format(count))

if __name__ == '__main__':
    app.run(debug=True)
