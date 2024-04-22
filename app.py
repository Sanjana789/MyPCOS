from flask import Flask, request, url_for, redirect, render_template, jsonify
from werkzeug.utils import secure_filename
import pickle
import os
import cv2
from PIL import Image
import numpy as np


app = Flask(__name__, template_folder="template")

reg = pickle.load(open("model1.pkl", "rb"))

@app.route("/")
def hello_worl():
     return render_template("main.html")

@app.route("/template/choose.html")
def hello_wo():
     return render_template("choose.html")

@app.route("/template/test.html")
def hello1():
     return render_template("test.html")


@app.route("/predict", methods=["POST"])
def home():
    data1 = float(request.form["a"])
    data2 = float(request.form["b"])
    data3 = float(request.form["c"])
    data4 = float(request.form["d"])
    d5 = float(request.form["e"])
    d6 = float(request.form["f"])
    d7 = float(request.form["g"])
    d8 = float(request.form["h"])
    d9 = float(request.form["i"])
    d10 = float(request.form["j"])
    d11 = float(request.form["k"])
    d12 = float(request.form["l"])
    d13 = float(request.form["m"])
    d14 = float(request.form["n"])
    d15 = float(request.form["o"])
    d16 = float(request.form["p"])
    d17 = float(request.form["q"])
    d18 = float(request.form["r"])
    d19 = float(request.form["s"])
    d20 = float(request.form["t"])
    d21 = float(request.form["u"])
    d22 = float(request.form["v"])
    d23 = float(request.form["w"])
    d24 = float(request.form["x"])
    d25 = float(request.form["y"])
    d26 = float(request.form["z"])
    d27 = float(request.form["za"])
    

    arr = np.array(
        [
            [
                data1,
                data2,
                data3,
                data4,
                d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,
                d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27
                            ]
        ]
    )
    pred = reg.predict(arr)
    print(pred)
    return render_template("index.html", data=pred)



    #ultrasound-------------------------------------------------------
reg1 = pickle.load(open("ultrasound.pkl", "rb"))

# Define image preprocessing function
def preprocess_image(image_path):
    SIZE = 128
    img = cv2.imread(image_path)
    img = cv2.resize(img, (SIZE, SIZE))
    img = img / 255.0
    return img

# Defining the labels 
labels = {'infected': 0, 'notinfected': 1}

@app.route("/template/test2.html")
def hello2():
     return render_template("test2.html")

@app.route('/predicts', methods=['POST'])
def predicts():
    # Get the uploaded file
    file = request.files['file']

    # Save the file temporarily
    file_path = 'temp_image.jpg'
    file.save(file_path)

    # Preprocess the image
    img = preprocess_image(file_path)

    # Reshape the image to match the input shape of the model
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = reg1.predict(img)

    # Get the predicted class label
    predicted_label = labels[prediction[0]]

    # Delete the temporary file
    os.remove(file_path)

    # Return the predicted class label
    #return jsonify({'result': predicted_label})
    print(prediction)
    return render_template("index2.html", datau=prediction)


if __name__ == "__main__":
    app.run(port=5000,debug=True)

