from flask import Flask, request, url_for, redirect, render_template, jsonify
from werkzeug.utils import secure_filename
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import cv2
import matplotlib.pyplot as plt
import joblib
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image
import numpy as np

from chatbot_model import chatbot_intents,pcoschatbotpredict

from chatbot_model import chatbot_intents,pcoschatbotpredict

app = Flask(__name__, template_folder="template")

reg = pickle.load(open("model1.pkl", "rb"))
assistant = chatbot_intents()

@app.route("/")
def hello_worl():
     return render_template("main.html")

@app.route("/template/choose.html")
def hello_wo():
     return render_template("choose.html")

@app.route("/template/test.html")
def hello1():
     return render_template("test.html")

@app.route("/template/nearestshop.html")
def hello3():
    return render_template("nearestshop.html")


@app.route("/template/gynac.html")
def hello4():
    return render_template("gynac.html")

@app.route("/template/diet.html")
def diet():
     return render_template("diet.html")

@app.route("/template/indexp.html")
def periodtracker():
    return render_template("indexp.html")

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
# Load your trained model
@app.route("/template/test2.html")
def hello2():
     return render_template("test2.html")

reg1 = joblib.load('xray.pkl')

# Initialize the VGG16 feature extractor
SIZE = 256
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

# Define preprocessing function
def preprocess_image(image):
    img = cv2.resize(image, (SIZE, SIZE))
    img = img / 255.0
    return img


@app.route('/predicts', methods=['POST'])
def predicts():
    # Get the uploaded file
    file = request.files['file']
    if not file:
        return "No file provided", 400

    # Save the file temporarily
    file_path = 'temp_image.jpg'
    file.save(file_path)

    # Preprocess the image
    img = cv2.imread(file_path)
    img = preprocess_image(img)

    # Use VGG16 to extract features
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    feature_extractor = vgg_model.predict(img)  # Extract features
    features = feature_extractor.reshape(1, -1)  # Flatten

    # Make prediction with the extracted features
    prediction = reg1.predict(features)

    # Define your labels
    labels = {0: 'infected', 1: 'notinfected'}


    predicted_label = labels.get(prediction[0], "Unknown")

    # Delete the temporary file
    os.remove(file_path)

    # Return the predicted class label in the template

    return render_template('index2.html', prediction=predicted_label)

#------------------------------------------------------
# New route to render the prediction result
#@app.route('/result')
#def show_result():
    # Extract the prediction from query parameters
  #  predictions = request.args.get('prediction', 'Unknown')
   # return render_template('index2.html', prediction=predictions)  # Render index2.html with the prediction
#--------------------------------------------------------


@app.route("/chatbot")
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    response = pcoschatbotpredict(assistant,message)
    return jsonify({"message": response})

@app.route("/chatbot")
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    response = pcoschatbotpredict(assistant,message)
    return jsonify({"message": response})


if __name__ == "__main__":
    app.run(port=5000,debug=True)