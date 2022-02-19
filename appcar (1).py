import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from flask import Flask, request, jsonify, render_template
import pickle

# create a flask app
app = Flask(__name__)


# load the pickel model
model = pickle.load(open("model1.pkl", "rb"))


@app.route("/")
def home():
    return render_template("C:/Users/DELL/Desktop/used car prediction new/index (1).html")

@app.route("/predict", methods = ['GET','POST'])
def predict():
    float_features = [int(x) or float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    return render_template("C:/Users/DELL/Desktop/used car prediction new/index (1).html", prediction_text = "the price for the car is {}".format(prediction))


if __name__ == "__main__":
    app.run()
   # app.run(debug = True)
    