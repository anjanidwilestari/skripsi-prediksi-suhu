from flask import Flask, render_template, request
import pickle
import sys
sys.path.append('../model-prediksi')
from model import NeuralNetwork
from model import Sigmoid
from model import Backpropagation


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    Nh = int(request.form['Nh'])
    
    prediction = model.predict_new_value([[Nh]])
    
    output = round(prediction[0], 3)
    return render_template('index.html', prediction_text=f"Prediksi Suhu berdasarkan Jumlah Awan {Nh} adalah {prediction} Celcius")


if __name__ == "__main__":
    app.run()