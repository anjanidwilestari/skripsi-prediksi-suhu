from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('../model-prediksi')
from model import NeuralNetwork, Sigmoid, Backpropagation

app = Flask(__name__)

# Memuat model dari file 'model.pkl'
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Memuat scaler dari file 'scaler.pkl'
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


@app.route("/")
def hello():
    return render_template('index-1.html')

@app.route("/predict", methods=['POST'])
def predict():
    Nh = float(request.form['Nh'])  # Ubah ke float
    
    # Normalisasi data input
    Nh_normalized = scaler.transform(np.array([[Nh]]))[:, 0]
    
    # Lakukan prediksi dengan model yang dimuat
    y_pred_normalized = model.predict_new_value([Nh_normalized])
    
    # Denormalisasi hasil prediksi
    y_pred_denormalized = scaler.inverse_transform(np.array(y_pred_normalized).reshape(-1, 1))[:, 0]
    
    output = round(y_pred_denormalized[0], 8)
    
    return render_template('index-1.html', prediction_text=f"Prediksi Suhu berdasarkan Jumlah Awan {Nh} adalah {output} Celcius")

if __name__ == "__main__":
    app.run()
