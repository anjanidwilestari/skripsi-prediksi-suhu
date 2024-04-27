from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
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
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Jika user mengirimkan input via text field
    if 'Nh' in request.form:
        Nh = float(request.form['Nh'])  # Ubah ke float
        output = make_prediction(Nh)
        return render_template('index.html', prediction_text=f"Prediksi Suhu berdasarkan Jumlah Awan {Nh} adalah {output} Celcius")

    # Jika user mengunggah file Excel
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction_text="Mohon pilih file terlebih dahulu.")
        if file:
            filename = secure_filename(file.filename)  # Perbaiki nama file
            if filename.endswith('.xlsx'):
                # Pastikan direktori 'uploads' ada
                if not os.path.exists('uploads'):
                    os.makedirs('uploads')
                file_path = os.path.join('uploads', filename)
                file.save(file_path)
                df = pd.read_excel(file_path)
                df['Prediction'] = df['Nh'].apply(make_prediction)
                # Pastikan direktori 'downloads' ada
                if not os.path.exists('downloads'):
                    os.makedirs('downloads')
                output_path = os.path.join('downloads', 'predictions.xlsx')
                df.to_excel(output_path, index=False)
                return send_file(output_path, as_attachment=True)

    return render_template('index.html', prediction_text="Mohon inputkan nilai Jumlah Awan atau unggah file Excel.")

def make_prediction(Nh):
    # Normalisasi data input
    Nh_normalized = scaler.transform(np.array([[Nh]]))[:, 0]

    # Lakukan prediksi dengan model yang dimuat
    y_pred_normalized = model.predict_new_value([Nh_normalized])

    # Denormalisasi hasil prediksi
    y_pred_denormalized = scaler.inverse_transform(np.array(y_pred_normalized).reshape(-1, 1))[:, 0]

    return round(y_pred_denormalized[0], 8)

if __name__ == "__main__":
    app.run(debug=True)
