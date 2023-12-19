from flask import Flask, render_template, request
import pandas as pd
import numpy as np
# import keras
import pickle
import joblib
from keras.models import load_model
model = load_model('model.h5')

app = Flask(__name__)

# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)
# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)

# model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features =  {
            'in_volume': float(request.form['in_volume']),
            'in_flowrate': float(request.form['in_flowrate']),
            'in_Pressure': float(request.form['in_Pressure']),
            'out_Volume': float(request.form['out_Volume']),
            'out_flowrate': float(request.form['out_flowrate']),
            'Pressure_out': float(request.form['Pressure_out'])
            }
        user_df = pd.DataFrame([features])
        user_scaled = scaler.transform(user_df)
        user_reshaped = user_scaled.reshape((user_scaled.shape[0], user_scaled.shape[1], 1))
        user_pred_proba = model.predict(user_reshaped)
        user_pred = np.argmax(user_pred_proba, axis=1)
        vol_pred = user_pred_proba[0, 0]
        pres_pred = user_pred_proba[0, 1]
        output = ""
        if vol_pred and pres_pred == 0.0:
            output = "no leak"
        elif (vol_pred and pres_pred == 1.0) or (vol_pred == 1.0 and pres_pred == 0.0):
            output = "leak"
        else:
            output = "theaft"
        

        return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
