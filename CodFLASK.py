from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Carga el modelo una vez
modelo = joblib.load('modelo_churn_pipeline.pkl')
encoder = joblib.load('target_encoder_churn.pkl')  # Solo si lo usas


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # JSON con los 20 campos
    df = pd.DataFrame([data])
    proba = modelo.predict_proba(df)[:, 1][0]
    pred = 1 if proba >= 0.4 else 0

    # Si tienes encoder, convierte a texto. Si no, queda en 0/1
    label = encoder.inverse_transform([pred])[0] if encoder else str(pred)
    return jsonify({
        'probability': round(proba, 4),
        'prediction': label
    })

if __name__ == '__main__':
    app.run(debug=True)
