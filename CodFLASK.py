from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carga el modelo y encoder (si existe)
modelo = joblib.load('modelo_churn_pipeline.pkl')

try:
    encoder = joblib.load('target_encoder_churn.pkl')
except Exception:
    encoder = None

# Define la lista de campos esperados (los 19 campos que usas en el sample_data)
expected_fields = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No JSON data received'}), 400

    # Validar que tenga todos los campos esperados
    missing_fields = [f for f in expected_fields if f not in data]
    if missing_fields:
        return jsonify({'error': f'Faltan campos: {missing_fields}'}), 400

    # Convertir a DataFrame para alimentar el pipeline
    df = pd.DataFrame([data])

    try:
        proba = modelo.predict_proba(df)[:, 1][0]
        pred = 1 if proba >= 0.4 else 0
    except Exception as e:
        return jsonify({'error': f'Error al hacer la predicción: {str(e)}'}), 500

    if encoder:
        label = encoder.inverse_transform([pred])[0]
    else:
        label = pred

    return jsonify({
        'probability': round(proba, 4),
        'prediction': label
    })

if __name__ == '__main__':
    app.run(debug=True)
