from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import os

# ============================================================
# CONFIGURACIÓN
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mejor_modelo_median.pkl")

try:
    modelo = joblib.load(MODEL_PATH)
    logging.info(f"Modelo cargado correctamente desde: {MODEL_PATH}")
except Exception as e:
    logging.error(f"No se pudo cargar el modelo: {str(e)}")
    modelo = None

# ============================================================
# CATEGORÍAS
# ============================================================

valid_categories = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "Yes", "No"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ],
}

numeric_fields = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
expected_fields = list(valid_categories.keys()) + numeric_fields

# ============================================================
# NORMALIZADOR
# ============================================================

def normalize(value: str):
    value = value.lower().replace("_", " ").replace("-", " ")
    return " ".join(value.split())

normalized_categories = {
    field: {normalize(v): v for v in allowed_values}
    for field, allowed_values in valid_categories.items()
}

# ============================================================
# VALIDACIÓN
# ============================================================

def validate_input(data):

    missing = [f for f in expected_fields if f not in data]
    if missing:
        return False, f"Faltan campos requeridos: {missing}"

    for f in numeric_fields:
        try:
            data[f] = float(data[f])
        except ValueError:
            return False, f"El campo '{f}' debe ser numérico."

        if data[f] < 0:
            return False, f"El campo '{f}' no puede ser negativo."

    if data["SeniorCitizen"] not in [0, 1]:
        return False, "SeniorCitizen debe ser 0 o 1."

    for field, allowed_dict in normalized_categories.items():
        raw_value = str(data[field])
        key = normalize(raw_value)

        if key not in allowed_dict:
            return False, (
                f"Valor inválido en '{field}'. "
                f"Valor recibido: '{raw_value}'. "
                f"Valores permitidos: {list(allowed_dict.values())}"
            )

        data[field] = allowed_dict[key]

    return True, None

# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if modelo is None:
        return jsonify({"error": "El modelo no está disponible"}), 500

    data = request.get_json()

    if not data:
        return jsonify({"error": "No se recibió JSON"}), 400

    ok, error_msg = validate_input(data)
    if not ok:
        return jsonify({"error": error_msg}), 400

    df = pd.DataFrame([data])

    try:
        proba = modelo.predict_proba(df)[:, 1][0]
        threshold = 0.40
        pred = int(proba >= threshold)
        label = "Churn" if pred == 1 else "No Churn"

        logging.info(f"Predicción: proba={proba:.4f}, pred={pred}")

        return jsonify({
            "success": True,
            "prediction": pred,
            "label": label,
            "probability": round(float(proba), 4),
            "threshold_used": threshold,
            "model": os.path.basename(MODEL_PATH)
        })

    except Exception as e:
        logging.error(f"Error en predicción: {str(e)}")
        return jsonify({"error": "Error procesando la predicción"}), 500

# ============================================================
# HEALTH CHECK (PRO)
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": modelo is not None
    })

# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Churn Prediction API",
        "status": "running",
        "model_loaded": modelo is not None
    })

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
