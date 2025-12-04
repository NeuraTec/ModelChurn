from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

# ============================================================
# CONFIGURACIÓN
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

MODEL_FILENAME = "mejor_modelo_median.pkl"

try:
    modelo = joblib.load(MODEL_FILENAME)
    logging.info(f"Modelo cargado correctamente: {MODEL_FILENAME}")
except Exception as e:
    logging.error(f"No se pudo cargar el modelo: {str(e)}")
    modelo = None

# ============================================================
# CATEGORÍAS PERMITIDAS (ORIGINALES)
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
# NORMALIZADOR FLEXIBLE
# ============================================================

def normalize(value: str):
    """
    Normaliza la entrada del usuario:
    - Todo a minúsculas
    - Reemplaza guiones y guiones bajos por espacios
    - Elimina espacios repetidos
    """
    value = value.lower().replace("_", " ").replace("-", " ")
    value = " ".join(value.split())  # elimina espacios dobles
    return value

# Crear diccionario normalizado para búsqueda rápida
normalized_categories = {
    field: {normalize(v): v for v in allowed_values}
    for field, allowed_values in valid_categories.items()
}

# ============================================================
# VALIDACIÓN DE ENTRADAS
# ============================================================

def validate_input(data):
    """Valida campos numéricos y categorías FLEXIBLES."""

    # 1️⃣ Campos faltantes
    missing = [f for f in expected_fields if f not in data]
    if missing:
        return False, f"Faltan campos requeridos: {missing}"

    # 2️⃣ Validación numérica y evitar negativos
    for f in numeric_fields:
        try:
            data[f] = float(data[f])
        except ValueError:
            return False, f"El campo '{f}' debe ser numérico."

        if data[f] < 0:
            return False, f"El campo '{f}' no puede ser negativo."

    # 3️⃣ Validación de SeniorCitizen
    if data["SeniorCitizen"] not in [0, 1]:
        return False, "SeniorCitizen debe ser 0 o 1."

    # 4️⃣ Validación FLEXIBLE de categorías
    for field, allowed_dict in normalized_categories.items():
        raw_value = str(data[field])
        key = normalize(raw_value)

        if key not in allowed_dict:
            return False, (
                f"Valor inválido en '{field}'. "
                f"Valor recibido: '{raw_value}'. "
                f"Valores permitidos: {list(allowed_dict.values())}"
            )

        # Reemplazar por el valor oficial
        data[field] = allowed_dict[key]

    return True, None

# ============================================================
# Punto de Predicción
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

        logging.info(f"Predicción realizada: proba={proba:.4f}, pred={pred}")

        return jsonify({
            "success": True,
            "prediction": pred,
            "label": label,
            "probability": round(float(proba), 4),
            "threshold_used": threshold,
            "model": MODEL_FILENAME
        })

    except Exception as e:
        logging.error(f"Error en predicción: {str(e)}")
        return jsonify({"error": "Hubo un error procesando la predicción"}), 500

# ============================================================
# ESTADO DEL SERVICIO
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Churn Prediction API",
        "status": "running",
        "model_loaded": modelo is not None
    })

# ============================================================
#INICIAR SERVIDOR
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

