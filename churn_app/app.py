import streamlit as st
import requests
import time

API_URL = "https://modelchurn.onrender.com/predict"

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="NeuraTec | Churn Analytics", layout="centered")

# =========================================================
# ESTADOS
# =========================================================
if "loading" not in st.session_state:
    st.session_state.loading = False

if "errors" not in st.session_state:
    st.session_state.errors = {}

# =========================================================
# FUNCIÓN API
# =========================================================
def llamar_api(data):
    intentos = 3

    for i in range(intentos):
        try:
            response = requests.post(API_URL, json=data, timeout=40)

            if response.status_code == 200:
                return response

            elif response.status_code == 429:
                time.sleep(3)

            else:
                return response

        except requests.exceptions.RequestException:
            time.sleep(3)

    return None

# =========================================================
# CSS (AGREGADO BORDE ROJO)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.main-title {
    text-align: center;
    background: linear-gradient(90deg, #1E2A38, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 60px !important;
}

.input-error input {
    border: 2px solid #ef4444 !important;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# FUNCION INPUT CON ERROR
# =========================================================
def input_con_error(label, key, placeholder=""):
    error = st.session_state.errors.get(key)

    if error:
        st.markdown('<div class="input-error">', unsafe_allow_html=True)

    value = st.text_input(label, key=key, placeholder=placeholder)

    if error:
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"<p style='color:#ef4444; font-size:12px;'>⚠ {error}</p>", unsafe_allow_html=True)

    return value

# =========================================================
# HEADER
# =========================================================
st.markdown('<h1 class="main-title">Análisis de Retención</h1>', unsafe_allow_html=True)

# =========================================================
# INPUTS
# =========================================================
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        tenure = input_con_error("Antigüedad del cliente (Meses)", "tenure", "Ej: 12")
        MonthlyCharges = input_con_error("Cargo Mensual ($)", "MonthlyCharges", "Ej: 75.5")
        TotalCharges = input_con_error("Cargos Totales ($)", "TotalCharges", "Ej: 900.25")

        gender = st.selectbox("Género", ["Selecciona...", "Male", "Female"])
        Partner = st.selectbox("¿Tiene Pareja?", ["Selecciona...", "Yes", "No"])
        Dependents = st.selectbox("¿Tiene Dependientes?", ["Selecciona...", "Yes", "No"])

    with col2:
        PhoneService = st.selectbox("Servicio Telefónico", ["Selecciona...", "Yes", "No"])
        InternetService = st.selectbox("Tipo de Internet", ["Selecciona...", "DSL", "Fiber optic", "No"])
        Contract = st.selectbox("Tipo de Contrato", ["Selecciona...", "Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Facturación Electrónica", ["Selecciona...", "Yes", "No"])
        PaymentMethod = st.selectbox("Método de Pago", [
            "Selecciona...",
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])

# =========================================================
# BOTÓN
# =========================================================
if st.button("Analizar Perfil del Cliente") and not st.session_state.loading:

    st.session_state.loading = True
    st.session_state.errors = {}

    # VALIDACIONES
    if not tenure:
        st.session_state.errors["tenure"] = "Este campo es obligatorio"
    elif not tenure.isdigit():
        st.session_state.errors["tenure"] = "Debe ser un número entero (ej: 12)"

    try:
        float(MonthlyCharges)
    except:
        st.session_state.errors["MonthlyCharges"] = "Debe ser un número válido (ej: 75.5)"

    try:
        float(TotalCharges)
    except:
        st.session_state.errors["TotalCharges"] = "Debe ser un número válido (ej: 900.25)"

    if "Selecciona..." in [
        gender, Partner, Dependents, PhoneService,
        InternetService, Contract, PaperlessBilling, PaymentMethod
    ]:
        st.warning("Seleccione todas las opciones")

    # SI HAY ERRORES
    if st.session_state.errors:
        st.warning("Corrige los campos en rojo")
        st.session_state.loading = False
        st.stop()

    # DATA
    data = {
        "SeniorCitizen": 0,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": "No",
        "InternetService": InternetService,
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod
    }

    with st.spinner("Procesando..."):
        response = llamar_api(data)

        if response and response.status_code == 200:
            result = response.json()
            prob = result["probability"]

            st.success("Predicción realizada")

            if prob > 0.5:
                st.error(f"ALTO RIESGO: {prob*100:.1f}%")
            else:
                st.success(f"BAJO RIESGO: {prob*100:.1f}%")

        elif response and response.status_code == 429:
            st.warning("⏳ El sistema está iniciando, intenta nuevamente")

        else:
            st.error("⚠️ No se pudo conectar con el servicio")

    st.session_state.loading = False
