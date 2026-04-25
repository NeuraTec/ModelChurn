import streamlit as st
import requests
import time

API_URL = "https://modelchurn.onrender.com/predict"

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="NeuraTec | Churn Analytics", layout="centered")

# =========================================================
# ESTADO
# =========================================================
if "loading" not in st.session_state:
    st.session_state.loading = False

if "errors" not in st.session_state:
    st.session_state.errors = {}

# =========================================================
# LÓGICA DE LIMPIEZA DINÁMICA
# =========================================================
def limpiar_error(key):
    if key not in st.session_state.errors:
        return

    valor = st.session_state.get(key, "")

    # Solo borra el error si el nuevo valor ya es correcto
    if key == "tenure":
        if valor and valor.strip().isdigit():
            del st.session_state.errors[key]
    elif key in ("MonthlyCharges", "TotalCharges"):
        try:
            float(valor)
            del st.session_state.errors[key]
        except:
            pass  # Sigue en rojo si el valor sigue siendo inválido

# =========================================================
# API
# =========================================================
def llamar_api(data):
    intentos = 3
    for _ in range(intentos):
        try:
            response = requests.post(API_URL, json=data, timeout=40)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                time.sleep(3)
            else:
                return response
        except:
            time.sleep(3)
    return None

# =========================================================
# ESTILOS
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

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 18px;
    margin-bottom: 30px;
}

div.stButton > button {
    display: block;
    margin: 0 auto;
    width: 100%;
    border-radius: 8px;
    height: 55px;
    background-color: #3b82f6;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

div.stButton > button:hover {
    background-color: #2563eb;
    transform: scale(1.02);
}

.result-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.error-text {
    color: #ef4444;
    font-size: 13px;
    font-weight: 600;
    margin-top: -15px;
    margin-bottom: 15px;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<h1 class="main-title">Análisis de Retención</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plataforma de Inteligencia Predictiva para la Toma de Decisiones</p>', unsafe_allow_html=True)

# =========================================================
# INPUT CON ERROR (JS INTEGRADO)
# =========================================================
def input_con_error(label, key, placeholder=""):
    error = st.session_state.errors.get(key)

    value = st.text_input(
        label,
        key=key,
        placeholder=placeholder,
        on_change=limpiar_error,
        args=(key,)
    )

    if error:
        st.markdown(f"""
        <style>
        div[data-testid="stTextInput"] input[aria-label="{label}"] {{
            border: 2px solid #ef4444 !important;
            background-color: #fff5f5 !important;
        }}
        </style>
        <span class='error-text'>⚠ {error}</span>
        """, unsafe_allow_html=True)

    return value

# =========================================================
# LAYOUT DE INPUTS
# =========================================================
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        tenure = input_con_error("Antigüedad del cliente (Meses)", "tenure", "Ej: 12")
        MonthlyCharges = input_con_error("Cargo Mensual ($)", "MonthlyCharges", "Ej: 75.5")
        TotalCharges = input_con_error("Cargos Totales Acumulados ($)", "TotalCharges", "Ej: 900.25")

        gender = st.selectbox("Género", ["Male", "Female"])
        Partner = st.selectbox("¿Tiene Pareja?", ["Yes", "No"])
        Dependents = st.selectbox("¿Tiene Dependientes?", ["Yes", "No"])

    with col2:
        PhoneService = st.selectbox("Servicio Telefónico", ["Yes", "No"])
        InternetService = st.selectbox("Tipo de Internet", ["DSL", "Fiber optic", "No"])
        Contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Facturación Electrónica", ["Yes", "No"])
        PaymentMethod = st.selectbox("Método de Pago", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

# =========================================================
# BOTÓN Y LÓGICA DE PROCESAMIENTO
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Analizar Perfil del Cliente") and not st.session_state.loading:

    st.session_state.loading = True
    st.session_state.errors = {}

    # VALIDACIÓN INICIAL AL CLIC
    if not tenure:
        st.session_state.errors["tenure"] = "Campo obligatorio"
    elif not tenure.isdigit():
        st.session_state.errors["tenure"] = "Solo números enteros"

    try:
        float(MonthlyCharges)
    except:
        st.session_state.errors["MonthlyCharges"] = "Debe ser un número válido"

    try:
        float(TotalCharges)
    except:
        st.session_state.errors["TotalCharges"] = "Debe ser un número válido"

    if st.session_state.errors:
        st.session_state.loading = False
        st.rerun()

    # PREPARACIÓN DE DATA
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

    with st.spinner('Procesando datos con el motor de IA...'):
        response = llamar_api(data)

        if response and response.status_code == 200:
            result = response.json()
            prob = result["probability"]
            st.toast("Análisis finalizado con éxito", icon="🎯")

            if prob > 0.5:
                st.markdown(f"""
                <div class="result-box" style="background-color:#f8d7da;">
                    <h2>ALTA PROBABILIDAD DE ABANDONO</h2>
                    <p style="font-size:30px; font-weight:bold; color:#721c24;">{prob*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box" style="background-color:#d4edda;">
                    <h2>BAJO RIESGO DE ABANDONO</h2>
                    <p style="font-size:30px; font-weight:bold; color:#155724;">{prob*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
        elif response and response.status_code == 429:
            st.warning("⏳ El sistema está iniciando. Intenta de nuevo en unos segundos.")
        else:
            st.error("⚠️ No se pudo conectar con el servicio.")

    st.session_state.loading = False

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<hr>
<div style='text-align:center; font-size:12px; color:#94a3b8;'>
© 2026 NeuraTec | Inteligencia Predictiva
</div>
""", unsafe_allow_html=True)
