import streamlit as st
import requests
import time

API_URL = "https://modelchurn.onrender.com/predict"

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="NeuraTec | Churn Analytics", layout="centered")

# =========================================================
# ESTADO INICIAL
# =========================================================
if "errors" not in st.session_state:
    st.session_state.errors = {}
if "show_warning" not in st.session_state:
    st.session_state.show_warning = False

# =========================================================
# ESTILOS CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    text-align: center;
    background: linear-gradient(90deg, #1E2A38, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800; font-size: 60px !important;
}

.subtitle { text-align: center; color: #64748b; font-size: 18px; margin-bottom: 30px; }

div.stButton > button {
    display: block; margin: 0 auto; width: 100%; border-radius: 8px;
    height: 55px; background-color: #3b82f6; color: white;
    font-size: 20px; font-weight: bold; border: none; transition: 0.3s;
}
div.stButton > button:hover { background-color: #2563eb; transform: scale(1.01); }

.error-text {
    color: #ef4444; font-size: 13px; font-weight: 600;
    margin-top: -15px; margin-bottom: 15px; display: block;
}

.result-box { padding: 30px; border-radius: 15px; text-align: center; margin-top: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<h1 class="main-title">Análisis de Retención</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plataforma de Inteligencia Predictiva para la Toma de Decisiones</p>', unsafe_allow_html=True)

# =========================================================
# COMPONENTE DE INPUT PERSONALIZADO CON BORDE ROJO 
# =========================================================
def input_con_error(label, key, placeholder=""):
    error = st.session_state.errors.get(key)
    
    # Selector CSS estándar usando el atributo aria-label del input
    if error:
        st.markdown(f"""
        <style>
        div[data-testid="stTextInput"] input[aria-label="{label}"] {{
            border: 2px solid #ef4444 !important;
            background-color: #fff5f5 !important;
        }}
        </style>
        """, unsafe_allow_html=True)

    val = st.text_input(label, key=key, placeholder=placeholder)
    
    if error:
        st.markdown(f"<span class='error-text'>⚠ {error}</span>", unsafe_allow_html=True)
    return val

# =========================================================
# FORMULARIO
# =========================================================
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        tenure = input_con_error("Antigüedad del cliente (Meses)", "tenure", "Ej: 12")
        MonthlyCharges = input_con_error("Cargo Mensual ($)", "MonthlyCharges", "Ej: 75.5")
        TotalCharges = input_con_error("Cargos Totales Acumulados ($)", "TotalCharges", "Ej: 900.25")
        
        gender = st.selectbox("Género", ["Masculino", "Femenino"], index=None, placeholder="Seleccione...")
        Partner = st.selectbox("¿Tiene Pareja?", ["Sí", "No"], index=None, placeholder="Seleccione...")
        Dependents = st.selectbox("¿Tiene Dependientes?", ["Sí", "No"], index=None, placeholder="Seleccione...")

    with col2:
        PhoneService = st.selectbox("Servicio Telefónico", ["Sí", "No"], index=None, placeholder="Seleccione...")
        InternetService = st.selectbox("Tipo de Internet", ["DSL", "Fibra óptica", "No"], index=None, placeholder="Seleccione...")
        Contract = st.selectbox("Tipo de Contrato", ["Mes a mes", "Un año", "Dos años"], index=None, placeholder="Seleccione...")
        PaperlessBilling = st.selectbox("Facturación Electrónica", ["Sí", "No"], index=None, placeholder="Seleccione...")
        PaymentMethod = st.selectbox("Método de Pago", [
            "Cheque electrónico", "Cheque por correo", "Transferencia bancaria (automática)", "Tarjeta de crédito (automática)"
        ], index=None, placeholder="Seleccione...")

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# BOTÓN Y VALIDACIÓN
# =========================================================
if st.button("Analizar Perfil del Cliente"):
    new_errors = {}
    
    # Validar campos numéricos
    if not tenure: new_errors["tenure"] = "Campo obligatorio"
    elif not tenure.isdigit(): new_errors["tenure"] = "Solo números enteros"
    
    try: float(MonthlyCharges)
    except: new_errors["MonthlyCharges"] = "Debe ser un número válido"
    
    try: float(TotalCharges)
    except: new_errors["TotalCharges"] = "Debe ser un número válido"
    
    # Validar selectores
    dropdowns = [gender, Partner, Dependents, PhoneService, InternetService, Contract, PaperlessBilling, PaymentMethod]
    st.session_state.show_warning = any(x is None for x in dropdowns)

    # Guardar errores y recargar para aplicar estilos inmediatamente
    st.session_state.errors = new_errors
    
    if new_errors or st.session_state.show_warning:
        st.rerun()

    # Si todo está OK, procesar con la API (Mapeo a Inglés)
    data = {
        "SeniorCitizen": 0, "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges), "TotalCharges": float(TotalCharges),
        "gender": "Male" if gender == "Masculino" else "Female",
        "Partner": "Yes" if Partner == "Sí" else "No",
        "Dependents": "Yes" if Dependents == "Sí" else "No",
        "PhoneService": "Yes" if PhoneService == "Sí" else "No",
        "MultipleLines": "No",
        "InternetService": {"DSL": "DSL", "Fibra óptica": "Fiber optic", "No": "No"}[InternetService],
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": {"Mes a mes": "Month-to-month", "Un año": "One year", "Dos años": "Two year"}[Contract],
        "PaperlessBilling": "Yes" if PaperlessBilling == "Sí" else "No",
        "PaymentMethod": {
            "Cheque electrónico": "Electronic check", "Cheque por correo": "Mailed check",
            "Transferencia bancaria (automática)": "Bank transfer (automatic)", "Tarjeta de crédito (automática)": "Credit card (automatic)"
        }[PaymentMethod]
    }

    with st.spinner('Procesando análisis...'):
        try:
            response = requests.post(API_URL, json=data, timeout=30)
            if response.status_code == 200:
                prob = response.json()["probability"]
                if prob > 0.5:
                    st.markdown(f'<div class="result-box" style="background-color:#f8d7da; color:#721c24;"><h2>ALTA PROBABILIDAD DE ABANDONO</h2><h3>{prob*100:.1f}%</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box" style="background-color:#d4edda; color:#155724;"><h2>BAJO RIESGO DE ABANDONO</h2><h3>{prob*100:.1f}%</h3></div>', unsafe_allow_html=True)
                    st.balloons()
            else:
                st.error("Error al obtener la predicción.")
        except:
            st.error("Servidor fuera de línea.")

# Mostrar advertencia si faltan selectores
if st.session_state.show_warning:
    st.warning("⚠️ Por favor, complete todas las selecciones de los menús desplegables.")

st.markdown("<hr><div style='text-align:center; font-size:12px; color:#94a3b8;'>© 2026 NeuraTec | Inteligencia Predictiva</div>", unsafe_allow_html=True)
