import streamlit as st
import requests

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
# LÓGICA DE VALIDACIÓN
# =========================================================
def validar_campo_live(key):
    valor = st.session_state[key]
    if key == "tenure":
        if not valor: st.session_state.errors[key] = "Campo obligatorio"
        elif not valor.isdigit(): st.session_state.errors[key] = "Solo números enteros"
        else: st.session_state.errors.pop(key, None)
    elif key in ["MonthlyCharges", "TotalCharges"]:
        if not valor: st.session_state.errors[key] = "Campo obligatorio"
        else:
            try:
                float(valor)
                st.session_state.errors.pop(key, None)
            except ValueError: st.session_state.errors[key] = "Debe ser un número válido"

# =========================================================
# ESTILOS CSS (INCLUYENDO TU NUEVA REGLA PARA LA "X")
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

/* --- FOCO AZUL Y ELIMINACIÓN DE SOBREBORDE --- */
[data-baseweb="input"], [data-baseweb="select"] {
    outline: none !important;
    border: none !important;
}

div[data-baseweb="input"] > div, 
div[data-baseweb="base-input"],
div[data-baseweb="select"] > div {
    border: 1px solid rgba(0,0,0,0.1) !important;
    transition: all 0.2s ease-in-out !important;
}

div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="base-input"]:focus-within,
div[data-baseweb="select"]:focus-within > div {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

/* --- TU AJUSTE: OCULTAR LA "X" DE LIMPIAR EN SELECTBOX --- */
div[role="button"][aria-label="Clear value"],
button[title="Clear value"],
button[aria-label="Clear value"],
div[data-baseweb="select"] [aria-label="Clear value"],
div[data-baseweb="select"] [title="Clear value"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

input { border: none !important; }

.error-text { color: #ef4444; font-size: 13px; font-weight: 600; margin-top: -15px; margin-bottom: 15px; display: block; }
.result-box { padding: 30px; border-radius: 15px; text-align: center; margin-top: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# =========================================================
# COMPONENTE DE INPUT PERSONALIZADO
# =========================================================
def input_con_error(label, key, placeholder=""):
    error = st.session_state.errors.get(key)
    if error:
        st.markdown(f"""<style>div[data-testid="stTextInput"] input[aria-label="{label}"] {{ border: 2px solid #ef4444 !important; background-color: #fff5f5 !important; }}</style>""", unsafe_allow_html=True)
    val = st.text_input(label, key=key, placeholder=placeholder, on_change=validar_campo_live, args=(key,))
    if error: st.markdown(f"<span class='error-text'>⚠ {error}</span>", unsafe_allow_html=True)
    return val

# =========================================================
# UI PRINCIPAL
# =========================================================
st.markdown('<h1 class="main-title">Análisis de Retención</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plataforma de Inteligencia Predictiva para la Toma de Decisiones</p>', unsafe_allow_html=True)

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

if st.button("Analizar Perfil del Cliente"):
    validar_campo_live("tenure")
    validar_campo_live("MonthlyCharges")
    validar_campo_live("TotalCharges")
    
    dropdowns = [gender, Partner, Dependents, PhoneService, InternetService, Contract, PaperlessBilling, PaymentMethod]
    st.session_state.show_warning = any(x is None for x in dropdowns)

    if st.session_state.errors or st.session_state.show_warning:
        st.rerun()

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
                    st.markdown(f'''
                        <div class="result-box" style="background-color:#f8d7da; color:#721c24; border: 1px solid #f5c6cb;">
                            <h2 style="margin:0;">ALTA PROBABILIDAD DE ABANDONO</h2>
                            <h3 style="margin:10px 0 0 0; font-size:40px;">{prob*100:.1f}%</h3>
                        </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                        <div class="result-box" style="background-color:#d4edda; color:#155724; border: 1px solid #c3e6cb;">
                            <h2 style="margin:0;">BAJO RIESGO DE ABANDONO</h2>
                            <h3 style="margin:10px 0 0 0; font-size:40px;">{prob*100:.1f}%</h3>
                        </div>
                    ''', unsafe_allow_html=True)
                    st.balloons()
            else: st.error("Error en la predicción.")
        except: st.error("Servidor no disponible.")

if st.session_state.show_warning:
    st.warning("⚠️ Por favor, complete todas las selecciones de los menús desplegables.")

st.markdown("<hr><div style='text-align:center; font-size:12px; color:#94a3b8;'>© 2026 NeuraTec | Inteligencia Predictiva</div>", unsafe_allow_html=True)
