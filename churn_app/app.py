import streamlit as st
import requests
import time

API_URL = "https://modelchurn.onrender.com/predict"

st.set_page_config(page_title="NeuraTec | Churn Analytics", layout="centered")

# =========================================================
# ESTADOS
# =========================================================
if "loading" not in st.session_state:
    st.session_state.loading = False

if "errors" not in st.session_state:
    st.session_state.errors = {}

# =========================================================
# CSS CORREGIDO (APUNTA BIEN AL INPUT)
# =========================================================
st.markdown("""
<style>
/* BORDE ROJO REAL EN INPUT */
div[data-testid="stTextInput"] input.error {
    border: 2px solid #ef4444 !important;
    box-shadow: 0 0 0 1px #ef4444 !important;
}

/* MENSAJE DE ERROR */
.error-text {
    color: #ef4444;
    font-size: 12px;
    margin-top: -10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# FUNCIÓN INPUT CON ERROR (VERSIÓN CORRECTA)
# =========================================================
def input_con_error(label, key, placeholder=""):
    error = st.session_state.errors.get(key)

    value = st.text_input(
        label,
        key=key,
        placeholder=placeholder,
    )

    # Inyectar clase error SOLO si hay error
    if error:
        st.markdown(f"""
        <script>
        const inputs = window.parent.document.querySelectorAll('input');
        inputs.forEach(el => {{
            if (el.getAttribute('aria-label') === "{label}") {{
                el.classList.add('error');
            }}
        }});
        </script>
        """, unsafe_allow_html=True)

        st.markdown(f"<div class='error-text'>⚠ {error}</div>", unsafe_allow_html=True)

    return value

# =========================================================
# HEADER
# =========================================================
st.markdown("## Análisis de Retención")

# =========================================================
# INPUTS
# =========================================================
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
        st.session_state.errors["tenure"] = "Campo obligatorio"
    elif not tenure.isdigit():
        st.session_state.errors["tenure"] = "Solo números enteros (ej: 12)"

    try:
        float(MonthlyCharges)
    except:
        st.session_state.errors["MonthlyCharges"] = "Debe ser número (ej: 75.5)"

    try:
        float(TotalCharges)
    except:
        st.session_state.errors["TotalCharges"] = "Debe ser número (ej: 900.25)"

    if "Selecciona..." in [
        gender, Partner, Dependents, PhoneService,
        InternetService, Contract, PaperlessBilling, PaymentMethod
    ]:
        st.warning("Selecciona todas las opciones")

    # BLOQUEAR SI HAY ERRORES
    if st.session_state.errors:
        st.session_state.loading = False
        st.rerun()

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
        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            prob = response.json()["probability"]
            st.success(f"Resultado: {prob*100:.1f}%")
        else:
            st.error("Error en API")

    st.session_state.loading = False
