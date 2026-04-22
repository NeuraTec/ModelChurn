import streamlit as st
import requests
import time

API_URL = "https://modelchurn.onrender.com/predict"

# =========================================================
# 1. CONFIGURACIÓN Y ESTILOS CSS MEJORADOS
# =========================================================
st.set_page_config(page_title="NeuraTec | Churn Analytics", layout="centered")

st.markdown("""
    <style>
    /* Importar una fuente más profesional */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Título elegante con degradado */
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #1E2A38, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 60px !important;
        margin-top: -30px !important;
        margin-bottom: 5px !important;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 18px;
        margin-bottom: 30px;
    }
    
    /* Botón de acción con efecto hover */
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
        border: none;
        color: white;
        transform: scale(1.02);
    }

    /* Cuadros de resultado más limpios */
    .result-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Encabezado visual
st.markdown('<h1 class="main-title">Análisis de Retención</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plataforma de Inteligencia Predictiva para la Toma de Decisiones</p>', unsafe_allow_html=True)

# =========================
# FUNCIONES DE APOYO
# =========================
def validar_texto(valor, opciones):
    return valor.strip().capitalize() in opciones

def mostrar_validacion(valor, valido):
    if valor == "": return
    if valido:
        st.markdown("<p style='color:#22c55e; font-size:12px; margin-top:-15px;'>✔ Campo correcto</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#ef4444; font-size:12px; margin-top:-15px;'>✖ Revisar formato</p>", unsafe_allow_html=True)

errores = {}

# =========================
# INPUTS ORGANIZADOS
# =========================
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.text_input("Antigüedad del cliente (Meses)")
        valido_tenure = tenure.isdigit() if tenure else False
        mostrar_validacion(tenure, valido_tenure)

        MonthlyCharges = st.text_input("Cargo Mensual ($)")
        try:
            float(MonthlyCharges)
            valido_mc = True
        except:
            valido_mc = False
        mostrar_validacion(MonthlyCharges, valido_mc)

        TotalCharges = st.text_input("Cargos Totales Acumulados ($)")
        try:
            float(TotalCharges)
            valido_tc = True
        except:
            valido_tc = False
        mostrar_validacion(TotalCharges, valido_tc)

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

# =========================
# BOTÓN Y PROCESAMIENTO
# =========================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Analizar Perfil del Cliente"):

    # Pequeña validación rápida
    if not tenure or not MonthlyCharges or not TotalCharges:
        st.warning("Por favor, complete la información financiera y de antigüedad para generar el análisis.")
    else:
        # Preparamos los datos
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

        # --- TEXTO DE CARGA MÁS CLARO Y PROFESIONAL ---
        with st.spinner('Procesando datos con el motor de IA...'):
            try:
                # Simulamos una pequeña pausa para que el cliente sienta que la IA "piensa"
                time.sleep(1) 
                response = requests.post(API_URL, json=data, timeout=30)
                
               

                if response.status_code == 200:
                    result = response.json()
                    prob = result["probability"]
                    pred = result["prediction"]

                    st.toast("Análisis finalizado con éxito", icon="🎯")

                    # RESULTADO VISUAL (Mantenemos la lógica de color que pediste)
                    riesgo = prob  # asumimos que 'prob' = probabilidad de abandono (churn)

                    if riesgo > 0.5:
                        st.markdown(f"""
                            <div class="result-box" style="background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24;">
                                <h2 style="margin:0;">ALTA PROBABILIDAD DE ABANDONO</h2>
                                <p style="font-size: 18px; opacity: 0.8;">Se recomienda activar protocolo de retención inmediato.</p>
                                <hr style="border-color: #f5c6cb;">
                                <p style="font-size: 32px; font-weight: 800; margin:0;">{riesgo*100:.1f}%</p>
                                <p style="font-size: 14px;">Índice de Riesgo</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="result-box" style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724;">
                                <h2 style="margin:0;">BAJO RIESGO DE ABANDONO</h2>
                                <p style="font-size: 18px; opacity: 0.8;">El cliente muestra indicadores sólidos de lealtad.</p>
                                <hr style="border-color: #c3e6cb;">
                                <p style="font-size: 32px; font-weight: 800; margin:0;">{riesgo*100:.1f}%</p>
                                <p style="font-size: 14px;">Índice de Riesgo</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                else:
                    st.error(f"Error API: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Error de conexión: {str(e)}")

# Pie de página profesional
st.markdown("""
    <br><br><hr>
    <div style='text-align: center; color: #94a3b8; font-size: 12px;'>
        © 2026 NeuraTec | Soluciones de Inteligencia Predictiva en Colombia<br>
        Generado mediante modelos de Machine Learning avanzados.
    </div>
""", unsafe_allow_html=True)
