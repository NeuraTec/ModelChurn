import requests

url = "https://modelchurn.onrender.com/predict"  # <- Esta es la URL de tu API

sample_data = {
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 820.5,
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}

response = requests.post(url, json=sample_data)

if response.status_code == 200:
    print("✅ Respuesta del modelo:")
    print(response.json())
else:
    print("❌ Error:", response.status_code)
    print(response.text)
