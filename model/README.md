# Modelo Entrenado – Churn Prediction

Este directorio contiene el modelo final ya entrenado para predecir el **abandono de clientes (churn)**.

## 🎯 ¿Para qué sirve este modelo?
- Predecir probabilidad de abandono.
- Priorizar clientes en riesgo.
- Integrar en una API o pipeline de producción.

## 👍 Modelo incluido
- `mejor_modelo_median.pkl`:  
  Modelo final optimizado con preprocesamiento, SMOTE y ajuste de hiperparámetros.

## ❌ Modelos descartados
Los modelos alternativos (mean, sin feature selection, etc.) no se incluyen porque no superaron el rendimiento del modelo final.

## 📌 Cómo usar el modelo
```python
import joblib
model = joblib.load("model/mejor_modelo_median.pkl")
pred = model.predict_proba(X)[:, 1]
