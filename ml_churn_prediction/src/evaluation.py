import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 👈 MODO SERVIDOR / TERMINAL
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score
)
from sklearn.model_selection import cross_val_predict

from src.config import PROJECT_ROOT
from src.io_utils import save_fig


def ajustar_threshold(best_model,logger,X_train,y_train,config,beta=2.0
):
    """
    Calcula el umbral de decisión óptimo usando validación cruzada y F-beta,
    y guarda los resultados en artifacts/metrics definidos en config.yaml
    """
    logger.info("\n📊 Ajustando threshold de decisión...")

    metrics_dir = PROJECT_ROOT / config["paths"]["metrics"]
    os.makedirs(metrics_dir, exist_ok=True)

    reports_dir = PROJECT_ROOT / config["paths"]["reports"]
    os.makedirs(reports_dir , exist_ok=True)
    cv_folds = config["split"]["cv_folds"]

    y_probs = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=cv_folds,
        method="predict_proba"
    )[:, 1]

    inicio = config["threshold"]["rango"]["inicio"]
    fin = config["threshold"]["rango"]["fin"]
    paso = config["threshold"]["rango"]["paso"]

    thresholds = np.arange(inicio, fin + paso, paso)
    resultados = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        resultados.append({
            "Threshold": round(t, 2),
            "Precision": round(precision_score(y_train, y_pred), 3),
            "Recall": round(recall_score(y_train, y_pred), 3),
            "F1-score": round(f1_score(y_train, y_pred), 3),
            f"F{int(beta)}-score": round(
                fbeta_score(y_train, y_pred, beta=beta), 3
            )
        })

    df = pd.DataFrame(resultados)

    # 📄 Guardar tabla
    df.to_csv(metrics_dir / "tabla_thresholds.csv", index=False)

    # 📈 Gráfico
    plt.figure(figsize=(10, 6))
    for col in ["Precision", "Recall", "F1-score", f"F{int(beta)}-score"]:
        plt.plot(df["Threshold"], df[col], label=col)

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    save_fig(plt.gcf(), reports_dir / "threshold_tuning_F2.png")
    plt.close()

    logger.info(
    "✅ Threshold tuning guardado | CSV: %s | Gráfico: %s",
    metrics_dir,
    reports_dir / "threshold_tuning_F2.png"
    )

    return df


def evaluar_en_test(modelo,logger,X_test,y_test,label_enc,config,threshold,beta=2.0,modelo_nombre="modelo"):
    """
    Evalúa el modelo final en el conjunto de prueba y:
    - Calcula métricas finales
    - Guarda matriz de confusión (PNG)
    - Guarda métricas finales (JSON)
    """
    # ================== Directorio de métricas ==================
    metrics_dir = PROJECT_ROOT / config["paths"]["metrics"]
    os.makedirs(metrics_dir, exist_ok=True)

    # ================== Predicciones ==================
    y_probs = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    # ================== Métricas ==================
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=beta)
    auc = roc_auc_score(y_test, y_probs)

    # ================== Matriz de confusión ==================
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_enc.classes_
    )

    title = (
        f"Matriz de Confusión (Test) | Th={threshold:.2f}\n"
        f"P:{precision:.3f}  R:{recall:.3f}  "
        f"F1:{f1:.3f}  F{int(beta)}:{f2:.3f}  AUC:{auc:.3f}"
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()

    output_cm = metrics_dir / f"matriz_confusion_{modelo_nombre}.png"
    save_fig(fig, output_cm)
    plt.close()

    logger.info("🧩 Matriz de confusión guardada en %s", output_cm)

    # ================== Guardar métricas finales (JSON) ==================
    metrics = {
        "model": modelo_nombre,
        "dataset": "test",
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        f"f{int(beta)}": round(f2, 3),
        "auc": round(auc, 3),
        "threshold": float(threshold),
        "evaluated_at": datetime.now().isoformat()
    }

    metrics_path = metrics_dir / f"metrics_final_{modelo_nombre}.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    logger.info("📊 Métricas finales guardadas en %s", metrics_path)

    # ------- classification report --------------------------------------
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=label_enc.classes_,
        zero_division=0
    )

    report_path = metrics_dir / f"classification_report_{modelo_nombre}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("📋 Classification report guardado en %s", report_path)


def analizar_probabilidades(
    modelo, logger, X_test, y_test_enc, config,
    threshold=0.4, modelo_nombre="modelo"
):
    """
    Analiza la distribución de las probabilidades de churn predichas.

    La función genera un histograma que compara las probabilidades predichas
    para las clases positiva y negativa, destacando el umbral de decisión elegido.
    """

    # ================== Directorio explainability ==================
    explain_dir = PROJECT_ROOT / config["paths"]["explainability"]
    os.makedirs(explain_dir, exist_ok=True)

    # ================== Predicciones ==================
    y_probs = modelo.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)

    df_probs = pd.DataFrame({
        "Probabilidad_Churn": y_probs,
        "Prediccion": y_preds,
        "Real": y_test_enc
    })

    # ================== Gráfico ==================
    plt.figure(figsize=(10, 6))

    plt.hist(
        df_probs[df_probs["Real"] == 1]["Probabilidad_Churn"],
        bins=25,
        alpha=0.6,
        label="Churn = Yes"
    )

    plt.hist(
        df_probs[df_probs["Real"] == 0]["Probabilidad_Churn"],
        bins=25,
        alpha=0.6,
        label="Churn = No"
    )

    plt.axvline(
        x=threshold,
        linestyle="--",
        label=f"Threshold = {threshold:.2f}"
    )

    plt.xlabel("Probabilidad predicha de Churn")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de probabilidades de Churn (Test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ================== Guardar ==================
    output_path = explain_dir / f"distribucion_probabilidades_{modelo_nombre}.png"
    save_fig(plt.gcf(), output_path)
    plt.close()

    logger.info("📈 Distribución de probabilidades guardada en %s", output_path)
