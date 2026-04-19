import os
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")  # 👈 MODO SERVIDOR / TERMINAL
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT
from src.io_utils import save_fig


def explicar_modelo_final_shap(pipeline,logger,X,config,modelo_nombre="modelo"): 
    """
    Genera explicaciones del modelo final utilizando SHAP.

    La función selecciona automáticamente el explicador SHAP adecuado según
    el tipo de modelo y guarda un gráfico resumen para su interpretación.
    """

    logger.info("🔍 Generando explicaciones SHAP...")

    steps = pipeline.named_steps

    # === Transformaciones ===
    X_t = steps["preprocessor"].transform(X)

    if "feature_selection" in steps:
        X_t = steps["feature_selection"].transform(X_t)

    # === Feature names ===
    try:
        feature_names = steps["preprocessor"].get_feature_names_out()
        if "feature_selection" in steps:
            feature_names = feature_names[
                steps["feature_selection"].get_support()
            ]
    except Exception:
        feature_names = np.arange(X_t.shape[1]).astype(str)
    #Modelo

    model_final = steps["model"]
    model_type = type(model_final).__name__.lower()

    # === Selección automática de SHAP ===
    try:
        if any(t in model_type for t in [
            "lgbm", "xgboost", "catboost",
            "randomforest", "gradientboosting"
        ]):
            explainer = shap.TreeExplainer(model_final)
            shap_values = explainer.shap_values(X_t)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        elif any(t in model_type for t in ["logistic", "linear"]):
            background = X_t[:100]
            explainer = shap.LinearExplainer(model_final, background)
            shap_values = explainer.shap_values(X_t)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        else:
            background = shap.sample(X_t, 200)
            explainer = shap.KernelExplainer(
                model_final.predict_proba, background
            )
            shap_values = explainer(X_t).values[:, 1]

    except Exception:
        background = shap.sample(X_t, 200)
        explainer = shap.KernelExplainer(
            model_final.predict_proba, background
        )
        shap_values = explainer(X_t).values[:, 1]

    # === Ruta  ===
    explain_dir = PROJECT_ROOT / config["paths"]["explainability"]
    os.makedirs(explain_dir, exist_ok=True)

    output_path = f"{explain_dir}/shap_summary_{modelo_nombre}.png"
    rng = np.random.default_rng(config.get("random_state", 42))
    # === Gráfico ===
    plt.figure()
    shap.summary_plot(
        shap_values,
        features=X_t,
        feature_names=feature_names,
        show=False,
        rng=rng
    )
    plt.tight_layout()

    save_fig(plt.gcf(), output_path)
    plt.close()

    logger.info("✅ SHAP guardado en %s", output_path)

    return output_path