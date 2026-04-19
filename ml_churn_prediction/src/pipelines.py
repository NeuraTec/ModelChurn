import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import RANDOM_STATE, PROJECT_ROOT

def construir_pipeline(model, preprocessor, use_fs=False, fs_threshold='median'):
    """
    Construye el pipeline completo con preprocesamiento, SMOTE y modelo final.
    Detalles:
    - SMOTE siempre se aplica después del preprocesador.
    - Si use_fs=True, se usa SelectFromModel con un RandomForest interno.
    Parámetros
    ----------
    model : estimator
        Modelo a entrenar.
    preprocessor : ColumnTransformer
        Preprocesador configurado.
    use_fs : bool
        Activa la selección de características.
    fs_threshold : str o float
        Umbral para SelectFromModel.
    Retorna
    -------
    ImbPipeline
        Pipeline final para entrenamiento y predicción.
    """



    steps = [
        ('preprocessor', preprocessor),

        # SMOTE siempre activo
        ('smote', SMOTE(random_state=RANDOM_STATE))
    ]

    # Selección de características SOLO si use_fs = True
    if use_fs:
        selector = SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            threshold=fs_threshold
        )
        steps.append(('feature_selection', selector))

    # Modelo final
    steps.append(('model', model))

    pipeline = ImbPipeline(steps)
    return pipeline

def mostrar_features_seleccionados(pipeline, X, modelo_nombre, logger, config):
    """
    Extrae y guarda las variables seleccionadas por el pipeline final.

    La función respeta todas las transformaciones del preprocesamiento y
    almacena los nombres finales de las variables seleccionadas en un archivo CSV
    para asegurar trazabilidad del modelo.
    """

    if 'feature_selection' not in pipeline.named_steps:
        logger.warning("⚠️ No existe etapa de selección de características en este pipeline.")
        return

    steps = pipeline.named_steps

    # ================== PREPROCESAMIENTO ==================
    X_transformed = steps['preprocessor'].transform(X)

    # ================== NOMBRES DE FEATURES ==================
    try:
        feature_names = steps['preprocessor'].get_feature_names_out()
    except Exception:
        feature_names = np.arange(X_transformed.shape[1]).astype(str)

    # ================== FEATURE SELECTION ==================
    selector = steps['feature_selection']
    mask = selector.get_support()

    if len(mask) != len(feature_names):
        logger.error("❌ ERROR: Tamaño del mask no coincide con el de las features.")
        logger.error("Features: %d, Mask: %d", len(feature_names), len(mask))
        return

    selected_features = np.array(feature_names)[mask]

    # ================== GUARDADO (CAMBIO CLAVE) ==================
    fs_dir = PROJECT_ROOT / config["paths"]["feature_selection"]
    fs_dir.mkdir(parents=True, exist_ok=True)

    archivo = fs_dir / f"features_seleccionadas_{modelo_nombre}.csv"

    df = pd.DataFrame(selected_features, columns=["feature"])
    df.to_csv(archivo, index=False)

    logger.info("✅ Features seleccionadas: %d", len(selected_features))
    logger.info("📁 Features guardadas en %s", archivo)


