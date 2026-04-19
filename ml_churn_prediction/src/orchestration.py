from sklearn import set_config
set_config(transform_output="pandas")


import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.config import PROJECT_ROOT
from src.logging_config import configurar_logging
from src.io_utils import cargar_config, cargar_datos
from src.eda import analizar_informacion_general, generar_reporte_eda
from src.preprocessing import limpiar_datos, dividir_datos, construir_preprocesador
from src.training import entrenar_modelos, seleccionar_modelo, guardar_modelo_final
from src.evaluation import (
    ajustar_threshold,
    evaluar_en_test,
    analizar_probabilidades
)
from src.explainability import explicar_modelo_final_shap
from src.pipelines import mostrar_features_seleccionados




def crear_carpeta_resultados(nombre_proyecto):
    """
    Crea la carpeta principal donde se almacenan los resultados del proyecto.

    Devuelve la ruta de la carpeta creada.
    """ 
    carpeta = Path(nombre_proyecto)
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


# ================== MAIN () ==================

def main(config_path="configs/config.yaml"):                                                                                                                                                                 
    """
    Ejecuta el pipeline completo de machine learning de principio a fin.

    El proceso incluye la carga de datos, análisis exploratorio, preprocesamiento,
    entrenamiento de modelos, evaluación, explicabilidad y guardado de artefactos,
    utilizando un archivo de configuración YAML.
    """
    # ================== CONFIG & LOGGING ==================
    config_path = PROJECT_ROOT / config_path
    config = cargar_config(config_path)

    RANDOM_STATE = config["random_state"]
    np.random.seed(RANDOM_STATE)

    logger = configurar_logging(PROJECT_ROOT)
    logger.info("🔧 Iniciando proceso con configuración YAML")

    # ================== PATHS ==================
    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir = artifacts_dir / "reports"

    dataset_path = PROJECT_ROOT / config["paths"]["dataset"]
    fs_threshold = config["feature_selection"]["threshold"]

    #  Cargar datos (RAW)
    df_raw = cargar_datos(dataset_path,logger)

    #  EDA RAW (diagnóstico)
    analizar_informacion_general(
        df=df_raw,
        logger=logger,
        carpeta=reports_dir / "eda/raw",
        nombre_dataset="dataset_raw"
    )

    # Limpieza
    df_limpio, resumen_limpieza = limpiar_datos(
        df_raw,
        reglas=config["limpieza"],
        logger=logger
    )
    # Guardar dataset limpio en data/processed
    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]
    df_limpio.to_csv(processed_path, index=False)
    logger.info(f"Dataset limpio guardado en {processed_path}")
    #  EDA CLEAN (dataset final)
    info_clean = analizar_informacion_general(
        df=df_limpio,
        logger=logger,
        carpeta=reports_dir / "eda/clean",
        nombre_dataset="dataset_limpio",
        resumen_limpieza=resumen_limpieza
   )

    #  EDA visual SOLO sobre clean
    generar_reporte_eda(
        df=df_limpio,
        logger=logger,
        num_cols=info_clean["num_cols"],
        cat_cols=info_clean["cat_cols"],
        carpeta=reports_dir / "eda/clean"
   )

    #  Split
    X = df_limpio.drop("Churn", axis=1)
    y = df_limpio["Churn"]
    X_train, X_test, y_train_enc, y_test_enc, label_enc = dividir_datos(X, y,logger=logger,)

    #  Preprocesamiento
    preprocessor = construir_preprocesador(X)
    cv = StratifiedKFold(
         n_splits=config["split"]["cv_folds"], shuffle=True, random_state=RANDOM_STATE )

    #  Entrenamiento
    resultados = entrenar_modelos(
        X_train,
        logger,
        y_train_enc,
        preprocessor,
        cv,
        fs_threshold=fs_threshold
    )
    # ✅ Selección del mejor modelo
    mejor_tag, best_model = seleccionar_modelo(resultados)

    # 📌 Guardar features seleccionadas (pipeline ya entrenado)
    mostrar_features_seleccionados(
        pipeline=best_model,
        X=X_train,
        modelo_nombre=f"{mejor_tag}_{fs_threshold}",
        logger=logger,
        config=config
   )

    # Explicabilidad (SHAP)
    explicar_modelo_final_shap(
        pipeline=best_model,
        logger=logger,
        X=X_train,
        config=config,
        modelo_nombre=f"{mejor_tag}_{fs_threshold}"
   )
    #  Ajuste de threshold
    threshold_df = ajustar_threshold(
        best_model=best_model,
        logger=logger,
        X_train=X_train,
        y_train=y_train_enc,
        config=config,
        beta=config["threshold"]["beta"]
   )

    threshold_final = threshold_df.loc[
        threshold_df[f"F{int(config['threshold']['beta'])}-score"].idxmax(),
        "Threshold"
    ]

    # Evaluación final
    evaluar_en_test(
        modelo=best_model,
        logger=logger,
        X_test=X_test,
        y_test=y_test_enc,
        label_enc=label_enc,
        config=config,
        threshold=threshold_final,
        beta=config["threshold"]["beta"],
        modelo_nombre=f"{mejor_tag}_{fs_threshold}"
    )

      # Análisis de probabilidades (explicabilidad negocio)
    analizar_probabilidades(
        modelo=best_model,
        logger=logger,
        X_test=X_test,
        y_test_enc=y_test_enc,
        config=config,
        threshold=threshold_final,
        modelo_nombre=f"{mejor_tag}_{fs_threshold}"
    )
    # . Guardar modelo
    guardar_modelo_final(
        best_model=best_model,
        logger=logger,
        config=config,
        fs_threshold=fs_threshold,
        mejor_tag=mejor_tag
    )

    logger.info("🎉 Proceso completado correctamente")

if __name__ == "__main__":
    main()