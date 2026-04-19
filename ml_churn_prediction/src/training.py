import os
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from src.pipelines import construir_pipeline
from src.config import f2_scorer, PROJECT_ROOT


def entrenar_modelos(X_train,logger, y_train, preprocessor, cv,fs_threshold='median',
                    scoring=f2_scorer):
    """
    Entrena múltiples modelos de clasificación mediante pipelines con o sin
    selección de características (Feature Selection) y búsqueda de hiperparámetros.

    Cada modelo se entrena usando GridSearchCV y se almacena junto con su mejor
    score y el mejor estimador encontrado.

    Parameters
    ----------
    X_train : array-like or DataFrame
        Matriz de características de entrenamiento.
    y_train : array-like or Series
        Etiquetas de entrenamiento.
    preprocessor : sklearn Transformer
        Preprocesador (ej. ColumnTransformer) a aplicar antes del modelo.
    cv : cross-validation generator
        Estrategia de validación cruzada (ej. StratifiedKFold).
    fs_threshold : {'median', 'mean'} or float, default='median'
        Umbral para la selección de características cuando `use_fs=True`.
    scoring : str or callable, default='f2'
        Métrica utilizada para la optimización de hiperparámetros.

    Returns
    -------
    dict
        Diccionario con los modelos entrenados.
        Cada clave corresponde al nombre del pipeline y el valor contiene:
            - grid : GridSearchCV entrenado
            - score : mejor score obtenido
            - model : mejor estimador (pipeline)
    """

    models = {
        'logistic_l1': {
            'model': LogisticRegression(
                solver='liblinear',
                penalty='l1'
            ),
            'param_grid': {'model__C': [0.01, 0.1, 1, 10]},
            'use_fs': False
        },

        'logistic_l2_no_fs': {
            'model': LogisticRegression(
                solver='liblinear',
                penalty='l2'
            ),
            'param_grid': {'model__C': [0.001, 0.01, 0.1, 1, 10]},
            'use_fs': False
        },

        'logistic_l2_fs': {
            'model': LogisticRegression(
                solver='liblinear',
                penalty='l2'
            ),
            'param_grid': {'model__C': [0.001, 0.01, 0.1, 1, 10]},
            'use_fs': True
        },

        'rf_fs': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [None, 10],
                'model__min_samples_split': [2, 5]
            },
            'use_fs': True
        },

        'lgbm_fs': {
            'model': LGBMClassifier(
                random_state=42,
                verbosity=-1
            ),
            'param_grid': {
                'model__n_estimators': [100],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.01, 0.1],
                'model__num_leaves': [15, 31]
            },
            'use_fs': True
        }
    }

    resultados = {}

    logger.info("🔄 Entrenando modelos (sin cross_val_predict)...")

    for name, info in models.items():

        logger.info(f"\n➡️ Entrenando pipeline: {name}")

        pipeline = construir_pipeline(
            info['model'],
            preprocessor,
            use_fs=info['use_fs'],
            fs_threshold=fs_threshold
        )

        # GridSearchCV ya ejecuta validación cruzada correctamente
        grid = GridSearchCV(
            pipeline,
            info['param_grid'],
            scoring=scoring,
            cv=cv,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        # 👉 MÉTRICA CORRECTA SIN DATA LEAKAGE
        best_score = grid.best_score_
        logger.info(f"{name} → Best CV F2-score: {best_score:.4f}")

        # Guardar resultados
        resultados[name] = {
            "grid": grid,
            "score": best_score,
            "model": grid.best_estimator_
        }

    return resultados
def seleccionar_modelo(resultados):
    """Selecciona el mejor modelo según el score guardado en 'resultados'."""
    mejor_tag = max(resultados, key=lambda t: resultados[t]['score'])
    best_model = resultados[mejor_tag]['model']
    return mejor_tag, best_model



def guardar_modelo_final(best_model,logger,config, fs_threshold, mejor_tag):
    """
    Guarda el modelo final en la carpeta definida en config.yaml
    """
    models_dir = PROJECT_ROOT / config["paths"]["models"]
    os.makedirs(models_dir, exist_ok=True)

    model_path = f"{models_dir}/mejor_modelo_{fs_threshold}.pkl"

    try:
        joblib.dump(best_model, model_path)
        logger.info("✅ Modelo guardado en %s", model_path)
    except Exception as e:
        logger.error("❌ Error guardando modelo: %s", e)
