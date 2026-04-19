import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


def limpiar_datos(df,logger, reglas):
    """
    Limpia un DataFrame basado en reglas explícitas y registra información detallada.
    Devuelve:
    - df limpio
    - resumen de limpieza con conversiones, nulos, columnas eliminadas y duplicados
    """
    df = df.copy()
    resumen = {}

    # --- Conversión numérica ---
    for col in reglas.get("convertir_numericas", []):
        if col in df.columns:
            antes = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors="coerce")
            despues = df[col].dtype
            resumen[f"convertido_{col}"] = f"{antes} → {despues}"
            nulos_despues = df[col].isnull().sum()
            resumen[f"nulos_despues_{col}"] = int(nulos_despues)
            if logger:
                logger.info("🔄 %s convertido de %s a %s", col, antes, despues)
                logger.info("💧 Nulos en %s después de convertir: %d", col, nulos_despues)

    # --- Eliminación de filas con nulos ---
    for col in reglas.get("eliminar_nulos", []):
        if col in df.columns:
            n_antes = df[col].isna().sum()
            df.dropna(subset=[col], inplace=True)
            resumen[f"filas_eliminadas_por_{col}"] = int(n_antes)
            if logger:
                logger.info("✅ Filas eliminadas por %s. Nulos restantes: %d", col, df[col].isnull().sum())

    # --- Eliminación de columnas ---
    for col in reglas.get("eliminar_columnas", []):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            resumen[f"columna_eliminada"] = col
            if logger:
                logger.info("🗑️ Columna eliminada: %s", col)

    # --- Eliminación de duplicados ---
    n_duplicados = df.duplicated().sum()
    if n_duplicados > 0:
        df.drop_duplicates(inplace=True)
        resumen["duplicados_eliminados"] = int(n_duplicados)
        if logger:
            logger.info("🗑️ Filas duplicadas eliminadas: %d", n_duplicados)

    return df, resumen


def dividir_datos(X, y,logger):
    """
    Divide el dataset en entrenamiento y prueba, y codifica las etiquetas.
    Operaciones realizadas:
    - Train/Test split (80%/20%) estratificado.
    - Codificación de las etiquetas usando LabelEncoder.
     Parámetros
    ----------
    X : pandas.DataFrame
        DataFrame de características.
    y : pandas.Series
        Variable objetivo.
    Retorna
    -------
    X_train : pandas.DataFrame
        Conjunto de entrenamiento de características.
    X_test : pandas.DataFrame
        Conjunto de prueba de características.
    y_train_enc : numpy.ndarray
        Etiquetas de entrenamiento codificadas.
    y_test_enc : numpy.ndarray
        Etiquetas de prueba codificadas.
    label_enc : LabelEncoder
        Objeto LabelEncoder ajustado, útil para decodificar predicciones.
    """

    logger.info("Dividiendo datos en train y test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    label_enc = LabelEncoder()
    y_train_enc = label_enc.fit_transform(y_train)
    y_test_enc = label_enc.transform(y_test)

    logger.info("Datos divididos: Train=%d, Test=%d", len(y_train_enc), len(y_test_enc))
    return X_train, X_test, y_train_enc, y_test_enc, label_enc


def construir_preprocesador(X):
    """
    Construye un preprocesador para datos mixtos (numéricos y categóricos).
    El preprocesador aplica:
    - StandardScaler a columnas numéricas
    - OneHotEncoder a columnas categóricas
    Parámetros
    ----------
    X : pandas.DataFrame
        Dataset original utilizado para identificar los tipos de columnas.
    Retorna
    -------
    sklearn.compose.ColumnTransformer
        Objeto preprocesador listo para usar dentro de un pipeline.
    """

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return preprocessor