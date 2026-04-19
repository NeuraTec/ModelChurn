import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 👈 MODO SERVIDOR / TERMINAL
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from src.io_utils import save_fig

def analizar_informacion_general(df,logger, carpeta, nombre_dataset, resumen_limpieza=None):
    """
    Analiza información estructural del dataset y genera un reporte más natural:
    - Información general
    - Nulos, duplicados, tipos de datos
    - Shape y columnas
    - Cambios aplicados por limpieza (opcional)
    """
    resumen = {}

    # 1️⃣ Información general
    info_text = f"Dataset '{nombre_dataset}' - Shape: {df.shape}\n\nColumnas:\n"
    info_text += ", ".join(df.columns) + "\n\n"

    # 2️⃣ Nulos
    nulos = df.isnull().sum()
    info_text += "💧 Nulos por columna:\n"
    for col, val in nulos.items():
        info_text += f"  {col}: {val}\n"
    resumen["nulos"] = nulos.to_dict()

    # 3️⃣ Duplicados
    duplicados = df.duplicated().sum()
    info_text += f"\n📄 Filas duplicadas: {duplicados}\n"
    resumen["duplicados"] = duplicados

    # 4️⃣ Tipos de datos
    info_text += "\n🔍 Tipos de datos:\n"
    for col, dtype in df.dtypes.items():
        info_text += f"  {col}: {dtype}\n"
    resumen["dtypes"] = df.dtypes.astype(str).to_dict()

    # 5️⃣ Variables numéricas y categóricas
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    resumen["num_cols"] = num_cols
    resumen["cat_cols"] = cat_cols

    # 6️⃣ Resumen de limpieza (opcional)
    if resumen_limpieza:
        info_text += "\n✨ Cambios realizados durante la limpieza:\n"
        for k, v in resumen_limpieza.items():
            info_text += f"  {k}: {v}\n"

    # Guardar reporte
    os.makedirs(carpeta, exist_ok=True)
    with open(f"{carpeta}/{nombre_dataset}_info.txt", "w", encoding="utf-8") as f:
        f.write(info_text)

    logger.info("📄 Reporte generado: %s/%s_info.txt", carpeta, nombre_dataset)
    return resumen

def generar_reporte_eda(df,logger, num_cols, cat_cols, carpeta,top_k_categoricas=5):
    
    """
    Genera un reporte de análisis exploratorio de datos (EDA) orientado a negocio.

    La función crea análisis visuales y estadísticos que incluyen:
    - Distribución de la variable objetivo
    - Histogramas de variables numéricas
    - Detección de outliers mediante el método IQR
    - Mapa de calor de correlaciones
    - Análisis chi-cuadrado entre variables categóricas y el objetivo

    Los resultados se guardan como imágenes y resúmenes en la carpeta indicada.
    """

    logger.info("🔍 Iniciando generación del reporte EDA...")
    os.makedirs(carpeta, exist_ok=True)

    resumen = {}

    # =========================
    # 1) Target Churn
    # =========================
    if "Churn" in df.columns:
        logger.info("🎯 Graficando distribución del target 'Churn'...")
        plt.figure(figsize=(6, 4))
        df["Churn"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
        plt.title("Distribución de Churn")
        save_fig(plt.gcf(), f"{carpeta}/target_Churn.png")
        plt.close()

        resumen["target_balance"] = df["Churn"].value_counts(normalize=True).to_dict()

    # =========================
    # 2) Histogramas numéricos
    # =========================
    logger.info("📊 Generando histogramas numéricos...")
    for col in num_cols:
        plt.figure()
        df[col].hist(bins=30, color="steelblue")
        plt.title(f"Histograma - {col}")
        save_fig(plt.gcf(), f"{carpeta}/hist_{col}.png")
        plt.close()
    # =========================
    # 3) Chi-cuadrado categóricas vs Churn (TOP K)
    # =========================
    if "Churn" in df.columns:
        logger.info("📐 [EDA | Chi-cuadrado] Seleccionando variables categóricas relevantes...")


        chi2_rows = []

        for col in cat_cols:
            if col == "Churn":
                continue

            # -------------------------
            # FILTRO DE ALTA CARDINALIDAD
            # -------------------------
            n_unique = df[col].nunique()
            ratio_unique = n_unique / len(df)
            if n_unique > 30 and ratio_unique > 0.05:
                logger.info(f"⛔ [EDA | Chi-cuadrado] {col} descartada por alta cardinalidad")

                continue

            # -------------------------
            # Tabla de contingencia
            # -------------------------
            contingency = pd.crosstab(
                df[col].fillna("MISSING"),  # tratar nulos
                df["Churn"]
            )

            # Evitar tablas inválidas
            if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
                continue

            # Test Chi-cuadrado
            chi2, p, _, _ = chi2_contingency(contingency)

            if p < 0.05:
                chi2_rows.append({
                    "feature": col,
                    "chi2": round(chi2, 3),
                    "p_value": round(p, 5)
                })

        # -------------------------
        # DataFrame con resultados
        # -------------------------
        chi2_df = pd.DataFrame(chi2_rows)

        if not chi2_df.empty:
            chi2_df = chi2_df.sort_values("chi2", ascending=False)

            # Seleccionar SOLO las TOP K
            top_features = chi2_df.head(top_k_categoricas)["feature"].tolist()

            resumen["features_categoricas_seleccionadas"] = top_features
            resumen["ranking_categoricas"] = (
                chi2_df.head(top_k_categoricas)
                .set_index("feature")
                .to_dict(orient="index")
            )


            logger.info(f"✅ [EDA | Chi-cuadrado] Features categóricas seleccionadas para el modelo: {top_features}")


            # -------------------------
            # Graficar SOLO las TOP K
            # -------------------------
            for col in top_features:
                fig, ax = plt.subplots(figsize=(8, 5))
                pd.crosstab(
                    df[col].fillna("MISSING"), df["Churn"], normalize="index"
                ).plot(kind="bar", stacked=True, ax=ax)

                ax.set_title(f"{col} vs Churn")
                ax.set_ylabel("Proporción")

                # Mover la leyenda fuera del gráfico a la derecha
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

                plt.tight_layout()
                save_fig(fig, f"{carpeta}/cat_{col}_vs_churn.png")
                logger.info(f"🖼️ [EDA | Chi-cuadrado] Gráfico guardado: cat_{col}_vs_churn.png")
                plt.close(fig)




    # =========================
    # 5) Outliers IQR
    # =========================
    logger.info("🔎 Detectando outliers por IQR...")
    outliers_resumen = {}

    for col in num_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = df[
            (df[col] < Q1 - 1.5 * IQR) |
            (df[col] > Q3 + 1.5 * IQR)
        ]
        outliers_resumen[col] = len(outliers)

    resumen["outliers_IQR"] = outliers_resumen

    # =========================
    # 6) Heatmap de correlaciones
    # =========================
    if len(num_cols) >= 2:
        logger.info("📈 Graficando mapa de correlaciones...")
        plt.figure(figsize=(10, 8))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm")
        plt.title("Mapa de Correlación")
        save_fig(plt.gcf(), f"{carpeta}/correlacion.png")
        plt.close()

    logger.info("✅ Reporte EDA generado correctamente en '%s'.", carpeta)

    return resumen
