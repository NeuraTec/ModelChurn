import pandas as pd
import yaml
from pathlib import Path

def cargar_config(ruta):
    """
    Carga parámetros del proyecto desde un archivo YAML.
    """
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_datos(ruta_csv, logger):
    """
    Carga un dataset CSV y registra su tamaño.
    """
    logger.info("📥 Cargando dataset desde %s", ruta_csv)
    df = pd.read_csv(ruta_csv)
    logger.info("✅ Dataset cargado. Shape: %s", df.shape)
    return df

def save_fig(fig, path):
    """
    Guarda una figura de matplotlib en alta resolución.
    Crea la ruta si no existe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")