import logging
from logging.handlers import RotatingFileHandler

def configurar_logging(project_root):
    """
    Configura el sistema de logging del proyecto.
    Guarda logs en archivo y consola de forma rotativa.
    """
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        force=True  # 🔥 CLAVE para VS Code
    )

    logger = logging.getLogger()

    file_handler = RotatingFileHandler(
        logs_dir / "entrenamiento.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8"
    )

    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)

    return logger
