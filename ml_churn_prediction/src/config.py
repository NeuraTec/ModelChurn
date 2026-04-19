from pathlib import Path

import numpy as np

from sklearn.metrics import make_scorer, fbeta_score

# ============================================================
# Proyecto
# ============================================================
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Fallback para notebooks / Colab
    PROJECT_ROOT = Path.cwd()

# ============================================================
# Reproducibilidad
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# Métricas
# ============================================================
f2_scorer = make_scorer(fbeta_score, beta=2)