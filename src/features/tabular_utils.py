import numpy as np
import pandas as pd

NUMERIC = ["age", "diameter_1", "diameter_2", "fitspatrick"]
CATEGORICAL = [
    "gender", "region", "background_father", "background_mother",
    "smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
    "has_piped_water", "has_sewage_system",
    "itch", "grew", "hurt", "changed", "bleed", "elevation",
]
FEATURES = NUMERIC + CATEGORICAL


def select_features(df):
    X = df[FEATURES].copy()
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATEGORICAL:
        X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return X
