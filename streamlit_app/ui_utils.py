import json
import os

import requests

from src.features.tabular_utils import CATEGORICAL, NUMERIC

API_URL = os.environ.get("API_URL", "http://localhost:8000")

CLASS_NAMES = {
    "ACK": "Queratosis actínica",
    "BCC": "Carcinoma basocelular",
    "MEL": "Melanoma",
    "NEV": "Nevo melanocítico",
    "SCC": "Carcinoma espinocelular",
    "SEK": "Queratosis seborreica",
}
MALIGNANT = {"BCC", "MEL", "SCC"}

FIELD_LABELS = {
    "age": "Edad", "gender": "Sexo", "region": "Región anatómica",
    "fitspatrick": "Fototipo (Fitzpatrick)", "diameter_1": "Diámetro mayor (mm)",
    "diameter_2": "Diámetro menor (mm)", "background_father": "Ascendencia paterna",
    "background_mother": "Ascendencia materna", "smoke": "Fuma", "drink": "Consume alcohol",
    "pesticide": "Exposición a pesticidas", "skin_cancer_history": "Antec. cáncer de piel",
    "cancer_history": "Antec. cáncer (otro)", "has_piped_water": "Agua corriente",
    "has_sewage_system": "Saneamiento", "itch": "Picazón", "grew": "Creció",
    "hurt": "Dolor", "changed": "Cambió", "bleed": "Sangró", "elevation": "Elevación",
}
GROUPS = {
    "Paciente": ["age", "gender", "fitspatrick", "background_father", "background_mother"],
    "Antecedentes": ["smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
                     "has_piped_water", "has_sewage_system"],
    "Lesión": ["region", "diameter_1", "diameter_2", "itch", "grew", "hurt", "changed",
               "bleed", "elevation"],
}

# campos a nivel paciente que se muestran en la ficha (no cambian entre lesiones)
PATIENT_FICHA = ["age", "gender", "fitspatrick", "background_father", "background_mother",
                 "smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
                 "has_piped_water", "has_sewage_system"]

# etiquetas legibles para valores del dataset
VALUE_LABELS = {"True": "Sí", "False": "No", "UNK": "No sabe",
                "MALE": "Masculino", "FEMALE": "Femenino"}


def label_value(v):
    return "—" if v is None else VALUE_LABELS.get(str(v), str(v))

def load_catalog(split="test"):
    from src.data.dataset import load_with_splits
    df = load_with_splits()
    df = df[df["img_path"].notna()].copy()
    if split:
        df = df[df["split"] == split].copy()
    df["nombre"] = df["patient_id"].astype(str)
    df["orden"] = df["patient_id"].str.extract(r"(\d+)").astype(float)
    return df.sort_values(["orden", "img_id"]).reset_index(drop=True)


def categorical_options(df):
    return {c: sorted(str(v) for v in df[c].dropna().unique()) for c in CATEGORICAL}


def is_numeric(field):
    return field in NUMERIC


def api_online():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).ok
    except requests.RequestException:
        return False


def call_predict(image_bytes, filename, clinical):
    files = {"file": (filename, image_bytes)}
    data = {"data": json.dumps(clinical)}
    r = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


def malignancy_score(probs):
    return sum(v for k, v in probs.items() if k in MALIGNANT)


def risk_level(score):
    if score >= 0.5:
        return "Alto", "#9b2226"
    if score >= 0.2:
        return "Moderado", "#a06a1b"
    return "Bajo", "#2e5d34"
