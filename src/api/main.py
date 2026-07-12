import io
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.features.tabular_utils import FEATURES
from src.models.predict import load_bundle, predict

app = FastAPI(
    title="Skin Lesion Classifier API",
    description="API multimodal (imagen + datos clínicos) para clasificar lesiones de piel — PAD-UFES-20.",
    version="0.1.0",
)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        try:
            _bundle = load_bundle()
        except FileNotFoundError:
            raise HTTPException(503, "Modelo no disponible. Entrená y colocá models/model.pt.")
    return _bundle


def clasificar(image_bytes, clinical):
    model, transformer, classes = get_bundle()
    clinical = {f: clinical.get(f) for f in FEATURES}
    probs = predict(io.BytesIO(image_bytes), clinical, model, transformer, classes)
    return {"prediccion": max(probs, key=probs.get), "probabilidades": probs}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Skin Lesion Classifier API. Ver /docs para la documentación."}


@app.get("/features")
def features():
    return {"features": FEATURES}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), data: str = Form("{}")):
    try:
        clinical = json.loads(data)
    except json.JSONDecodeError:
        raise HTTPException(422, "El campo 'data' debe ser JSON válido.")
    return clasificar(await file.read(), clinical)


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...), data: str = Form("[]")):
    try:
        records = json.loads(data)
    except json.JSONDecodeError:
        raise HTTPException(422, "El campo 'data' debe ser un JSON con una lista de registros.")
    resultados = []
    for i, file in enumerate(files):
        clinical = records[i] if i < len(records) else {}
        r = clasificar(await file.read(), clinical)
        resultados.append({"filename": file.filename, **r})
    return {"resultados": resultados}
