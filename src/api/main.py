from fastapi import FastAPI

app = FastAPI(
    title="Skin Lesion Classifier API",
    description="API multimodal (imagen + datos clínicos) para clasificar lesiones de piel — PAD-UFES-20.",
    version="0.0.1",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Skin Lesion Classifier API. Ver /docs para la documentación."}
