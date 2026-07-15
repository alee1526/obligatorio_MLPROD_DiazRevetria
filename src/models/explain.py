"""Explicabilidad del modelo multimodal: Grad-CAM (imagen) y SHAP (tabular).

Uso local (requiere data/ y models/model.pt). No forma parte del contenedor de
serving; se consume desde notebooks/06_explainability.ipynb.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import load_with_splits
from src.features.tabular_utils import FEATURES, select_features
from src.models.predict import SERVE_TRANSFORM

# ---------------------------------------------------------------- utilidades


def prepare_inputs(image_path, clinical, transformer):
    """Imagen y ficha clínica -> tensores, con el mismo transform que el serving."""
    pil = Image.open(image_path).convert("RGB")
    image = SERVE_TRANSFORM(pil).unsqueeze(0)
    X = select_features(pd.DataFrame([clinical]))
    tabular = torch.tensor(transformer.transform(X), dtype=torch.float32)
    return pil, image, tabular


def row_to_clinical(row):
    return {c: row[c] for c in FEATURES}


# ------------------------------------------------------------------ Grad-CAM


def find_target_layer(model):
    """Última capa convolucional de la rama de imagen (mapa 7x7 en EfficientNet-B0)."""
    backbone = model.image.model
    for name in ("bn2", "conv_head"):
        layer = getattr(backbone, name, None)
        if layer is not None:
            return layer
    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1]
    convs = [m for m in backbone.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No se encontró una capa convolucional en la rama de imagen.")
    return convs[-1]


def gradcam(model, image, tabular, class_idx=None, layer=None):
    """Mapa de activación de la clase `class_idx` (por defecto, la predicha).

    El backward se hace sobre el logit del modelo *completo*, así que el mapa
    refleja la predicción multimodal (imagen + tabular), no la rama de imagen
    aislada.

    Devuelve (cam, class_idx, probs) con cam normalizado a [0, 1] y del tamaño
    de la imagen de entrada.
    """
    model.eval()
    layer = layer or find_target_layer(model)
    saved = {}

    def forward_hook(_module, _inputs, output):
        saved["activations"] = output
        output.register_hook(lambda grad: saved.__setitem__("gradients", grad))

    handle = layer.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            # El backbone está congelado (requires_grad=False): sin esto no se
            # construye el grafo y no habría gradientes que engancharle al hook.
            image = image.clone().requires_grad_(True)
            logits = model(image, tabular)
            probs = F.softmax(logits, dim=1).squeeze(0).detach().numpy()
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1))
            model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    activations = saved["activations"]
    gradients = saved.get("gradients")
    if gradients is None:
        raise RuntimeError("No se capturaron gradientes en la capa objetivo.")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.numpy(), int(class_idx), probs


def overlay_cam(pil_image, cam, alpha=0.45, cmap="jet"):
    """Superpone el mapa sobre la imagen original y devuelve un PIL.Image."""
    from matplotlib import colormaps

    base = pil_image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    heat = (colormaps[cmap](cam)[..., :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * np.asarray(base, dtype=np.float32) + alpha * heat
    return Image.fromarray(blended.astype(np.uint8))


# ---------------------------------------------------------------------- SHAP


def image_embedding(model, image):
    """Salida de la rama de imagen: fija para una imagen dada."""
    with torch.no_grad():
        return model.image(image)


def tabular_predict_fn(model, img_feat):
    """f(X_transformada) -> probabilidades, con la imagen congelada.

    Reusa el embedding ya calculado y solo re-corre la rama tabular y la cabeza
    de fusión. Es exactamente equivalente al forward completo, pero evita pasar
    la CNN en cada una de las miles de evaluaciones que hace KernelExplainer.
    """
    def f(X):
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            tab_feat = model.tabular(X)
            fused = torch.cat([img_feat.expand(X.shape[0], -1), tab_feat], dim=1)
            return F.softmax(model.head(fused), dim=1).numpy()

    return f


def build_background(transformer, n=100, seed=42, split="train"):
    """Muestra de referencia para SHAP, tomada del split de entrenamiento."""
    df = load_with_splits()
    df = df[df["split"] == split]
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return transformer.transform(select_features(df))


def transformed_feature_names(transformer):
    return list(transformer.get_feature_names_out())


def _base_feature(name, features=FEATURES):
    """'cat__gender_MALE' -> 'gender'; 'num__missingindicator_age' -> 'age'."""
    stem = name.split("__", 1)[-1]
    if stem.startswith("missingindicator_"):
        stem = stem[len("missingindicator_"):]
    matches = [f for f in features if stem == f or stem.startswith(f + "_")]
    return max(matches, key=len) if matches else None


def group_by_feature(values, names, features=FEATURES):
    """Suma las columnas one-hot / indicadoras de vuelta a la feature original.

    SHAP es aditivo, así que sumar las contribuciones de las columnas que salen
    de una misma variable da la contribución de esa variable.
    """
    grouped = dict.fromkeys(features, 0.0)
    for value, name in zip(np.asarray(values).ravel(), names):
        base = _base_feature(name, features)
        if base is not None:
            grouped[base] += float(value)
    return pd.Series(grouped).sort_values(key=np.abs, ascending=False)
