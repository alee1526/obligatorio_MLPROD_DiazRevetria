# Clasificador de lesiones de piel (PAD-UFES-20)

> README inicial. El proyecto está en construcción; esta descripción y las
> instrucciones de uso se irán completando a medida que avancen las fases.

Sistema de Machine Learning end-to-end que clasifica lesiones de piel en 6
diagnósticos (BCC, MEL, SCC, ACK, NEV, SEK) combinando **imágenes clínicas** y
**datos tabulares** del paciente. El proyecto abarca desde la construcción del
dataset y el análisis exploratorio hasta el entrenamiento de un modelo
multimodal y su despliegue como API (predicción online y batch), con una
interfaz de usuario y contenedores Docker.

Dataset: [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) — 2.298
imágenes de smartphone de 1.373 pacientes, con metadatos clínicos asociados.

## Estructura del repositorio

```
src/
  api/           API de predicción (serving)
  data/          descarga del dataset, join imágenes↔tabular y split
  features/      preprocesamiento compartido entre entrenamiento y serving
  models/        modelos de imagen, tabular y fusión multimodal
streamlit_app/   interfaz de usuario
docker/          Dockerfile.api, Dockerfile.ui
requirements/    dependencias: base / prod / ui / dev
notebooks/       análisis exploratorio (EDA)
data/            datasets (fuera de control de versiones)
informe/         informe de la entrega
```

La lógica sigue el flujo de los datos: `data` obtiene y divide, `features`
transforma de forma idéntica en train y en producción, `models` entrena y
guarda los artefactos, y `api` los sirve.
