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

## Despliegue en AWS Academy

Despliegue de la app (API FastAPI + UI Streamlit) en una **instancia EC2** del
**AWS Academy Learner Lab**, corriendo con `docker compose`. Se eligió EC2 por
ser el camino de menor fricción: reaprovecha los contenedores del repo y **no
requiere crear roles IAM** (lo que descarta Elastic Beanstalk y ECS/Fargate en
el Learner Lab). El modelo es una EfficientNet-B0 multimodal (~20 MB) que corre
en **CPU**, así que no se necesita GPU.

Archivos de despliegue:

```
docker-compose.prod.yml     compose de producción (solo publica la UI :8501)
deploy/aws/install-docker.sh  instala Docker + Compose en Amazon Linux 2023
deploy/aws/user-data.sh       igual, pero para pegar como "User data" al lanzar
deploy/aws/.env.example       variables de entorno (documentación)
```

### Limitaciones del Learner Lab a tener en cuenta

- La sesión dura ~4 h y hay que apretar **Start Lab** cada vez; al cerrarse, la
  instancia **se detiene** (no se borra) y su **IP pública cambia** al reiniciar.
- **IAM restringido**: no se pueden crear roles/usuarios (solo existe `LabRole`).
- Región **us-east-1**; **sin instancias GPU**; sin Route 53 ni dominios propios
  (se accede por **IP pública**); presupuesto en créditos limitado.
- **Detené la instancia (Stop, no Terminate)** al terminar para no gastar crédito
  ni perder el trabajo.

### Requisito previo: artefactos del modelo

El `Dockerfile.api` copia `models/` a la imagen. Antes de construir, la carpeta
`models/` debe contener:

```
models/model.pt
models/tabular_transformer.joblib
```

Si no los tenés, generalos con `python -m src.features.preprocess` y
`python -m src.models.train` (ver flujo de entrenamiento). `models/` está en
`.gitignore`, así que estos archivos se suben a la instancia aparte (paso 5).

> Nota: `data/` está en `.dockerignore`, por lo que dentro del contenedor el
> selector "Paciente registrado" queda vacío; en AWS la UI funciona con **carga
> manual de imagen**.

### Pasos desde cero

1. **Iniciar el lab.** En AWS Academy entrá al curso → **Start Lab** (esperá el
   punto verde) → **AWS** para abrir la consola. Verificá que la región sea
   **N. Virginia (us-east-1)**.

2. **Lanzar la instancia EC2.** Consola → **EC2** → **Launch instance**:
   - **AMI**: Amazon Linux 2023.
   - **Tipo**: `t3.medium` (2 vCPU / 4 GB).
   - **Key pair**: creá una nueva (`.pem`) y descargala (la usás para SSH).
   - **Storage**: subí el disco raíz a **30 GB** (gp3).
   - **Security group** (crear nuevo) con dos reglas de entrada:
     - `SSH` (TCP 22) — origen **My IP**.
     - `Custom TCP` **8501** — origen **Anywhere-IPv4 (0.0.0.0/0)**.
   - *(Opcional)* **Advanced details → User data**: pegá el contenido de
     `deploy/aws/user-data.sh` para que Docker quede instalado solo en el primer
     arranque (te saltás el paso 4).
   - **Launch instance** y anotá la **IPv4 pública**.

3. **Conectarse por SSH** (desde tu máquina, en la carpeta del `.pem`):
   ```bash
   chmod 400 mi-clave.pem
   ssh -i mi-clave.pem ec2-user@<IP-PUBLICA>
   ```

4. **Instalar Docker** en la instancia *(saltear si usaste User data en el paso 2)*:
   ```bash
   # (subí primero el repo — paso 5 — o cloná; el script está en deploy/aws/)
   ./deploy/aws/install-docker.sh
   newgrp docker        # o cerrá y reabrí la sesión SSH
   ```

5. **Subir el código y el modelo a la instancia.** Desde tu máquina (no dentro
   del SSH), copiá el proyecto y los artefactos. Opción simple con `scp`:
   ```bash
   # todo el proyecto (incluye models/ si ya tiene los artefactos)
   scp -i mi-clave.pem -r "obligatorio_MLPROD_DiazRevetria" ec2-user@<IP-PUBLICA>:~/app
   ```
   Alternativa: `git clone` del repo en la instancia y luego `scp` solo los dos
   archivos de `models/` (que no están en git):
   ```bash
   scp -i mi-clave.pem models/model.pt models/tabular_transformer.joblib \
       ec2-user@<IP-PUBLICA>:~/app/models/
   ```

6. **Construir y levantar** (dentro del SSH, en la carpeta del proyecto):
   ```bash
   cd ~/app
   docker compose -f docker-compose.prod.yml up -d --build
   ```
   Verificá el estado (esperá a que `api` quede `healthy`):
   ```bash
   docker compose -f docker-compose.prod.yml ps
   ```

7. **Acceder a la UI** desde el navegador:
   ```
   http://<IP-PUBLICA>:8501
   ```
   Subí una imagen de lesión, completá la ficha clínica y presioná *Analizar*.

8. **Al terminar la sesión.** Parar los contenedores es opcional; lo importante:
   EC2 → seleccioná la instancia → **Instance state → Stop** (no *Terminate*).
   En la próxima sesión: **Start Lab**, arrancá la instancia (**Start**), tomá la
   **nueva IP pública** y volvé al paso 7.
