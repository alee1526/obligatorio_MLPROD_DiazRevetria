#!/usr/bin/env bash
# EC2 user data (Amazon Linux 2023). Pegar este contenido en
# "Advanced details -> User data" al LANZAR la instancia.
#
# AWS lo ejecuta automaticamente UNA sola vez, como root, en el primer arranque.
# Deja Docker + Compose v2 instalados y el servicio corriendo, para que al
# conectarte por SSH solo tengas que subir el codigo y hacer 'docker compose up'.
#
# Nota: corre como root (sin sudo). No se re-ejecuta en los Stop/Start
# posteriores; Docker ya queda instalado en el disco EBS.
set -euxo pipefail

# --- Docker Engine ---
dnf update -y
dnf install -y docker
systemctl enable --now docker

# --- Plugin Compose v2 ---
ARCH="$(uname -m)"
mkdir -p /usr/libexec/docker/cli-plugins
curl -SL \
  "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${ARCH}" \
  -o /usr/libexec/docker/cli-plugins/docker-compose
chmod +x /usr/libexec/docker/cli-plugins/docker-compose

# --- Permitir usar docker sin sudo al usuario por defecto de la AMI ---
usermod -aG docker ec2-user || true

# Log de verificacion (queda en /var/log/cloud-init-output.log dentro de la instancia)
docker --version
docker compose version
