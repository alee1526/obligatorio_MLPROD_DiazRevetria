#!/usr/bin/env bash
# Instala Docker Engine + plugin Compose v2 en una instancia EC2 con
# Amazon Linux 2023. Ejecutar una sola vez tras conectarse por SSH:
#
#   chmod +x deploy/aws/install-docker.sh
#   ./deploy/aws/install-docker.sh
#
# (Tambien sirve como "user data" al lanzar la instancia; en ese caso corre
#  como root y el paso newgrp no aplica.)
set -euo pipefail

echo ">> Actualizando paquetes e instalando Docker..."
sudo dnf update -y
sudo dnf install -y docker

echo ">> Habilitando y arrancando el servicio Docker..."
sudo systemctl enable --now docker

echo ">> Instalando el plugin Compose v2..."
ARCH="$(uname -m)"
sudo mkdir -p /usr/libexec/docker/cli-plugins
sudo curl -SL \
  "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${ARCH}" \
  -o /usr/libexec/docker/cli-plugins/docker-compose
sudo chmod +x /usr/libexec/docker/cli-plugins/docker-compose

echo ">> Permitiendo usar docker sin sudo para el usuario actual..."
sudo usermod -aG docker "${USER:-ec2-user}" || true

echo ""
docker --version
sudo docker compose version
echo ""
echo ">> Listo. Cerra sesion SSH y volve a conectarte (o corre 'newgrp docker')"
echo "   para usar 'docker' sin sudo en esta sesion."
