#!/usr/bin/env bash
set -euo pipefail
towel_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
host_ip="${1:-}"
if [[ ! "$host_ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then
  echo 'Usage: make certificate HOST_IP=192.168.1.10' >&2
  exit 2
fi
cd "$towel_root/teleop"
if [[ -e cert.pem || -e key.pem ]]; then
  echo 'Certificate/key already exists; leaving it unchanged.'
  exit 0
fi
umask 077
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem -subj '/CN=newton-towel-sim' \
  -addext "subjectAltName=IP:$host_ip,IP:127.0.0.1,DNS:localhost"
echo 'Created teleop/cert.pem and key.pem. Trust the certificate on the headset before entering WebXR.'
