#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/students/sushaoqi/CR/main}"
DEST_DIR="${ROOT}/pretrained/sam"
CKPT="${DEST_DIR}/sam_vit_b_01ec64.pth"
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

mkdir -p "${DEST_DIR}"

if [[ -s "${CKPT}" ]]; then
  echo "SAM ViT-B checkpoint already exists: ${CKPT}"
  ls -lh "${CKPT}"
  exit 0
fi

TMP="${CKPT}.tmp"
rm -f "${TMP}"

if command -v wget >/dev/null 2>&1; then
  wget -O "${TMP}" "${URL}"
elif command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "${TMP}" "${URL}"
else
  echo "Neither wget nor curl is available." >&2
  exit 1
fi

mv "${TMP}" "${CKPT}"
echo "Downloaded SAM ViT-B checkpoint:"
ls -lh "${CKPT}"
