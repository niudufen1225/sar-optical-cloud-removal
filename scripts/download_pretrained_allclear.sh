#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/students/sushaoqi/CR/main"
EXTERNAL_DIR="${ROOT}/external"
PRETRAINED_DIR="${ROOT}/pretrained"

mkdir -p "${EXTERNAL_DIR}" "${PRETRAINED_DIR}"

if [ ! -d "${EXTERNAL_DIR}/SoftShadow/.git" ]; then
  git clone --depth 1 https://github.com/Xinrui014/SoftShadow.git "${EXTERNAL_DIR}/SoftShadow"
fi

if [ ! -d "${EXTERNAL_DIR}/TG-ECNet/.git" ]; then
  git clone --depth 1 https://github.com/LeeX54946/TG-ECNet.git "${EXTERNAL_DIR}/TG-ECNet"
fi

SAM_CKPT="${PRETRAINED_DIR}/sam_vit_h_4b8939.pth"
if [ ! -f "${SAM_CKPT}" ]; then
  wget -O "${SAM_CKPT}" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

echo "SoftShadow repo: ${EXTERNAL_DIR}/SoftShadow"
echo "TG-ECNet repo:    ${EXTERNAL_DIR}/TG-ECNet"
echo "SAM checkpoint:  ${SAM_CKPT}"
