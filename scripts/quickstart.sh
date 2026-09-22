#!/usr/bin/env bash
# Calibrus — full quickstart: dataset -> train -> calibrate -> test infer
set -euo pipefail

CONFIG="${1:-config.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "[quickstart] $CONFIG not found. Copy config.yaml.example -> config.yaml and adapt it."
    exit 1
fi

mkdir -p data checkpoints

echo "[quickstart] 1/4 — building the dataset..."
python3 -m calibrus.data.dataset --config "$CONFIG" --out data/dataset.pt

echo "[quickstart] 2/4 — training..."
python3 -m calibrus.train --config "$CONFIG" --dataset data/dataset.pt --out checkpoints/calibrus_model.pt

echo "[quickstart] 3/4 — calibration..."
python3 -m calibrus.calibrate --dataset data/dataset.pt --checkpoint checkpoints/calibrus_model.pt \
    --out checkpoints/calibrus_model_calibrated.pt

echo "[quickstart] 4/4 — quick inference test..."
python3 -m calibrus.infer --checkpoint checkpoints/calibrus_model_calibrated.pt \
    --features '{}' --text "test signal"

echo "[quickstart] done. Calibrated model -> checkpoints/calibrus_model_calibrated.pt"
