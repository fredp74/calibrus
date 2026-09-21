#!/usr/bin/env bash
# Calibrus — quickstart complet: dataset -> train -> calibrate -> infer de test
set -euo pipefail

CONFIG="${1:-config.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "[quickstart] $CONFIG introuvable. Copie config.yaml.example -> config.yaml et adapte-le."
    exit 1
fi

mkdir -p data checkpoints

echo "[quickstart] 1/4 — construction du dataset..."
python3 -m calibrus.data.dataset --config "$CONFIG" --out data/dataset.pt

echo "[quickstart] 2/4 — entraînement..."
python3 -m calibrus.train --config "$CONFIG" --dataset data/dataset.pt --out checkpoints/calibrus_model.pt

echo "[quickstart] 3/4 — calibration..."
python3 -m calibrus.calibrate --dataset data/dataset.pt --checkpoint checkpoints/calibrus_model.pt \
    --out checkpoints/calibrus_model_calibrated.pt

echo "[quickstart] 4/4 — test d'inférence rapide..."
python3 -m calibrus.infer --checkpoint checkpoints/calibrus_model_calibrated.pt \
    --features '{}' --text "test signal"

echo "[quickstart] terminé. Modèle calibré -> checkpoints/calibrus_model_calibrated.pt"
