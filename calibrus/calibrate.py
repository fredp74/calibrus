"""
calibrus/calibrate.py — Calibration post-hoc du modèle via temperature scaling.

Principe: après entraînement, le modèle est souvent sur- ou sous-confiant.
On apprend un seul scalaire T (température) sur le split de VALIDATION qui,
appliqué aux logits avant softmax (logits / T), minimise le NLL — sans
toucher aux poids du modèle ni à son accuracy.

On mesure l'ECE (Expected Calibration Error) avant/après pour vérifier
que la calibration a réellement amélioré les choses.

Usage:
    python3 -m calibrus.calibrate --dataset data/dataset.pt \
        --checkpoint checkpoints/calibrus_model.pt \
        --out checkpoints/calibrus_model_calibrated.pt
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from calibrus.model import CalibrusModel
from calibrus.train import SplitDataset


@torch.no_grad()
def collect_logits_labels(model, loader, device):
    """Fait un forward pass complet sur un split et récupère tous les logits + labels."""
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        x_num = batch["x_num"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["y"].to(device)

        out = model(x_num, input_ids, attention_mask)
        all_logits.append(out["class_logits"].cpu())
        all_labels.append(y.cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor,
                                n_bins: int = 15) -> float:
    """
    ECE: moyenne pondérée, sur des bins de confiance, de |accuracy - confiance moyenne|.
    Plus c'est proche de 0, plus le modèle est bien calibré.
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return ece.item()


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                     max_iter: int = 50, lr: float = 0.01) -> float:
    """
    Apprend le scalaire T par LBFGS en minimisant le NLL sur (logits/T, labels).
    """
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    nll_criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / temperature.clamp(min=0.05)
        loss = nll_criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.clamp(min=0.05).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[calibrate] device: {device}")

    data = torch.load(args.dataset)
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = CalibrusModel.from_dataset_meta(data).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_ds = SplitDataset(data["splits"]["val"])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print("[calibrate] collecte des logits sur le split de validation...")
    logits, labels = collect_logits_labels(model, val_loader, device)

    probs_before = torch.softmax(logits, dim=-1)
    ece_before = expected_calibration_error(probs_before, labels)
    print(f"[calibrate] ECE avant calibration: {ece_before:.4f}")

    print("[calibrate] fit de la température...")
    temperature = fit_temperature(logits, labels)
    print(f"[calibrate] température apprise: T={temperature:.4f}")

    probs_after = torch.softmax(logits / temperature, dim=-1)
    ece_after = expected_calibration_error(probs_after, labels)
    print(f"[calibrate] ECE après calibration: {ece_after:.4f}")

    if ece_after >= ece_before:
        print("[calibrate] ATTENTION: la calibration n'a pas réduit l'ECE. "
              "Vérifie la taille du split de validation (trop petit = bruit) "
              "ou le comportement du modèle avant de déployer en confiance.")

    ckpt["temperature"] = temperature
    ckpt["ece_before"] = ece_before
    ckpt["ece_after"] = ece_after
    torch.save(ckpt, args.out)
    print(f"[calibrate] checkpoint calibré sauvegardé -> {args.out}")


if __name__ == "__main__":
    main()
