"""
calibrus/calibrate.py — Post-hoc calibration of the model via temperature scaling.

Principle: after training, the model is often over- or under-confident.
We learn a single scalar T (temperature) on the VALIDATION split which,
applied to the logits before softmax (logits / T), minimizes the NLL — without
touching the model's weights or its accuracy.

We measure the ECE (Expected Calibration Error) before/after to verify
that calibration actually improved things.

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
    """Runs a full forward pass over a split and collects all logits + labels."""
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
    ECE: weighted average, over confidence bins, of |accuracy - average confidence|.
    The closer to 0, the better calibrated the model is.
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
    Learns the scalar T via LBFGS by minimizing the NLL on (logits/T, labels).
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

    print("[calibrate] collecting logits on the validation split...")
    logits, labels = collect_logits_labels(model, val_loader, device)

    probs_before = torch.softmax(logits, dim=-1)
    ece_before = expected_calibration_error(probs_before, labels)
    print(f"[calibrate] ECE before calibration: {ece_before:.4f}")

    print("[calibrate] fitting temperature...")
    temperature = fit_temperature(logits, labels)
    print(f"[calibrate] learned temperature: T={temperature:.4f}")

    probs_after = torch.softmax(logits / temperature, dim=-1)
    ece_after = expected_calibration_error(probs_after, labels)
    print(f"[calibrate] ECE after calibration: {ece_after:.4f}")

    if ece_after >= ece_before:
        print("[calibrate] WARNING: calibration did not reduce the ECE. "
              "Check the size of the validation split (too small = noisy) "
              "or the model's behavior before deploying with confidence.")

    ckpt["temperature"] = temperature
    ckpt["ece_before"] = ece_before
    ckpt["ece_after"] = ece_after
    torch.save(ckpt, args.out)
    print(f"[calibrate] calibrated checkpoint saved -> {args.out}")


if __name__ == "__main__":
    main()
