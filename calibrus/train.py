"""
calibrus/train.py — Multi-task training (route classification + score
regression) of the CalibrusModel, with the temporal split already done upstream
(dataset.py).

Usage:
    python3 -m calibrus.train --config config.yaml --dataset data/dataset.pt \
        --out checkpoints/calibrus_model.pt
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from calibrus.model import CalibrusModel


class SplitDataset(Dataset):
    """Wraps a split (dict of tensors) into a standard PyTorch Dataset."""

    def __init__(self, split: dict):
        self.x_num = split["x_num"]
        self.input_ids = split["input_ids"]
        self.attention_mask = split["attention_mask"]
        self.y = split["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x_num": self.x_num[idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "y": self.y[idx],
        }


def pseudo_target_score(y: torch.Tensor) -> torch.Tensor:
    """
    Regression target derived from the class label, in the absence of a
    continuous score provided by the user: short=-1, hold=0, long=+1.
    TODO: if you have a real continuous score (e.g. normalized future return)
    in your data, replace this function to use that column instead.
    """
    mapping = torch.tensor([-1.0, 0.0, 1.0])
    return mapping[y]


def evaluate(model, loader, device, class_criterion, reg_criterion, reg_weight):
    model.eval()
    total_loss, total_class_loss, total_reg_loss = 0.0, 0.0, 0.0
    correct, n = 0, 0

    with torch.no_grad():
        for batch in loader:
            x_num = batch["x_num"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["y"].to(device)

            out = model(x_num, input_ids, attention_mask)
            class_loss = class_criterion(out["class_logits"], y)
            reg_target = pseudo_target_score(y).to(device)
            reg_loss = reg_criterion(out["score"], reg_target)
            loss = class_loss + reg_weight * reg_loss

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_class_loss += class_loss.item() * bs
            total_reg_loss += reg_loss.item() * bs
            correct += (out["class_logits"].argmax(dim=-1) == y).sum().item()
            n += bs

    return {
        "loss": total_loss / n,
        "class_loss": total_class_loss / n,
        "reg_loss": total_reg_loss / n,
        "accuracy": correct / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg.get("train", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    data = torch.load(args.dataset)
    train_ds = SplitDataset(data["splits"]["train"])
    val_ds = SplitDataset(data["splits"]["val"])

    batch_size = train_cfg.get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CalibrusModel.from_dataset_meta(
        data,
        freeze_text_encoder=train_cfg.get("freeze_text_encoder", False),
    ).to(device)

    lr = train_cfg.get("learning_rate", 2e-5)
    epochs = train_cfg.get("epochs", 10)
    reg_weight = train_cfg.get("reg_loss_weight", 0.5)
    patience = train_cfg.get("early_stopping_patience", 3)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    class_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            x_num = batch["x_num"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            out = model(x_num, input_ids, attention_mask)

            class_loss = class_criterion(out["class_logits"], y)
            reg_target = pseudo_target_score(y).to(device)
            reg_loss = reg_criterion(out["score"], reg_target)
            loss = class_loss + reg_weight * reg_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device, class_criterion, reg_criterion, reg_weight)

        print(f"[epoch {epoch}/{epochs}] train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "feat_cols": data["feat_cols"],
                "labels": data["labels"],
                "norm_mean": data["norm_mean"],
                "norm_std": data["norm_std"],
                "text_model": data["text_model"],
                "max_len": data["max_len"],
                "val_loss": best_val_loss,
                "epoch": epoch,
            }, args.out)
            print(f"  -> new best model saved ({args.out})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[train] early stopping (no improvement for {patience} epochs)")
                break

    print(f"[train] done. best val_loss={best_val_loss:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
