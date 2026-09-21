"""
calibrus/data/dataset.py — Pipeline complet: charge (loaders) -> features -> split
temporel -> tenseurs prêts pour l'entraînement. Point d'entrée: build_dataset(cfg).

Usage direct:
    python3 -m calibrus.data.dataset --config config.yaml --out data/dataset.pt
"""

import argparse
import yaml
import torch
import pandas as pd
from transformers import AutoTokenizer

from calibrus.data.loaders import load_from_config, DataLoadError
from calibrus.data.features import prepare_features, FeatureError

LABELS = {"short": 0, "hold": 1, "long": 2}
DEFAULT_TEXT_MODEL = "distilbert-base-uncased"


class DatasetBuildError(Exception):
    pass


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in df.columns:
        raise DatasetBuildError(
            "Colonne 'label' manquante. Chaque ligne doit avoir un label dans "
            f"{list(LABELS.keys())}."
        )
    df = df.copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    if df["label"].isna().any():
        raise DatasetBuildError("Labels manquants détectés — nettoie tes données avant.")
    bad = set(df["label"].unique()) - set(LABELS.keys())
    if bad:
        raise DatasetBuildError(
            f"Labels inconnus: {bad}. Attendu: {list(LABELS.keys())}"
        )
    return df


def ensure_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "text" not in df.columns:
        df["text"] = ""
    df["text"] = df["text"].fillna("").astype(str)
    return df


def temporal_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    if train_end == 0 or val_end == train_end or val_end == n:
        raise DatasetBuildError(
            f"Dataset trop petit ({n} lignes) pour un split "
            f"train={train_frac}/val={val_frac}/test={1-train_frac-val_frac}. "
            "Ajoute plus de données ou ajuste les fractions dans config.yaml."
        )
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


def compute_norm_stats(train_df: pd.DataFrame, feat_cols: list):
    mean = train_df[feat_cols].mean()
    std = train_df[feat_cols].std().replace(0, 1.0)
    return mean, std


def build_split_tensors(df, feat_cols, mean, std, tokenizer, max_len):
    x_num = torch.tensor(((df[feat_cols] - mean) / std).values, dtype=torch.float32)
    enc = tokenizer(
        df["text"].tolist(), padding="max_length", truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    y = torch.tensor(df["label"].map(LABELS).values, dtype=torch.long)
    return {
        "x_num": x_num,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "y": y,
    }


def build_dataset(cfg: dict) -> dict:
    df = load_from_config(cfg)
    df = prepare_features(df, cfg)
    df = validate_labels(df)
    df = ensure_text_column(df)

    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])

    split_cfg = cfg.get("split", {})
    train_frac = split_cfg.get("train_frac", 0.7)
    val_frac = split_cfg.get("val_frac", 0.15)

    train_df, val_df, test_df = temporal_split(df, train_frac, val_frac)
    print(f"[dataset] split temporel -> train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    mean, std = compute_norm_stats(train_df, feat_cols)

    text_model = cfg.get("features", {}).get("text_model", DEFAULT_TEXT_MODEL)
    max_len = cfg.get("features", {}).get("max_text_len", 128)
    tokenizer = AutoTokenizer.from_pretrained(text_model)

    splits = {}
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        splits[name] = build_split_tensors(d, feat_cols, mean, std, tokenizer, max_len)

    return {
        "splits": splits,
        "feat_cols": feat_cols,
        "labels": LABELS,
        "norm_mean": mean.to_dict(),
        "norm_std": std.to_dict(),
        "text_model": text_model,
        "max_len": max_len,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    try:
        dataset = build_dataset(cfg)
    except (DataLoadError, FeatureError, DatasetBuildError) as e:
        print(f"[ERREUR] {e}")
        raise SystemExit(1)

    torch.save(dataset, args.out)
    print(f"[dataset] sauvegardé -> {args.out}")


if __name__ == "__main__":
    main()
