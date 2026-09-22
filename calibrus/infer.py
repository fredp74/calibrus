"""
calibrus/infer.py — "Calibrus-style" entry point: takes a state (numeric features +
text), returns a typed decision with score and calibrated confidence.

CLI usage (quick test):
    python3 -m calibrus.infer --checkpoint checkpoints/calibrus_model_calibrated.pt \
        --features '{"feat_rsi_14": 28.4, "feat_macd": 0.12}' \
        --text "Fed signals rate cut, market rallies"

Programmatic usage:
    from calibrus.infer import CalibrusPredictor
    predictor = CalibrusPredictor("checkpoints/calibrus_model_calibrated.pt")
    result = predictor.predict(features={...}, text="...")
    # -> {"route": "long", "score": 0.73, "confidence": 0.88, "probs": {...}}
"""

import argparse
import json
import torch
from transformers import AutoTokenizer

from calibrus.model import CalibrusModel

INV_LABELS = {0: "short", 1: "hold", 2: "long"}


class InferenceError(Exception):
    pass


class CalibrusPredictor:
    """Loads a calibrated checkpoint once, then serves fast predictions."""

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.feat_cols = ckpt["feat_cols"]
        self.norm_mean = ckpt["norm_mean"]
        self.norm_std = ckpt["norm_std"]
        self.max_len = ckpt["max_len"]
        self.temperature = ckpt.get("temperature", 1.0)

        if "temperature" not in ckpt:
            print("[infer] WARNING: no temperature found in the checkpoint — "
                  "uncalibrated model (did you run calibrate.py?). Defaulting to T=1.0.")

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt["text_model"])
        self.model = CalibrusModel.from_dataset_meta({
            "feat_cols": self.feat_cols, "text_model": ckpt["text_model"],
        }).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        print(f"[infer] model loaded ({checkpoint_path}), T={self.temperature:.4f}, "
              f"device={self.device}")

    def _build_numeric_tensor(self, features: dict) -> torch.Tensor:
        missing = [c for c in self.feat_cols if c not in features]
        if missing:
            raise InferenceError(
                f"Missing features: {missing}. Expected: {self.feat_cols}"
            )
        raw = [features[c] for c in self.feat_cols]
        mean = [self.norm_mean[c] for c in self.feat_cols]
        std = [self.norm_std[c] for c in self.feat_cols]
        normed = [(r - m) / s for r, m, s in zip(raw, mean, std)]
        return torch.tensor([normed], dtype=torch.float32)

    @torch.no_grad()
    def predict(self, features: dict, text: str = "") -> dict:
        x_num = self._build_numeric_tensor(features).to(self.device)

        enc = self.tokenizer(
            [text or ""], padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        out = self.model(x_num, input_ids, attention_mask)

        calibrated_logits = out["class_logits"] / self.temperature
        probs = torch.softmax(calibrated_logits, dim=-1)[0]

        route_idx = int(probs.argmax().item())
        confidence = float(probs[route_idx].item())
        score = float(out["score"][0].item())

        return {
            "route": INV_LABELS[route_idx],
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "probs": {INV_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features", required=True, help='JSON string of features, e.g.: {"feat_rsi_14": 28.4}')
    ap.add_argument("--text", default="", help="Optional text (news/context)")
    args = ap.parse_args()

    try:
        features = json.loads(args.features)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] --features must be valid JSON: {e}")

    predictor = CalibrusPredictor(args.checkpoint)

    try:
        result = predictor.predict(features=features, text=args.text)
    except InferenceError as e:
        raise SystemExit(f"[ERROR] {e}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
