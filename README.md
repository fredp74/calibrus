# Calibrus

An open-source "System One" decision model for algorithmic trading:
route (long/short/hold), a continuous score, and **calibrated confidence**,
built from market features (OHLCV) and optional text (news, sentiment).

## Why

A standard classifier will say "long" with 99% confidence even when it's
wrong half the time. Calibrus explicitly calibrates that confidence
(Temperature Scaling + Expected Calibration Error (ECE) measurement) so that "90% confidence" actually
means "right 9 times out of 10" — useful for deciding when to act
automatically and when to escalate to human review.

## Installation

```bash
git clone <repo-url> calibrus
cd calibrus
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (with sample data)

```bash
cp config.yaml.example config.yaml
bash scripts/quickstart.sh
```

This builds the dataset, trains, calibrates, and runs a test inference.
The sample data (`examples/sample_data.csv`) is tiny — just enough to
verify the pipeline runs end to end. Swap in your own data for real training.

## Using your own data

### Option 1 — CSV
Required columns: `timestamp, open, high, low, close, volume, text, label`
(`text` can be empty, `label` is one of {short, hold, long}).
Set `data_source.type: csv` and the `path` in `config.yaml`.

### Option 2 — MariaDB
```bash
mysql -u root -p < scripts/setup_db.sql
export CALIBRUS_DB_PASSWORD="your_password"
```
Set `data_source.type: mariadb` in `config.yaml`.

**Security**: never put a password in plain text in `config.yaml` if you
plan to commit it — always use the `CALIBRUS_DB_PASSWORD` environment
variable instead.

### Automatic technical indicators
If you only have raw OHLCV, enable `features.compute_ta.enabled: true` in
`config.yaml` — indicators (`rsi`, `macd`, `bbands`, `ema`, `atr`, `obv`)
are computed automatically. If you already have your own features, name them `feat_*` in your data and disable `compute_ta`.

## Pipeline overview
calibrus.data.dataset -> builds the dataset (loads data, computes features, temporal split)
calibrus.train -> trains the hybrid model (text + numeric)
calibrus.calibrate -> calibrates confidence (temperature scaling) + measures ECE
calibrus.infer -> serves typed predictions


Each step can be run separately:
```bash
python3 -m calibrus.data.dataset --config config.yaml --out data/dataset.pt
python3 -m calibrus.train --config config.yaml --dataset data/dataset.pt --out checkpoints/calibrus_model.pt
python3 -m calibrus.calibrate --dataset data/dataset.pt --checkpoint checkpoints/calibrus_model.pt --out checkpoints/calibrus_model_calibrated.pt
python3 -m calibrus.infer --checkpoint checkpoints/calibrus_model_calibrated.pt --features '{"feat_rsi_14": 28.4}' --text "Fed signals rate cut"
```

## Output

```json
{
  "route": "long",
  "score": 0.73,
  "confidence": 0.88,
  "probs": {"short": 0.05, "hold": 0.07, "long": 0.88}
}
```

## Known limitations

- Temporal split is mandatory (never random shuffle) — but this doesn't
  protect against look-ahead bias if your own feature calculations use
  future data. Double-check your feature engineering.
- The temporal split means **little data = no real test possible**.
  A few hundred rows isn't enough for a reliable evaluation.
- Calibration (temperature scaling) is learned on the validation split;
  if that split is small, the learned temperature will be noisy. Check the
  `ece_before`/`ece_after` values logged by `calibrate.py` before trusting
  the model.
- This is not financial advice or a ready-to-use trading system — it's a
  research pipeline that needs rigorous validation (full backtesting,
  transaction costs, slippage) before any real-world use.

## License

MIT — see LICENSE.

---

*A research project by [Algotradingresearch.com](https://algotradingresearch.com)*
