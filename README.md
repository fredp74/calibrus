# Calibrus

Un modèle de décision "System One" open-source pour le trading algorithmique :
route (long/short/hold), score continu et **confiance calibrée**, à partir de
features de marché (OHLCV) et de texte optionnel (news, sentiment).

## Pourquoi

Un classifieur normal dit "long" avec 99% de confiance même quand il se trompe
une fois sur deux. Calibrus calibre explicitement cette confiance (temperature
scaling + mesure de l'ECE) pour que "90% de confiance" veuille vraiment dire
"a raison 9 fois sur 10" — utile pour décider quand agir automatiquement et
quand escalader vers une revue humaine.

## Installation

```bash
git clone <repo-url> calibrus
cd calibrus
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (avec les données d'exemple)

```bash
cp config.yaml.example config.yaml
bash scripts/quickstart.sh
```

Ça construit le dataset, entraîne, calibre, et lance une inférence de test.
Les données d'exemple (`examples/sample_data.csv`) sont minuscules — juste
pour vérifier que le pipeline tourne. Remplace par tes propres données pour
un vrai entraînement.

## Utiliser tes propres données

### Option 1 — CSV
Colonnes requises : `timestamp, open, high, low, close, volume, text, label`
(`text` peut être vide, `label` dans {short, hold, long}).
Configure `data_source.type: csv` et le `path` dans `config.yaml`.

### Option 2 — MariaDB
```bash
mysql -u root -p < scripts/setup_db.sql
export CALIBRUS_DB_PASSWORD="ton_mot_de_passe"
```
Configure `data_source.type: mariadb` dans `config.yaml`.

**Sécurité** : ne mets jamais de mot de passe en clair dans `config.yaml` si tu
comptes le committer — utilise toujours la variable d'environnement
`CALIBRUS_DB_PASSWORD`.

### Indicateurs techniques automatiques
Si tu n'as que de l'OHLCV brut, active `features.compute_ta.enabled: true` dans
`config.yaml` — les indicateurs (`rsi`, `macd`, `bbands`, `ema`, `atr`, `obv`)
sont calculés automatiquement. Si tu as déjà tes propres features, nomme-les
`feat_*` dans tes données et désactive `compute_ta`.

## Pipeline en détail
calibrus.data.dataset -> construit le dataset (charge, calcule les features, split temporel)
calibrus.train -> entraîne le modèle hybride (texte + numérique)
calibrus.calibrate -> calibre la confiance (temperature scaling) + mesure l'ECE
calibrus.infer -> sert des prédictions typées


Chaque étape peut être lancée séparément :
```bash
python3 -m calibrus.data.dataset --config config.yaml --out data/dataset.pt
python3 -m calibrus.train --config config.yaml --dataset data/dataset.pt --out checkpoints/calibrus_model.pt
python3 -m calibrus.calibrate --dataset data/dataset.pt --checkpoint checkpoints/calibrus_model.pt --out checkpoints/calibrus_model_calibrated.pt
python3 -m calibrus.infer --checkpoint checkpoints/calibrus_model_calibrated.pt --features '{"feat_rsi_14": 28.4}' --text "Fed signals rate cut"
```

## Sortie

```json
{
  "route": "long",
  "score": 0.73,
  "confidence": 0.88,
  "probs": {"short": 0.05, "hold": 0.07, "long": 0.88}
}
```

## Limites connues

- Split temporel obligatoire (jamais de shuffle random) — mais ça ne protège
  pas contre le look-ahead bias si tes features elles-mêmes utilisent des
  données futures. Vérifie tes calculs de features.
- Le split temporel signifie que **peu de données = pas de vrai test possible**.
  Quelques centaines de lignes ne suffisent pas pour une évaluation fiable.
- La calibration (temperature scaling) est apprise sur le split de validation ;
  si ce split est petit, la température apprise sera bruitée. Vérifie
  `ece_before`/`ece_after` loggés par `calibrate.py` avant de faire confiance
  au modèle.
- Ceci n'est pas un conseil financier ni un système de trading prêt à l'emploi
  — c'est un pipeline de recherche à valider rigoureusement (backtest complet,
  coûts de transaction, slippage) avant tout usage réel.

## License

MIT — voir LICENSE.
