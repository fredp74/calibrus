"""
calibrus/data/features.py — Calcul automatique de features techniques depuis OHLCV
brut, ou passthrough si l'utilisateur fournit déjà des colonnes feat_*.
"""

import pandas as pd

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

SUPPORTED_INDICATORS = {"rsi", "macd", "bbands", "ema", "atr", "obv"}


class FeatureError(Exception):
    pass


def has_ohlcv(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in OHLCV_COLS)


def has_existing_features(df: pd.DataFrame) -> bool:
    return any(c.startswith("feat_") for c in df.columns)


def compute_indicators(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
    """
    Calcule les indicateurs demandés depuis OHLCV et les ajoute comme colonnes feat_*.
    Nécessite: pip install pandas-ta
    """
    try:
        import pandas_ta as ta
    except ImportError:
        raise FeatureError(
            "Le package 'pandas-ta' est requis pour le calcul auto d'indicateurs. "
            "Installe avec: pip install pandas-ta"
        )

    if not has_ohlcv(df):
        missing = [c for c in OHLCV_COLS if c not in df.columns]
        raise FeatureError(
            f"Colonnes OHLCV manquantes pour calculer les indicateurs: {missing}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    unknown = set(indicators) - SUPPORTED_INDICATORS
    if unknown:
        raise FeatureError(
            f"Indicateurs inconnus: {unknown}. Supportés: {sorted(SUPPORTED_INDICATORS)}"
        )

    df = df.copy()
    added_cols = []

    if "rsi" in indicators:
        df["feat_rsi_14"] = ta.rsi(df["close"], length=14)
        added_cols.append("feat_rsi_14")

    if "macd" in indicators:
        macd = ta.macd(df["close"])
        df["feat_macd"] = macd["MACD_12_26_9"]
        df["feat_macd_signal"] = macd["MACDs_12_26_9"]
        df["feat_macd_hist"] = macd["MACDh_12_26_9"]
        added_cols += ["feat_macd", "feat_macd_signal", "feat_macd_hist"]

    if "bbands" in indicators:
        bb = ta.bbands(df["close"], length=20)
        df["feat_bb_upper"] = bb["BBU_20_2.0"]
        df["feat_bb_lower"] = bb["BBL_20_2.0"]
        df["feat_bb_width"] = bb["BBB_20_2.0"]
        added_cols += ["feat_bb_upper", "feat_bb_lower", "feat_bb_width"]

    if "ema" in indicators:
        df["feat_ema_9"] = ta.ema(df["close"], length=9)
        df["feat_ema_21"] = ta.ema(df["close"], length=21)
        added_cols += ["feat_ema_9", "feat_ema_21"]

    if "atr" in indicators:
        df["feat_atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        added_cols.append("feat_atr_14")

    if "obv" in indicators:
        df["feat_obv"] = ta.obv(df["close"], df["volume"])
        added_cols.append("feat_obv")

    n_before = len(df)
    df = df.dropna(subset=added_cols).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[features] {n_dropped} lignes supprimées (NaN de warm-up des indicateurs)")

    print(f"[features] Indicateurs calculés: {added_cols}")
    return df


def prepare_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Point d'entrée unique. Logique de décision:
    1. Si compute_ta activé dans la config -> calcule depuis OHLCV (erreur si OHLCV absent)
    2. Sinon, si des feat_* existent déjà -> passthrough
    3. Sinon -> erreur explicite (rien à faire tourner)
    """
    ta_cfg = cfg.get("features", {}).get("compute_ta", {})
    compute_enabled = ta_cfg.get("enabled", False)

    if compute_enabled:
        indicators = ta_cfg.get("indicators", [])
        if not indicators:
            raise FeatureError("compute_ta.enabled=true mais aucune liste 'indicators' fournie.")
        df = compute_indicators(df, indicators)

    if not has_existing_features(df):
        raise FeatureError(
            "Aucune colonne feat_* trouvée après traitement. "
            "Active 'compute_ta' dans config.yaml ou fournis des colonnes feat_* toi-même."
        )

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    print(f"[features] {len(feat_cols)} features prêtes: {feat_cols}")
    return df
