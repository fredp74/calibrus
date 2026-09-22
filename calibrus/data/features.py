"""
calibrus/data/features.py — Automatic computation of technical features from raw
OHLCV, or passthrough if the user already provides feat_* columns.
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
    Computes the requested indicators from OHLCV and adds them as feat_* columns.
    Requires: pip install pandas-ta
    """
    try:
        import pandas_ta as ta
    except ImportError:
        raise FeatureError(
            "The 'pandas-ta' package is required for automatic indicator computation. "
            "Install with: pip install pandas-ta"
        )

    if not has_ohlcv(df):
        missing = [c for c in OHLCV_COLS if c not in df.columns]
        raise FeatureError(
            f"Missing OHLCV columns needed to compute indicators: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    unknown = set(indicators) - SUPPORTED_INDICATORS
    if unknown:
        raise FeatureError(
            f"Unknown indicators: {unknown}. Supported: {sorted(SUPPORTED_INDICATORS)}"
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
        print(f"[features] {n_dropped} rows dropped (NaN from indicator warm-up)")

    print(f"[features] Computed indicators: {added_cols}")
    return df


def prepare_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Single entry point. Decision logic:
    1. If compute_ta is enabled in the config -> compute from OHLCV (error if OHLCV is missing)
    2. Otherwise, if feat_* columns already exist -> passthrough
    3. Otherwise -> explicit error (nothing to run)
    """
    ta_cfg = cfg.get("features", {}).get("compute_ta", {})
    compute_enabled = ta_cfg.get("enabled", False)

    if compute_enabled:
        indicators = ta_cfg.get("indicators", [])
        if not indicators:
            raise FeatureError("compute_ta.enabled=true but no 'indicators' list was provided.")
        df = compute_indicators(df, indicators)

    if not has_existing_features(df):
        raise FeatureError(
            "No feat_* column found after processing. "
            "Enable 'compute_ta' in config.yaml or provide feat_* columns yourself."
        )

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    print(f"[features] {len(feat_cols)} features ready: {feat_cols}")
    return df
