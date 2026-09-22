"""
calibrus/data/loaders.py — Data source adapters for Calibrus.
Each loader returns a DataFrame with, at minimum: timestamp + raw columns
(OHLCV and/or feat_* already present) + optionally 'text' and 'label'.
"""

import os
import pandas as pd

REQUIRED_BASE_COLS = {"timestamp"}


class DataLoadError(Exception):
    pass


def _validate_base(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    missing = REQUIRED_BASE_COLS - set(df.columns)
    if missing:
        raise DataLoadError(
            f"[{source_name}] Missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise DataLoadError(f"[csv] File not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = _validate_base(df, "csv")
    print(f"[loaders] CSV loaded: {path} ({len(df)} rows, columns: {list(df.columns)})")
    return df


def load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise DataLoadError(f"[parquet] File not found: {path}")
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = _validate_base(df, "parquet")
    print(f"[loaders] Parquet loaded: {path} ({len(df)} rows)")
    return df


def load_mariadb(host: str, port: int, database: str, table: str,
                  user: str, password: str, query: str = None) -> pd.DataFrame:
    """
    Requires: pip install sqlalchemy mysqlclient
    The user can provide a custom query (`query`), otherwise the whole table
    is used by default.
    """
    try:
        import sqlalchemy
    except ImportError:
        raise DataLoadError(
            "[mariadb] The 'sqlalchemy' package is required. "
            "Install with: pip install sqlalchemy mysqlclient"
        )

    conn_str = f"mysql+mysqldb://{user}:{password}@{host}:{port}/{database}"
    engine = sqlalchemy.create_engine(conn_str)

    sql = query if query else f"SELECT * FROM `{table}`"
    try:
        df = pd.read_sql(sql, engine)
    except Exception as e:
        raise DataLoadError(f"[mariadb] Query failed on {database}.{table}: {e}")
    finally:
        engine.dispose()

    df.columns = [c.strip().lower() for c in df.columns]
    df = _validate_base(df, "mariadb")
    print(f"[loaders] MariaDB loaded: {database}.{table} ({len(df)} rows)")
    return df


def load_from_config(cfg: dict) -> pd.DataFrame:
    """Single entry point — dispatches based on config.yaml['data_source']['type']."""
    source = cfg.get("data_source", {})
    source_type = source.get("type")

    if source_type == "csv":
        return load_csv(source["csv"]["path"])

    elif source_type == "parquet":
        return load_parquet(source["parquet"]["path"])

    elif source_type == "mariadb":
        mcfg = source["mariadb"]
        password = mcfg.get("password") or os.environ.get("CALIBRUS_DB_PASSWORD", "")
        if not password:
            raise DataLoadError(
                "[mariadb] Missing password. Set 'password' in config.yaml "
                "or the CALIBRUS_DB_PASSWORD environment variable."
            )
        return load_mariadb(
            host=mcfg["host"], port=mcfg.get("port", 3306),
            database=mcfg["database"], table=mcfg["table"],
            user=mcfg["user"], password=password,
            query=mcfg.get("query"),
        )

    else:
        raise DataLoadError(
            f"Unknown source type: '{source_type}'. Expected: csv | parquet | mariadb"
        )
