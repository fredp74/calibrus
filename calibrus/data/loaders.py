"""
calibrus/data/loaders.py — Adaptateurs de source de données pour Calibrus.
Chaque loader retourne un DataFrame avec au minimum: timestamp + colonnes brutes
(OHLCV et/ou feat_* déjà présentes) + optionnellement 'text' et 'label'.
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
            f"[{source_name}] Colonnes obligatoires manquantes: {missing}. "
            f"Colonnes trouvées: {list(df.columns)}"
        )
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise DataLoadError(f"[csv] Fichier introuvable: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = _validate_base(df, "csv")
    print(f"[loaders] CSV chargé: {path} ({len(df)} lignes, colonnes: {list(df.columns)})")
    return df


def load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise DataLoadError(f"[parquet] Fichier introuvable: {path}")
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = _validate_base(df, "parquet")
    print(f"[loaders] Parquet chargé: {path} ({len(df)} lignes)")
    return df


def load_mariadb(host: str, port: int, database: str, table: str,
                  user: str, password: str, query: str = None) -> pd.DataFrame:
    """
    Nécessite: pip install sqlalchemy mysqlclient
    L'user peut fournir une requête custom (`query`) ou on prend toute la table par défaut.
    """
    try:
        import sqlalchemy
    except ImportError:
        raise DataLoadError(
            "[mariadb] Le package 'sqlalchemy' est requis. "
            "Installe avec: pip install sqlalchemy mysqlclient"
        )

    conn_str = f"mysql+mysqldb://{user}:{password}@{host}:{port}/{database}"
    engine = sqlalchemy.create_engine(conn_str)

    sql = query if query else f"SELECT * FROM `{table}`"
    try:
        df = pd.read_sql(sql, engine)
    except Exception as e:
        raise DataLoadError(f"[mariadb] Échec de la requête sur {database}.{table}: {e}")
    finally:
        engine.dispose()

    df.columns = [c.strip().lower() for c in df.columns]
    df = _validate_base(df, "mariadb")
    print(f"[loaders] MariaDB chargé: {database}.{table} ({len(df)} lignes)")
    return df


def load_from_config(cfg: dict) -> pd.DataFrame:
    """Point d'entrée unique — dispatch selon config.yaml['data_source']['type']."""
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
                "[mariadb] Mot de passe manquant. Définis 'password' dans config.yaml "
                "ou la variable d'environnement CALIBRUS_DB_PASSWORD."
            )
        return load_mariadb(
            host=mcfg["host"], port=mcfg.get("port", 3306),
            database=mcfg["database"], table=mcfg["table"],
            user=mcfg["user"], password=password,
            query=mcfg.get("query"),
        )

    else:
        raise DataLoadError(
            f"Type de source inconnu: '{source_type}'. Attendu: csv | parquet | mariadb"
        )
