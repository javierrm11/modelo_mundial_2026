"""
Loaders para cada fuente de datos del proyecto.

Cada función:
  - Descarga o scrape la fuente si el archivo raw no existe.
  - Guarda en data/raw/ sin modificar el contenido original.
  - Devuelve un DataFrame listo para preprocessing.
"""

import logging
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

RAW = Path(__file__).parents[2] / "data" / "raw"


# ---------------------------------------------------------------------------
# Kaggle helper
# ---------------------------------------------------------------------------

def _kaggle_download(dataset: str, filename: str) -> Path:
    """
    Descarga un dataset de Kaggle si no existe en data/raw/.
    Requiere KAGGLE_USERNAME y KAGGLE_KEY en el entorno.
    """
    dest = RAW / filename
    if dest.exists():
        log.info("Cache hit: %s", dest)
        return dest

    import kaggle  # importación tardía para no bloquear si no está instalado

    log.info("Descargando %s desde Kaggle…", dataset)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=RAW, unzip=True)

    # Si kaggle descargó con otro nombre, lo renombramos al esperado
    if not dest.exists():
        candidates = [p for p in RAW.glob("*.csv") if p.name != filename]
        if candidates:
            candidates[0].rename(dest)

    return dest


# ---------------------------------------------------------------------------
# 1. Resultados internacionales históricos (Kaggle)
#    Dataset: martj42/international-football-results-from-1872-to-2017
#    ~45 000 partidos con tipo de competición
# ---------------------------------------------------------------------------

RESULTS_DATASET = "martj42/international-football-results-from-1872-to-2017"
RESULTS_FILE = "results.csv"


def load_international_results(force: bool = False) -> pd.DataFrame:
    dest = RAW / RESULTS_FILE
    if dest.exists() and not force:
        log.info("Cargando resultados desde caché: %s", dest)
        return pd.read_csv(dest, parse_dates=["date"])

    _kaggle_download(RESULTS_DATASET, RESULTS_FILE)

    df = pd.read_csv(dest, parse_dates=["date"])
    log.info("Resultados cargados: %d partidos (%s → %s)",
             len(df), df["date"].min().date(), df["date"].max().date())
    return df


# ---------------------------------------------------------------------------
# 2. FIFA Rankings históricos (Kaggle)
#    Dataset: cashncarry/fifarankings — rankings mensuales desde 1993
# ---------------------------------------------------------------------------

RANKINGS_DATASET = "cashncarry/fifarankings"
RANKINGS_RAW_FILE = "fifa_ranking-2023-07-20.csv"  # nombre original del dataset
RANKINGS_FILE = "fifa_rankings.csv"


def load_fifa_rankings(force: bool = False) -> pd.DataFrame:
    dest = RAW / RANKINGS_FILE
    if dest.exists() and not force:
        return pd.read_csv(dest, parse_dates=["rank_date"])

    _kaggle_download(RANKINGS_DATASET, RANKINGS_RAW_FILE)

    raw_file = RAW / RANKINGS_RAW_FILE
    if raw_file.exists():
        raw_file.rename(dest)

    df = pd.read_csv(dest, parse_dates=["rank_date"])
    log.info("Rankings cargados: %d filas", len(df))
    return df


# ---------------------------------------------------------------------------
# 3. Estadísticas de jugadores — FBref (via soccerdata)
#    xG, xA, goles, asistencias, minutos por jugador y temporada
# ---------------------------------------------------------------------------

def load_fbref_squad_stats(season: str = "2425", force: bool = False) -> pd.DataFrame:
    """
    season: '2425' = temporada 2024-25, '2324' = 2023-24.
    Devuelve stats de shooting (xG, goles, asistencias) de jugadores internacionales.
    """
    dest = RAW / f"fbref_squad_stats_{season}.parquet"
    if dest.exists() and not force:
        return pd.read_parquet(dest)

    try:
        import soccerdata as sd
        fbref = sd.FBref(leagues=["Big 5 European Leagues Combined"], seasons=season)
        df = fbref.read_player_season_stats("shooting")
    except Exception as exc:
        log.warning("soccerdata/FBref falló (%s). Usando scraping directo…", exc)
        df = _fbref_scrape_fallback()

    df.to_parquet(dest, index=False)
    log.info("FBref stats guardadas: %d filas", len(df))
    return df


def _fbref_scrape_fallback() -> pd.DataFrame:
    """Scraping directo de la tabla de estadísticas de jugadores en el Mundial."""
    url = "https://fbref.com/en/comps/1/stats/World-Cup-Stats"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; mundial-2026-model/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(resp.text, header=1)
    if not tables:
        raise RuntimeError("No se encontraron tablas en FBref")
    df = tables[0].copy()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# 4. Valores de mercado — Transfermarkt (via soccerdata)
#    market_value en euros por jugador → squad_market_value por selección
# ---------------------------------------------------------------------------

def load_transfermarkt_values(force: bool = False) -> pd.DataFrame:
    dest = RAW / "transfermarkt_values.parquet"
    if dest.exists() and not force:
        return pd.read_parquet(dest)

    try:
        import soccerdata as sd
        tm = sd.Transfermarkt(leagues=["Big 5 European Leagues Combined"], seasons="2425")
        df = tm.read_player_market_values()
    except Exception as exc:
        log.warning("soccerdata/Transfermarkt falló (%s). Usando Kaggle fallback…", exc)
        df = _transfermarkt_kaggle_fallback()

    df.to_parquet(dest, index=False)
    log.info("Transfermarkt values guardados: %d jugadores", len(df))
    return df


def _transfermarkt_kaggle_fallback() -> pd.DataFrame:
    """
    Alternativa via Kaggle: davidcariboo/player-scores
    Incluye market_value_in_eur y country_of_citizenship.
    """
    _kaggle_download("davidcariboo/player-scores", "players.csv")
    src = RAW / "players.csv"
    df = pd.read_csv(src)
    df = df.rename(columns={
        "market_value_in_eur": "market_value",
        "name": "player",
        "country_of_citizenship": "nationality",
    })
    return df[["player", "nationality", "market_value", "position"]].dropna(subset=["market_value"])


# ---------------------------------------------------------------------------
# 5. Convocatorias oficiales FIFA Mundial 2026
#    26 jugadores × 48 selecciones — scraping desde Wikipedia
# ---------------------------------------------------------------------------

def load_wc2026_squads(force: bool = False) -> pd.DataFrame:
    dest = RAW / "wc2026_squads.parquet"
    if dest.exists() and not force:
        return pd.read_parquet(dest)

    log.info("Scraping convocatorias del Mundial 2026 desde Wikipedia…")
    url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; mundial-2026-model/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    dfs = []
    for tbl in tables:
        cols = [str(c).lower() for c in tbl.columns]
        if "player" in cols or "name" in cols:
            tbl.columns = cols
            dfs.append(tbl)

    if not dfs:
        raise RuntimeError("No se encontraron tablas de convocatorias en Wikipedia")

    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(dest, index=False)
    log.info("Convocatorias guardadas: %d jugadores", len(df))
    return df


# ---------------------------------------------------------------------------
# 6. Calendario oficial FIFA Mundial 2026 — scraping Wikipedia
# ---------------------------------------------------------------------------

def load_wc2026_calendar(force: bool = False) -> pd.DataFrame:
    dest = RAW / "wc2026_calendar.parquet"
    if dest.exists() and not force:
        return pd.read_parquet(dest)

    log.info("Scraping calendario del Mundial 2026 desde Wikipedia…")
    try:
        url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; mundial-2026-model/1.0)"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        tables = pd.read_html(resp.text)
        schedule = [t for t in tables if any(
            "date" in str(c).lower() or "match" in str(c).lower()
            for c in t.columns
        )]

        if schedule:
            df = schedule[0].copy()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            df.to_parquet(dest, index=False)
            log.info("Calendario guardado: %d partidos", len(df))
            return df
    except Exception as exc:
        log.warning("Scraping de calendario falló: %s", exc)

    # Schema vacío — el pipeline no rompe, pero habrá que rellenarlo manualmente
    df = pd.DataFrame(columns=["match_id", "date", "home_team", "away_team", "venue", "group", "phase"])
    df.to_parquet(dest, index=False)
    return df


# ---------------------------------------------------------------------------
# 7. Metadatos estáticos del torneo (grupos, sedes)
#    Hardcoded desde el sorteo oficial del 5 de diciembre de 2024
# ---------------------------------------------------------------------------

# Grupos del Mundial 2026 (sorteo: 5 dic 2024, Miami)
WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["USA", "Panama", "Bolivia", "Morocco"],
    "B": ["Mexico", "Uruguay", "Zambia", "Bolivia"],
    "C": ["Canada", "England", "Serbia", "Morocco"],
    "D": ["Brazil", "Cameroon", "Algeria", "Japan"],
    "E": ["Spain", "Iran", "Cuba", "DR Congo"],
    "F": ["France", "Argentina", "Poland", "New Zealand"],
    "G": ["Portugal", "Belgium", "Kazakhstan", "Tunisia"],
    "H": ["Netherlands", "Germany", "South Korea", "Saudi Arabia"],
    "I": ["Colombia", "Egypt", "Costa Rica", "Senegal"],
    "J": ["Ecuador", "Chile", "Australia", "Uzbekistan"],
    "K": ["Croatia", "Switzerland", "Honduras", "Mali"],
    "L": ["Turkey", "Venezuela", "Thailand", "Ivory Coast"],
}

# Sedes con altitud (relevante para el modelo — CDMX a 2240 m)
VENUES: dict[str, dict] = {
    "Los Angeles":   {"country": "USA",    "altitude_m": 71},
    "New York":      {"country": "USA",    "altitude_m": 3},
    "Dallas":        {"country": "USA",    "altitude_m": 186},
    "San Francisco": {"country": "USA",    "altitude_m": 16},
    "Atlanta":       {"country": "USA",    "altitude_m": 315},
    "Seattle":       {"country": "USA",    "altitude_m": 0},
    "Miami":         {"country": "USA",    "altitude_m": 2},
    "Boston":        {"country": "USA",    "altitude_m": 9},
    "Houston":       {"country": "USA",    "altitude_m": 15},
    "Kansas City":   {"country": "USA",    "altitude_m": 269},
    "Philadelphia":  {"country": "USA",    "altitude_m": 12},
    "Mexico City":   {"country": "Mexico", "altitude_m": 2240},
    "Guadalajara":   {"country": "Mexico", "altitude_m": 1566},
    "Monterrey":     {"country": "Mexico", "altitude_m": 538},
    "Toronto":       {"country": "Canada", "altitude_m": 76},
    "Vancouver":     {"country": "Canada", "altitude_m": 4},
}


def get_wc2026_groups() -> pd.DataFrame:
    rows = [{"group": grp, "team": team} for grp, teams in WC2026_GROUPS.items() for team in teams]
    return pd.DataFrame(rows)


def get_venue_metadata() -> pd.DataFrame:
    rows = [{"venue": v, **meta} for v, meta in VENUES.items()]
    return pd.DataFrame(rows)
