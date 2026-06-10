"""
Descarga todas las fuentes de datos a data/raw/.

Uso:
    python scripts/download_data.py                  # todas las fuentes
    python scripts/download_data.py --only results   # solo resultados históricos
    python scripts/download_data.py --force          # re-descarga aunque exista caché

Requiere KAGGLE_USERNAME y KAGGLE_KEY en .env (ver .env.example).
"""

import argparse
import logging
import sys
from pathlib import Path

# Añade la raíz del proyecto al path para poder importar src/
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.loaders import (
    load_international_results,
    load_fifa_rankings,
    load_fbref_squad_stats,
    load_transfermarkt_values,
    load_wc2026_squads,
    load_wc2026_calendar,
    get_wc2026_groups,
    get_venue_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SOURCES = {
    "results":       ("Resultados históricos internacionales (Kaggle)",    load_international_results),
    "rankings":      ("FIFA Rankings históricos (Kaggle)",                  load_fifa_rankings),
    "fbref":         ("Estadísticas de jugadores — FBref",                  load_fbref_squad_stats),
    "transfermarkt": ("Valores de mercado — Transfermarkt",                 load_transfermarkt_values),
    "squads":        ("Convocatorias oficiales FIFA 2026",                   load_wc2026_squads),
    "calendar":      ("Calendario oficial FIFA 2026",                       load_wc2026_calendar),
}


def main():
    parser = argparse.ArgumentParser(description="Descarga las fuentes de datos del proyecto.")
    parser.add_argument("--only", choices=list(SOURCES.keys()), help="Descargar solo una fuente")
    parser.add_argument("--force", action="store_true", help="Re-descargar aunque exista caché")
    args = parser.parse_args()

    targets = {args.only: SOURCES[args.only]} if args.only else SOURCES

    ok, failed = [], []

    for key, (label, loader) in targets.items():
        log.info("━━━ %s ━━━", label)
        try:
            kwargs = {"force": args.force}
            if key == "fbref":
                kwargs["season"] = "2425"
            df = loader(**kwargs)
            log.info("OK — %d filas\n", len(df))
            ok.append(key)
        except Exception as exc:
            log.error("FALLÓ: %s\n", exc)
            failed.append((key, exc))

    # Los metadatos estáticos no requieren descarga, solo los mostramos
    groups = get_wc2026_groups()
    venues = get_venue_metadata()
    log.info("Grupos del Mundial 2026: %d equipos registrados en data/loaders.py", len(groups))
    log.info("Sedes: %d venues con altitud registrada", len(venues))

    print("\n" + "=" * 60)
    print(f"  Completadas: {len(ok)}/{len(targets)}")
    if failed:
        print(f"  Fallidas: {[k for k, _ in failed]}")
        print("\n  Pasos para resolver errores de Kaggle:")
        print("  1. pip install kaggle")
        print("  2. Rellena KAGGLE_USERNAME y KAGGLE_KEY en .env")
        print("     (Obtén la API key en https://www.kaggle.com/settings → API)")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
