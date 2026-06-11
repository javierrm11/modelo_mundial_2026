"""
Paso 7 — Dataset final.
Una fila por partido entre equipos del Mundial 2026.

Columnas:
  date, home_team, away_team, home_score, away_score, tournament, neutral
  resultado          : H / D / A
  elo_home_pre       : Elo local antes del partido
  elo_away_pre       : Elo visitante antes del partido
  elo_diff           : elo_home_pre - elo_away_pre
  home_<feat>        : features de plantilla del local (squad_features.csv)
  away_<feat>        : features de plantilla del visitante
"""

import csv
import os

RESULTS_PATH = (
    "datos/historico de partidos/"
    "[International Football Results from 1872 to 2026/results.csv"
)
ELO_PATH    = "datos/master/elo_por_partido.csv"
SQUAD_PATH  = "datos/master/squad_features.csv"
MASTER_PATH = "datos/master/team_name_master.csv"
OUT_PATH    = "datos/master/dataset_final.csv"

os.makedirs("datos/master", exist_ok=True)

# Conjunto de equipos del Mundial 2026 (canonical)
wc_teams: set[str] = set()
with open(MASTER_PATH, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        wc_teams.add(row["canonical"])

# squad features indexadas por canonical
sq: dict[str, dict] = {}
with open(SQUAD_PATH, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        sq[row["canonical"]] = row

# Elo indexado por (date, home_team, away_team)
elo_idx: dict[tuple, dict] = {}
with open(ELO_PATH, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        elo_idx[(row["date"], row["home_team"], row["away_team"])] = row

SQ_FEATS = [
    "squad_mv_total", "squad_mv_avg", "squad_mv_median",
    "caps_avg", "goals_total", "avg_age",
]

FIELDS = (
    ["date", "home_team", "away_team", "home_score", "away_score",
     "tournament", "neutral", "resultado",
     "elo_home_pre", "elo_away_pre", "elo_diff"] +
    [f"home_{f}" for f in SQ_FEATS] +
    [f"away_{f}" for f in SQ_FEATS]
)


def resultado(hs: int, as_: int) -> str:
    if hs > as_: return "H"
    if hs < as_: return "A"
    return "D"


rows_out, skipped = [], 0

with open(RESULTS_PATH, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        home = row["home_team"]
        away = row["away_team"]

        if home not in wc_teams or away not in wc_teams:
            skipped += 1
            continue

        date = row["date"]
        elo_row = elo_idx.get((date, home, away))
        if elo_row is None:
            skipped += 1
            continue

        try:
            hs  = int(row["home_score"])
            as_ = int(row["away_score"])
        except ValueError:
            skipped += 1
            continue

        eh = float(elo_row["elo_home_pre"])
        ea = float(elo_row["elo_away_pre"])

        sq_h = sq.get(home, {})
        sq_a = sq.get(away, {})

        out: dict = {
            "date":        date,
            "home_team":   home,
            "away_team":   away,
            "home_score":  hs,
            "away_score":  as_,
            "tournament":  row.get("tournament", ""),
            "neutral":     row.get("neutral", ""),
            "resultado":   resultado(hs, as_),
            "elo_home_pre": round(eh, 2),
            "elo_away_pre": round(ea, 2),
            "elo_diff":     round(eh - ea, 2),
        }
        for feat in SQ_FEATS:
            out[f"home_{feat}"] = sq_h.get(feat, "")
            out[f"away_{feat}"] = sq_a.get(feat, "")

        rows_out.append(out)

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows_out)

from collections import Counter
dist = Counter(r["resultado"] for r in rows_out)

print(f"OK {OUT_PATH}")
print(f"  Partidos incluidos : {len(rows_out):,}")
print(f"  Partidos omitidos  : {skipped:,}")
print(f"  Resultado H/D/A    : {dist['H']} / {dist['D']} / {dist['A']}")

# Preview de las primeras filas
print("\nPrimeras 5 filas:")
header = ["date","home_team","away_team","resultado","elo_home_pre","elo_away_pre","elo_diff"]
print("  " + "  ".join(f"{h:<22}" for h in header))
for r in rows_out[:5]:
    print("  " + "  ".join(str(r[h])[:22].ljust(22) for h in header))