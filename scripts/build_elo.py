"""
Paso 5 — Elo dinámico.
Itera results.csv cronológicamente y calcula el Elo de cada equipo partido a partido.
Guarda el Elo PRE-partido como feature (no el posterior).

K-factors:
  40 — FIFA World Cup (fase final)
  30 — eliminatorias, torneos continentales
  20 — amistosos y el resto
"""

import csv
import os
from collections import defaultdict

RESULTS_PATH = (
    "datos/historico de partidos/"
    "[International Football Results from 1872 to 2026/results.csv"
)
OUT_PATH = "datos/master/elo_por_partido.csv"

ELO_INICIO = 1500
K_MUNDIAL  = 40
K_COMP     = 30   # competición seria
K_AMISTOSO = 20

MUNDIAL_EXACT = {"FIFA World Cup"}

COMP_KEYWORDS = [
    "qualif", "qualifier",          # eliminatorias de cualquier torneo
    "cup of nations",               # AFCON y variantes
    "copa am",                      # Copa América (con/sin tilde)
    "gold cup",                     # CONCACAF Gold Cup
    "nations league",               # UEFA Nations League
    "asian cup",                    # AFC Asian Cup
    "confederation",                # Confederations Cup / CONMEBOL etc.
    "olympic",                      # Juegos Olímpicos
    "world cup",                    # captura variantes no-FIFA
    "championship",                 # UEFA Euro, etc.
]


def k_factor(tournament: str) -> int:
    if tournament in MUNDIAL_EXACT:
        return K_MUNDIAL
    t = tournament.lower()
    if any(kw in t for kw in COMP_KEYWORDS):
        return K_COMP
    return K_AMISTOSO


def expected(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def puntos(home_score: int, away_score: int):
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


os.makedirs("datos/master", exist_ok=True)

elo: dict[str, float] = defaultdict(lambda: ELO_INICIO)

with open(RESULTS_PATH, encoding="utf-8") as f:
    partidos = list(csv.DictReader(f))

partidos.sort(key=lambda r: r["date"])

rows_out = []
for row in partidos:
    home = row["home_team"]
    away = row["away_team"]
    try:
        hs  = int(row["home_score"])
        as_ = int(row["away_score"])
    except (ValueError, KeyError):
        continue

    tournament = row.get("tournament", "")
    K = k_factor(tournament)

    eh_pre = elo[home]
    ea_pre = elo[away]

    exp_h = expected(eh_pre, ea_pre)
    exp_a = 1.0 - exp_h
    res_h, res_a = puntos(hs, as_)

    elo[home] = eh_pre + K * (res_h - exp_h)
    elo[away] = ea_pre + K * (res_a - exp_a)

    rows_out.append({
        "date":          row["date"],
        "home_team":     home,
        "away_team":     away,
        "home_score":    hs,
        "away_score":    as_,
        "tournament":    tournament,
        "K":             K,
        "elo_home_pre":  round(eh_pre,    2),
        "elo_away_pre":  round(ea_pre,    2),
        "elo_home_post": round(elo[home], 2),
        "elo_away_post": round(elo[away], 2),
    })

FIELDS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "K",
    "elo_home_pre", "elo_away_pre",
    "elo_home_post", "elo_away_post",
]
with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows_out)

print(f"OK {OUT_PATH}  ({len(rows_out):,} partidos)")

top20 = sorted(elo.items(), key=lambda x: -x[1])[:20]
print("\nTop 20 Elo final:")
for i, (team, e) in enumerate(top20, 1):
    print(f"  {i:2}. {team:<30} {round(e, 1)}")