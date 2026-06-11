"""
Paso 6 — Features de plantilla.
Convocatoria (player_id de football-data.org = player_id de Transfermarkt)
  → join con players.csv
  → agrega por selección

Salida: datos/master/squad_features.csv
  canonical, squad_size, players_matched,
  squad_mv_total, squad_mv_avg, squad_mv_median,
  caps_avg, goals_total, avg_age
"""

import csv
import os
from collections import defaultdict
from datetime import date

CONV_PATH    = "datos/jugadores/convocatoria/convocatoria.csv"
PLAYERS_PATH = "datos/jugadores/transfermarket/players.csv"
MASTER_PATH  = "datos/master/team_name_master.csv"
OUT_PATH     = "datos/master/squad_features.csv"

FECHA_REF = date(2026, 6, 11)   # inicio del Mundial

os.makedirs("datos/master", exist_ok=True)

# fdorg_id → canonical
id_to_canonical: dict[int, str] = {}
with open(MASTER_PATH, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        id_to_canonical[int(row["fdorg_id"])] = row["canonical"]

# Índices de players.csv
players_by_id:   dict[int, dict] = {}
players_by_name: dict[str, dict] = {}   # nombre normalizado → fila

def normalize_name(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()

with open(PLAYERS_PATH, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            players_by_id[int(row["player_id"])] = row
        except ValueError:
            pass
        name_key = normalize_name(row.get("name", ""))
        if name_key:
            players_by_name[name_key] = row

def lookup_player(pid: int, pname: str) -> dict | None:
    p = players_by_id.get(pid)
    if p:
        return p
    return players_by_name.get(normalize_name(pname))

# convocatoria → agrupar por equipo (canonical)
squads: dict[str, list[tuple]] = defaultdict(list)
with open(CONV_PATH, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        tid = int(row["team_id"])
        canonical = id_to_canonical.get(tid, row["team_name"])
        squads[canonical].append((int(row["player_id"]), row["player_name"]))


def to_float(val) -> float | None:
    try:
        v = float(val)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def to_int(val, default=0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def parse_dob(val: str) -> date | None:
    if not val:
        return None
    try:
        return date.fromisoformat(val[:10])
    except ValueError:
        return None


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


rows_out = []
for canonical, squad in sorted(squads.items()):
    mv_vals, caps_vals, goals_vals, ages = [], [], [], []
    n_matched = 0

    for pid, pname in squad:
        p = lookup_player(pid, pname)
        if p is None:
            continue
        n_matched += 1

        mv = to_float(p.get("market_value_in_eur"))
        if mv is not None:
            mv_vals.append(mv)

        caps_vals.append(to_int(p.get("international_caps")))
        goals_vals.append(to_int(p.get("international_goals")))

        dob = parse_dob(p.get("date_of_birth", ""))
        if dob:
            ages.append((FECHA_REF - dob).days / 365.25)

    rows_out.append({
        "canonical":        canonical,
        "squad_size":       len(squad),
        "players_matched":  n_matched,
        "squad_mv_total":   round(sum(mv_vals))              if mv_vals   else "",
        "squad_mv_avg":     round(sum(mv_vals)/len(mv_vals)) if mv_vals   else "",
        "squad_mv_median":  round(median(mv_vals))           if mv_vals   else "",
        "caps_avg":         round(sum(caps_vals)/len(caps_vals), 1) if caps_vals else "",
        "goals_total":      sum(goals_vals)                  if goals_vals else "",
        "avg_age":          round(sum(ages)/len(ages), 1)    if ages      else "",
    })

FIELDS = [
    "canonical", "squad_size", "players_matched",
    "squad_mv_total", "squad_mv_avg", "squad_mv_median",
    "caps_avg", "goals_total", "avg_age",
]
with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows_out)

print(f"OK {OUT_PATH}  ({len(rows_out)} equipos)")
print()
print(f"{'Equipo':<30} {'match':>5}  {'MV_total':>14}  {'caps_avg':>8}  {'edad':>5}")
print("-" * 70)
for r in rows_out:
    mv = f"{r['squad_mv_total']:,}" if r["squad_mv_total"] != "" else "—"
    print(
        f"{r['canonical']:<30} "
        f"{r['players_matched']:>2}/{r['squad_size']:<2}  "
        f"{mv:>14}  "
        f"{str(r['caps_avg']):>8}  "
        f"{str(r['avg_age']):>5}"
    )