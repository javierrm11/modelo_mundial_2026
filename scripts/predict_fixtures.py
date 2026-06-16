"""
Prediccion partido a partido del Mundial 2026.

Lee un CSV de partidos (datos/master/fixtures.csv por defecto) y, usando el
modelo de goles (goal_model.pkl) y el estado actual de cada seleccion
(team_state_2026.csv), predice cada partido:

  - 1X2 (local / empate / visitante)
  - goles esperados (lambda) de cada equipo
  - los 5 marcadores mas probables
  - en partidos de eliminatoria: P(pasa de ronda) resolviendo el empate
    con prorroga + penaltis ponderados por Elo

Uso:
  python scripts/predict_fixtures.py                  # lee datos/master/fixtures.csv
  python scripts/predict_fixtures.py otro_fixtures.csv
"""

import os
import sys
import csv
import pickle
import unicodedata
import difflib
from math import exp, factorial
import numpy as np
import pandas as pd

MODEL_PATH   = "modelos/goal_model.pkl"
STATE_PATH   = "datos/master/team_state_2026.csv"
DEFAULT_FIX  = "datos/master/fixtures.csv"
OUT_PATH     = "datos/master/predicciones_partidos.csv"

MAX_GOLES = 10
CLIP_LO, CLIP_HI = 0.05, 6.0
FASES_KO = {"eliminatoria", "ko", "knockout", "elim"}
ANFITRIONES = {"Mexico", "United States", "Canada"}   # sedes del Mundial 2026
NEUTRAL_TOKENS = {"", "neutral", "no", "none", "-"}

# ----------------------------------------------------------------------
# Carga de modelo y estado de equipos
# ----------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
model      = bundle["model"]
FEATURES   = bundle["features"]
CALIBRATOR = bundle.get("calibrator")   # Platt multiclase (puede faltar en pkl viejos)
RHO        = bundle.get("rho", 0.0)     # parametro Dixon-Coles

state = pd.read_csv(STATE_PATH)
ST = {}
for _, r in state.iterrows():
    ST[r["team"]] = {
        "elo": float(r["elo"]),
        "mv":  float(r["squad_mv_total"]) if pd.notna(r["squad_mv_total"]) else np.nan,
        "caps": float(r["caps_avg"])      if pd.notna(r["caps_avg"])       else np.nan,
        "age": float(r["avg_age"])        if pd.notna(r["avg_age"])        else np.nan,
    }
CANONICOS = list(ST.keys())

# ----------------------------------------------------------------------
# Resolucion de nombres (alias en espanol + sin tildes + sugerencia)
# ----------------------------------------------------------------------
ALIAS = {
    "espana": "Spain", "spain": "Spain",
    "brasil": "Brazil", "brazil": "Brazil",
    "estados unidos": "United States", "usa": "United States", "eeuu": "United States",
    "corea del sur": "South Korea", "korea del sur": "South Korea",
    "costa de marfil": "Ivory Coast",
    "paises bajos": "Netherlands", "holanda": "Netherlands",
    "chequia": "Czech Republic", "republica checa": "Czech Republic",
    "arabia saudi": "Saudi Arabia", "arabia saudita": "Saudi Arabia",
    "catar": "Qatar", "qatar": "Qatar",
    "rd congo": "DR Congo", "republica democratica del congo": "DR Congo",
    "alemania": "Germany", "francia": "France", "inglaterra": "England",
    "belgica": "Belgium", "croacia": "Croatia", "suiza": "Switzerland",
    "tunez": "Tunisia", "turquia": "Turkey", "sudafrica": "South Africa",
    "egipto": "Egypt", "japon": "Japan", "mexico": "Mexico", "noruega": "Norway",
    "panama": "Panama", "escocia": "Scotland", "cabo verde": "Cape Verde",
    "curazao": "Curaçao", "jordania": "Jordan", "irak": "Iraq", "iran": "Iran",
    "uzbekistan": "Uzbekistan", "haiti": "Haiti", "nueva zelanda": "New Zealand",
    "bosnia y herzegovina": "Bosnia and Herzegovina", "suecia": "Sweden",
    "marruecos": "Morocco", "argelia": "Algeria", "ecuador": "Ecuador",
    "colombia": "Colombia", "uruguay": "Uruguay", "paraguay": "Paraguay",
    "australia": "Australia", "austria": "Austria", "ghana": "Ghana",
    "senegal": "Senegal", "portugal": "Portugal", "argentina": "Argentina",
}

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.strip().lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

# indice normalizado de los nombres canonicos
_CANON_NORM = {_norm(t): t for t in CANONICOS}

def resolver(nombre: str) -> str:
    key = _norm(nombre)
    if key in _CANON_NORM:
        return _CANON_NORM[key]
    if key in ALIAS:
        return ALIAS[key]
    # sugerencia
    cand = difflib.get_close_matches(key, list(_CANON_NORM.keys()), n=1, cutoff=0.6)
    sug = f" Quizas quisiste decir '{_CANON_NORM[cand[0]]}'." if cand else ""
    raise ValueError(f"Equipo no reconocido: '{nombre}'.{sug}")

# ----------------------------------------------------------------------
# Modelo de goles -> lambdas -> matriz de marcadores
# ----------------------------------------------------------------------
def construir_features(home: str, away: str, local=None):
    """local = equipo con ventaja de campo (o None si neutral)."""
    h, a = ST[home], ST[away]
    fila_h = {
        "elo_diff": h["elo"] - a["elo"],
        "is_home": 1 if local == home else 0,
        "log_mv": np.log1p(h["mv"]), "log_mv_opp": np.log1p(a["mv"]),
        "caps": h["caps"], "caps_opp": a["caps"],
        "age": h["age"], "age_opp": a["age"],
    }
    fila_a = {
        "elo_diff": a["elo"] - h["elo"],
        "is_home": 1 if local == away else 0,
        "log_mv": np.log1p(a["mv"]), "log_mv_opp": np.log1p(h["mv"]),
        "caps": a["caps"], "caps_opp": h["caps"],
        "age": a["age"], "age_opp": h["age"],
    }
    return pd.DataFrame([fila_h, fila_a])[FEATURES]

def predecir_lambdas(home: str, away: str, local=None):
    pred = model.predict(construir_features(home, away, local))
    lh, la = float(pred[0]), float(pred[1])
    return float(np.clip(lh, CLIP_LO, CLIP_HI)), float(np.clip(la, CLIP_LO, CLIP_HI))

def auto_local(home: str, away: str):
    """Si solo uno de los dos es anfitrion, le da ventaja de campo."""
    anfitriones = {home, away} & ANFITRIONES
    return next(iter(anfitriones)) if len(anfitriones) == 1 else None

def determinar_local(home: str, away: str, valor):
    """Decide quien tiene ventaja de campo a partir de la columna 'local':
       - vacia o ausente -> automatico (anfitrion del Mundial 2026).
       - 'neutral'/'no'   -> fuerza campo neutral.
       - un equipo        -> ese equipo es local.
    """
    s = None if valor is None else str(valor).strip().lower()
    if s in (None, ""):
        return auto_local(home, away)
    if s in NEUTRAL_TOKENS:
        return None
    try:
        loc = resolver(valor)
        if loc in (home, away):
            return loc
        print(f"    (aviso: 'local={valor}' no juega este partido; se trata como neutral)")
    except ValueError:
        print(f"    (aviso: 'local={valor}' no reconocido; se trata como neutral)")
    return None

def poisson_pmf(lmbda, k):
    return exp(-lmbda) * lmbda**k / factorial(k)

def tau_dc(x, y, lh, la, rho):
    """Factor de correccion de Dixon-Coles para los 4 marcadores bajos."""
    if x == 0 and y == 0: return 1.0 - lh * la * rho
    if x == 0 and y == 1: return 1.0 + lh * rho
    if x == 1 and y == 0: return 1.0 + la * rho
    if x == 1 and y == 1: return 1.0 - rho
    return 1.0

def matriz_marcadores(lh, la, maxg=MAX_GOLES, rho=RHO):
    ph = np.array([poisson_pmf(lh, k) for k in range(maxg + 1)])
    pa = np.array([poisson_pmf(la, k) for k in range(maxg + 1)])
    m = np.outer(ph, pa)
    if rho != 0.0:
        for x in (0, 1):
            for y in (0, 1):
                m[x, y] *= tau_dc(x, y, lh, la, rho)
    return m / m.sum()

def probs_1x2(m):
    p_home = np.tril(m, -1).sum()
    p_draw = np.trace(m)
    p_away = np.triu(m, 1).sum()
    return p_home, p_draw, p_away

def calibrar(ph, pd_, pa):
    """Aplica el calibrador de Platt al 1X2 crudo. Sin calibrador, lo deja igual."""
    if CALIBRATOR is None:
        return ph, pd_, pa
    P = np.array([[ph, pd_, pa]])
    c = CALIBRATOR.predict_proba(np.log(np.clip(P, 1e-6, 1.0)))[0]
    return float(c[0]), float(c[1]), float(c[2])

def top_marcadores(m, n=5):
    idx = np.dstack(np.unravel_index(np.argsort(m.ravel())[::-1], m.shape))[0]
    return [((int(i), int(j)), float(m[i, j])) for i, j in idx[:n]]

# ----------------------------------------------------------------------
# Eliminatoria: P(home pasa) con prorroga (lambda*1/3) + penaltis por Elo
# ----------------------------------------------------------------------
def p_penaltis_home(home, away):
    return 1.0 / (1.0 + 10 ** ((ST[away]["elo"] - ST[home]["elo"]) / 400.0))

def resolver_eliminatoria(home, away, lh, la, reg_probs):
    ph, pdraw, pa = reg_probs                      # 1X2 reglamentario ya calibrado
    # prorroga: 30 min ~ 1/3 del partido
    eh, ed, ea = probs_1x2(matriz_marcadores(lh / 3.0, la / 3.0))
    p_pen = p_penaltis_home(home, away)
    p_home_pasa = ph + pdraw * (eh + ed * p_pen)
    return p_home_pasa, 1.0 - p_home_pasa

# ----------------------------------------------------------------------
# Procesar el fichero de fixtures
# ----------------------------------------------------------------------
def main():
    fix_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FIX
    if not os.path.exists(fix_path):
        print(f"No existe el fichero de fixtures: {fix_path}")
        sys.exit(1)

    with open(fix_path, encoding="utf-8") as f:
        partidos = list(csv.DictReader(f))

    salida = []
    print(f"\n=== Predicciones ({len(partidos)} partidos) ===")

    for p in partidos:
        try:
            home = resolver(p["home_team"])
            away = resolver(p["away_team"])
        except ValueError as e:
            print(f"\n  [SALTADO] {e}")
            continue

        fase = (p.get("fase") or "grupo").strip().lower()
        es_ko = fase in FASES_KO
        local = determinar_local(home, away, p.get("local"))

        lh, la = predecir_lambdas(home, away, local)
        m = matriz_marcadores(lh, la)
        ph, pdraw, pa = calibrar(*probs_1x2(m))      # 1X2 calibrado
        tops = top_marcadores(m, 5)

        sede = f"local: {local}" if local else "neutral"
        print(f"\n  {home}  vs  {away}   [{'eliminatoria' if es_ko else 'grupo'} · {sede}]")
        print(f"    1X2:  {home} {ph*100:5.1f}%  |  Empate {pdraw*100:5.1f}%  |  {away} {pa*100:5.1f}%")
        print(f"    Goles esperados:  {lh:.2f} - {la:.2f}")
        marc = "  ".join(f"{i}-{j} ({pr*100:.1f}%)" for (i, j), pr in tops)
        print(f"    Marcadores probables:  {marc}")

        fila = {
            "home_team": home, "away_team": away, "fase": fase,
            "local": local or "neutral",
            "lambda_home": round(lh, 3), "lambda_away": round(la, 3),
            "p_home": round(ph, 4), "p_draw": round(pdraw, 4), "p_away": round(pa, 4),
            "marcador_top1": f"{tops[0][0][0]}-{tops[0][0][1]}",
            "p_pasa_home": "", "p_pasa_away": "",
        }

        if es_ko:
            ph_pasa, pa_pasa = resolver_eliminatoria(home, away, lh, la, (ph, pdraw, pa))
            print(f"    Eliminatoria:  {home} pasa {ph_pasa*100:5.1f}%  |  {away} pasa {pa_pasa*100:5.1f}%")
            fila["p_pasa_home"] = round(ph_pasa, 4)
            fila["p_pasa_away"] = round(pa_pasa, 4)

        salida.append(fila)

    if salida:
        campos = ["home_team", "away_team", "fase", "local",
                  "lambda_home", "lambda_away",
                  "p_home", "p_draw", "p_away", "marcador_top1",
                  "p_pasa_home", "p_pasa_away"]
        with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(salida)
        print(f"\nOK {OUT_PATH}  ({len(salida)} partidos)")

if __name__ == "__main__":
    main()
