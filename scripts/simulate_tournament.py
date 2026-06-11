"""
Paso 8b - Simulacion Monte Carlo del Mundial 2026.

Usa el modelo de goles (goal_model.pkl) y el estado actual de cada
seleccion (team_state_2026.csv) para simular el torneo completo miles
de veces y estimar la probabilidad de cada resultado.

Formato 2026: 48 equipos, 12 grupos de 4.
  - Fase de grupos: liguilla (todos contra todos en el grupo).
  - Avanzan: 1o y 2o de cada grupo (24) + los 8 mejores terceros = 32.
  - Eliminatorias: dieciseisavos -> octavos -> cuartos -> semis -> final.

El cuadro de grupos se lee de datos/master/groups_2026.csv. Si no existe,
se genera un sorteo sembrado por Elo (4 bombos) que DEBES reemplazar por
el sorteo oficial cuando se conozca.

Uso:
  python scripts/simulate_tournament.py [n_simulaciones]
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

MODEL_PATH  = "modelos/goal_model.pkl"
STATE_PATH  = "datos/master/team_state_2026.csv"
GROUPS_PATH = "datos/master/groups_2026.csv"
OUT_PATH    = "datos/master/predicciones_mundial2026.csv"

N_SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. Cargar modelo y estado de los equipos
# ----------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
model    = bundle["model"]
FEATURES = bundle["features"]

state = pd.read_csv(STATE_PATH)
teams = state["team"].tolist()
n = len(teams)
idx = {t: i for i, t in enumerate(teams)}

elo = state["elo"].to_numpy(dtype=float)
mv  = pd.to_numeric(state["squad_mv_total"], errors="coerce").to_numpy(dtype=float)
cap = pd.to_numeric(state["caps_avg"],       errors="coerce").to_numpy(dtype=float)
age = pd.to_numeric(state["avg_age"],        errors="coerce").to_numpy(dtype=float)

# ----------------------------------------------------------------------
# 2. Matriz de goles esperados L[i][j] = goles de i contra j (neutral)
# ----------------------------------------------------------------------
rows = []
for i in range(n):
    for j in range(n):
        rows.append({
            "elo_diff":   elo[i] - elo[j],
            "is_home":    0,                      # todo campo neutral
            "log_mv":     np.log1p(mv[i]),
            "log_mv_opp": np.log1p(mv[j]),
            "caps":       cap[i],
            "caps_opp":   cap[j],
            "age":        age[i],
            "age_opp":    age[j],
        })
L = model.predict(pd.DataFrame(rows)[FEATURES]).reshape(n, n)
L = np.clip(L, 0.05, 6.0)   # acotar lambdas a un rango razonable

# ----------------------------------------------------------------------
# 3. Cuadro de grupos (genera un sorteo sembrado por Elo si no existe)
# ----------------------------------------------------------------------
def generar_sorteo_sembrado():
    orden = sorted(range(n), key=lambda i: -elo[i])
    pots = [orden[k*12:(k+1)*12] for k in range(4)]
    for p in pots:
        RNG.shuffle(p)
    grupos = "ABCDEFGHIJKL"
    filas = []
    for g in range(12):
        for pot in pots:
            filas.append({"group": grupos[g], "team": teams[pot[g]]})
    df = pd.DataFrame(filas)
    df.to_csv(GROUPS_PATH, index=False)
    print(f"  (sorteo sembrado por Elo generado en {GROUPS_PATH} - "
          f"reemplazar por el sorteo oficial)")
    return df

if os.path.exists(GROUPS_PATH):
    groups_df = pd.read_csv(GROUPS_PATH)
else:
    groups_df = generar_sorteo_sembrado()

groups = {}
for g, sub in groups_df.groupby("group"):
    groups[g] = [idx[t] for t in sub["team"]]

# ----------------------------------------------------------------------
# 4. Utilidades de simulacion
# ----------------------------------------------------------------------
def jugar(i, j):
    """Devuelve (goles_i, goles_j) muestreando dos Poisson independientes."""
    return RNG.poisson(L[i, j]), RNG.poisson(L[j, i])

def penaltis(i, j):
    """Desempate por penaltis ponderado por Elo. Devuelve el ganador."""
    p_i = 1.0 / (1.0 + 10 ** ((elo[j] - elo[i]) / 400.0))
    return i if RNG.random() < p_i else j

def ganador_ko(i, j):
    gi, gj = jugar(i, j)
    if gi > gj: return i
    if gj > gi: return j
    return penaltis(i, j)

def bracket_seed_order(size):
    """Orden de siembra estandar (1 vs size, mitades equilibradas)."""
    order = [1]
    while len(order) < size:
        m = len(order) * 2 + 1
        order = [x for e in order for x in (e, m - e)]
    return order

SEED_ORDER_32 = bracket_seed_order(32)

# ----------------------------------------------------------------------
# 5. Una simulacion completa del torneo
# ----------------------------------------------------------------------
def simular_torneo():
    terceros = []          # (idx, pts, gd, gf) de cada 3o de grupo
    clasificados = {}      # idx -> status (0=1o, 1=2o, 2=3o)
    seed_stat = {}         # idx -> (status, pts, gd, gf) para sembrar el cuadro

    for g, eqs in groups.items():
        pts = {e: 0 for e in eqs}
        gf  = {e: 0 for e in eqs}
        ga  = {e: 0 for e in eqs}
        for a in range(4):
            for b in range(a + 1, 4):
                i, j = eqs[a], eqs[b]
                gi, gj = jugar(i, j)
                gf[i] += gi; ga[i] += gj
                gf[j] += gj; ga[j] += gi
                if gi > gj:   pts[i] += 3
                elif gj > gi: pts[j] += 3
                else:         pts[i] += 1; pts[j] += 1
        # ordenar: pts, dif de goles, goles a favor, elo
        rank = sorted(eqs, key=lambda e: (pts[e], gf[e] - ga[e], gf[e], elo[e]),
                      reverse=True)
        for pos, e in enumerate(rank):
            if pos < 2:
                clasificados[e] = pos
                seed_stat[e] = (pos, pts[e], gf[e] - ga[e], gf[e])
            elif pos == 2:
                terceros.append((e, pts[e], gf[e] - ga[e], gf[e]))

    # 8 mejores terceros
    terceros.sort(key=lambda t: (t[1], t[2], t[3], elo[t[0]]), reverse=True)
    for e, p, gd, gfv in terceros[:8]:
        clasificados[e] = 2
        seed_stat[e] = (2, p, gd, gfv)

    qualifiers = list(clasificados.keys())   # 32 equipos

    # sembrar 1..32: 1os mejor que 2os mejor que 3os; luego pts, dif, gf, elo
    ranked = sorted(qualifiers,
                    key=lambda e: (seed_stat[e][0],
                                   -seed_stat[e][1],
                                   -seed_stat[e][2],
                                   -seed_stat[e][3],
                                   -elo[e]))
    seed_to_team = {s: ranked[s - 1] for s in range(1, 33)}
    ko = [seed_to_team[s] for s in SEED_ORDER_32]   # orden del cuadro

    # rondas: 32 -> 16 -> 8 -> 4 -> 2 -> 1
    ronda_alcanzada = {e: "grupos" for e in range(n)}
    for e in qualifiers:
        ronda_alcanzada[e] = "R32"

    etiquetas = ["R32", "R16", "QF", "SF", "F"]
    nivel = 0
    while len(ko) > 1:
        siguiente = []
        for k in range(0, len(ko), 2):
            w = ganador_ko(ko[k], ko[k + 1])
            siguiente.append(w)
            ronda_alcanzada[w] = etiquetas[min(nivel + 1, len(etiquetas) - 1)]
        ko = siguiente
        nivel += 1
    campeon = ko[0]
    ronda_alcanzada[campeon] = "Campeon"
    return campeon, qualifiers, ronda_alcanzada

# ----------------------------------------------------------------------
# 6. Monte Carlo
# ----------------------------------------------------------------------
print(f"Simulando {N_SIMS:,} torneos...")
camp   = np.zeros(n)
finals = np.zeros(n)
semis  = np.zeros(n)
cuartos = np.zeros(n)
octavos = np.zeros(n)
r32    = np.zeros(n)

ORD = {"grupos": 0, "R32": 1, "R16": 2, "QF": 3, "SF": 4, "F": 5, "Campeon": 6}

for s in range(N_SIMS):
    campeon, quals, ronda = simular_torneo()
    camp[campeon] += 1
    for e in range(n):
        lvl = ORD[ronda[e]]
        if lvl >= 1: r32[e]     += 1
        if lvl >= 2: octavos[e] += 1
        if lvl >= 3: cuartos[e] += 1
        if lvl >= 4: semis[e]   += 1
        if lvl >= 5: finals[e]  += 1

# ----------------------------------------------------------------------
# 7. Resultados
# ----------------------------------------------------------------------
res = pd.DataFrame({
    "team":      teams,
    "elo":       np.round(elo, 1),
    "campeon_%": np.round(100 * camp    / N_SIMS, 2),
    "final_%":   np.round(100 * finals  / N_SIMS, 2),
    "semis_%":   np.round(100 * semis   / N_SIMS, 2),
    "cuartos_%": np.round(100 * cuartos / N_SIMS, 2),
    "octavos_%": np.round(100 * octavos / N_SIMS, 2),
    "pasa_grupo_%": np.round(100 * r32 / N_SIMS, 2),
}).sort_values("campeon_%", ascending=False).reset_index(drop=True)

res.to_csv(OUT_PATH, index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(f"\n=== Probabilidades Mundial 2026 ({N_SIMS:,} simulaciones) ===\n")
print(res.head(20).to_string(index=False))
print(f"\nOK {OUT_PATH}")