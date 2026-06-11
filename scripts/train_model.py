"""
Paso 8a - Modelo de goles (Poisson).

Entrena un modelo que predice los goles esperados (lambda) de un equipo
contra otro, dado Elo y features de plantilla. A partir de dos lambdas
(local y visitante) se deriva cualquier marcador y el 1X2.

  - Formato largo: cada partido -> 2 filas (una por equipo que marca).
  - Modelo: LightGBM con objetivo 'poisson'.
  - Validacion: split temporal (entrena < CORTE, valida >= CORTE).

Salidas:
  modelos/goal_model.pkl          - modelo + metadatos
  datos/master/team_state_2026.csv - Elo actual + plantilla por seleccion
"""

import os
import pickle
import numpy as np
import pandas as pd
from math import exp, factorial
import lightgbm as lgb

DATASET_PATH = "datos/master/dataset_final.csv"
ELO_PATH     = "datos/master/elo_por_partido.csv"
SQUAD_PATH   = "datos/master/squad_features.csv"
MODEL_PATH   = "modelos/goal_model.pkl"
STATE_PATH   = "datos/master/team_state_2026.csv"

CORTE_VALID = "2018-01-01"   # partidos >= esta fecha = validacion
MAX_GOLES   = 10             # truncado para la matriz de marcadores

os.makedirs("modelos", exist_ok=True)

# ----------------------------------------------------------------------
# 1. Cargar y construir formato largo (perspectiva del equipo que marca)
# ----------------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)
df["date"] = pd.to_datetime(df["date"])

def lado(df, scorer):
    """scorer = 'home' o 'away'. Devuelve filas desde la perspectiva del que marca."""
    opp = "away" if scorer == "home" else "home"
    out = pd.DataFrame()
    out["date"]        = df["date"]
    out["goals"]       = df[f"{scorer}_score"]
    out["elo"]         = df[f"elo_{scorer}_pre"]
    out["elo_opp"]     = df[f"elo_{opp}_pre"]
    out["elo_diff"]    = df[f"elo_{scorer}_pre"] - df[f"elo_{opp}_pre"]
    # ventaja de campo: 1 si este equipo juega en casa y NO es campo neutral
    es_neutral = df["neutral"].astype(str).str.upper().eq("TRUE")
    out["is_home"]     = ((scorer == "home") & ~es_neutral).astype(int)
    # plantilla (log del valor de mercado para comprimir la escala)
    out["log_mv"]      = np.log1p(pd.to_numeric(df[f"{scorer}_squad_mv_total"], errors="coerce"))
    out["log_mv_opp"]  = np.log1p(pd.to_numeric(df[f"{opp}_squad_mv_total"], errors="coerce"))
    out["caps"]        = pd.to_numeric(df[f"{scorer}_caps_avg"], errors="coerce")
    out["caps_opp"]    = pd.to_numeric(df[f"{opp}_caps_avg"], errors="coerce")
    out["age"]         = pd.to_numeric(df[f"{scorer}_avg_age"], errors="coerce")
    out["age_opp"]     = pd.to_numeric(df[f"{opp}_avg_age"], errors="coerce")
    return out

long_df = pd.concat([lado(df, "home"), lado(df, "away")], ignore_index=True)
long_df = long_df.dropna(subset=["goals"])

FEATURES = [
    "elo_diff", "is_home",
    "log_mv", "log_mv_opp",
    "caps", "caps_opp",
    "age", "age_opp",
]

train = long_df[long_df["date"] < CORTE_VALID]
valid = long_df[long_df["date"] >= CORTE_VALID]

print(f"Filas entrenamiento: {len(train):,}")
print(f"Filas validacion   : {len(valid):,}")

# ----------------------------------------------------------------------
# 2. Entrenar LightGBM Poisson
# ----------------------------------------------------------------------
model = lgb.LGBMRegressor(
    objective="poisson",
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)
model.fit(
    train[FEATURES], train["goals"],
    eval_set=[(valid[FEATURES], valid["goals"])],
    eval_metric="poisson",
    callbacks=[lgb.early_stopping(40, verbose=False)],
)

# ----------------------------------------------------------------------
# 3. Validacion derivando el 1X2 sobre los partidos de validacion
# ----------------------------------------------------------------------
def poisson_pmf(lmbda, k):
    return exp(-lmbda) * lmbda**k / factorial(k)

def matriz_1x2(lh, la):
    """P(local), P(empate), P(visitante) con dos Poisson independientes."""
    ph = np.array([poisson_pmf(lh, k) for k in range(MAX_GOLES + 1)])
    pa = np.array([poisson_pmf(la, k) for k in range(MAX_GOLES + 1)])
    m = np.outer(ph, pa)
    p_home = np.tril(m, -1).sum()
    p_draw = np.trace(m)
    p_away = np.triu(m, 1).sum()
    s = p_home + p_draw + p_away
    return p_home / s, p_draw / s, p_away / s

# reconstruir los partidos de validacion en formato partido (no largo)
val_matches = df[df["date"] >= CORTE_VALID].copy()
Xh = lado(val_matches, "home")[FEATURES]
Xa = lado(val_matches, "away")[FEATURES]
val_matches["lh"] = model.predict(Xh)
val_matches["la"] = model.predict(Xa)

eps = 1e-12
logloss, brier, aciertos, n = 0.0, 0.0, 0, 0
for _, r in val_matches.iterrows():
    ph, pd_, pa = matriz_1x2(r["lh"], r["la"])
    probs = {"H": ph, "D": pd_, "A": pa}
    y = r["resultado"]
    logloss += -np.log(max(probs[y], eps))
    brier   += sum((probs[k] - (1.0 if k == y else 0.0))**2 for k in probs)
    if max(probs, key=probs.get) == y:
        aciertos += 1
    n += 1

print("\n--- Validacion 1X2 (derivado del modelo de goles) ---")
print(f"  Partidos     : {n}")
print(f"  Log-loss     : {logloss / n:.4f}")
print(f"  Brier score  : {brier / n:.4f}")
print(f"  Accuracy     : {aciertos / n:.3f}")

# baseline tonto: clase mayoritaria
maj = val_matches["resultado"].value_counts(normalize=True)
print(f"  (baseline clase mayoritaria acc={maj.max():.3f}, "
      f"log-loss uniforme={np.log(3):.4f})")

print("\nImportancia de features:")
for f, imp in sorted(zip(FEATURES, model.feature_importances_),
                     key=lambda x: -x[1]):
    print(f"  {f:<14} {imp}")

# ----------------------------------------------------------------------
# 4. Guardar modelo + estado actual de cada seleccion (para simular 2026)
# ----------------------------------------------------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": model, "features": FEATURES, "max_goles": MAX_GOLES}, f)
print(f"\nOK {MODEL_PATH}")

# Elo actual = elo_post del partido mas reciente de cada equipo
elo = pd.read_csv(ELO_PATH, parse_dates=["date"])
home_e = elo[["date", "home_team", "elo_home_post"]].rename(
    columns={"home_team": "team", "elo_home_post": "elo"})
away_e = elo[["date", "away_team", "elo_away_post"]].rename(
    columns={"away_team": "team", "elo_away_post": "elo"})
elo_long = pd.concat([home_e, away_e], ignore_index=True)
latest = (elo_long.sort_values("date")
                  .groupby("team", as_index=False)
                  .last()[["team", "elo"]])

squad = pd.read_csv(SQUAD_PATH).rename(columns={"canonical": "team"})
state = squad.merge(latest, on="team", how="left")
state = state[["team", "elo", "squad_mv_total", "caps_avg", "avg_age", "goals_total"]]
state.to_csv(STATE_PATH, index=False)
print(f"OK {STATE_PATH}  ({len(state)} selecciones)")