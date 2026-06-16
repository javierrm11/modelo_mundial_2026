"""
Paso 8a - Modelo de goles (Poisson) + calibracion probabilistica.

Entrena un modelo que predice los goles esperados (lambda) de un equipo
contra otro, dado Elo y features de plantilla. A partir de dos lambdas
(local y visitante) se deriva cualquier marcador y el 1X2.

  - Formato largo: cada partido -> 2 filas (una por equipo que marca).
  - Modelo: LightGBM con objetivo 'poisson'.
  - Validacion: split temporal (entrena < CORTE, valida >= CORTE).
  - Calibracion: el 1X2 derivado de dos Poisson independientes suele estar
    mal calibrado (infraestima los empates). Se ajusta un calibrador de
    Platt multiclase (regresion logistica sobre los log-probs) usando
    predicciones OUT-OF-FOLD con validacion cruzada temporal, de modo que
    el calibrador nunca ve datos que el modelo uso para entrenar.

Salidas:
  modelos/goal_model.pkl           - modelo + calibrador + metadatos
  datos/master/team_state_2026.csv - Elo actual + plantilla por seleccion
"""

import os
import pickle
import numpy as np
import pandas as pd
from math import exp, factorial
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

DATASET_PATH = "datos/master/dataset_final.csv"
ELO_PATH     = "datos/master/elo_por_partido.csv"
SQUAD_PATH   = "datos/master/squad_features.csv"
MODEL_PATH   = "modelos/goal_model.pkl"
STATE_PATH   = "datos/master/team_state_2026.csv"

CORTE_VALID = "2018-01-01"   # partidos >= esta fecha = validacion
MAX_GOLES   = 10             # truncado para la matriz de marcadores
N_FOLDS     = 5              # folds temporales para la calibracion OOF
CLIP_LO, CLIP_HI = 0.05, 6.0
RES2IDX = {"H": 0, "D": 1, "A": 2}   # orden de clases en el 1X2

os.makedirs("modelos", exist_ok=True)

# ----------------------------------------------------------------------
# 1. Cargar dataset
# ----------------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)
df["date"] = pd.to_datetime(df["date"])

FEATURES = [
    "elo_diff", "is_home",
    "log_mv", "log_mv_opp",
    "caps", "caps_opp",
    "age", "age_opp",
]

def lado(d, scorer):
    """scorer = 'home'/'away'. Filas desde la perspectiva del que marca."""
    opp = "away" if scorer == "home" else "home"
    out = pd.DataFrame()
    out["date"]    = d["date"]
    out["goals"]   = d[f"{scorer}_score"]
    out["elo_diff"] = d[f"elo_{scorer}_pre"] - d[f"elo_{opp}_pre"]
    es_neutral = d["neutral"].astype(str).str.upper().eq("TRUE")
    out["is_home"] = ((scorer == "home") & ~es_neutral).astype(int)
    out["log_mv"]     = np.log1p(pd.to_numeric(d[f"{scorer}_squad_mv_total"], errors="coerce"))
    out["log_mv_opp"] = np.log1p(pd.to_numeric(d[f"{opp}_squad_mv_total"],    errors="coerce"))
    out["caps"]     = pd.to_numeric(d[f"{scorer}_caps_avg"], errors="coerce")
    out["caps_opp"] = pd.to_numeric(d[f"{opp}_caps_avg"],    errors="coerce")
    out["age"]      = pd.to_numeric(d[f"{scorer}_avg_age"],  errors="coerce")
    out["age_opp"]  = pd.to_numeric(d[f"{opp}_avg_age"],     errors="coerce")
    return out

def long_from_wide(wide):
    """Convierte partidos (formato ancho) en formato largo de entrenamiento."""
    return pd.concat([lado(wide, "home"), lado(wide, "away")],
                     ignore_index=True).dropna(subset=["goals"])

# ----------------------------------------------------------------------
# 2. Modelo de goles y derivacion del 1X2
# ----------------------------------------------------------------------
def make_model():
    return lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=350,
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

def entrenar(wide):
    m = make_model()
    long = long_from_wide(wide)
    m.fit(long[FEATURES], long["goals"])
    return m

def poisson_pmf(lmbda, k):
    return exp(-lmbda) * lmbda**k / factorial(k)

def tau_dc(x, y, lh, la, rho):
    """Factor de correccion de Dixon-Coles para los 4 marcadores bajos."""
    if x == 0 and y == 0: return 1.0 - lh * la * rho
    if x == 0 and y == 1: return 1.0 + lh * rho
    if x == 1 and y == 0: return 1.0 + la * rho
    if x == 1 and y == 1: return 1.0 - rho
    return 1.0

def matriz_marcadores(lh, la, rho=0.0):
    """Matriz de marcadores (filas=local, cols=visitante) con correccion DC."""
    ph = np.array([poisson_pmf(lh, k) for k in range(MAX_GOLES + 1)])
    pa = np.array([poisson_pmf(la, k) for k in range(MAX_GOLES + 1)])
    m = np.outer(ph, pa)
    if rho != 0.0:
        for x in (0, 1):
            for y in (0, 1):
                m[x, y] *= tau_dc(x, y, lh, la, rho)
    return m / m.sum()

def matriz_1x2(lh, la, rho=0.0):
    """P(local), P(empate), P(visitante) a partir de la matriz de marcadores."""
    m = matriz_marcadores(lh, la, rho)
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()

def predict_1x2(model, wide, rho=0.0):
    """Devuelve P[N,3] (H,D,A) para un conjunto de partidos en formato ancho."""
    lh = np.clip(model.predict(lado(wide, "home")[FEATURES]), CLIP_LO, CLIP_HI)
    la = np.clip(model.predict(lado(wide, "away")[FEATURES]), CLIP_LO, CLIP_HI)
    return np.array([matriz_1x2(a, b, rho) for a, b in zip(lh, la)])

def estimar_rho(model, wide):
    """Estima rho (Dixon-Coles) por maxima verosimilitud sobre los marcadores bajos."""
    lh = np.clip(model.predict(lado(wide, "home")[FEATURES]), CLIP_LO, CLIP_HI)
    la = np.clip(model.predict(lado(wide, "away")[FEATURES]), CLIP_LO, CLIP_HI)
    xs = wide["home_score"].to_numpy()
    ys = wide["away_score"].to_numpy()
    mask = (xs <= 1) & (ys <= 1)
    cx, cy, cl, cm = xs[mask], ys[mask], lh[mask], la[mask]

    def neg_loglik(rho):
        t = np.ones(len(cx))
        t = np.where((cx == 0) & (cy == 0), 1.0 - cl * cm * rho, t)
        t = np.where((cx == 0) & (cy == 1), 1.0 + cl * rho, t)
        t = np.where((cx == 1) & (cy == 0), 1.0 + cm * rho, t)
        t = np.where((cx == 1) & (cy == 1), 1.0 - rho, t)
        if np.any(t <= 0):
            return 1e9
        return -np.sum(np.log(t))

    res = minimize_scalar(neg_loglik, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)

# ----------------------------------------------------------------------
# 3. Calibracion (Platt multiclase sobre los log-probabilidades)
# ----------------------------------------------------------------------
def fit_calibrador(P_raw, y):
    cal = LogisticRegression(C=1.0, max_iter=2000)
    cal.fit(np.log(np.clip(P_raw, 1e-6, 1.0)), y)
    return cal

def aplicar_calibrador(cal, P_raw):
    return cal.predict_proba(np.log(np.clip(P_raw, 1e-6, 1.0)))

def calibrador_oof(wide_train, rho=0.0, n_folds=N_FOLDS):
    """Predicciones out-of-fold con CV temporal -> calibrador sin fuga."""
    w = wide_train.sort_values("date").reset_index(drop=True)
    idx = np.array_split(np.arange(len(w)), n_folds)
    P_list, y_list = [], []
    for i in range(1, n_folds):                       # ventana expansiva
        tr = w.iloc[np.concatenate(idx[:i])]
        va = w.iloc[idx[i]]
        m_i = entrenar(tr)
        P_list.append(predict_1x2(m_i, va, rho))
        y_list.append(va["resultado"].map(RES2IDX).to_numpy())
    P_oof = np.vstack(P_list)
    y_oof = np.concatenate(y_list)
    return fit_calibrador(P_oof, y_oof)

# ----------------------------------------------------------------------
# 4. Metricas
# ----------------------------------------------------------------------
def eval_probs(P, y):
    eps = 1e-12
    ll = -np.mean(np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0)))
    onehot = np.eye(3)[y]
    brier = np.mean(np.sum((P - onehot) ** 2, axis=1))
    acc = np.mean(P.argmax(1) == y)
    return ll, brier, acc

def ece(P, y, bins=10):
    """Expected Calibration Error sobre la confianza (prob de la clase predicha)."""
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if m.sum() > 0:
            e += abs(correct[m].mean() - conf[m].mean()) * m.sum()
    return e / len(y)

def tabla_fiabilidad(P, y, bins=5):
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    filas = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if m.sum() > 0:
            filas.append((f"{lo:.1f}-{hi:.1f}", int(m.sum()),
                          conf[m].mean(), correct[m].mean()))
    return filas

# ----------------------------------------------------------------------
# 5. Entrenamiento + evaluacion temporal honesta
# ----------------------------------------------------------------------
train_wide = df[df["date"] <  CORTE_VALID]
test_wide  = df[df["date"] >= CORTE_VALID]
print(f"Partidos entrenamiento: {len(train_wide):,}")
print(f"Partidos validacion   : {len(test_wide):,}")

model = entrenar(train_wide)

# Dixon-Coles: estimar rho (correlacion en los marcadores bajos)
RHO = estimar_rho(model, train_wide)
print(f"\nParametro Dixon-Coles  rho = {RHO:+.4f}  "
      f"({'sube' if RHO < 0 else 'baja'} la probabilidad de 0-0 y 1-1)")

print("Ajustando calibrador (CV temporal out-of-fold)...")
cal = calibrador_oof(train_wide, RHO)

y_test = test_wide["resultado"].map(RES2IDX).to_numpy()
P_ind  = predict_1x2(model, test_wide, rho=0.0)   # Poisson independiente
P_dc   = predict_1x2(model, test_wide, rho=RHO)   # + Dixon-Coles
P_full = aplicar_calibrador(cal, P_dc)            # + calibracion

print("\n--- Validacion 1X2 sobre el test (>= 2018) ---")
print(f"{'':22}{'Log-loss':>10}{'Brier':>9}{'Accuracy':>10}{'ECE':>8}")
for nombre, P in [("Poisson independiente", P_ind),
                  ("+ Dixon-Coles", P_dc),
                  ("+ Dixon-Coles + calib.", P_full)]:
    ll, br, ac = eval_probs(P, y_test)
    print(f"{nombre:22}{ll:>10.4f}{br:>9.4f}{ac:>10.3f}{ece(P, y_test):>8.4f}")
print(f"{'Baseline (uniforme)':22}{np.log(3):>10.4f}{'':>9}"
      f"{test_wide['resultado'].value_counts(normalize=True).max():>10.3f}")

print("\nEfecto de Dixon-Coles en los marcadores bajos (medias en el test):")
def prob_marcador(P_wide, rho, x, y):
    lh = np.clip(model.predict(lado(P_wide, "home")[FEATURES]), CLIP_LO, CLIP_HI)
    la = np.clip(model.predict(lado(P_wide, "away")[FEATURES]), CLIP_LO, CLIP_HI)
    return np.mean([matriz_marcadores(a, b, rho)[x, y] for a, b in zip(lh, la)])
real = test_wide["home_score"].astype(str) + "-" + test_wide["away_score"].astype(str)
for (x, y) in [(0, 0), (1, 1), (1, 0), (0, 1)]:
    p_ind = prob_marcador(test_wide, 0.0, x, y)
    p_dc  = prob_marcador(test_wide, RHO, x, y)
    p_real = (real == f"{x}-{y}").mean()
    print(f"  {x}-{y}:  indep {p_ind*100:4.1f}%  ->  DC {p_dc*100:4.1f}%   (real {p_real*100:4.1f}%)")

print("\nFiabilidad del modelo final (confianza vs acierto real):")
print(f"  {'rango':>10}{'n':>6}{'conf_media':>12}{'acierto':>10}")
for rango, nbin, conf, acc in tabla_fiabilidad(P_full, y_test):
    print(f"  {rango:>10}{nbin:>6}{conf:>12.3f}{acc:>10.3f}")

# ----------------------------------------------------------------------
# 6. Guardar modelo + calibrador + rho + estado de cada seleccion
# ----------------------------------------------------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": model, "calibrator": cal, "rho": RHO,
                 "features": FEATURES, "max_goles": MAX_GOLES}, f)
print(f"\nOK {MODEL_PATH}")

elo = pd.read_csv(ELO_PATH, parse_dates=["date"])
home_e = elo[["date", "home_team", "elo_home_post"]].rename(
    columns={"home_team": "team", "elo_home_post": "elo"})
away_e = elo[["date", "away_team", "elo_away_post"]].rename(
    columns={"away_team": "team", "elo_away_post": "elo"})
elo_long = pd.concat([home_e, away_e], ignore_index=True)
latest = (elo_long.sort_values("date").groupby("team", as_index=False)
                  .last()[["team", "elo"]])

squad = pd.read_csv(SQUAD_PATH).rename(columns={"canonical": "team"})
state = squad.merge(latest, on="team", how="left")
state = state[["team", "elo", "squad_mv_total", "caps_avg", "avg_age", "goals_total"]]
state.to_csv(STATE_PATH, index=False)
print(f"OK {STATE_PATH}  ({len(state)} selecciones)")