"""
Paso 8a - Modelo de goles (Poisson / sobre-disperso) + calibracion probabilistica.

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
# end of module docstring
"""

import os
import pickle
import numpy as np
import pandas as pd
from math import exp, factorial, comb, log, lgamma
from sklearn.model_selection import ParameterSampler
import lightgbm as lgb
import random
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

DATASET_PATH = "datos/master/dataset_final_fifa.csv"
ELO_PATH     = "datos/master/elo_por_partido.csv"
SQUAD_PATH   = "datos/master/squad_features.csv"
MODEL_PATH   = "modelos/goal_model.pkl"
STATE_PATH   = "datos/master/team_state_2026.csv"

CORTE_VALID = "2020-01-01"   # partidos >= esta fecha = validacion
MAX_GOLES   = 10             # truncado para la matriz de marcadores
N_FOLDS     = 5              # folds temporales para la calibracion OOF
CLIP_LO, CLIP_HI = 0.05, 6.0
RES2IDX = {"H": 0, "D": 1, "A": 2}   # orden de clases en el 1X2
HALF_LIVES  = [4, 8, 12, 20]  # candidatos de semivida (anios) para el decaimiento
NB_ALPHA = 2                # dispersion global: Var = mu + NB_ALPHA * mu^2

os.makedirs("modelos", exist_ok=True)

# ----------------------------------------------------------------------
# 1. Cargar dataset
# ----------------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)
df["date"] = pd.to_datetime(df["date"])
REFERENCIA = df["date"].max()   # "hoy" para medir la antiguedad de cada partido

def peso_temporal(dates, half_life):
    """Peso por decaimiento exponencial: un partido de hace `half_life` anios
       pesa la mitad. half_life=None -> sin decaimiento (pesos uniformes)."""
    if half_life is None:
        return None
    anios = (REFERENCIA - pd.to_datetime(dates)).dt.days / 365.25
    return 0.5 ** (anios / half_life)

FEATURES = [
    "elo_diff", "elo_diff_squared", "is_home",
    "log_mv", "log_mv_opp", "mv_ratio",
    "caps", "caps_opp", "caps_diff_squared",
    "age", "age_opp",
    "fifa_points_diff",
    "days_rest", "days_rest_opp",
    "recent_form_5", "recent_form_5_opp",
]

def lado(d, scorer):
    """scorer = 'home'/'away'. Filas desde la perspectiva del que marca."""
    opp = "away" if scorer == "home" else "home"
    out = pd.DataFrame()
    out["date"]    = d["date"]
    out["goals"]   = d[f"{scorer}_score"]
    
    # Elo features
    elo_diff = d[f"elo_{scorer}_pre"] - d[f"elo_{opp}_pre"]
    out["elo_diff"] = elo_diff
    out["elo_diff_squared"] = elo_diff ** 2
    
    es_neutral = d["neutral"].astype(str).str.upper().eq("TRUE")
    out["is_home"] = ((scorer == "home") & ~es_neutral).astype(int)
    
    # Market value features
    mv_scorer = np.log1p(pd.to_numeric(d[f"{scorer}_squad_mv_total"], errors="coerce"))
    mv_opp = np.log1p(pd.to_numeric(d[f"{opp}_squad_mv_total"], errors="coerce"))
    out["log_mv"] = mv_scorer
    out["log_mv_opp"] = mv_opp
    out["mv_ratio"] = mv_scorer - mv_opp  # logarithmic ratio (equivalent to log(mv_scorer / mv_opp))
    
    # Experience features
    caps_scorer = pd.to_numeric(d[f"{scorer}_caps_avg"], errors="coerce")
    caps_opp = pd.to_numeric(d[f"{opp}_caps_avg"], errors="coerce")
    caps_diff = caps_scorer - caps_opp
    out["caps"] = caps_scorer
    out["caps_opp"] = caps_opp
    out["caps_diff_squared"] = caps_diff ** 2
    
    # Age features
    out["age"] = pd.to_numeric(d[f"{scorer}_avg_age"], errors="coerce")
    out["age_opp"] = pd.to_numeric(d[f"{opp}_avg_age"], errors="coerce")
    
    # FIFA rankings difference: from scorer's perspective (scorer - opponent)
    scorer_fifa = pd.to_numeric(d.get(f"{scorer}_fifa_points", 0), errors="coerce").fillna(0)
    opp_fifa = pd.to_numeric(d.get(f"{opp}_fifa_points", 0), errors="coerce").fillna(0)
    out["fifa_points_diff"] = scorer_fifa - opp_fifa

    # Fatigue / momentum features
    out["days_rest"] = pd.to_numeric(d.get(f"{scorer}_days_rest", 0), errors="coerce").fillna(0)
    out["days_rest_opp"] = pd.to_numeric(d.get(f"{opp}_days_rest", 0), errors="coerce").fillna(0)
    out["recent_form_5"] = pd.to_numeric(d.get(f"{scorer}_recent_form_5", 0), errors="coerce").fillna(0)
    out["recent_form_5_opp"] = pd.to_numeric(d.get(f"{opp}_recent_form_5", 0), errors="coerce").fillna(0)
    
    return out

def long_from_wide(wide):
    """Convierte partidos (formato ancho) en formato largo de entrenamiento."""
    return pd.concat([lado(wide, "home"), lado(wide, "away")],
                     ignore_index=True).dropna(subset=["goals"])

# ----------------------------------------------------------------------
# 2. Modelo de goles y derivacion del 1X2
# ----------------------------------------------------------------------
def make_model(params=None):
    """Crea un LGBM con parámetros por defecto que pueden ser sobreescritos por `params`."""
    defaults = dict(
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
    if params:
        defaults.update(params)
    return lgb.LGBMRegressor(**defaults)

def entrenar(wide, half_life=None, model_params=None):
    m = make_model(model_params)
    long = long_from_wide(wide)
    m.fit(long[FEATURES], long["goals"],
          sample_weight=peso_temporal(long["date"], half_life))
    return m

def poisson_pmf(lmbda, k):
    return exp(-lmbda) * lmbda**k / factorial(k)

def negbin_pmf(k, mu, alpha):
    """Negative binomial parametrizada por media mu y dispersion alpha."""
    if alpha <= 0:
        return poisson_pmf(mu, k)
    r = 1.0 / alpha
    log_p = (
        lgamma(k + r) - lgamma(r) - lgamma(k + 1)
        + r * log(r / (r + mu))
        + k * log(mu / (r + mu))
    )
    return exp(log_p)

# ----- Poisson bivariante (Karlis & Ntzoufras) -----------------------------
# X = W1 + W3,  Y = W2 + W3,  con Wk ~ Poisson(lk) independientes.
# lambda3 es el termino de covarianza (cov(X,Y) = lambda3 >= 0); modela la
# correlacion entre los goles de ambos equipos de forma general (sustituye a
# Dixon-Coles). Con lambda3 = 0 se reduce al Poisson independiente.
FACT = [factorial(k) for k in range(MAX_GOLES + 1)]

def biv_score_proba(x, y, mu_h, mu_a, lam3):
    """P(X=x, Y=y) del Poisson bivariante."""
    l3 = min(lam3, 0.95 * min(mu_h, mu_a))
    l1 = max(mu_h - l3, 1e-9)
    l2 = max(mu_a - l3, 1e-9)
    s = 0.0
    for k in range(min(x, y) + 1):
        s += (comb(x, k) * comb(y, k) * FACT[k]) * (l3 / (l1 * l2))**k
    return exp(-(l1 + l2 + l3)) * l1**x / FACT[x] * l2**y / FACT[y] * s

def matriz_marcadores(mu_h, mu_a, lam3=0.0, nb_alpha=0.0):
    """Matriz de marcadores (filas=local, cols=visitante)."""
    if nb_alpha > 0:
        p_h = np.array([negbin_pmf(x, mu_h, nb_alpha) for x in range(MAX_GOLES + 1)])
        p_a = np.array([negbin_pmf(y, mu_a, nb_alpha) for y in range(MAX_GOLES + 1)])
        M = np.outer(p_h, p_a)
    else:
        M = np.array([[biv_score_proba(x, y, mu_h, mu_a, lam3)
                       for y in range(MAX_GOLES + 1)]
                      for x in range(MAX_GOLES + 1)])
    return M / M.sum()

def matriz_1x2(mu_h, mu_a, lam3=0.0, nb_alpha=0.0):
    """P(local), P(empate), P(visitante) a partir de la matriz de marcadores."""
    m = matriz_marcadores(mu_h, mu_a, lam3, nb_alpha)
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()

def predict_1x2(model, wide, lam3=0.0, nb_alpha=0.0):
    """Devuelve P[N,3] (H,D,A) para un conjunto de partidos en formato ancho."""
    lh = np.clip(model.predict(lado(wide, "home")[FEATURES]), CLIP_LO, CLIP_HI)
    la = np.clip(model.predict(lado(wide, "away")[FEATURES]), CLIP_LO, CLIP_HI)
    return np.array([matriz_1x2(a, b, lam3, nb_alpha) for a, b in zip(lh, la)])

def estimar_lambda3(model, wide, half_life=None):
    """Estima lambda3 (covarianza del bivariante) por maxima verosimilitud
       ponderada sobre los marcadores reales."""
    mu_h = np.clip(model.predict(lado(wide, "home")[FEATURES]), CLIP_LO, CLIP_HI)
    mu_a = np.clip(model.predict(lado(wide, "away")[FEATURES]), CLIP_LO, CLIP_HI)
    xs = np.minimum(wide["home_score"].to_numpy(), MAX_GOLES)
    ys = np.minimum(wide["away_score"].to_numpy(), MAX_GOLES)
    w = peso_temporal(wide["date"], half_life)
    w = np.ones(len(wide)) if w is None else w.to_numpy()

    def neg_loglik(c):
        ll = 0.0
        for i in range(len(xs)):
            p = biv_score_proba(int(xs[i]), int(ys[i]), mu_h[i], mu_a[i], c)
            ll += w[i] * log(max(p, 1e-300))
        return -ll

    res = minimize_scalar(neg_loglik, bounds=(0.0, 0.6), method="bounded")
    return float(res.x)

# ----------------------------------------------------------------------
# 3. Calibracion (Platt multiclase sobre los log-probabilidades)
# ----------------------------------------------------------------------
def fit_calibrador(P_raw, y, weights=None):
    cal = LogisticRegression(C=1.0, max_iter=2000)
    cal.fit(np.log(np.clip(P_raw, 1e-6, 1.0)), y, sample_weight=weights)
    return cal

def aplicar_calibrador(cal, P_raw):
    return cal.predict_proba(np.log(np.clip(P_raw, 1e-6, 1.0)))

def calibrador_oof(wide_train, rho=0.0, half_life=None, nb_alpha=0.0, n_folds=N_FOLDS):
    """Predicciones out-of-fold con CV temporal -> calibrador sin fuga."""
    w = wide_train.sort_values("date").reset_index(drop=True)
    idx = np.array_split(np.arange(len(w)), n_folds)
    P_list, y_list, wt_list = [], [], []
    for i in range(1, n_folds):                       # ventana expansiva
        tr = w.iloc[np.concatenate(idx[:i])]
        va = w.iloc[idx[i]]
        m_i = entrenar(tr, half_life)
        P_list.append(predict_1x2(m_i, va, rho, nb_alpha))
        y_list.append(va["resultado"].map(RES2IDX).to_numpy())
        wv = peso_temporal(va["date"], half_life)
        wt_list.append(np.ones(len(va)) if wv is None else wv.to_numpy())
    P_oof = np.vstack(P_list)
    y_oof = np.concatenate(y_list)
    wt_oof = np.concatenate(wt_list)
    return fit_calibrador(P_oof, y_oof, wt_oof if half_life is not None else None)

def oof_logloss(wide_train, half_life, model_params=None, n_folds=N_FOLDS):
    """Log-loss out-of-fold (CV temporal) para elegir la semivida sin tocar el test."""
    w = wide_train.sort_values("date").reset_index(drop=True)
    idx = np.array_split(np.arange(len(w)), n_folds)
    ll_sum, n_sum = 0.0, 0
    for i in range(1, n_folds):
        tr = w.iloc[np.concatenate(idx[:i])]
        va = w.iloc[idx[i]]
        m_i = entrenar(tr, half_life, model_params)
        P = predict_1x2(m_i, va, 0.0, NB_ALPHA)
        y = va["resultado"].map(RES2IDX).to_numpy()
        ll, _, _ = eval_probs(P, y)
        ll_sum += ll * len(y); n_sum += len(y)
    return ll_sum / n_sum

def tune_hyperparams(wide_train, half_life, n_iter=20, random_state=42):
    """Random search temporal para hiperparámetros de LightGBM.

    Usa `oof_logloss` con ventana expansiva para comparar configuraciones.
    """
    param_dist = {
        'num_leaves': [15, 31, 63],
        'min_child_samples': [10, 30, 50, 100],
        'learning_rate': [0.01, 0.03, 0.05],
        'n_estimators': [200, 350, 500],
        'reg_lambda': [0.0, 0.5, 1.0],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }
    sampler = ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state)
    best = None
    best_score = float('inf')
    for i, params in enumerate(sampler):
        score = oof_logloss(wide_train, half_life, model_params=params)
        print(f"Tuning {i+1}/{n_iter}: score={score:.4f} params={params}")
        if score < best_score:
            best_score = score
            best = params
    print(f"\nMejor params (oof): {best} score={best_score:.4f}")
    return best, best_score
def estimar_nb_alpha(wide_train, half_life=None, n_folds=N_FOLDS):
    """Estima una dispersion global por momentos usando predicciones out-of-fold."""
    w = wide_train.sort_values("date").reset_index(drop=True)
    idx = np.array_split(np.arange(len(w)), n_folds)
    numerador, denominador = 0.0, 0.0
    for i in range(1, n_folds):
        tr = w.iloc[np.concatenate(idx[:i])]
        va = w.iloc[idx[i]]
        m_i = entrenar(tr, half_life)
        long_va = long_from_wide(va)
        mu = np.clip(m_i.predict(long_va[FEATURES]), CLIP_LO, CLIP_HI)
        y = long_va["goals"].to_numpy(dtype=float)
        wv = peso_temporal(long_va["date"], half_life)
        wt = np.ones(len(long_va)) if wv is None else wv.to_numpy()
        numerador += np.sum(wt * (((y - mu) ** 2) - y))
        denominador += np.sum(wt * (mu ** 2))
    if denominador <= 0:
        return 0.0
    return float(max(numerador / denominador, 0.0))

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

# Decaimiento temporal: elegir la semivida por CV temporal OOF (sin tocar el test)
print("\nSeleccion de decaimiento temporal (log-loss OOF en el train):")
ll_por_hl = {}
for hl in [None] + HALF_LIVES:
    ll_por_hl[hl] = oof_logloss(train_wide, hl)
    etq = "sin decaimiento" if hl is None else f"half-life = {hl} anios"
    print(f"  {etq:24} log-loss OOF = {ll_por_hl[hl]:.4f}")
BEST_HL = min(ll_por_hl, key=ll_por_hl.get)
print(f"  -> elegido: "
      f"{'sin decaimiento' if BEST_HL is None else f'half-life = {BEST_HL} anios'}")

# Si se llama con 'tune', ejecutar la búsqueda de hiperparámetros temporal
import sys
if 'tune' in sys.argv:
    print("Ejecutando tuning temporal de hiperparámetros...")
    best, score = tune_hyperparams(train_wide, BEST_HL, n_iter=20, random_state=42)
    print(f"Tuning completado. Mejor score {score:.4f}, params: {best}")
    sys.exit(0)

model         = entrenar(train_wide, BEST_HL)     # modelo final (con decaimiento)
model_nodecay = entrenar(train_wide, None)        # referencia sin decaimiento

NB_ALPHA = estimar_nb_alpha(train_wide, BEST_HL)
print(f"\nSobredispersion global NB_ALPHA = {NB_ALPHA:.4f}  "
      f"(si es > 0, ensancha la cola de marcadores)")

print("Ajustando calibrador (CV temporal out-of-fold)...")
cal = calibrador_oof(train_wide, 0.0, BEST_HL, NB_ALPHA)

y_test = test_wide["resultado"].map(RES2IDX).to_numpy()
P_nodec = predict_1x2(model_nodecay, test_wide, lam3=0.0, nb_alpha=0.0)   # sin decaimiento
P_poiss = predict_1x2(model, test_wide, lam3=0.0, nb_alpha=0.0)           # Poisson independiente
P_nb    = predict_1x2(model, test_wide, lam3=0.0, nb_alpha=NB_ALPHA)       # NB sobre-disperso
P_full  = aplicar_calibrador(cal, P_nb)                                    # + calibracion

print("\n--- Validacion 1X2 sobre el test (>= 2018) ---")
print(f"{'':28}{'Log-loss':>10}{'Brier':>9}{'Accuracy':>10}{'ECE':>8}")
for nombre, P in [("Poisson indep. sin decay", P_nodec),
                  ("Poisson indep. con decay", P_poiss),
                  ("NegBin sobre-disperso", P_nb),
                  ("+ calibracion (final)", P_full)]:
    ll, br, ac = eval_probs(P, y_test)
    print(f"{nombre:28}{ll:>10.4f}{br:>9.4f}{ac:>10.3f}{ece(P, y_test):>8.4f}")
print(f"{'Baseline (uniforme)':28}{np.log(3):>10.4f}{'':>9}"
      f"{test_wide['resultado'].value_counts(normalize=True).max():>10.3f}")

print("\nEfecto de la sobre-dispersion en los marcadores bajos (medias en el test):")
def prob_marcador(P_wide, nb_alpha, x, y):
    lh = np.clip(model.predict(lado(P_wide, "home")[FEATURES]), CLIP_LO, CLIP_HI)
    la = np.clip(model.predict(lado(P_wide, "away")[FEATURES]), CLIP_LO, CLIP_HI)
    return np.mean([matriz_marcadores(a, b, 0.0, nb_alpha)[x, y] for a, b in zip(lh, la)])
real = test_wide["home_score"].astype(str) + "-" + test_wide["away_score"].astype(str)
for (x, y) in [(0, 0), (1, 1), (1, 0), (0, 1)]:
    p_ind = prob_marcador(test_wide, 0.0, x, y)
    p_nb = prob_marcador(test_wide, NB_ALPHA, x, y)
    p_real = (real == f"{x}-{y}").mean()
    print(f"  {x}-{y}:  indep {p_ind*100:4.1f}%  ->  NB {p_nb*100:4.1f}%   (real {p_real*100:4.1f}%)")

print("\nFiabilidad del modelo final (confianza vs acierto real):")
print(f"  {'rango':>10}{'n':>6}{'conf_media':>12}{'acierto':>10}")
for rango, nbin, conf, acc in tabla_fiabilidad(P_full, y_test):
    print(f"  {rango:>10}{nbin:>6}{conf:>12.3f}{acc:>10.3f}")

# ----------------------------------------------------------------------
# 6. Guardar modelo + calibrador + rho + estado de cada seleccion
# ----------------------------------------------------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": model, "calibrator": cal, "lambda3": 0.0,
                 "score_model": "negbin", "nb_alpha": NB_ALPHA,
                 "half_life": BEST_HL,
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