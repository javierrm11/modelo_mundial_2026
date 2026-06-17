# Mundial 2026 — Predicción con ML

Sistema de predicción del Mundial 2026 construido paso a paso, desde la recopilación de datos hasta el modelo final.

> 48 selecciones · 104 partidos · 11 jun – 19 jul 2026

---

## Estado actual

| Paso | Estado | Descripción |
|---|---|---|
| 1. Datos históricos de partidos | ✅ | `results.csv` — 45k partidos desde 1872 (Kaggle) |
| 2. Convocatorias 2026 | ✅ | 48 selecciones × 26 jugadores vía football-data.org API |
| 3. Rankings FIFA | ✅ | `fifa_mens_rank.csv` — ranking histórico FIFA masculino |
| 4. Datos de jugadores | ✅ | `players.csv` + `player_valuations.csv` (Transfermarkt, Kaggle) |
| 5. Tabla maestra de nombres | ✅ | `team_name_master.csv` — 48 equipos, 3 fuentes unificadas |
| 6. Elo dinámico | ✅ | `elo_por_partido.csv` — 49,400 partidos con Elo pre/post |
| 7. Features de plantilla | ✅ | `squad_features.csv` — valor de mercado, caps, edad por selección |
| 8. Dataset final + rankings | ✅ | `dataset_final_fifa.csv` — 7,527 partidos con Elo, squad, Elo y rankings FIFA |
| 9. Modelo + simulación | ✅ | Poisson (LightGBM) + Monte Carlo del torneo → `predicciones_mundial2026.csv` |

---

## Datos

### Partidos históricos
- **Fuente:** Kaggle — *International Football Results from 1872 to 2026*
- **Archivo:** `datos/historico de partidos/[International Football Results from 1872 to 2026/results.csv`
- **Columnas:** `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`

### Convocatorias FIFA 2026
- **Fuente:** football-data.org API (token en `.env`)
- **Script:** `scripts/fetch_convocatoria.py`
- **Archivo:** `datos/jugadores/convocatoria/convocatoria.csv`
- **Columnas:** `team_id, team_name, player_id, player_name, position, date_of_birth, nationality, shirt_number`

### Rankings FIFA masculinos
- **Fuente:** FIFA Men's World Rankings
- **Archivo:** `datos/historico de partidos/FIFA World Rankings/fifa_mens_rank.csv`
- **Columnas:** `date, semester, rank, team, acronym, total.points, previous.points, diff.points`
- **Integración:** Incorporados en `dataset_final_fifa.csv` como features `home_fifa_rank`, `home_fifa_points`, `away_fifa_rank`, `away_fifa_points`. El modelo usa `fifa_points_diff` (diferencia de puntos FIFA) como predictor activo de goles esperados.

### Dataset final (`dataset_final_fifa.csv`)
Este es el dataset que alimenta el entrenamiento del modelo. No todas sus
columnas entran como features: algunas son metadatos para ordenar, unir o
interpretar los partidos, y otras son el objetivo que el modelo intenta
predecir.

| Columna | Significado | Uso en el modelo |
|---|---|---|
| `date` | Fecha del partido | Metadato temporal. Sirve para ordenar el histórico y aplicar validación temporal. |
| `home_team` | Equipo local | Metadato de identidad. Necesario para emparejar y trazar cada partido. |
| `away_team` | Equipo visitante | Metadato de identidad. Necesario para emparejar y trazar cada partido. |
| `home_score` | Goles del local | Objetivo histórico. Se usa para entrenar y evaluar el modelo de goles. |
| `away_score` | Goles del visitante | Objetivo histórico. Se usa para entrenar y evaluar el modelo de goles. |
| `tournament` | Tipo de partido (Mundial, eliminatoria, amistoso...) | Contexto. Permite interpretar el partido y se usa indirectamente en la construcción del Elo. |
| `neutral` | Indica si el partido fue en campo neutral | Contexto. Ayuda a distinguir ventaja de local y partidos sin anfitrión. |
| `resultado` | Resultado 1X2 del partido | Etiqueta auxiliar para análisis y validación. No es feature del modelo de goles. |
| `elo_home_pre` | Elo del local antes del partido | Feature. Resume la fuerza competitiva reciente del local. |
| `elo_away_pre` | Elo del visitante antes del partido | Feature. Resume la fuerza competitiva reciente del visitante. |
| `elo_diff` | Diferencia de Elo prepartido | Feature principal. Captura la ventaja relativa entre equipos. |
| `home_days_rest` | Días desde el último partido del local | Feature. Captura fatiga y carga competitiva reciente. |
| `away_days_rest` | Días desde el último partido del visitante | Feature. Captura fatiga y carga competitiva reciente. |
| `home_recent_form_5` | Media de puntos del local en sus últimos 5 partidos | Feature. Resume forma reciente sin depender de un solo resultado. |
| `away_recent_form_5` | Media de puntos del visitante en sus últimos 5 partidos | Feature. Resume forma reciente sin depender de un solo resultado. |
| `home_squad_mv_total` | Valor total de mercado de la plantilla local | Feature. Proxy de calidad base del equipo. |
| `home_squad_mv_avg` | Valor medio de mercado de la plantilla local | Feature auxiliar. Captura profundidad/calidad media de la convocatoria. |
| `home_squad_mv_median` | Mediana del valor de mercado local | Feature auxiliar. Reduce el peso de estrellas extremas. |
| `home_caps_avg` | Media de internacionalidades del local | Feature. Mide experiencia internacional del grupo. |
| `home_goals_total` | Goles internacionales acumulados del local | Feature auxiliar. Aproxima capacidad ofensiva histórica de la plantilla. |
| `home_avg_age` | Edad media del local | Feature. Captura madurez o juventud del equipo. |
| `away_squad_mv_total` | Valor total de mercado de la plantilla visitante | Feature. Proxy de calidad base del visitante. |
| `away_squad_mv_avg` | Valor medio de mercado de la plantilla visitante | Feature auxiliar. Complementa el valor total con la calidad media. |
| `away_squad_mv_median` | Mediana del valor de mercado visitante | Feature auxiliar. Hace la señal más robusta a outliers. |
| `away_caps_avg` | Media de internacionalidades del visitante | Feature. Resume experiencia del visitante. |
| `away_goals_total` | Goles internacionales acumulados del visitante | Feature auxiliar. Señal adicional de producción ofensiva histórica. |
| `away_avg_age` | Edad media del visitante | Feature. Captura el perfil físico/táctico del visitante. |
| `home_fifa_rank` | Ranking FIFA del local | Feature auxiliar. Señal externa de fuerza relativa. |
| `home_fifa_points` | Puntos FIFA del local | Feature. Es la versión más informativa del ranking FIFA. |
| `away_fifa_rank` | Ranking FIFA del visitante | Feature auxiliar. Señal externa de fuerza relativa. |
| `away_fifa_points` | Puntos FIFA del visitante | Feature. Es la versión más informativa del ranking FIFA. |

De todas estas columnas, el modelo final se queda solo con las variables que
aportan señal predictiva estable y disponibles antes del partido. En concreto,
usa `elo_diff`, `elo_diff_squared`, `is_home`, `log_mv`, `log_mv_opp`,
`mv_ratio`, `caps`, `caps_opp`, `caps_diff_squared`, `age`, `age_opp`,
`fifa_points_diff`, `days_rest`, `days_rest_opp`, `recent_form_5` y
`recent_form_5_opp`. El resto de columnas se conservan para trazabilidad,
análisis y construcción del dataset.

### Jugadores — Transfermarkt
- **Fuente:** Kaggle — *Football Data from Transfermarkt*
- **Archivos:** `datos/jugadores/transfermarket/players.csv`, `player_valuations.csv`
- **Features clave:** `market_value_in_eur, international_caps, international_goals, position, date_of_birth`

---

## Pipeline de construcción

El problema central es que el modelo necesita una fila por partido con todas las features de ambos equipos. Los datos viven en niveles distintos y hay que alinearlos.

```
results.csv (nivel partido)
    └── + Elo calculado partido a partido
    └── + features de plantilla (convocatoria → players → agregado por equipo)
    └── = tabla final: partido × features_local × features_visitante × resultado
```

### El problema de los nombres
Cada fuente usa nombres distintos para el mismo equipo:

| results.csv | football-data.org | Transfermarkt |
|---|---|---|
| USA | United States | Vereinigte Staaten |
| South Korea | Korea Republic | Südkorea |
| Bosnia-Herzegovina | Bosnia-Herzegovina | Bosnien-Herzegowina |

**Paso 1** antes de todo: tabla maestra `team_name_master.csv` que unifica las tres fuentes.

### Elo dinámico
Se calcula sobre `results.csv` iterando cronológicamente. Cada equipo empieza en 1500.

```
K = 40 (Mundiales) / 30 (eliminatorias) / 20 (amistosos)
expected = 1 / (1 + 10^((elo_rival - elo_equipo) / 400))
nuevo_elo = elo + K * (resultado - expected)
```

El Elo **antes de cada partido** se guarda como feature — no el posterior.

### Features de plantilla (nivel equipo)
Desde `convocatoria.csv` → join `players.csv` → agregado por selección:

- `squad_market_value_total`
- `squad_market_value_avg`
- `international_caps_avg`
- `international_goals_total`

El join usa `player_id` y, como respaldo, el nombre normalizado (sin tildes/
mayúsculas), porque los IDs de football-data.org y Transfermarkt no coinciden.
Cobertura típica: 19–26 de los 26 jugadores por selección.

> ⚠️ Las features de plantilla son **constantes por equipo** (valores de la
> convocatoria 2026 aplicados a todo el histórico). Sirven como proxy de la
> "fuerza base" de la selección, no como feature temporalmente realista. El Elo
> sí es temporal (pre-partido).

### Modelo (paso 8 y 9)
**Modelo de goles (LightGBM + opciones sobre-dispersas)** (`scripts/train_model.py`):

Se escoge este enfoque porque el problema es de conteo de goles, con muchas
interacciones no lineales entre variables y bastante ruido entre selecciones.
`LightGBM` captura bien esas relaciones sin exigir una parametrización manual
muy rígida, mientras que `Poisson` respeta que los goles sean enteros no
negativos. Cuando la distribución real tiene más varianza que la Poisson,
`nb_alpha` permite ensanchar la cola de marcadores y evitar que el modelo se
quede demasiado "estrecho".

- Formato largo: cada partido → 2 filas (perspectiva de cada equipo que marca).
- `LightGBM` con objetivo `poisson` predice los goles esperados (λ) a partir de:
  - **Elo dinámico**: `elo_diff`, `elo_diff_squared` (efecto no-lineal)
  - **Valor de plantilla**: `log_mv`, `log_mv_opp`, `mv_ratio` (ratio logarítmico)
  - **Experiencia internacional**: `caps`, `caps_opp`, `caps_diff_squared` (efecto no-lineal)
  - **Edad media**: `age`, `age_opp`
  - **Ventaja de local**: `is_home`
  - **Rankings FIFA**: `fifa_points_diff` (diferencia de puntos FIFA)
  - **Fatiga y forma reciente**: `days_rest`, `days_rest_opp`, `recent_form_5`, `recent_form_5_opp`
- **Decaimiento temporal**: cada partido pesa `0.5^(años/semivida)`; la semivida
  se elige por CV temporal *out-of-fold* (ej. 8 años en la configuración actual).
- **Sobre-dispersión (NegBin)** opcional: se estima una `nb_alpha` global por
  momentos/OoF y, si `nb_alpha>0`, las predicciones de marcadores usan la
  distribución Negative-Binomial marginal en vez de Poisson independiente.
  Esto ensancha la cola de marcadores (más probabilidad para 2-0, 3-0, 0-3, ...)
  sin forzar sesgos artificiales.
- **Poisson bivariante** (Karlis & Ntzoufras) sigue disponible y modela la
  correlación entre goles en el paquete final cuando `nb_alpha==0`.
- **Calibración probabilística** (Platt multiclase con CV temporal *out-of-fold*):
  pulido final del 1X2. Decaimiento, bivariante/NegBin y calibración se combinan.
- **Explicabilidad (SHAP)**: el predictor muestra *por qué* da cada pronóstico,
  vía las contribuciones nativas de LightGBM (`pred_contrib`).

Validación y tuning

- El script `train_model.py` realiza validación temporal para elegir la semivida
  y puede ejecutar una búsqueda aleatoria temporal de hiperparámetros:

```bash
# Buscar hiperparámetros (Random search temporal, 20 iters)
python scripts/train_model.py tune
```

La mejor configuración OOF se puede usar para reentrenar definitivamente el
modelo y el bundle guardado (`modelos/goal_model.pkl`) incluye ahora metadatos:
`nb_alpha` (si ≠ 0), `score_model` (`negbin` cuando procede) y las `features`.

Ejemplo real de salida al entrenar:

```text
Partidos entrenamiento: 6,787
Partidos validacion   : 740

Seleccion de decaimiento temporal (log-loss OOF en el train):
  sin decaimiento          log-loss OOF = 1.0363
  half-life = 4 anios      log-loss OOF = 1.0620
  half-life = 8 anios      log-loss OOF = 1.0418
  half-life = 12 anios     log-loss OOF = 1.0392
  half-life = 20 anios     log-loss OOF = 1.0375
  -> elegido: sin decaimiento

Sobredispersion global NB_ALPHA = 0.1209  (si es > 0, ensancha la cola de marcadores)
Ajustando calibrador (CV temporal out-of-fold)...

--- Validacion 1X2 sobre el test (>= 2018) ---
                              Log-loss    Brier  Accuracy     ECE
Poisson indep. sin decay        1.0157   0.6099     0.497  0.0597
Poisson indep. con decay        1.0157   0.6099     0.497  0.0597
NegBin sobre-disperso           1.0137   0.6082     0.497  0.0462
+ calibracion (final)           1.0101   0.6055     0.497  0.0289
Baseline (uniforme)             1.0986              0.442

Efecto de la sobre-dispersion en los marcadores bajos (medias en el test):
  0-0:  indep  6.6%  ->  NB  8.2%   (real 10.7%)
  1-1:  indep 10.5%  ->  NB  9.7%   (real 11.8%)
  1-0:  indep  9.7%  ->  NB 10.2%   (real 11.6%)
  0-1:  indep  7.7%  ->  NB  8.2%   (real  6.6%)

Fiabilidad del modelo final (confianza vs acierto real):
       rango     n  conf_media   acierto
     0.2-0.4   125       0.374     0.352
     0.4-0.6   389       0.488     0.458
     0.6-0.8   215       0.669     0.637
     0.8-1.0    11       0.829     0.818

OK modelos/goal_model.pkl
OK datos/master/team_state_2026.csv  (48 selecciones)
```

Como leer este bloque:

- `Partidos entrenamiento` y `Partidos validacion`: corte temporal del dataset; se entrena con el pasado y se valida con partidos mas recientes.
- `Seleccion de decaimiento temporal`: compara semividas por log-loss OOF en train; gana la menor perdida (en este ejemplo, sin decaimiento).
- `NB_ALPHA`: parametro global de sobre-dispersion. Si es mayor que 0, la distribucion de marcadores permite mas varianza que Poisson puro.
- Tabla de validacion 1X2, `Log-loss`: penaliza probabilidades mal asignadas. Mas bajo es mejor.
- Tabla de validacion 1X2, `Brier`: error cuadratico medio en probabilidades 1X2. Mas bajo es mejor.
- Tabla de validacion 1X2, `Accuracy`: porcentaje de aciertos en la clase mas probable. Mas alto es mejor.
- Tabla de validacion 1X2, `ECE` (Expected Calibration Error): diferencia entre confianza predicha y frecuencia real. Mas bajo es mejor.
- `Baseline (uniforme)`: referencia naive que asigna 1/3 a H-D-A; sirve para medir ganancia real del modelo.
- `Efecto de la sobre-dispersion`: compara frecuencias de marcadores bajos entre Poisson independiente y NegBin frente al dato real.
- `Fiabilidad del modelo final`: por rangos de confianza, muestra si la probabilidad media que predice el modelo coincide con el acierto observado.
- `OK modelos/goal_model.pkl`: confirma que el modelo entrenado se guardo correctamente.
- `OK datos/master/team_state_2026.csv`: confirma que se actualizo el estado de las 48 selecciones para prediccion/simulacion.

**Simulación Monte Carlo** (`scripts/simulate_tournament.py`):

- Precalcula la matriz de goles esperados entre las 48 selecciones.
- Simula el torneo completo (12 grupos de 4 → 1º, 2º y 8 mejores terceros →
  dieciseisavos → final) **10.000 veces** muestreando marcadores con el **Poisson
  bivariante** (las correcciones del modelo entran también en la simulación) y
  resolviendo empates de eliminatoria por penaltis ponderados por Elo.
- Salida: `predicciones_mundial2026.csv` con % de campeón, final, semis, etc.
  Favoritas actuales: **Argentina 15 %, España 15 %, Francia 14 %**.

> El cuadro de grupos se lee de `datos/master/groups_2026.csv`. Si no existe se
> genera un **sorteo sembrado por Elo** (4 bombos) — reemplázalo por el sorteo
> oficial cuando se conozca. El cuadro de eliminatorias se siembra por
> rendimiento en la fase de grupos (aproximación del bracket oficial).
> Los **anfitriones** (México, EE. UU., Canadá) tienen **ventaja de local** en
> sus partidos de la fase de grupos (+0,36 goles de media, vía la feature `is_home`).

### Predicción partido a partido (`scripts/predict_fixtures.py`)
Para predecir partidos concretos del cuadro real sin simular todo el torneo.
Lee `datos/master/fixtures.csv` (editable):

```
home_team,away_team,fase,local
Spain,Brazil,grupo,
Mexico,Croatia,grupo,
Mexico,United States,eliminatoria,neutral
```

Por cada partido da el **1X2**, los **goles esperados (λ)** de cada equipo, los
**5 marcadores más probables** y una explicación **SHAP** de *por qué* (cuánto
pesan el Elo, el valor de plantilla, la ventaja de local…). En las filas marcadas
`eliminatoria` resuelve el empate con prórroga (λ·⅓) + penaltis ponderados por Elo
y da el **P(pasa de ronda)**. Acepta alias en español (`España`, `Países Bajos`…).

La columna opcional **`local`** controla la ventaja de campo: vacía → automática
(el anfitrión del Mundial juega en casa); `neutral` → fuerza campo neutral; un
equipo → ese equipo es local. Salida en `datos/master/predicciones_partidos.csv`.

Ejemplo de salida:

```text
=== Predicciones (4 partidos) ===

  Portugal  vs  DR Congo   [grupo · neutral]
    1X2:  Portugal  81.1%  |  Empate  14.1%  |  DR Congo   4.8%
    Goles esperados:  3.49 - 0.83
    Marcadores probables:  3-0 (8.5%)  2-0 (8.3%)  4-0 (7.1%)  3-1 (6.4%)  2-1 (6.3%)
    Por que Portugal:  Elo x1.44  descanso rival x1.21  forma reciente rival x1.19
    Por que DR Congo:  Elo x0.68  ratio valor x0.86  descanso rival x1.16

  England  vs  Croatia   [grupo · neutral]
    1X2:  England  69.2%  |  Empate  21.4%  |  Croatia   9.4%
    Goles esperados:  3.41 - 1.49
    Marcadores probables:  3-1 (6.1%)  2-1 (6.1%)  4-1 (5.0%)  3-0 (4.8%)  2-0 (4.8%)
    Por que England:  forma reciente rival x1.25  descanso rival x1.22  Elo x1.20
    Por que Croatia:  Elo x0.86  descanso rival x1.16  forma reciente rival x1.14

  Ghana  vs  Panama   [grupo · neutral]
    1X2:  Ghana  26.3%  |  Empate  28.4%  |  Panama  45.3%
    Goles esperados:  1.04 - 1.84
    Marcadores probables:  0-1 (10.7%)  1-1 (9.9%)  0-2 (9.1%)  1-2 (8.4%)  0-0 (7.1%)
    Por que Ghana:  Elo x0.68  descanso rival x1.15  forma reciente rival x1.13
    Por que Panama:  Elo x1.43  descanso rival x1.22  ratio valor x0.84

  Uzbekistan  vs  Colombia   [grupo · neutral]
    1X2:  Uzbekistan  15.8%  |  Empate  21.1%  |  Colombia  63.1%
    Goles esperados:  0.94 - 3.07
    Marcadores probables:  0-2 (8.5%)  0-3 (7.9%)  1-2 (7.2%)  0-1 (6.8%)  1-3 (6.6%)
    Por que Uzbekistan:  Elo x0.71  valor plantilla x0.86  descanso rival x1.15
    Por que Colombia:  Elo x1.42  descanso rival x1.21  forma reciente rival x1.20

OK datos/master/predicciones_partidos.csv  (4 partidos)
```

La línea `OK datos/master/predicciones_partidos.csv` indica que el archivo de
salida se escribió correctamente y que el script terminó sin errores, dejando
el CSV listo para abrirlo o reutilizarlo en el histórico.

Histórico: el script también **anexa** las mismas filas a
`datos/master/historial_predicciones_partidos.csv` (crea la cabecera si no existe),
manteniendo un histórico cronológico de las ejecuciones de predicción.

---

## Cómo ejecutar

```bash
# Pipeline completo (pasos 4 → 9, incluye rankings FIFA)
python scripts/run_pipeline.py

# O pasos individuales:

# Paso 7b — Integrar rankings FIFA (genera dataset_final_fifa.csv)
python scripts/merge_fifa_rankings.py

# Paso 7c — Generar estado 2026 con rankings FIFA (para predicciones)
python scripts/build_team_state_fifa.py

# Paso 8 — Entrenamiento: seleccionar semivida temporal
python scripts/train_model.py

# Paso 8 (opcional) — Tuning de hiperparámetros (20 iteraciones, CV temporal)
python scripts/train_model.py tune

# Paso 9 — Simulación del torneo (10.000 veces por defecto)
python scripts/simulate_tournament.py 10000

# Predicción partido a partido (lee datos/master/fixtures.csv)
# Genera datos/master/predicciones_partidos.csv y anexa al historial
python scripts/predict_fixtures.py
```

---

## Estructura del repositorio

```
Modelo Mundial/
├── .env                          # FOOTBALL_DATA_API token
├── readme.md
│
├── datos/
│   ├── historico de partidos/
│   │   ├── [International Football Results from 1872 to 2026/
│   │   │   └── results.csv       # Partidos 1872-2026
│   │   └── FIFA World Rankings/
│   │       └── fifa_mens_rank.csv # Ranking FIFA masculino histórico
│   ├── jugadores/
│   │   ├── convocatoria/
│   │   │   └── convocatoria.csv  # 48 selecciones × 26 jugadores
│   │   └── transfermarket/
│   │       ├── players.csv
│   │       └── player_valuations.csv
│   └── master/                   # Generado por el pipeline
│       ├── team_name_master.csv  # Correspondencia de nombres (3 fuentes)
│       ├── elo_por_partido.csv   # Elo pre/post para los 49k partidos
│       ├── squad_features.csv    # Features de plantilla por selección
│       ├── dataset_final.csv     # Dataset base (7,527 partidos)
│       ├── dataset_final_fifa.csv # Dataset + rankings FIFA (feature activa)
│       ├── team_state_2026.csv   # Elo actual + plantilla + FIFA por selección
│       ├── groups_2026.csv       # Cuadro de grupos (editable)
│       ├── predicciones_mundial2026.csv  # Salida de la simulación
│       ├── fixtures.csv          # Partidos a predecir (editable)
│       ├── predicciones_partidos.csv     # Salida partido a partido
│       └── historial_predicciones_partidos.csv # Histórico acumulado de predicciones
│
├── modelos/
│   └── goal_model.pkl            # Modelo entrenado (incluye metadatos: nb_alpha, score_model)
│
└── scripts/
    ├── fetch_convocatoria.py     # Descarga convocatorias de football-data.org
    ├── build_name_master.py      # Paso 4 — tabla maestra de nombres
    ├── build_elo.py              # Paso 5 — Elo dinámico
    ├── build_squad_features.py   # Paso 6 — features de plantilla
    ├── build_dataset.py          # Paso 7 — dataset final
    ├── merge_fifa_rankings.py    # Paso 7b — merge rankings FIFA → dataset_final_fifa.csv
    ├── build_team_state_fifa.py  # Paso 7c — estado 2026 con rankings FIFA
    ├── train_model.py            # Paso 8 — modelo de goles (LightGBM Poisson)
    ├── simulate_tournament.py    # Paso 9 — simulación Monte Carlo
    ├── predict_fixtures.py       # Predicción partido a partido
    └── run_pipeline.py           # Ejecuta pasos 4-9 en orden
```

---

## Posibles mejoras

Ideas para seguir desarrollando el proyecto, de mayor a menor impacto:

- **Más features contextuales**: días de descanso entre partidos, distancia de
  viaje, altitud (Ciudad de México), clima.
- **Forma reciente y bajas**: racha de resultados de los últimos meses y
  lesiones/sanciones de última hora.
- **Modelado ataque / defensa**: entrenar modelos separados para fuerza ofensiva
  y defensiva por selección y combinarlos mejora la cola de marcadores altos.
- **Búsqueda de hiperparámetros temporal**: tuning (`train_model.py tune`) con
  validación expandida suele mejorar lambdas y la capacidad de predecir scores altos.
- **Intervalos de incertidumbre** en las probabilidades de campeón (error de
  Monte Carlo / bandas de confianza), no solo el porcentaje puntual.
- **Pipeline automatizado**: refrescar Elo y convocatorias vía API de forma
  programada para mantener el modelo siempre al día durante el torneo.

---

## Stack

| | |
|---|---|
| Lenguaje | Python 3.10+ |
| Datos | pandas, numpy |
| API | requests, python-dotenv |
| Modelo | scikit-learn, LightGBM |
| Entorno | venv |

---

## Autor

**Javier Molero**
