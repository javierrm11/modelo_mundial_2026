# Mundial 2026 — Predicción con ML

Sistema de predicción del Mundial 2026 construido paso a paso, desde la recopilación de datos hasta el modelo final.

> 48 selecciones · 104 partidos · 11 jun – 19 jul 2026

---

## Estado actual

| Paso | Estado | Descripción |
|---|---|---|
| 1. Datos históricos de partidos | ✅ | `results.csv` — 45k partidos desde 1872 (Kaggle) |
| 2. Convocatorias 2026 | ✅ | 48 selecciones × 26 jugadores vía football-data.org API |
| 3. Datos de jugadores | ✅ | `players.csv` + `player_valuations.csv` (Transfermarkt, Kaggle) |
| 4. Tabla maestra de nombres | ✅ | `team_name_master.csv` — 48 equipos, 3 fuentes unificadas |
| 5. Elo dinámico | ✅ | `elo_por_partido.csv` — 49,400 partidos con Elo pre/post |
| 6. Features de plantilla | ✅ | `squad_features.csv` — valor de mercado, caps, edad por selección |
| 7. Dataset final | ✅ | `dataset_final.csv` — 7,527 partidos entre equipos del Mundial |
| 8. Modelo + simulación | ✅ | Poisson (LightGBM) + Monte Carlo del torneo → `predicciones_mundial2026.csv` |

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

### Modelo (paso 8)
**Modelo de goles (LightGBM + opciones sobre-dispersas)** (`scripts/train_model.py`):

- Formato largo: cada partido → 2 filas (perspectiva de cada equipo que marca).
- `LightGBM` con objetivo `poisson` predice los goles esperados (λ) de un equipo
  contra otro a partir de `elo_diff`, ventaja de campo y features de plantilla.
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

Histórico: el script también **anexa** las mismas filas a
`datos/master/historial_predicciones_partidos.csv` (crea la cabecera si no existe),
manteniendo un histórico cronológico de las ejecuciones de predicción.

---

## Cómo ejecutar

```bash
# Pipeline completo (pasos 4 → 9)
python scripts/run_pipeline.py

# Buscar hiperparámetros (tuning temporal — recomendable antes de reentrenar)
python scripts/train_model.py tune

# Reentrenar el modelo final (usa los params hallados si quieres aplicarlos)
python scripts/train_model.py

# Solo re-simular el torneo (con N simulaciones)
python scripts/simulate_tournament.py 10000

# Predecir partidos concretos (lee datos/master/fixtures.csv).
# El script genera datos/master/predicciones_partidos.csv y anexa al historial.
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
│   │   └── [International Football Results from 1872 to 2026/
│   │       └── results.csv       # Partidos 1872-2026
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
│       ├── dataset_final.csv     # Dataset listo para el modelo
│       ├── team_state_2026.csv   # Elo actual + plantilla por selección
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
    ├── train_model.py            # Paso 8 — modelo de goles (Poisson)
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
