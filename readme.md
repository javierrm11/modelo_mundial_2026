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
**Modelo de goles Poisson** (`scripts/train_model.py`):

- Formato largo: cada partido → 2 filas (perspectiva de cada equipo que marca).
- `LightGBM` con objetivo `poisson` predice los goles esperados (λ) de un equipo
  contra otro a partir de `elo_diff`, ventaja de campo y features de plantilla.
- Validación **temporal** (entrena < 2018, valida ≥ 2018, incluye Mundiales
  2018 y 2022). Métricas sobre el 1X2 derivado de los dos λ:

  | | Modelo | Baseline |
  |---|---|---|
  | Log-loss | **1.00** | 1.10 (uniforme) |
  | Accuracy | **50.5 %** | 44.8 % (clase mayoritaria) |

**Simulación Monte Carlo** (`scripts/simulate_tournament.py`):

- Precalcula la matriz de goles esperados entre las 48 selecciones.
- Simula el torneo completo (12 grupos de 4 → 1º, 2º y 8 mejores terceros →
  dieciseisavos → final) **10.000 veces** muestreando marcadores con Poisson y
  resolviendo empates de eliminatoria por penaltis ponderados por Elo.
- Salida: `predicciones_mundial2026.csv` con % de campeón, final, semis, etc.

> El cuadro de grupos se lee de `datos/master/groups_2026.csv`. Si no existe se
> genera un **sorteo sembrado por Elo** (4 bombos) — reemplázalo por el sorteo
> oficial cuando se conozca. El cuadro de eliminatorias se siembra por
> rendimiento en la fase de grupos (aproximación del bracket oficial).

---

## Cómo ejecutar

```bash
# Pipeline completo (pasos 4 → 9)
python scripts/run_pipeline.py

# Solo re-simular el torneo (con N simulaciones)
python scripts/simulate_tournament.py 10000
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
│       └── predicciones_mundial2026.csv  # Salida de la simulación
│
├── modelos/
│   └── goal_model.pkl            # Modelo Poisson entrenado
│
└── scripts/
    ├── fetch_convocatoria.py     # Descarga convocatorias de football-data.org
    ├── build_name_master.py      # Paso 4 — tabla maestra de nombres
    ├── build_elo.py              # Paso 5 — Elo dinámico
    ├── build_squad_features.py   # Paso 6 — features de plantilla
    ├── build_dataset.py          # Paso 7 — dataset final
    ├── train_model.py            # Paso 8 — modelo de goles (Poisson)
    ├── simulate_tournament.py    # Paso 9 — simulación Monte Carlo
    └── run_pipeline.py           # Ejecuta pasos 4-9 en orden
```

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
