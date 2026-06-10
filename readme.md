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
| 4. Tabla maestra de nombres | pendiente | Unificar nombres de equipos entre fuentes |
| 5. Elo dinámico | pendiente | Calcular sobre `results.csv` partido a partido |
| 6. Features de plantilla | pendiente | Agregar stats de jugadores a nivel selección |
| 7. Dataset final | pendiente | Unir todo a nivel partido |
| 8. Modelo | pendiente | |

---

## Datos

### Partidos históricos
- **Fuente:** Kaggle — *International Football Results from 1872 to 2026*
- **Archivo:** `datos/historico de partidos/results.csv`
- **Columnas:** `date, home_team, away_team, home_score, away_score, tournament, neutral`

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

---

## Estructura del repositorio

```
Modelo Mundial/
├── .env                          # FOOTBALL_DATA_API token
├── readme.md
│
├── datos/
│   ├── historico de partidos/
│   │   └── results.csv           # Partidos 1872-2026
│   └── jugadores/
│       ├── convocatoria/
│       │   └── convocatoria.csv  # 48 selecciones × 26 jugadores
│       └── transfermarket/
│           ├── players.csv
│           └── player_valuations.csv
│
└── scripts/
    └── fetch_convocatoria.py     # Descarga convocatorias de football-data.org
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
