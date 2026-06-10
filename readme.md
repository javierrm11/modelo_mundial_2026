# Mundial 2026 — Predicción con IA

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Sistema de predicción del Mundial de Fútbol 2026 basado en un modelo híbrido de **regresión de Poisson + Gradient Boosting** sobre features combinadas de **selección y jugador**, con simulación Monte Carlo del cuadro completo y recalibración por rondas.

> El Mundial 2026 se celebra del **11 de junio al 19 de julio de 2026** en Estados Unidos, México y Canadá. Es la primera edición con **48 selecciones** distribuidas en 12 grupos de 4, jugándose **104 partidos** en total.

---

## Tabla de contenidos

1. [Objetivo del proyecto](#objetivo-del-proyecto)
2. [Arquitectura del sistema](#arquitectura-del-sistema)
3. [Stack tecnológico](#stack-tecnológico)
4. [Estructura del repositorio](#estructura-del-repositorio)
5. [Fuentes de datos](#fuentes-de-datos)
6. [Modelado](#modelado)
7. [Sistema de recalibración](#sistema-de-recalibración)
8. [Roadmap](#roadmap)
9. [Instalación y uso](#instalación-y-uso)
10. [Resultados](#resultados)
11. [Disclaimer](#disclaimer)
12. [Autor](#autor)

---

## Objetivo del proyecto

Construir un sistema reproducible que produzca:

- **Probabilidades partido a partido** para los 104 encuentros del torneo.
- **Cuadro completo predicho** (grupos, octavos, cuartos, semifinales, final) con probabilidad de avance por selección en cada ronda.
- **Predicción del campeón** con intervalo de confianza derivado de la simulación Monte Carlo.

El sistema se diseña para **recalibrarse al final de cada fase**, integrando los resultados reales y emitiendo una predicción actualizada de lo que queda del torneo.

---

## Arquitectura del sistema

El proyecto se estructura en tres componentes desacoplados que se ejecutan en pipeline:

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   1. PREDICTOR      │    │   2. SIMULADOR       │    │  3. RECALIBRADOR    │
│  partido único      │───▶│  Monte Carlo del     │───▶│  ronda a ronda      │
│                     │    │  torneo completo     │    │                     │
│ • Poisson → goles   │    │ • 10.000 simulaciones│    │ • Actualiza Elo     │
│ • GBM → P(victoria) │    │ • Bracket dinámico   │    │ • Refresca forma    │
│ • Ensemble          │    │ • P(avance) por ronda│    │ • Re-simula         │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### 1. Predictor de partido único

Toma dos selecciones y devuelve la distribución de probabilidad del resultado:

- **Modelo de goles** (Poisson regression bivariada): predice `λ_local` y `λ_visitante`, los goles esperados de cada equipo.
- **Modelo de resultado** (Gradient Boosting con LightGBM): predice directamente `P(victoria local)`, `P(empate)`, `P(victoria visitante)`.
- **Ensemble**: media calibrada de ambos modelos, con peso ajustable por validación cruzada.

### 2. Simulador Monte Carlo del torneo

Ejecuta 10.000 simulaciones del torneo desde el estado actual:

- Para la **fase de grupos**, simula los 72 partidos (12 grupos × 6 partidos/grupo) y aplica los criterios oficiales de desempate FIFA.
- Para las **eliminatorias**, simula cada cruce respetando el bracket. Si hay empate en 90 minutos, simula prórroga (multiplicador 0.5 a los λ) y penaltis (modelo aparte basado en estadísticas históricas).
- Agrega los resultados → `P(cada selección alcanza la fase X)`.

### 3. Sistema de recalibración

Al cerrarse cada fase, los resultados reales se integran:

- **Elo dinámico**: cada partido jugado actualiza el rating de las dos selecciones según el algoritmo de FIFA Women's Ranking adaptado (K=60 para Mundial).
- **Forma reciente**: el feature de últimos 5 partidos se desplaza con los datos nuevos.
- **Re-simulación**: con el bracket parcialmente resuelto, solo se simulan los partidos restantes.

---

## Stack tecnológico

| Capa | Herramienta |
|---|---|
| Lenguaje | Python 3.11+ |
| Manipulación de datos | pandas, NumPy |
| Modelos | scikit-learn, LightGBM, statsmodels (Poisson) |
| Simulación | NumPy (vectorizada) + scipy.stats |
| Visualización | matplotlib, seaborn, plotly |
| Notebooks de exploración | Jupyter |
| Tests | pytest |
| Gestión de entorno | uv / venv + requirements.txt |
| Configuración | python-dotenv |

---

## Estructura del repositorio

```
mundial-2026-prediction/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/                    # CSVs originales sin modificar
│   ├── interim/                # Datos limpios y unificados
│   └── processed/              # Datasets finales con features listos
│
├── notebooks/
│   ├── 01_exploration.ipynb    # EDA de fuentes de datos
│   ├── 02_features.ipynb       # Diseño y validación de features
│   ├── 03_modeling.ipynb       # Entrenamiento y comparación de modelos
│   └── 04_simulation.ipynb     # Monte Carlo y resultados
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── loaders.py          # Carga desde cada fuente (CSV, scraping, API)
│   │   ├── elo.py              # Cálculo del Elo dinámico
│   │   └── preprocessing.py    # Limpieza y unificación
│   │
│   ├── features/
│   │   ├── team_features.py    # Features a nivel selección
│   │   ├── player_features.py  # Features agregadas a nivel jugador
│   │   └── match_features.py   # Features del enfrentamiento (H2H, etc.)
│   │
│   ├── models/
│   │   ├── poisson_model.py    # Regresión de Poisson para goles
│   │   ├── gbm_model.py        # LightGBM para resultado
│   │   └── ensemble.py         # Combinación de ambos modelos
│   │
│   ├── simulation/
│   │   ├── monte_carlo.py      # Motor de simulación
│   │   ├── group_stage.py      # Reglas de desempate FIFA
│   │   ├── knockout.py         # Bracket eliminatorio
│   │   └── tiebreaker.py       # Prórroga y penaltis
│   │
│   └── utils/
│       ├── config.py
│       └── viz.py              # Heatmaps, brackets, gráficos
│
├── scripts/
│   ├── train.py                # Entrena los modelos desde cero
│   ├── predict.py              # Genera la predicción del torneo
│   └── recalibrate.py          # Re-entrena tras cada fase
│
├── outputs/
│   ├── predictions/            # JSON/CSV con probabilidades por versión
│   └── figures/                # Visualizaciones para contenido
│
└── tests/
    └── ...                     # pytest sobre src/
```

---

## Fuentes de datos

### Histórico de partidos (entrenamiento)

- **[International Football Results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)** — Kaggle. Base del dataset histórico, con ~45k partidos internacionales etiquetados por tipo (Mundial, eliminatorias, Nations League, Copa América, amistoso).
- **FIFA Rankings históricos** — para feature de ranking previo al partido.
- **Cálculo de Elo dinámico** propio sobre el dataset anterior.

**Ventana temporal**: últimos 4 años (ciclo mundialista 2022 → 2026) para evitar contaminación generacional.

**Ponderación**: partidos competitivos (eliminatorias, Nations League, Copa América, Mundiales, Eurocopa) con peso `1.0`. Partidos amistosos ponderados a `0.3` durante el entrenamiento.

### Datos a nivel jugador (features de plantilla)

- **[FBref](https://fbref.com)** — xG, xA, minutos, goles y métricas avanzadas por jugador y temporada.
- **[Transfermarkt](https://www.transfermarkt.com)** — valor de mercado por jugador.
- **Convocatorias oficiales FIFA** del Mundial 2026 — los 26 jugadores por selección.

### Configuración del torneo

- **Calendario oficial FIFA 2026** — sedes, horarios, condiciones (altitud Ciudad de México, calor en sedes USA).
- **Bracket oficial** tras el sorteo.

---

## Modelado

### Por qué Poisson + Gradient Boosting

El modelado de fútbol tiene un consenso académico bastante asentado: **los goles siguen aproximadamente una distribución de Poisson**, y la regresión de Poisson bivariada (Dixon-Coles, 1997) sigue siendo la referencia. El Gradient Boosting añade no-linealidades y captura interacciones entre features que la Poisson no modela bien.

**Las redes neuronales no se eligieron** porque en problemas tabulares con pocos miles de muestras, el gradient boosting suele igualarlas o superarlas con mucho menos coste computacional y mejor interpretabilidad.

### Features a nivel selección

| Feature | Descripción |
|---|---|
| `elo_team` | Elo dinámico actualizado hasta el partido. |
| `fifa_ranking` | Ranking FIFA en el mes del partido. |
| `form_last_5` | Puntos ponderados de los últimos 5 partidos competitivos. |
| `goals_for_avg` | Media de goles a favor en los últimos 10 partidos. |
| `goals_against_avg` | Media de goles en contra en los últimos 10 partidos. |
| `days_rest` | Días desde el último partido. |
| `is_host` | 1 si la selección juega en su país anfitrión. |
| `confederation` | UEFA / CONMEBOL / CONCACAF / etc. |

### Features agregadas a nivel jugador

Se computan sobre los **11 titulares más probables** ponderados por minutos jugados con la selección en los últimos 12 meses:

| Feature | Descripción |
|---|---|
| `squad_market_value` | Suma del valor de mercado de la convocatoria. |
| `xg_per90_avg` | Media de xG/90 minutos de los delanteros titulares. |
| `xa_per90_avg` | Media de xA/90 de centrocampistas titulares. |
| `league_strength_avg` | Coeficiente medio de fuerza de las ligas de los titulares. |
| `gk_save_pct` | % de paradas del portero titular. |
| `defenders_xga` | xGA acumulado por la defensa titular. |

### Features del enfrentamiento

| Feature | Descripción |
|---|---|
| `elo_diff` | Diferencia de Elo entre las dos selecciones. |
| `h2h_last_5` | Resultados directos en los últimos 5 enfrentamientos. |
| `same_confederation` | 1 si comparten confederación. |
| `venue_altitude` | Altitud de la sede (relevante para CDMX, Guadalajara). |

### Variable objetivo

- **Modelo Poisson**: número de goles de cada equipo.
- **Modelo GBM**: clase del resultado en formato one-hot (`local_win`, `draw`, `away_win`).

### Validación

- División temporal: train hasta 2024, validación 2025, test sobre clasificación europea 2025–2026.
- Métricas: log loss, Brier score, accuracy. Comparación contra baseline de cuotas de casas de apuestas.

---

## Sistema de recalibración

Tras cada ronda jugada del Mundial se ejecuta `scripts/recalibrate.py`:

1. Ingesta los resultados oficiales de los partidos disputados.
2. Actualiza el Elo dinámico de las selecciones implicadas.
3. Refresca las features de forma reciente.
4. Re-ejecuta el simulador Monte Carlo **únicamente sobre los partidos pendientes**.
5. Emite un nuevo informe de predicciones versionado: `outputs/predictions/v1_pre.json`, `v2_post_groups.json`, `v3_post_r16.json`, etc.

---

## Roadmap

| Fase | Estado | Descripción |
|---|---|---|
| **v0 — MVP pre-torneo** | en desarrollo | Modelo solo a nivel selección (Elo + ranking FIFA + forma). Poisson básico. Primera predicción del torneo antes del 11/6. |
| **v1 — Modelo híbrido** | pendiente | Añadir features de plantilla (xG, valor de mercado, fuerza de liga). Ensemble Poisson + GBM. |
| **v2 — Recalibración** | pendiente | Pipeline automatizado de actualización tras fase de grupos. |
| **v3 — Recalibración por ronda** | pendiente | Actualización tras octavos, cuartos, semifinales. |
| **v4 — Post-mortem** | pendiente | Análisis comparativo predicción vs realidad. Identificación de fallos del modelo. |

---

## Instalación y uso

```bash
# Clonar el repositorio
git clone https://github.com/molerodev/mundial-2026-prediction.git
cd mundial-2026-prediction

# Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copiar y rellenar variables de entorno
cp .env.example .env

# Descargar todas las fuentes de datos a data/raw/
python scripts/download_data.py

# Entrenar el modelo
python scripts/train.py

# Generar predicción del torneo completo
python scripts/predict.py --output outputs/predictions/v1_pre.json

# Tras cada fase, recalibrar
python scripts/recalibrate.py --phase groups
```

---

## Resultados

> Esta sección se actualizará con cada versión del modelo.

### v0 — Predicción pre-torneo (pendiente)

- Campeón más probable: _por determinar_
- Top 4 favoritos: _por determinar_
- Probabilidades por grupo: _por determinar_

---

## Disclaimer

Este proyecto tiene **fines educativos y de divulgación**. Las predicciones generadas por modelos de Machine Learning aplicados al fútbol tienen un margen de error significativo y **no constituyen asesoramiento de apuestas**.

---

## Autor

**Javier Molero** — Desarrollador full-stack y creador de contenido técnico.

- Web: [javierm.dev](https://javierm.dev)
- YouTube: [@molerodev](https://youtube.com/@molerodev)
- LinkedIn: _por añadir_

---

## Licencia

MIT License. Ver `LICENSE` para más detalles.