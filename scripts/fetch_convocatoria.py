import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("FOOTBALL_DATA_API")
BASE = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": TOKEN}
OUT = r"datos\jugadores\convocatoria\convocatoria.csv"
MAX_RETRIES = 4


def get(path):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            wait = 15 * (attempt + 1)
            print(f"    [reintento {attempt+1}/{MAX_RETRIES}] error de red, esperando {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"No se pudo conectar tras {MAX_RETRIES} intentos: {path}")


# Cargar progreso previo si existe
if os.path.exists(OUT):
    df_prev = pd.read_csv(OUT)
    done_ids = set(df_prev["team_id"].unique())
    rows = df_prev.to_dict("records")
    print(f"Retomando — {len(done_ids)} equipos ya descargados\n")
else:
    done_ids = set()
    rows = []

# 1. Equipos del Mundial 2026
data = get("/competitions/WC/teams?season=2026")
teams = data["teams"]
print(f"{len(teams)} selecciones en total\n")

# 2. Squad de cada equipo (saltando los ya descargados)
for team in teams:
    if team["id"] in done_ids:
        print(f"  [skip] {team['name']}")
        continue

    time.sleep(7)
    squad_data = get(f"/teams/{team['id']}")
    squad = squad_data.get("squad", [])
    for p in squad:
        rows.append({
            "team_id":       team["id"],
            "team_name":     team["name"],
            "player_id":     p.get("id"),
            "player_name":   p.get("name"),
            "position":      p.get("position"),
            "date_of_birth": p.get("dateOfBirth"),
            "nationality":   p.get("nationality"),
            "shirt_number":  p.get("shirtNumber"),
        })
    print(f"  {team['name']}: {len(squad)} jugadores")

    # Guardar después de cada equipo por si vuelve a fallar
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")

print(f"\nListo — {len(rows)} jugadores guardados en {OUT}")
