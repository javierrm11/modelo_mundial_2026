"""
Paso 4 — Tabla maestra de nombres.
Genera datos/master/team_name_master.csv con la correspondencia de nombres
de los 48 equipos del Mundial 2026 entre las tres fuentes:
  - canonical   : nombre en results.csv (clave usada en todo el pipeline)
  - fdorg_id    : ID en football-data.org (= team_id en convocatoria.csv)
  - fdorg_name  : nombre en football-data.org / convocatoria.csv
  - tm_name     : country_of_citizenship en players.csv (Transfermarkt)
"""

import csv
import os

# (canonical, fdorg_id, fdorg_name, tm_name)
TEAMS = [
    ("Algeria",               778,  "Algeria",           "Algeria"),
    ("Argentina",             762,  "Argentina",          "Argentina"),
    ("Australia",             779,  "Australia",          "Australia"),
    ("Austria",               816,  "Austria",            "Austria"),
    ("Belgium",               805,  "Belgium",            "Belgium"),
    ("Bosnia and Herzegovina",1060, "Bosnia-Herzegovina", "Bosnia-Herzegovina"),
    ("Brazil",                764,  "Brazil",             "Brazil"),
    ("Canada",                828,  "Canada",             "Canada"),
    ("Cape Verde",           1930,  "Cape Verde Islands", "Cape Verde"),
    ("Colombia",              818,  "Colombia",           "Colombia"),
    ("DR Congo",             1934,  "Congo DR",           "DR Congo"),
    ("Croatia",               799,  "Croatia",            "Croatia"),
    ("Curaçao",              9460,  "Curaçao",            "Curacao"),
    ("Czech Republic",        798,  "Czechia",            "Czech Republic"),
    ("Ecuador",               791,  "Ecuador",            "Ecuador"),
    ("Egypt",                 825,  "Egypt",              "Egypt"),
    ("England",               770,  "England",            "England"),
    ("France",                773,  "France",             "France"),
    ("Germany",               759,  "Germany",            "Germany"),
    ("Ghana",                 763,  "Ghana",              "Ghana"),
    ("Haiti",                 836,  "Haiti",              "Haiti"),
    ("Iran",                  840,  "Iran",               "Iran"),
    ("Iraq",                 8062,  "Iraq",               "Iraq"),
    ("Ivory Coast",          1935,  "Ivory Coast",        "Cote d'Ivoire"),
    ("Japan",                 766,  "Japan",              "Japan"),
    ("Jordan",               8049,  "Jordan",             "Jordan"),
    ("Mexico",                769,  "Mexico",             "Mexico"),
    ("Morocco",               815,  "Morocco",            "Morocco"),
    ("Netherlands",          8601,  "Netherlands",        "Netherlands"),
    ("New Zealand",           783,  "New Zealand",        "New Zealand"),
    ("Norway",               8872,  "Norway",             "Norway"),
    ("Panama",               1836,  "Panama",             "Panama"),
    ("Paraguay",              761,  "Paraguay",           "Paraguay"),
    ("Portugal",              765,  "Portugal",           "Portugal"),
    ("Qatar",                8030,  "Qatar",              "Qatar"),
    ("Saudi Arabia",          801,  "Saudi Arabia",       "Saudi Arabia"),
    ("Scotland",             8873,  "Scotland",           "Scotland"),
    ("Senegal",               804,  "Senegal",            "Senegal"),
    ("South Africa",          774,  "South Africa",       "South Africa"),
    ("South Korea",           772,  "South Korea",        "Korea, South"),
    ("Spain",                 760,  "Spain",              "Spain"),
    ("Sweden",                792,  "Sweden",             "Sweden"),
    ("Switzerland",           788,  "Switzerland",        "Switzerland"),
    ("Tunisia",               802,  "Tunisia",            "Tunisia"),
    ("Turkey",                803,  "Turkey",             "Turkey"),
    ("United States",         771,  "United States",      "United States"),
    ("Uruguay",               758,  "Uruguay",            "Uruguay"),
    ("Uzbekistan",           8070,  "Uzbekistan",         "Uzbekistan"),
]

OUT_PATH = "datos/master/team_name_master.csv"
os.makedirs("datos/master", exist_ok=True)

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["canonical", "fdorg_id", "fdorg_name", "tm_name"])
    for row in TEAMS:
        writer.writerow(row)

print(f"OK {OUT_PATH}  ({len(TEAMS)} equipos)")