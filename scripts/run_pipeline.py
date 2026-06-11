"""
Ejecuta los pasos 4-7 del pipeline en orden.

  python scripts/run_pipeline.py
"""

import subprocess
import sys
import time

PASOS = [
    ("Paso 4 - Tabla maestra de nombres",  "scripts/build_name_master.py"),
    ("Paso 5 - Elo dinamico",              "scripts/build_elo.py"),
    ("Paso 6 - Features de plantilla",     "scripts/build_squad_features.py"),
    ("Paso 7 - Dataset final",             "scripts/build_dataset.py"),
]

python = sys.executable

for titulo, script in PASOS:
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([python, script], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nERROR en {script} (codigo {result.returncode})")
        sys.exit(result.returncode)
    print(f"  ({elapsed:.1f}s)")

print(f"\n{'='*60}")
print("  Pipeline completo.")
print(f"{'='*60}")