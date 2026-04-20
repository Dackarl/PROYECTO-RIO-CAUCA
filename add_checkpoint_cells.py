import json, sys, ast
sys.stdout.reconfigure(encoding='utf-8')

with open('rio_cauca_ml_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Celda GUARDAR — va después de celda 56 (resumenes_optuna)
# índice actual 56 → insertar en posición 57
save_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# GUARDAR ARTEFACTOS (ejecutar una vez terminado el entrenamiento)\n",
        "import joblib, os\n",
        "\n",
        "ruta = os.path.join(os.getcwd(), 'artefactos_checkpoint.pkl')\n",
        "joblib.dump(artefactos, ruta, compress=3)\n",
        "print('Artefactos guardados en:', ruta)\n",
        "print('Claves guardadas:', list(artefactos.keys()))\n",
    ]
}

# Celda CARGAR — va justo después de la celda de guardar (posición 58)
load_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# CARGAR ARTEFACTOS (usar en lugar de re-entrenar)\n",
        "# Ejecuta SOLO esta celda si ya tienes artefactos_checkpoint.pkl guardado\n",
        "import joblib, os\n",
        "\n",
        "ruta = os.path.join(os.getcwd(), 'artefactos_checkpoint.pkl')\n",
        "assert os.path.exists(ruta), f'No existe {ruta} — ejecuta primero el entrenamiento y guarda'\n",
        "\n",
        "artefactos = joblib.load(ruta)\n",
        "print('Artefactos cargados desde:', ruta)\n",
        "print('Claves disponibles:', list(artefactos.keys()))\n",
        "\n",
        "# Verificar campeones\n",
        "if 'campeones_por_objetivo' in artefactos:\n",
        "    for obj, info in artefactos['campeones_por_objetivo'].items():\n",
        "        est_ok = info.get('estimator') is not None\n",
        "        print(f'  {obj}: modelo={info[\"modelo\"]} | esc={info[\"escenario\"]} | estimator={\"OK\" if est_ok else \"None\"}')\n",
        "else:\n",
        "    print('[AVISO] No hay campeones_por_objetivo en el checkpoint')\n",
    ]
}

# Insertar después del índice 56 (resumenes_optuna)
nb['cells'].insert(57, save_cell)
nb['cells'].insert(58, load_cell)

# Verificar sintaxis de ambas celdas
for label, cell in [('GUARDAR', save_cell), ('CARGAR', load_cell)]:
    src = ''.join(cell['source'])
    ast.parse(src)
    print(f'Celda {label}: sintaxis OK')

with open('rio_cauca_ml_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Notebook guardado — total celdas: {len(nb["cells"])}')
print()
print('Nueva estructura:')
print('  Celda 57 → GUARDAR artefactos  (ejecutar después de entrenar)')
print('  Celda 58 → CARGAR artefactos   (ejecutar en sesiones futuras en vez de re-entrenar)')
