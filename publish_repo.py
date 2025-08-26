# publish_repo.py
import subprocess, shlex, pathlib, textwrap

USER_NAME   = "Weimar Cortes"   # tu nombre real o el que quieras que aparezca en commits
USER_EMAIL  = "49415645+Dackarl@users.noreply.github.com"  # tu email noreply de GitHub
REPO_URL    = "https://github.com/Dackarl/PROYECTO-RIO-CAUCA.git"  # la URL de tu repo

def run(cmd):
    print("+", cmd)
    subprocess.run(shlex.split(cmd), check=True)

# escribe archivos clave para despliegue
pathlib.Path("requirements.txt").write_text(
    "streamlit\npandas\nnumpy\nscikit-learn\nmatplotlib\njoblib\nopenpyxl\n", encoding="utf-8"
)
pathlib.Path(".gitignore").write_text(textwrap.dedent("""
.venv/
__pycache__/
.ipynb_checkpoints/
.DS_Store
*.log
*.tmp
artifacts/
artifacts_compare_diccionario/
models/
_backup_run_*/
*.pkl
*.joblib
*.sav
*.parquet
Copia de Calidad_del_agua_del_Rio_Cauca.csv
""").strip("\n"), encoding="utf-8")
pathlib.Path("README.md").write_text(
    "# Río Cauca · Dashboard\n\nApp de Streamlit: `streamlit_app.py`.\n", encoding="utf-8"
)

# configura git y publica
try:
    run(f'git config --global user.name "{USER_NAME}"')
    run(f'git config --global user.email "{USER_EMAIL}"')
except Exception:
    pass

if not pathlib.Path(".git").exists():
    run("git init -b main")

run("git add .")
run('git commit -m "pub: estructura + requirements + app"')

try:
    run(f"git remote add origin {REPO_URL}")
except Exception:
    pass

run("git branch -M main")
run("git push -u origin main")
print("listo: repo publicado en GitHub")
