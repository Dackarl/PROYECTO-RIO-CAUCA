
import os, glob, json, csv
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Río Cauca • Dashboard", layout="wide")

def find_latest_artifacts(base="artifacts"):
    if not os.path.isdir(base): return None
    runs = sorted(glob.glob(os.path.join(base, "*")), key=os.path.getmtime)
    return runs[-1] if runs else None

def load_artifact_tables(run_dir, target):
    out = {"metrics": None, "sensitivity": None, "features": None, "sens_plot": None, "model_path": None}
    if not run_dir: return out
    tdir = os.path.join(run_dir, target)
    if not os.path.isdir(tdir): return out
    m = os.path.join(tdir, "metrics.csv")
    s = os.path.join(tdir, "sensitivity.csv")
    f = os.path.join(tdir, "feature_names.json")
    p = os.path.join(tdir, "sensitivity_top15.png")
    mp = os.path.join(tdir, "model.joblib")
    if os.path.exists(m):
        try: out["metrics"] = pd.read_csv(m)
        except Exception: pass
    if os.path.exists(s):
        try: out["sensitivity"] = pd.read_csv(s)
        except Exception: pass
    if os.path.exists(f):
        try: out["features"] = json.load(open(f, "r", encoding="utf-8"))
        except Exception: pass
    out["sens_plot"]  = p if os.path.exists(p)  else None
    out["model_path"] = mp if os.path.exists(mp) else None
    return out

def _parse_bands(cell: str):
    import re, math
    if not isinstance(cell, str): return []
    t = cell.strip().lower()
    if "no aplica" in t or "varía" in t or "varia" in t: return []
    out=[]
    for part in re.split(r"[;,/]| o ", t):
        s=part.strip()
        if not s: continue
        m=re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*$", s)
        if m:
            a,b=float(m.group(1)),float(m.group(2))
            if b<a: a,b=b,a
            out.append(dict(lo=a,hi=b,lo_inc=True,hi_inc=True)); continue
        m=re.match(r"^\s*(≥|>)\s*(-?\d+(?:\.\d+)?)\s*$", s)
        if m:
            op,a=m.group(1),float(m.group(2))
            out.append(dict(lo=a,hi=math.inf,lo_inc=(op=="≥"),hi_inc=False)); continue
        m=re.match(r"^\s*(≤|<)\s*(-?\d+(?:\.\d+)?)\s*$", s)
        if m:
            op,a=m.group(1),float(m.group(2))
            out.append(dict(lo=-math.inf,hi=a,lo_inc=False,hi_inc=(op=="≤"))); continue
    return out

def _fmt_rango(b):
    import math
    li="[" if b["lo_inc"] else "("
    ri="]" if b["hi_inc"] else ")"
    lo="-∞" if b["lo"]==-math.inf else f"{b['lo']:g}"
    hi="+∞" if b["hi"]== math.inf else f"{b['hi']:g}"
    return f"{li}{lo}, {hi}{ri}"

@st.cache_data(show_spinner=False)
def cargar_diccionario(path="diccionario.xlsx", hoja="Hoja1"):
    if not os.path.exists(path): return None, None
    df = pd.read_excel(path, sheet_name=hoja)
    col_param = [c for c in df.columns if ("parámetro" in c.lower() or "parametro" in c.lower())][0]
    col_def   = [c for c in df.columns if ("definición" in c.lower() or "definicion" in c.lower())][0]
    col_rel   = [c for c in df.columns if ("relación"  in c.lower() or "relacion"  in c.lower())][0]
    col_ref   = [c for c in df.columns if "referencia" in c.lower()][0]
    col_verde   = next((c for c in df.columns if "verde"   in c.lower()), None)
    col_amar    = next((c for c in df.columns if "amarill" in c.lower()), None)
    col_naranja = next((c for c in df.columns if "naranja" in c.lower()), None)
    col_rojo    = next((c for c in df.columns if "rojo"    in c.lower()), None)
    dicc, reglas = {}, {}
    for _, r in df.iterrows():
        nombre = str(r[col_param]).strip()
        if not nombre or nombre.lower()=="nan": continue
        dicc[nombre] = {
            "definicion": r.get(col_def,""),
            "relacion_contaminacion": r.get(col_rel,""),
            "referencia": r.get(col_ref,"")
        }
        if all([col_verde,col_amar,col_naranja,col_rojo]):
            reglas[nombre] = {
                "Verde":   _parse_bands(str(r.get(col_verde,""))),
                "Amarillo":_parse_bands(str(r.get(col_amar,""))),
                "Naranja": _parse_bands(str(r.get(col_naranja,""))),
                "Rojo":    _parse_bands(str(r.get(col_rojo,"")))
            }
    return dicc, reglas

def render_diccionario(path="diccionario.xlsx", hoja="Hoja1"):
    dicc, reglas = cargar_diccionario(path, hoja)
    if dicc is None:
        st.info("No se encontró diccionario.xlsx"); return
    st.subheader("Diccionario de variables")
    q = st.text_input("Buscar por nombre")
    opciones = sorted([k for k in dicc.keys() if q.lower() in k.lower()] or dicc.keys(), key=str.casefold)
    var = st.selectbox("Variable", opciones, index=0)
    info = dicc.get(var, {})
    st.markdown(f"**Definición**  \n{info.get('definicion','(sin definición)')}")
    st.markdown(f"**Relación con la contaminación**  \n{info.get('relacion_contaminacion','(sin información)')}")
    st.markdown(f"**Referencia**  \n{info.get('referencia','(sin referencia)')}")
    if reglas and var in reglas:
        filas=[{"Categoria": c, "Rangos": ", ".join(_fmt_rango(b) for b in bs)} for c,bs in reglas[var].items() if bs]
        if filas: st.dataframe(pd.DataFrame(filas), use_container_width=True)

st.sidebar.button("Reiniciar (limpiar caché)", on_click=lambda: (st.cache_data.clear(), st.rerun()))

@st.cache_data(show_spinner=False)
def load_raw_csv(path):
    if not os.path.exists(path): return None, "no existe"
    try:
        sample = open(path, "r", encoding="utf-8", errors="ignore").read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        df = pd.read_csv(path, sep=dialect.delimiter, engine="python", on_bad_lines="skip", encoding_errors="ignore")
        return df, f"delimitador='{dialect.delimiter}'"
    except Exception:
        try:
            df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip", encoding_errors="ignore")
            return df, "forzado sep=';'"
        except Exception as e2:
            return None, f"error: {e2}"

def main():
    st.title("Río Cauca · Dashboard")
    latest = find_latest_artifacts("artifacts")
    st.sidebar.write("Carpeta de artefactos detectada:")
    st.sidebar.code(latest if latest else "no encontrada", language="bash")
    target = st.sidebar.selectbox("Variable objetivo", ["DEMANDA BIOQUIMICA DE OXIGENO (mg O2-l)", "pH"], index=0)

    tabs = st.tabs(["Resumen","Métricas y sensibilidad","Comparación con diccionario","Datos base","Diccionario"])

    with tabs[0]:
        art = load_artifact_tables(latest, target)
        met = art["metrics"]
        st.subheader("Estado general")
        c1,c2,c3 = st.columns(3)
        if met is not None and "R2" in met: c1.metric("R² promedio", round(met["R2"].mean(),3))
        if met is not None and "RMSE" in met: c2.metric("RMSE promedio", round(met["RMSE"].mean(),3))
        if met is not None and "MAE" in met: c3.metric("MAE promedio", round(met["MAE"].mean(),3))
        if art["sens_plot"]: st.image(art["sens_plot"], use_column_width=True, caption="Top 15 por importancia")
        if art["features"]: st.caption("Variables usadas por el modelo"); st.write(art["features"])

    with tabs[1]:
        art = load_artifact_tables(latest, target)
        if art["metrics"] is not None: st.dataframe(art["metrics"], use_container_width=True)
        if art["sensitivity"] is not None: st.dataframe(art["sensitivity"].head(30), use_container_width=True)
        if art["sens_plot"]: st.image(art["sens_plot"], use_column_width=True)

    with tabs[2]:
        base = os.path.join("artifacts_compare_diccionario", target)
        perm = os.path.join(base, "permutation_vs_diccionario.csv")
        top  = os.path.join(base, "pi_top20.png")
        desac = os.path.join(base, "top_desacuerdos.csv")
        if os.path.exists(perm): st.dataframe(pd.read_csv(perm), use_container_width=True)
        if os.path.exists(top): st.image(top, use_column_width=True)
        if os.path.exists(desac): st.dataframe(pd.read_csv(desac), use_container_width=True)

    with tabs[3]:
        data_path = "Copia de Calidad_del_agua_del_Rio_Cauca.csv"
        df, info = load_raw_csv(data_path)
        if df is not None:
            st.caption(f"lectura: {info}")
            st.write(f"Filas: {len(df):,} · Columnas: {df.shape[1]}")
            st.dataframe(df.head(200), use_container_width=True)
        else:
            st.info("No se pudo leer el CSV; revisa separador, comillas o filas corruptas.")

    with tabs[4]:
        render_diccionario("diccionario.xlsx","Hoja1")

if __name__ == "__main__":
    main()
