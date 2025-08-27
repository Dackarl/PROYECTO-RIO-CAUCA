
import os, glob, json, csv
import pandas as pd
import streamlit as st
import seaborn as sns
import pandas as _pd
PD_NA = _pd.NA

st.set_page_config(page_title="Río Cauca • Dashboard", layout="wide")

def find_latest_artifacts(base="artifacts"):
    if not os.path.isdir(base): return None
    runs = sorted(glob.glob(os.path.join(base, "*")), key=os.path.getmtime)
    return runs[-1] if runs else None

def load_artifact_tables(run_dir, target):
    out = {"metrics": None, "sensitivity": None, "features": None, "sens_plot": None}
    if not run_dir: return out
    tdir = os.path.join(run_dir, target)
    if not os.path.isdir(tdir): return out
    try:
        m = os.path.join(tdir, "metrics.csv")
        s = os.path.join(tdir, "sensitivity.csv")
        f = os.path.join(tdir, "feature_names.json")
        p = os.path.join(tdir, "sensitivity_top15.png")
        if os.path.exists(m): out["metrics"] = pd.read_csv(m)
        if os.path.exists(s): out["sensitivity"] = pd.read_csv(s)
        if os.path.exists(f): out["features"] = json.load(open(f, "r", encoding="utf-8"))
        out["sens_plot"] = p if os.path.exists(p) else None
    except Exception:
        pass
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
    g = df.columns.str.lower()
    col_param = df.columns[g.str.contains("parámetro|parametro")][0]
    col_def   = df.columns[g.str.contains("definición|definicion")][0]
    col_rel   = df.columns[g.str.contains("relación|relacion")][0]
    col_ref   = df.columns[g.str.contains("referencia")][0]
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

    tabs = st.tabs(["Datos base","Diccionario","Métricas y sensibilidad","Resumen"])

    with tabs[0]:
        st.title("Datos base · EDA")

        # 1) Cargar tu CSV
        data_path = "Calidad_del_agua_del_Rio_Cauca.csv"
        df_raw, info = load_raw_csv(data_path)

        if df_raw is None:
            st.info("No se pudo leer el CSV; revisa separador, comillas o filas corruptas.")
        else:
            st.caption(f"lectura: {info}")
            st.write(f"Filas: {len(df_raw):,} · Columnas: {df_raw.shape[1]}")

            # 2) Limpieza básica
            df_eda = df_raw.copy()
            for c in df_eda.columns:
                if df_eda[c].dtype == "object":
                    s = df_eda[c].astype(str).str.strip()
                    s = s.replace({"None": pd.NA, "nan": pd.NA, "": pd.NA})
                    s = s.str.replace(".", "", regex=False)
                    s = s.str.replace(",", ".", regex=False)
                    num = pd.to_numeric(s, errors="coerce")
                    if num.notna().sum() > 0:
                        df_eda[c] = num
            df_eda.dropna(axis="columns", how="all", inplace=True)

            st.subheader("Muestra del dataset limpio")
            st.dataframe(df_eda.head(200), use_container_width=True)

            # 3) Tabla de estadísticas
            import numpy as np
            def build_stats_table(df_num: pd.DataFrame) -> pd.DataFrame:
                est = df_num.describe().T
                est["Tipo de dato"] = df_num.dtypes
                est["IQR"] = est["75%"] - est["25%"]
                est["MAD"] = (df_num - df_num.mean()).abs().mean()
                est["CV"]  = est["std"] / est["mean"]
                est["Skewness"]  = df_num.skew(numeric_only=True)
                est["Kurtosis"]  = df_num.kurtosis(numeric_only=True)

                n = len(df_num)
                est["SE.Skewness"] = np.sqrt((6*n*(n-1))/((n-2)*(n+1)*(n+3)))
                est["Pct.Valid"] = (est["count"]/n)*100

                est = est.rename(columns={
                    "count":"N.Valid","mean":"Mean","std":"Std.Dev","min":"Min",
                    "25%":"Q1","50%":"Median","75%":"Q3","max":"Max"
                })

                cols = ["Tipo de dato","N.Valid","Pct.Valid","Mean","Std.Dev","Min",
                        "Q1","Median","Q3","Max","IQR","MAD","CV","Skewness","SE.Skewness","Kurtosis"]
                est = est[cols]

                def _fmt(x):
                    if pd.isna(x): return ""
                    if isinstance(x,(int,np.integer)): return f"{x}"
                    if isinstance(x,(float,np.floating)):
                        return f"{int(x)}" if float(x).is_integer() else f"{x:.2f}"
                    return x

                est_fmt = est.copy()
                for c in est_fmt.columns:
                    if c not in ("Tipo de dato",):
                        est_fmt[c] = est_fmt[c].map(_fmt)
                est_fmt = est_fmt.reset_index().rename(columns={"index":"Variable"})
                return est_fmt

            df_num = df_eda.select_dtypes(include="number")
            stats_tbl = build_stats_table(df_num)

            st.subheader("Estadísticas completas (post-limpieza/normalización)")
            st.dataframe(stats_tbl, use_container_width=True)

            st.download_button(
                "Descargar estadísticas (CSV)",
                data=stats_tbl.to_csv(index=False).encode("utf-8"),
                file_name="estadisticas_completas.csv",
                mime="text/csv",
            )

            # 4) Auditoría rápida de completitud (EDA puro, solo numéricas)
            st.subheader("Variables con menos del 80% de datos válidos")

            # quedarnos únicamente con columnas numéricas
            df_eda_num = df_eda.select_dtypes(include='number')

            n = len(df_eda_num)
            n_valid = df_eda_num.notna().sum()
            pct_valid = (n_valid / n * 100).round(2)
            tipo = df_eda_num.dtypes.astype(str)

            audit = (
                pd.DataFrame({
                    "Variable": pct_valid.index,
                    "Pct.Valid": pct_valid.values,
                    "N.Valid": n_valid.values,
                    "Tipo de dato": tipo.values
                })
                .sort_values("Pct.Valid")
                .reset_index(drop=True)
            )

            umbral = 80
            audit["Estado"] = np.where(audit["Pct.Valid"] < umbral, "❌ Bajo umbral", "✅ OK")

            st.write(f"Total de variables auditadas: {len(audit)}")
            st.dataframe(audit, use_container_width=True)

            # Enunciado final (solo diagnóstico, no filtra aún)
            n_supervivientes = (audit["Pct.Valid"] >= umbral).sum()
            st.info(f"Con un umbral del {umbral}%, quedarían **{n_supervivientes} variables numéricas** para continuar.")

            # ===== PASO 5 y 6: Nulos antes / Imputación KNN / Nulos después =====
            import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import KNNImputer

            st.subheader("Mapa de nulos ANTES de la imputación")

            # base para imputación: solo numéricas del EDA
            df_eda_num = df_eda.select_dtypes(include="number").copy()

            # definir filtrado por completitud (umbral fijo)
            umbral = 80
            pct_valid_eda = (df_eda_num.notna().sum() / len(df_eda_num) * 100)
            cols_ok = pct_valid_eda[pct_valid_eda >= umbral].index.tolist()
            df_filtrado = df_eda_num[cols_ok].copy()

            # resumen + heatmap antes
            tabla_nulos_antes = (
                pd.DataFrame({
                    "Variable": df_filtrado.columns,
                    "Nulos": df_filtrado.isna().sum().values,
                    "% Nulos": (df_filtrado.isna().sum().values / len(df_filtrado) * 100).round(2)
                }).sort_values("% Nulos", ascending=False)
            )
            st.dataframe(tabla_nulos_antes, use_container_width=True)

            fig, ax = plt.subplots(figsize=(12,6))
            sns.heatmap(df_filtrado.isna(), cbar=False, cmap="viridis", yticklabels=False, ax=ax)
            ax.set_title("Mapa de valores nulos · ANTES de la imputación")
            ax.set_xlabel("Variables"); ax.set_ylabel("Registros")
            st.pyplot(fig)

            st.caption(f"Con umbral {umbral}%, variables consideradas: {len(df_filtrado.columns)}")

            st.subheader("Imputación KNN y verificación")

            # máscara de nulos para identificar imputados
            mask_imputed = df_filtrado.isna()

            # escalar → imputar → desescalar
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(df_filtrado), columns=df_filtrado.columns)
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            X_imp_scaled = pd.DataFrame(imputer.fit_transform(X_scaled), columns=X_scaled.columns)
            df_imputed = pd.DataFrame(scaler.inverse_transform(X_imp_scaled), columns=df_filtrado.columns, index=df_filtrado.index)

            # verificación posterior
            fig, ax = plt.subplots(figsize=(12,6))
            sns.heatmap(df_imputed.isna(), cbar=False, cmap="viridis", yticklabels=False, ax=ax)
            ax.set_title("Mapa de valores nulos · DESPUÉS de la imputación")
            ax.set_xlabel("Variables"); ax.set_ylabel("Registros")
            st.pyplot(fig)

            imputaciones = int(mask_imputed.sum().sum())
            st.success(f"Imputación completada. Celdas imputadas: {imputaciones}. Columnas en df_filtrado: {len(df_filtrado.columns)}")

    with tabs[1]:
        render_diccionario("diccionario.xlsx","Hoja1")

    with tabs[2]:
        art = load_artifact_tables(latest, target)
        if art["metrics"] is not None: st.dataframe(art["metrics"], use_container_width=True)
        if art["sensitivity"] is not None: st.dataframe(art["sensitivity"].head(30), use_container_width=True)
        if art["sens_plot"]: st.image(art["sens_plot"], use_column_width=True)

    with tabs[3]:
        art = load_artifact_tables(latest, target)
        met = art["metrics"]
        st.subheader("Estado general del modelo")
        c1,c2,c3 = st.columns(3)
        if met is not None and "R2" in met: c1.metric("R² promedio", round(met["R2"].mean(),3))
        if met is not None and "RMSE" in met: c2.metric("RMSE promedio", round(met["RMSE"].mean(),3))
        if met is not None and "MAE" in met: c3.metric("MAE promedio", round(met["MAE"].mean(),3))
        if art["features"]: st.caption("Variables usadas por el modelo"); st.write(art["features"])

if __name__ == "__main__":
    main()

# streamlit run streamlit_app.py --PARA EJECUTAR EN TERMINAL
## EJECUTAR LINEA POR LINEA EN TERMINAL PARA ACTUALIZAR EN GITHUB
# git add .
# git commit -m "actualización de archivos"
# git push origin main