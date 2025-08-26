# AUTOGENERADO desde diccionario.xlsx (Hoja1)
import math, re, pandas as pd

def _parse_bands(cell:str):
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

def cargar_diccionario(path="diccionario.xlsx", hoja="Hoja1"):
    df = pd.read_excel(path, sheet_name=hoja)
    col_param = "PARÁMETRO (UNIDAD)"
    col_def   = "DEFINICIÓN CONCISA"
    col_rel   = "RELACIÓN CON LA CONTAMINACIÓN (ENFOQUE ECOLÓGICO)"
    col_ref   = "REFERENCIA CLAVE (APA 7)"
    col_verde   = "CALIDAD EXCELENTE / SIN EVIDENCIA DE CONTAMINACIÓN (VERDE)" if "CALIDAD EXCELENTE / SIN EVIDENCIA DE CONTAMINACIÓN (VERDE)"!="None" else None
    col_amar    = "EVIDENCIA LIGERA DE CONTAMINACIÓN (AMARILLO)" if "EVIDENCIA LIGERA DE CONTAMINACIÓN (AMARILLO)"!="None" else None
    col_naranja = "EVIDENCIA CLARA DE CONTAMINACIÓN (NARANJA)" if "EVIDENCIA CLARA DE CONTAMINACIÓN (NARANJA)"!="None" else None
    col_rojo    = "ALTA CONTAMINACIÓN / CONDICIÓN CRÍTICA (ROJO)" if "ALTA CONTAMINACIÓN / CONDICIÓN CRÍTICA (ROJO)"!="None" else None

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

diccionario_variables, reglas_por_parametro = cargar_diccionario()
