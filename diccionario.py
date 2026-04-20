"""
Módulo que carga diccionario_variables y reglas_por_parametro desde diccionario.xlsx.
"""
import math
import re
import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _parse_bands(cell: str):
    if not isinstance(cell, str):
        return []
    t = cell.strip().lower()
    if "no aplica" in t or "varía" in t or "varia" in t:
        return []
    out = []
    for part in re.split(r"[;,/]| o ", t):
        s = part.strip()
        if not s:
            continue
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[-\u2013]\s*(-?\d+(?:\.\d+)?)\s*$", s)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if b < a:
                a, b = b, a
            out.append(dict(lo=a, hi=b, lo_inc=True, hi_inc=True))
            continue
        m = re.match(r"^\s*([\u2265>])\s*(-?\d+(?:\.\d+)?)\s*$", s)
        if m:
            op, a = m.group(1), float(m.group(2))
            out.append(dict(lo=a, hi=math.inf, lo_inc=(op == "\u2265"), hi_inc=False))
            continue
        m = re.match(r"^\s*([\u2264<])\s*(-?\d+(?:\.\d+)?)\s*$", s)
        if m:
            op, a = m.group(1), float(m.group(2))
            out.append(dict(lo=-math.inf, hi=a, lo_inc=False, hi_inc=(op == "\u2264")))
            continue
    return out


def _cargar(path=None, hoja="Hoja1"):
    if path is None:
        path = os.path.join(_HERE, "diccionario.xlsx")
    df = pd.read_excel(path, sheet_name=hoja)

    col_param   = [c for c in df.columns if "par\u00e1metro" in c.lower() or "parametro" in c.lower()][0]
    col_def     = [c for c in df.columns if "definici\u00f3n" in c.lower() or "definicion" in c.lower()][0]
    col_rel     = [c for c in df.columns if "relaci\u00f3n" in c.lower() or "relacion" in c.lower()][0]
    col_ref     = [c for c in df.columns if "referencia" in c.lower()][0]

    col_verde   = next((c for c in df.columns if "verde"   in c.lower()), None)
    col_amar    = next((c for c in df.columns if "amarill" in c.lower()), None)
    col_naranja = next((c for c in df.columns if "naranja" in c.lower()), None)
    col_rojo    = next((c for c in df.columns if "rojo"    in c.lower()), None)

    dic_vars  = {}
    reglas    = {}

    for _, r in df.iterrows():
        nombre = str(r[col_param]).strip()
        if not nombre or nombre.lower() == "nan":
            continue
        dic_vars[nombre] = {
            "definicion": r.get(col_def, ""),
            "relacion_contaminacion": r.get(col_rel, ""),
            "referencia": r.get(col_ref, ""),
        }
        if all([col_verde, col_amar, col_naranja, col_rojo]):
            reglas[nombre] = {
                "V": _parse_bands(str(r.get(col_verde,   ""))),
                "A": _parse_bands(str(r.get(col_amar,    ""))),
                "N": _parse_bands(str(r.get(col_naranja, ""))),
                "R": _parse_bands(str(r.get(col_rojo,    ""))),
            }

    return dic_vars, reglas


diccionario_variables, reglas_por_parametro = _cargar()
