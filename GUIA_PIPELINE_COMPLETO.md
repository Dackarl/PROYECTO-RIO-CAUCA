# Guía del Pipeline Completo - Análisis de Calidad del Agua del Río Cauca

## Resumen General

Este notebook implementa un **pipeline completo de Machine Learning** para predecir **DBO (Demanda Bioquímica de Oxígeno)** y **pH** en el Río Cauca, utilizando datos de múltiples estaciones de monitoreo. El pipeline integra las siguientes etapas:

1. **Exploración de Datos (EDA)**
2. **Preprocesamiento y Filtrado VIF**
3. **Selección de Variables (RFE)**
4. **Entrenamiento Base (6 modelos × 4 escenarios)**
5. **Optimización con Grid Search + Optuna**
6. **Evaluación Avanzada (Permutation Importance, SHAP, Curvas de Aprendizaje)**

---

## Estructura del Pipeline

### **Fase 1: Preparación de Datos**

#### Celdas 1-46: EDA y Preprocesamiento
- **Carga de datos**: `Calidad_del_agua_del_Rio_Cauca.csv`
- **Análisis exploratorio**: Estadísticas, distribuciones, outliers
- **Correlaciones**: Pearson y Spearman
- **Imputación**: KNN Imputer (k=5)
- **Filtrado VIF**: Elimina multicolinealidad (umbral VIF < 5.0)

**Output clave**: `df_vif_filtered` (base para todo el modelado)

---

### **Fase 2: Selección de Variables (RFE)**

#### Celda 48: Recursive Feature Elimination
```python
# Ejecuta RFE con 3 modelos (Linear, Ridge, SVR)
# Genera 3 conjuntos de variables por votación mayoritaria
```

**3 Escenarios RFE generados**:
1. **RFE-DBO**: Variables seleccionadas para predecir DBO
2. **RFE-pH**: Variables seleccionadas para predecir pH
3. **RFE-común(DBO∩pH)**: Variables comunes/unión entre ambos

**Outputs**:
- `rfe_resultados`: Diccionario con selecciones por objetivo
- `rfe_comun`: Lista de variables comunes
- `rfe_tablas`: Tablas formateadas para display

---

### **Fase 3: Entrenamiento Base**

#### Celda 50: 6 Modelos × 4 Escenarios × 2 Objetivos = 48 Combinaciones

**6 Modelos**:
1. `LinearRegression` (con StandardScaler)
2. `DecisionTree (CART)`
3. `RandomForest` (300 estimators)
4. `SVR (RBF)` (con MinMaxScaler)
5. `MLPRegressor` (red neuronal)
6. `XGBoost` (gradient boosting)

**4 Escenarios**:
1. **VIF**: Todas las variables VIF-filtradas (~200 variables)
2. **RFE-DBO**: Variables seleccionadas por RFE para DBO
3. **RFE-pH**: Variables seleccionadas por RFE para pH
4. **RFE-común(DBO∩pH)**: Variables compartidas

**4 Métricas por combinación**:
- **R² (R2)**: Coeficiente de determinación (0-1, mayor es mejor)
- **RMSE**: Root Mean Squared Error (menor es mejor)
- **MSE**: Mean Squared Error (menor es mejor)
- **MAE**: Mean Absolute Error (menor es mejor)

**Outputs**:
- `artefactos["resumen_global_4esc"]`: Tabla con 48 filas de resultados
- `artefactos["modelos_entrenados_4esc"]`: Modelos entrenados guardados
- `artefactos["splits"]`: Train/Test splits (80/20)

**Tiempo estimado**: ~2-3 minutos

---

### **Fase 4: Visualización Base**

#### Celdas 51-52: Top-5 Modelos por R²

**Para cada objetivo (DBO y pH)**:
1. **Gráfico R²**: Barras de los 5 mejores modelos por R²
2. **Gráfico RMSE/MAE**: Comparación de errores

**Características visuales**:
- Paleta elegante: `#2C5F8D` (R²), `#C9546C` (RMSE), `#5FA55A` (MAE)
- Bordes con `edgecolor` y `linewidth=1.2`
- Alpha 0.85 para transparencia
- Escalado dinámico de ejes Y

---

### **Fase 5: Optimización de Hiperparámetros**

#### Celda 54: Grid Search + Optuna

**Estrategia**:
- **Grid Search**: Para LinearRegression, CART, RandomForest, SVR, MLP
- **Optuna (Bayesian Optimization)**: Para XGBoost (40 trials)

**Espacios de búsqueda**:

```python
# Ejemplo: RandomForest
{
    "n_estimators": [300, 600, 900],
    "max_depth": [None, 8, 16],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# Ejemplo: XGBoost (Optuna)
{
    "n_estimators": [300-1200],
    "max_depth": [3-10],
    "learning_rate": [0.01-0.2] (log scale),
    "subsample": [0.6-1.0],
    "colsample_bytree": [0.6-1.0],
    "reg_lambda": [0.001-10.0] (log scale),
    "min_child_weight": [1-8],
    "gamma": [0.0-2.0]
}
```

**Outputs**:
- `artefactos["tabla_tuning_4esc"]`: Resultados con tuning (48 filas)
- `artefactos["modelos_tuning_4esc"]`: Modelos optimizados
- `artefactos["splits_tuning_4esc"]`: Splits usados
- `artefactos["metas_tuning_4esc"]`: Metadatos (best_params, studies)

**Tiempo estimado**: ~15-20 minutos (depende de hardware)

**Ventajas**:
- Mejora R² entre 5-15% típicamente
- Reduce RMSE/MAE significativamente
- Explora espacio de hiperparámetros inteligentemente
- Mantiene compatibilidad total con pipeline base

---

### **Fase 6: Visualización Post-Tuning**

#### Celda 55: Top-5 (Auto-detecta Base o Tuning)

**Lógica inteligente**:
```python
if 'tabla_tuning_4esc' in artefactos:
    df = artefactos['tabla_tuning_4esc']  # Usa resultados con tuning
else:
    df = artefactos['resumen_global_4esc']  # Usa resultados base
```

**Mismo formato elegante que celdas 51-52**, pero ahora muestra:
- Resultados optimizados (si ejecutaste tuning)
- Mejores R², menores RMSE/MAE

---

### **Fase 7: Evaluación Avanzada**

#### Celda 57: Permuted Importance + SHAP

**Para los 2 mejores modelos (1 DBO + 1 pH)**:

1. **Permuted Importance**:
   - 30 repeticiones de permutación
   - Top-10 variables más importantes
   - Incluye desviación estándar

2. **SHAP (solo modelos tipo árbol)**:
   - TreeExplainer o Explainer genérico
   - Top-10 variables por |mean(SHAP)|
   - Identifica impacto direccional

**Outputs**:
- `artefactos["compacto_pi"]`: Tabla PI
- `artefactos["compacto_shap"]`: Tabla SHAP

---

#### Celda 58: Curvas de Aprendizaje

**Para cada objetivo (DBO, pH)**:
- Cross-validation con 5 folds
- 9 puntos de 10% a 100% del dataset
- Muestra RMSE train vs CV
- Identifica overfitting/underfitting

**Interpretación**:
- Train << CV: Overfitting
- Train ≈ CV (ambos altos): Underfitting
- Train ≈ CV (ambos bajos): Buen ajuste

---

#### Celda 59: SHAP Visualizaciones Detalladas

**2 tipos de gráficos por objetivo**:

1. **Beeswarm**:
   - Cada punto = 1 muestra
   - Color = valor de la variable
   - Distribución de impactos SHAP

2. **Bar (mean |SHAP|)**:
   - Top-15 variables
   - Importancia global promedio
   - Barras horizontales con valores

---

## Flujo de Ejecución Recomendado

### **Opción A: Pipeline Completo con Tuning**

```plaintext
1. Celdas 1-46  → Preparación y VIF
2. Celda 48     → RFE (3 tablas)
3. Celda 50     → Entrenamiento Base (referencia)
4. Celdas 51-52 → Visualización Base (opcional, comparar después)
5. Celda 54     → TUNING (Grid + Optuna)
6. Celda 55     → Visualización Top-5 (con tuning)
7. Celda 57     → PI + SHAP (mejores modelos)
8. Celda 58     → Curvas de Aprendizaje
9. Celda 59     → SHAP Detallado
```

**Tiempo total**: ~25-30 minutos

---

### **Opción B: Pipeline Rápido (Sin Tuning)**

```plaintext
1. Celdas 1-46  → Preparación y VIF
2. Celda 48     → RFE
3. Celda 50     → Entrenamiento Base
4. Celdas 51-52 → Visualización Top-5
5. Celda 57     → PI + SHAP (opcional)
```

**Tiempo total**: ~5 minutos

---

## Comparación Base vs Tuning

### **Mejoras Esperadas con Tuning**:

| Modelo          | R² Base | R² Tuning | Mejora |
|-----------------|---------|-----------|--------|
| XGBoost         | 0.85    | 0.92      | +8%    |
| RandomForest    | 0.82    | 0.88      | +7%    |
| SVR             | 0.78    | 0.84      | +8%    |
| MLP             | 0.75    | 0.82      | +9%    |
| CART            | 0.72    | 0.76      | +6%    |
| LinearRegression| 0.68    | 0.68      | 0%     |

*(Valores ilustrativos, varían según dataset)*

---

## Casos de Uso

### **1. Desarrollo y Experimentación**
```python
# Ejecutar solo celdas 1-52 (sin tuning)
# Iterar rápidamente probando diferentes umbrales VIF
```

### **2. Producción/Presentación Final**
```python
# Ejecutar pipeline completo con tuning (celdas 1-59)
# Usar mejores modelos optimizados
# Generar todos los gráficos profesionales
```

### **3. Análisis de Variables**
```python
# Ejecutar hasta celda 59
# Enfocarse en SHAP para identificar variables críticas
# Usar PI para validación cruzada de importancia
```

---

## Interpretación de Resultados

### **Tabla de Resultados (resumen_global_4esc o tabla_tuning_4esc)**

| Columna | Descripción |
|---------|-------------|
| `escenario` | VIF, RFE-DBO, RFE-pH, RFE-común(DBO∩pH) |
| `objetivo` | DBO o pH |
| `modelo` | Uno de los 6 modelos |
| `search` | "grid" o "optuna" (solo en tuning) |
| `R2` | Coeficiente de determinación (0-1) |
| `RMSE` | Raíz del error cuadrático medio |
| `MSE` | Error cuadrático medio |
| `MAE` | Error absoluto medio |
| `n_vars` | Número de variables usadas |
| `vars` | Lista de variables (string) |
| `best_params` | Mejores hiperparámetros (solo tuning) |

### **Cómo Identificar el Mejor Modelo**

1. **Ordenar por RMSE** (ascendente): El más bajo es mejor
2. **Verificar R²**: Debe ser alto (>0.80 es excelente)
3. **Comparar MAE**: Debe estar alineado con RMSE
4. **Revisar n_vars**: Menos variables = modelo más simple (mejor generalización)

**Ejemplo de fila ideal**:
```
escenario: RFE-DBO
objetivo: DEMANDA BIOQUIMICA DE OXIGENO (mg O2/l)
modelo: XGBoost
R2: 0.9234
RMSE: 1.2567
MAE: 0.8934
n_vars: 45
```

---

## Solución de Problemas Comunes

### **Error: "Faltan artefactos en memoria"**
```python
# Solución: Ejecutar celda 50 (entrenamiento base)
# o celda 54 (tuning) primero
```

### **Warning: "Optuna not available"**
```python
# XGBoost usará Grid Search en lugar de Optuna
# Instalación opcional: pip install optuna
```

### **Error: "No hay split para (objetivo, escenario)"**
```python
# Solución: Re-ejecutar celda de entrenamiento (50 o 54)
# Los splits se generan automáticamente
```

### **Gráficos SHAP vacíos**
```python
# SHAP solo funciona con modelos tipo árbol
# Si mejor modelo es Linear/SVR/MLP → No habrá SHAP
# Permuted Importance funciona con todos los modelos
```

---

## Archivos Generados

### **En memoria (artefactos)**:
- `resumen_global_4esc`: Resultados base
- `tabla_tuning_4esc`: Resultados con tuning
- `modelos_entrenados_4esc`: Modelos base
- `modelos_tuning_4esc`: Modelos optimizados
- `splits` / `splits_tuning_4esc`: Train/Test splits
- `compacto_pi`: Permuted Importance
- `compacto_shap`: SHAP values

### **En disco (opcional)**:
```python
# Puedes guardar modelos con:
import joblib
joblib.dump(modelos_tuning_4esc, 'modelos_optimizados.pkl')

# Exportar resultados:
tabla_tuning_4esc.to_csv('resultados_tuning.csv', index=False)
```

---

## Conceptos Clave

### **VIF (Variance Inflation Factor)**
- Detecta multicolinealidad entre variables
- VIF < 5: Buena independencia
- VIF > 10: Alta correlación (eliminar)

### **RFE (Recursive Feature Elimination)**
- Elimina variables recursivamente
- Usa votación de 3 modelos (diversidad)
- Produce conjuntos más compactos

### **Grid Search**
- Búsqueda exhaustiva en grilla
- Evalúa todas las combinaciones
- Más lento pero completo

### **Optuna**
- Optimización bayesiana
- Aprende de trials anteriores
- Más eficiente para espacios grandes

### **Permuted Importance**
- Mide degradación al permutar variables
- Independiente del modelo
- Robusto pero costoso (30 repeticiones)

### **SHAP (SHapley Additive exPlanations)**
- Basado en Teoría de Juegos
- Explica contribución de cada variable
- Beeswarm: distribución por muestra
- Bar: importancia global

---

## Próximos Pasos Sugeridos

1. **Ensambles**: Combinar mejores modelos (Stacking, Voting)
2. **Feature Engineering**: Crear interacciones polinomiales
3. **Análisis Temporal**: Si hay series de tiempo (datos por fecha)
4. **Validación Geográfica**: Separar por estaciones de medición
5. **Deploy**: Crear API con FastAPI o Streamlit para predicciones

---

## Referencias

- **Scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/
- **Optuna**: https://optuna.org/
- **SHAP**: https://shap.readthedocs.io/

---

## Checklist de Ejecución

- [ ] Celdas 1-46 ejecutadas (VIF filtrado completado)
- [ ] Celda 48 ejecutada (3 tablas RFE mostradas)
- [ ] Celda 50 ejecutada (48 modelos base entrenados)
- [ ] Celda 54 ejecutada (tuning completado, ~20 min)
- [ ] Celda 55 ejecutada (Top-5 visualizados)
- [ ] Celda 57 ejecutada (PI + SHAP calculados)
- [ ] Celda 58 ejecutada (Curvas de aprendizaje mostradas)
- [ ] Celda 59 ejecutada (SHAP detallado visualizado)
- [ ] Resultados exportados (opcional)
- [ ] Modelos guardados (opcional)

---

**Versión**: 1.0  
**Fecha**: 2026-02-04  
**Proyecto**: Análisis de Calidad del Agua - Río Cauca  
**Autor**: Equipo de Análisis
