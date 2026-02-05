# Análisis de Calidad del Agua - Río Cauca

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de Machine Learning para predecir dos indicadores críticos de calidad del agua en el Río Cauca:
- **DBO (Demanda Bioquímica de Oxígeno)**: Indicador de contaminación orgánica
- **pH**: Medida de acidez/alcalinidad del agua

El análisis utiliza datos fisicoquímicos históricos del río y emplea técnicas avanzadas de selección de características, optimización de hiperparámetros y explicabilidad de modelos.

## Estructura del Repositorio

```
PROYECTO-RIO-CAUCA/
├── General_1.1.ipynb              # Notebook principal con pipeline completo
├── Calidad_del_agua_del_Rio_Cauca.csv  # Dataset principal
├── diccionario.xlsx                # Diccionario de datos de variables
├── requirements.txt                # Dependencias del proyecto
├── README.md                       # Este archivo
├── GUIA_PIPELINE_COMPLETO.md      # Guía detallada del pipeline
└── .gitignore                      # Archivos excluidos del repositorio
```

## Pipeline de Machine Learning

### 1. Preparación de Datos
- Carga y limpieza del dataset
- Estandarización de nombres de columnas
- Imputación de valores faltantes con KNN (k=5)
- Análisis exploratorio de datos (EDA)

### 2. Reducción de Multicolinealidad
- Cálculo de VIF (Variance Inflation Factor)
- Eliminación iterativa de variables con VIF > 10.0
- Objetivo: Reducir redundancia entre predictores

### 3. Selección de Características (RFE)
Se evaluaron 4 escenarios por objetivo utilizando Recursive Feature Elimination:
- **Escenario 1**: 5 variables
- **Escenario 2**: 10 variables
- **Escenario 3**: 15 variables
- **Escenario 4**: 20 variables

### 4. Modelos Evaluados
Para cada escenario se entrenaron 6 modelos:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regressor
5. Random Forest Regressor
6. XGBoost Regressor

### 5. Optimización de Hiperparámetros
- **Grid Search**: Búsqueda exhaustiva en espacio de hiperparámetros
- **Optuna**: Optimización bayesiana con TPE Sampler (40 trials)

### 6. Interpretabilidad y Explicabilidad
- **Feature Importance**: Importancia nativa de modelos basados en árboles
- **SHAP Values**: Explicabilidad global y local usando Shapley Additive Explanations
- **Permutation Importance**: Importancia por perturbación de características

## Resultados

### DBO (Demanda Bioquímica de Oxígeno)
- **Mejor modelo**: Random Forest (Escenario 3 - 15 variables)
- **Desempeño**: R² = 0.91, RMSE = bajo
- **Variables más influyentes**:
  - Sulfatos
  - Conductividad
  - Bicarbonatos
  - Sólidos Disueltos Totales
  - Alcalinidad

### pH
- **Mejor modelo**: XGBoost (Escenario 2 - 10 variables)
- **Desempeño**: R² = 0.36, RMSE moderado
- **Variables más influyentes**:
  - Nitratos
  - Bicarbonatos
  - Potasio
  - Conductividad
  - Cloruros

## Instalación y Uso

### Requisitos
- Python 3.8+
- Jupyter Notebook o JupyterLab

### Instalación de dependencias
```bash
pip install -r requirements.txt
```

### Ejecución
1. Abrir el notebook `General_1.1.ipynb`
2. Ejecutar las celdas secuencialmente
3. Los resultados y gráficas se generan automáticamente
4. Las gráficas de presentación se exportan a `presentacion_graficas/`

## Tecnologías Utilizadas

- **Análisis de datos**: pandas, numpy
- **Visualización**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, XGBoost
- **Optimización**: Optuna
- **Interpretabilidad**: SHAP
- **Ambiente**: Jupyter Notebook

## Archivos Generados (No incluidos en el repositorio)

Los siguientes archivos se generan durante la ejecución pero están excluidos del repositorio:
- `models/`: Modelos entrenados (.pkl, .joblib)
- `presentacion_graficas/`: Gráficas exportadas para presentaciones
- `__pycache__/`: Archivos de caché de Python
- `.venv/`: Entorno virtual

## Notas Técnicas

- **Random State**: Se utiliza `random_state=42` en todos los procesos para garantizar reproducibilidad
- **División de datos**: 80% entrenamiento, 20% prueba
- **Umbral VIF**: 10.0 para eliminación de multicolinealidad
- **Almacenamiento**: Los artefactos (modelos, divisiones, resultados) se guardan en un diccionario `artefactos` para fácil acceso

## Documentación Adicional

Para una guía detallada del pipeline, consultar `GUIA_PIPELINE_COMPLETO.md`.

## Contribución

Este proyecto fue desarrollado como parte de un análisis académico de calidad del agua del Río Cauca.

## Licencia

Proyecto académico - Análisis de Calidad del Agua del Río Cauca
