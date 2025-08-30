INICIO: Preparación del Entorno y Carga de Datos

  //--- Parte 1: Cargar y verificar las configuraciones ---

  // Se importa un módulo personalizado que contiene definiciones y reglas.
  // Es como abrir tu libro de apuntes antes de empezar a trabajar.
  IMPORTAR el módulo 'diccionario'.
  ASEGURARSE de cargar la versión más reciente del módulo 'diccionario' (para evitar usar una versión antigua en caché).

  // Se realizan varias comprobaciones para asegurar que todo está en orden.
  VERIFICAR que el módulo 'diccionario' se cargó correctamente:
    - CONFIRMAR que la lista de 'diccionario_variables' existe dentro del módulo.
    - INSPECCIONAR los contenidos del módulo para ver qué elementos están disponibles.
    - COMPARAR el número de 'variables' con el número de 'reglas' para asegurar que coinciden (en este caso, hay 53 de cada uno).

  //--- Parte 2: Cargar el conjunto de datos principal ---

  // Ahora, se carga el archivo con los datos de calidad del agua.
  LEER el archivo 'Calidad_del_agua_del_Rio_Cauca.csv'.
    - ESPECIFICAR que el separador de columnas es un punto y coma ';'.
  GUARDAR toda la información en una tabla principal llamada 'df'.

  // Se echa un primer vistazo a los datos.
  MOSTRAR las primeras 5 filas de la tabla 'df' para confirmar que se cargó correctamente.

FIN: Preparación y Carga de Datos

// A partir de la tabla de datos original ('df'), se crea una nueva versión.
  CREAR una nueva tabla llamada 'df_sin_columnas'.
  ELIMINAR de esta nueva tabla las columnas 'FECHA DE MUESTREO' y 'ESTACIONES',
  porque no son necesarias para el análisis posterior.

  //--- Parte 3: Carga y Procesamiento del Diccionario de Variables ---
  
  // Se define un proceso inteligente para leer un archivo Excel que funciona
  // como una base de conocimiento o "diccionario" de todas las variables.
  DEFINIR la función 'cargar_diccionario':
    1. ABRIR el archivo Excel 'diccionario.xlsx'.
    2. IDENTIFICAR automáticamente las columnas importantes (parámetro, definición, rangos de color, etc.) sin importar su orden en el archivo.
    3. RECORRER el archivo fila por fila.
    4. PARA CADA fila (que representa un parámetro de calidad del agua):
        a. EXTRAER su nombre, definición, y otra información descriptiva.
        b. INTERPRETAR los textos que definen los rangos de calidad (ej. '10 - 20', '>50', '≤ 5').
        c. ALMACENAR toda esta información de forma organizada en dos memorias: una para las descripciones ('diccionario_variables') y otra para las reglas de colores ('reglas_por_parametro').

  // Se ejecuta la función anterior para cargar toda la base de conocimiento en el programa.
  EJECUTAR 'cargar_diccionario' y guardar los resultados.

  //--- Parte 4: Creación de la Interfaz de Usuario Interactiva ---

  // Se crean los componentes visuales de la herramienta.
  CREAR una caja de texto para que el usuario pueda **buscar** un parámetro.
  CREAR un menú desplegable para **seleccionar** un parámetro de la lista cargada del Excel.
  CREAR espacios de texto vacíos para **mostrar** la información detallada:
    - Definición
    - Relación con la contaminación
    - Referencia
    - Rangos por color

  // Se programa el comportamiento interactivo de la herramienta.
  PROGRAMAR la lógica para que, **CUANDO el usuario seleccione un parámetro** del menú desplegable:
    1. BUSCAR toda la información de ese parámetro en la memoria.
    2. MOSTRAR la definición, relación y referencia en sus espacios correspondientes.
    3. FORMATEAR y MOSTRAR los rangos de calidad con sus colores y emojis (ej. 🟢 Verde, 🟡 Amarillo, 🟠 Naranja, 🔴 Rojo).

  PROGRAMAR la lógica para que, **CUANDO el usuario escriba algo** en la caja de búsqueda:
    1. FILTRAR las opciones del menú desplegable para que solo muestren las que coinciden con la búsqueda.

  // Finalmente, se ensambla y muestra la herramienta.
  ORGANIZAR todos estos elementos visuales en una sola interfaz vertical.
  MOSTRAR la interfaz interactiva completa al usuario.

FIN: Limpieza y Creación del Explorador

INICIO: Análisis, Limpieza y Filtrado de Datos

  //--- Parte 5: Limpieza Profunda y Estandarización de Tipos de Datos ---

  // Se crea una copia de la tabla para trabajar de forma segura sin alterar el original.
  CREAR una nueva tabla 'df_eda' como una copia de 'df_sin_columnas'.

  // Se realiza una limpieza exhaustiva para asegurar que todos los datos sean numéricos.
  RECORRER cada columna de la tabla 'df_eda':
    1. CONVERTIR todos los valores de la columna a texto.
    2. REEMPLAZAR las comas (',') por puntos ('.') para estandarizar el formato de los decimales.
    3. INTENTAR convertir el texto resultante a un valor numérico.
    4. Si un valor no se puede convertir (porque es un texto como 'N/A'), marcarlo como 'dato faltante'.

  // Se eliminan las columnas que puedan haber quedado completamente vacías tras la limpieza.
  ELIMINAR cualquier columna que contenga ÚNICAMENTE datos faltantes.

  // Se verifica el resultado.
  MOSTRAR un resumen de la tabla 'df_eda' para confirmar que todas las columnas son de tipo numérico.

  //--- Parte 6: Cálculo de Estadísticas Descriptivas Completas ---

  // Se calcula un perfil estadístico detallado para cada variable.
  PARA CADA columna en la tabla 'df_eda', CALCULAR un conjunto completo de estadísticas. Esto incluye:
    - Medidas básicas: media, mediana, desviación estándar, mínimo y máximo.
    - Medidas de distribución: Rango Intercuartílico (IQR), asimetría (Skewness) y curtosis.
    - Medidas de calidad: Porcentaje de datos válidos (no faltantes).
    - Y otras medidas avanzadas como el Coeficiente de Variación (CV).

  // Se mejora la presentación de la tabla de estadísticas para que sea más legible.
  RENOMBRAR algunas columnas para que sean más claras (ej. '50%' se convierte en 'Mediana').
  ORGANIZAR las columnas de la tabla de estadísticas en un orden lógico y predefinido.
  APLICAR un formato visual a los números (ej. redondear a 2 decimales) para que la tabla final sea limpia y profesional.
  
  // Se muestra el resultado final del análisis.
  MOSTRAR la tabla de estadísticas completa y formateada.

  //--- Parte 7: Filtrado de Variables por Calidad de Datos ---

  // Se toma una decisión basada en la cantidad de datos faltantes.
  ESTABLECER un 'umbral de calidad' del 80%.
  (Esto significa que solo confiaremos en las variables que tengan al menos el 80% de sus datos completos).

  // Se identifican las variables que no pasan el filtro.
  IDENTIFICAR y mostrar una lista de las variables que tienen MENOS del 80% de datos válidos.

  // Se crea el conjunto de datos final y limpio.
  SELECCIONAR la lista de todas las variables que SÍ cumplen o superan el umbral del 80%.
  CREAR una nueva tabla final, 'df_filtrado', que contenga ÚNICAMENTE estas columnas de alta calidad.
  
  // Se informa el resultado del filtrado.
  MOSTRAR un mensaje indicando cuántas variables fueron mantenidas (ej. 27 de 39) y mostrar sus nombres.

FIN: Análisis y Filtrado Completados

INICIO: Selección de Variables Objetivo y Preparación de Datos Finales

  //--- Parte 8: Estandarización de Nombres de Variables ---

  // Se crea una función para simplificar y limpiar los nombres de las columnas.
  DEFINIR un proceso de 'limpieza de texto' para nombres de columnas que hace lo siguiente:
    1. Convierte todo a minúsculas.
    2. Elimina acentos (ej. 'oxígeno' -> 'oxigeno').
    3. Quita símbolos especiales como (), /, -, _.
    4. Normaliza los espacios para que solo haya uno entre palabras.
  
  // Se aplica esta limpieza a todas las columnas del conjunto de datos filtrado.
  APLICAR el proceso de limpieza a cada nombre de columna en la tabla 'df_filtrado'.
  GUARDAR un mapa que relacione cada nombre original con su versión simplificada.
  MOSTRAR esta lista para verificar que la limpieza funcionó correctamente.

  //--- Parte 9: Selección Interactiva de la(s) Variable(s) Objetivo ---

  // Se construye una interfaz para que el usuario elija qué variable(s) quiere predecir.
  CREAR una interfaz de usuario interactiva:
    - PARA CADA variable disponible, MOSTRAR una caja de selección (checkbox) con su nombre.
      - Añadir una marca (✅ o ❌) para indicar si la variable tiene información de contexto en el 'diccionario' cargado previamente.
    - CREAR un botón de 'Confirmar selección'.

  // Se programa el comportamiento de la interfaz.
  PROGRAMAR la lógica para que, CUANDO el usuario presione el botón 'Confirmar':
    1. VERIFICAR que se hayan seleccionado 1 o 2 variables (ni más, ni menos).
    2. Si la selección es válida, GUARDAR los nombres de las variables elegidas en una memoria global llamada 'objetivos'.
    3. MOSTRAR un mensaje de confirmación con las variables que se seleccionaron.

  // Se muestra la herramienta al usuario.
  MOSTRAR esta interfaz de checkboxes y el botón.

  //--- Parte 10: Preparación Final de Datos para Modelado (Creación de X y y) ---

  // Se define el proceso final para dejar los datos listos para un modelo de Machine Learning.
  DEFINIR un proceso automatizado ('get_X_y_para') que, dado un 'objetivo', hace lo siguiente:
    1. **Separar Objetivo (y):** Tomar la columna 'objetivo' y guardarla como el vector 'y'. Se eliminan de 'y' las filas que tengan datos faltantes.
    2. **Separar Predictoras (X):** Crear la tabla 'X' con todas las demás columnas que no son el objetivo.
    3. **Alinear Datos:** Asegurarse de que la tabla 'X' contenga exactamente las mismas filas que el vector 'y'.
    4. **Limpiar Predictoras (X):** Realizar una limpieza final en 'X':
        a. Quitar cualquier columna que no sea numérica.
        b. Eliminar columnas que sean constantes (tienen el mismo valor en todas las filas), ya que no aportan información.
    5. **Verificación Final:** Realizar una última alineación para garantizar que 'X' y 'y' son perfectamente compatibles.
    6. **Devolver** los conjuntos de datos 'X' (predictores) y 'y' (objetivo) listos para el modelo.

  // Se ejecuta el proceso anterior con las variables que el usuario seleccionó.
  VERIFICAR si el usuario ya seleccionó sus 'objetivos' con la interfaz interactiva.
  SI lo hizo, ENTONCES PARA CADA 'objetivo' seleccionado:
    - EJECUTAR el proceso de preparación para obtener su 'X' y 'y' correspondientes.
    - MOSTRAR el tamaño (número de filas y columnas) de los conjuntos 'X' y 'y' resultantes.

FIN: Datos Preparados y Listos para Modelar

INICIO: Tratamiento de Datos Faltantes y Análisis de Outliers

  //--- Parte 11: Tratamiento de Datos Faltantes (Imputación KNN) ---
  
  // 11.1: Diagnóstico Inicial
  // Antes de arreglar el problema, primero se mide su magnitud.
  CALCULAR y MOSTRAR una tabla con el conteo y porcentaje de datos faltantes para cada variable.
  GENERAR un 'mapa de calor' visual para identificar patrones en los datos faltantes (dónde el amarillo indica un dato faltante).
  
  // 11.2: Proceso de Imputación
  // Se rellenan los huecos de forma inteligente.
  PREPARAR los datos para la imputación, escalando todas las variables a un rango estándar.
  APLICAR el algoritmo de imputación 'K-Nearest Neighbors' (KNN).
    // Este método 'adivina' los valores faltantes basándose en los 5 registros más similares que sí tienen datos.
  REVERTIR el escalado para devolver los datos a su escala original, ahora sin valores faltantes.

  // 11.3: Verificación
  // Se comprueba que la operación fue exitosa.
  GENERAR un segundo 'mapa de calor' para confirmar visualmente que no quedan datos faltantes (el mapa debe ser de un solo color sólido).

  //--- Parte 12: Análisis y Tratamiento de Valores Atípicos (Outliers) ---

// 12.1: Inspección Visual Inicial (Boxplots)
// (Este paso se mantiene como lo describimos antes)
PARA CADA variable en los datos (ya imputados):
  1. APLICAR una transformación logarítmica 'segura'.
  2. GENERAR un gráfico de 'caja y bigotes' (boxplot) para identificar visualmente la presencia de outliers.

// 12.2: Inspección Visual de Distribuciones (Histogramas)
DEFINIR una regla para decidir si una variable necesita una transformación logarítmica (basado en su asimetría o "skewness").
DEFINIR un proceso de 'preparación para graficar' que:
  a. Aplica la transformación logarítmica solo si la regla anterior lo indica.
  b. 'Winsoriza' los datos para recortar los valores más extremos y facilitar la visualización.

RECORRER todas las variables numéricas en páginas (ej. 16 gráficos por página).
PARA CADA variable:
  1. APLICAR el proceso de 'preparación para graficar'.
  2. GENERAR un histograma para visualizar su distribución.
MOSTRAR las páginas de gráficos con un título general.

// 12.3: Análisis Cuantitativo de Variables Problemáticas
SELECCIONAR las variables que parecen más problemáticas.
PARA CADA una de estas variables:
  a. MOSTRAR los 10 valores más altos para inspección manual.
  b. CALCULAR un informe estadístico con el número y porcentaje exacto de outliers (método IQR).

// 12.4: Demostración de Tratamiento en Boxplots
DEFINIR y APLICAR el proceso de transformación 'robusta' (log + winsorize) a las variables problemáticas.
GENERAR nuevos boxplots para mostrar cómo la técnica reduce los outliers extremos.

FIN: Imputación y Análisis de Outliers Completado

// 13.1: Preparación y Análisis de Correlación
// Se examina la relación entre las variables predictoras.
PREPARAR un conjunto de datos numérico que contenga únicamente las variables predictoras (excluyendo las variables objetivo).
CALCULAR las matrices de correlación entre todos los predictores, usando los métodos de Pearson y Spearman.
GENERAR 'mapas de calor' (heatmaps) para visualizar estas correlaciones, mostrando las relaciones más fuertes.
IDENTIFICAR y MOSTRAR las parejas de predictores que tienen una correlación muy fuerte (ej. mayor a 0.8).

// 13.2: Filtro Automático por Colinealidad (Heurística)
// Se elimina la redundancia de forma inteligente.
DEFINIR un proceso automático para eliminar una variable de cada par altamente correlacionado (ej. Spearman > 0.85).
ESTABLECER una regla inteligente para decidir cuál variable conservar de cada par:
  - **Opción A (si hay objetivos definidos):** Se conservará el predictor que esté más fuertemente correlacionado con la variable objetivo.
  - **Opción B (si no hay objetivos definidos):** Como plan B, se conservará el predictor que, en promedio, tenga la correlación más baja con el resto de predictores.

APLICAR esta regla a todos los pares conflictivos para crear una lista de variables a eliminar.
CREAR una nueva tabla ('df_corr_filtered') eliminando estas variables redundantes y MOSTRAR cuántas variables se conservaron.

//--- Parte 14: Reducción de Multicolinealidad con VIF Iterativo ---

// 14.1: Proceso Iterativo de VIF
// Se aplica un método más avanzado para refinar la selección.
APLICAR el método del 'Factor de Inflación de la Varianza' (VIF) para detectar y eliminar la multicolinealidad restante.
INICIAR un proceso repetitivo (bucle) que se ejecuta hasta que no queden variables con VIF alto:
  1. En cada 'iteración', CALCULAR el VIF para todas las variables restantes.
  2. IDENTIFICAR la variable con el VIF más alto.
  3. **Condición de Parada:** Si el VIF más alto ya es aceptable (ej. < 10), DETENER el proceso.
  4. **Regla de Protección:** Si la variable con el VIF más alto es una variable objetivo, DETENER el proceso para no eliminarla.
  5. **Eliminación:** Si ninguna de las condiciones anteriores se cumple, ELIMINAR la variable con el VIF más alto y volver al paso 1.

MOSTRAR un resumen de las variables eliminadas en cada iteración y la lista final de variables conservadas.

// 14.2: Análisis Final de Correlación (Objetivo vs. Predictores)
// Se verifica la relación de los predictores finales con el objetivo.
COMO verificación final, PARA CADA variable objetivo:
  CALCULAR y MOSTRAR una lista ordenada de la correlación entre el objetivo y cada uno de los predictores finales.

FIN: Selección de Características Completada

//--- Parte 15: Pipeline Automatizado de Modelado y Evaluación ---

// 15.1: Definición de Herramientas y Modelos
// Se definen los procesos y los 'competidores' del torneo de modelado.
DEFINIR un conjunto de procesos auxiliares para automatizar tareas:
  - **Proceso 'Seleccionar Datos de Entrada':** Decide automáticamente cuál es el mejor conjunto de datos de predictores (X) para usar (dando preferencia al que pasó el filtro VIF).
  - **Proceso 'Construir X y y':** Toma un objetivo y prepara los conjuntos de datos finales y alineados para el modelado.
  - **Proceso 'Evaluar Modelo':** Una rutina estándar que entrena un modelo, mide su rendimiento (con R2, RMSE, MAE) y guarda los resultados.

CREAR un 'registro de modelos' con una lista de todos los algoritmos que competirán:
  - Incluir modelos como Regresión Lineal, Árbol de Decisión, Random Forest, SVR y una Red Neuronal (MLP).
  - Configurar automáticamente los modelos que necesitan escalado de datos dentro de un 'pipeline'.
  - Incluir opcionalmente el modelo XGBoost si está instalado en el sistema.

// 15.2: Proceso Principal de Entrenamiento y Evaluación
// El corazón del pipeline: un bucle que lo hace todo para cada objetivo.
VERIFICAR que el usuario haya seleccionado 1 o 2 variables objetivo previamente.
INICIAR un bucle principal que se ejecutará PARA CADA una de las variables objetivo seleccionadas:
  
  1. **Preparar Datos:** Usar el proceso 'Construir X y y' para obtener los datos listos para el objetivo actual.
  2. **Dividir Datos:** Separar los datos en un conjunto de entrenamiento (80%) y uno de prueba (20%).
  3. **Competencia de Modelos:** Iniciar un sub-bucle PARA CADA modelo en el 'registro de modelos':
      a. Entrenar y evaluar el modelo usando el proceso 'Evaluar Modelo'.
      b. Guardar los resultados de rendimiento (R2, RMSE, MAE).
      c. Guardar el modelo ya entrenado para uso futuro.
  4. **Consolidar Resultados:** Crear una tabla de resumen con el rendimiento de todos los modelos para el objetivo actual. Ordenarla por el mejor RMSE (menor error).
  5. **Analizar Importancia de Variables:**
      a. Seleccionar un modelo de referencia (preferiblemente Random Forest o el de mejor rendimiento).
      b. Calcular la 'Importancia por Permutación' para entender qué variables predictoras fueron las más influyentes para las predicciones de ese modelo.
  6. **Mostrar Resultados:** Imprimir en pantalla la tabla de rendimiento de los modelos y la tabla con las variables más importantes para el objetivo actual.

// 15.3: Resumen Global y Almacenamiento de Artefactos
// Se consolidan y guardan todos los resultados finales.
UNA VEZ que el bucle principal ha terminado (procesado todos los objetivos):
  - UNIR los resultados de todos los objetivos en una única tabla de 'Resumen Global'.
  - MOSTRAR esta tabla resumen.

CREAR un 'paquete' final llamado 'artefactos' que contenga todos los productos importantes del pipeline:
  - Los resultados detallados por objetivo.
  - El resumen global.
  - Todos los modelos ya entrenados.
  - Las tablas de importancia de variables.

INFORMAR al usuario que estos artefactos están listos en memoria para ser explorados sin necesidad de volver a ejecutar todo el proceso.

FIN: Pipeline de Modelado y Evaluación Completado

//--- Parte 16: Optimización de Hiperparámetros con Búsqueda Exhaustiva (GridSearchCV) ---

// 16.1: Definición de Herramientas y Espacios de Búsqueda
// Se preparan las herramientas de diagnóstico y el "menú" de opciones para cada modelo.
DEFINIR procesos auxiliares para el diagnóstico:
  - "'Alerta de Sobreajuste'": Una función que revisa si el rendimiento en los datos de entrenamiento es excesivamente mejor que en los de prueba, lo cual es una señal de alarma. 🚨

PARA CADA tipo de modelo (Random Forest, XGBoost, etc.):
  - DEFINIR un "'espacio de búsqueda'": una lista de diferentes valores para sus configuraciones más importantes (ej. para un árbol: `profundidad máxima`, `mínimo de muestras por hoja`, etc.).
  - Seleccionar rangos de búsqueda razonables para evitar configuraciones extremas que tiendan al sobreajuste.

// 16.2: Proceso Principal de Optimización
// Se inicia un torneo para encontrar la mejor versión de cada modelo.
INICIAR un bucle principal que se ejecutará PARA CADA una de las variables objetivo.
  1. PREPARAR y dividir los datos en conjuntos de entrenamiento y prueba para el objetivo actual.
  2. INICIAR un sub-bucle PARA CADA modelo y su 'espacio de búsqueda' definido:
      a. CONFIGURAR un proceso de 'Búsqueda en Rejilla' (GridSearchCV).
      b. INSTRUIR al proceso para que pruebe **todas las combinaciones posibles** de los hiperparámetros definidos.
      c. USAR validación cruzada (CV de 5 pliegues) para evaluar cada combinación de forma robusta.
      d. ESPECIFICAR que el objetivo es encontrar la combinación que minimice el error (RMSE).
      e. EJECUTAR la búsqueda, lo cual puede tomar un tiempo considerable. ⏳

// 16.3: Diagnóstico y Consolidación de Resultados
// Para cada modelo, se analiza al ganador de la búsqueda.
UNA VEZ que la búsqueda para un modelo termina:
  1. OBTENER el modelo con la mejor combinación de hiperparámetros encontrada. 🏆
  2. EVALUAR su rendimiento en los datos de entrenamiento y en los de prueba.
  3. LLAMAR a la 'Alerta de Sobreajuste' para verificar la salud del modelo.
  4. MOSTRAR un informe detallado en pantalla con: la mejor configuración, el rendimiento en train/test/CV y las posibles alertas.
  5. GUARDAR el rendimiento del modelo optimizado en una lista de resultados.

// 16.4: Resumen Global de Modelos Optimizados
// Se presenta la tabla final con los campeones de cada categoría.
UNA VEZ que el bucle principal ha procesado todos los modelos para todos los objetivos:
  - UNIR todos los resultados en una única tabla de 'Resumen Global'.
  - MOSTRAR esta tabla final, ordenada por el mejor rendimiento, para comparar los modelos ya optimizados.

FIN: Optimización de Hiperparámetros Completada

//--- Parte 17: Optimización Inteligente de Hiperparámetros con Optuna ---

// 17.1: Configuración y Definición de "Estudios"
// Se prepara el entorno para una búsqueda guiada e inteligente.
ESTABLECER los parámetros de la búsqueda: número de intentos por modelo (`N_TRIALS`), número de pliegues para la validación cruzada, y una semilla para la reproducibilidad.
DEFINIR una "función objetivo" (o 'estudio') PARA CADA tipo de modelo:
  - Cada 'estudio' le describe a Optuna qué hiperparámetros probar y en qué rangos (ej. `profundidad del árbol` entre 2 y 20).
  - La misión del 'estudio' es construir un modelo con los parámetros sugeridos por Optuna y devolver su rendimiento (error RMSE) medido con validación cruzada.

// 17.2: Proceso Principal de Optimización Inteligente
// Se le pide a Optuna que encuentre la mejor configuración para cada modelo.
INICIAR un bucle principal que se ejecutará PARA CADA una de las variables objetivo.
  1. PREPARAR y dividir los datos en conjuntos de entrenamiento y prueba para el objetivo actual.
  2. INICIAR un sub-bucle PARA CADA 'estudio' de modelo definido:
      a. CREAR un 'estudio' de Optuna con el objetivo de **minimizar** el error.
      b. INICIAR el proceso de optimización (`study.optimize`). Optuna llamará a la "función objetivo" repetidamente (`N_TRIALS` veces), usando su inteligencia para proponer mejores combinaciones de hiperparámetros en cada intento. 🚀

// 17.3: Evaluación Final y Almacenamiento de Resultados
// Se toma la mejor configuración encontrada y se evalúa en datos nunca antes vistos.
UNA VEZ que Optuna termina la búsqueda para un modelo:
  1. OBTENER la mejor combinación de hiperparámetros encontrada.
  2. RECONSTRUIR el modelo usando esa configuración óptima.
  3. ENTRENAR este modelo final con todos los datos de entrenamiento.
  4. EVALUAR su rendimiento definitivo en el conjunto de prueba (datos que nunca vio durante la optimización).
  5. GUARDAR los resultados (rendimiento en CV, en Test y los mejores parámetros) en una lista.
  6. ALMACENAR el modelo final ya entrenado en memoria.

// 17.4: Resumen Global y Guardado de Artefactos
// Se consolidan los resultados y se guardan los productos finales.
UNA VEZ que el bucle principal ha optimizado todos los modelos para todos los objetivos:
  - UNIR todos los resultados en una única tabla de 'Resumen Global'.
  - MOSTRAR esta tabla final para comparar los modelos optimizados.
  - SI está activado, GUARDAR los artefactos finales en disco para uso futuro:
    - La tabla de resumen (en formato Parquet).
    - Todos los mejores modelos entrenados (en un archivo Pickle).
  - INFORMAR al usuario de la ubicación de los archivos guardados. 💾

FIN: Optimización Inteligente Completada

//--- Parte 18: Sistema de Registro de Corridas (Logger) para Experimentación ---

// 18.1: Arquitectura del Logger
// Se establece una "bitácora" para guardar los resultados de los experimentos.
// Esta bitácora tiene dos modos de operación:
// - En Memoria: Un registro rápido que vive mientras el programa se ejecuta, ideal para análisis interactivo.
// - Persistente (en Disco): La capacidad opcional de guardar cada experimento en archivos para que los resultados no se pierdan al cerrar el programa.

// 18.2: Proceso de Registro de una Nueva Corrida (`log_run`)
// Se define el proceso principal para registrar un nuevo experimento.
1. **Generar Metadata:** Al registrar una corrida, primero se crea una "etiqueta" con su información clave:
   - Un ID único basado en la fecha y hora exacta. 🕒
   - La lista de variables objetivo.
   - La lista de variables predictoras.
   - Una "firma digital" (hash) única para el conjunto de predictores, que permite compararlos de forma fiable.

2. **Implementar Lógica de Sobrescritura:** Para mantener la bitácora limpia en aplicaciones interactivas:
   - Antes de guardar, verificar si ya existe una corrida anterior con la **misma combinación exacta** de objetivos y predictores.
   - Si existe, eliminar el registro antiguo para reemplazarlo con el nuevo. 🔄

3. **Guardar en Memoria:** Almacenar la metadata y la tabla de resultados del experimento en la bitácora en memoria.

4. **Guardar en Disco (Opcional):** Si se solicita persistencia:
   - Guardar la metadata y los resultados en archivos Parquet separados.
   - Actualizar un archivo de "índice" central que sirve como un catálogo rápido de todos los experimentos guardados. 📇

// 18.3: Procesos de Consulta (`list_runs` y `get_run`)
// Se definen procesos para poder consultar la bitácora fácilmente.
- **Proceso "Listar Corridas":** Ofrece una vista de resumen de todos los experimentos registrados, ya sea leyendo desde la memoria o desde el índice en disco.
- **Proceso "Obtener Corrida":** Permite recuperar todos los detalles (metadata y tabla de resultados) de un experimento específico usando su ID único.

FIN: Sistema de Registro Definido

//--- Parte 19: Diagnóstico Profundo e Interpretabilidad del Modelo Final ---

// 19.1: Definición del "Panel de Diagnóstico"
// Se define un conjunto de herramientas de análisis avanzado para "interrogar" a cualquier modelo.
- **Curvas de Aprendizaje:** Para diagnosticar si el modelo tiene "hambre" de más datos o si está sobreajustando.
- **Análisis SHAP:** Una "radiografía" del modelo para ver la contribución exacta (positiva o negativa) de cada variable en cada predicción.  Röntgen
- **Importancia por Permutación:** Para identificar las variables más críticas, barajando sus valores y midiendo cuánto "empeora" el rendimiento.
- **Gráficos de Dependencia Parcial (PDP):** Para visualizar la relación que el modelo ha aprendido entre una variable y el resultado (ej. "a más temperatura, más...").
- **Análisis de Sensibilidad (±10%):** Una prueba práctica para medir qué tanto cambia la predicción si alteramos cada variable de entrada en un 10%.

// 19.2: Selección del Modelo "Campeón"
// Se define un proceso final de "campeonato" para elegir al mejor modelo absoluto.
1. REUNIR a todos los modelos finalistas de las optimizaciones anteriores (GridSearch y Optuna).
2. RE-ENTRENAR y EVALUAR a todos los candidatos en un mismo conjunto de datos de prueba para asegurar una comparación justa.
3. DECLARAR como "campeón" al modelo con el menor error (RMSE). 🏆

// 19.3: Proceso de "Examen Final" del Modelo
// El modelo ganador es sometido a un examen exhaustivo.
INICIAR un bucle final que se ejecutará PARA CADA una de las variables objetivo.
  1. **Elegir al Campeón:** Ejecutar el proceso de "campeonato" para el objetivo actual y anunciar al ganador y su rendimiento final.
  2. **Someter al Examen:** Aplicar el "Panel de Diagnóstico" completo al modelo campeón:
      a. GENERAR y MOSTRAR sus Curvas de Aprendizaje.
      b. GENERAR y MOSTRAR los gráficos de análisis SHAP.
      c. IDENTIFICAR sus variables más importantes y GENERAR los Gráficos de Dependencia Parcial para ellas.
      d. REALIZAR y MOSTRAR el Análisis de Sensibilidad.

FIN: Diagnóstico y Análisis de Interpretabilidad Completado