# Proyecto Final - Ciencia de Datos (Videojuegos)

## 📌 Abstracto y Motivación
La industria de los videojuegos es uno de los sectores de entretenimiento más rentables a nivel mundial. Comprender qué factores influyen en el éxito comercial de un título resulta clave para inversores, estudios de desarrollo y distribuidores.  

Este proyecto busca responder a una pregunta central: **¿qué variables permiten predecir el rendimiento en ventas globales de un videojuego?**.  
La audiencia principal son ejecutivos de la industria, analistas de mercado y equipos de planificación estratégica que requieren información confiable para la toma de decisiones.

## 📊 Dataset
- Fuente: dataset de ventas de videojuegos con información histórica (hasta ~2016).  
- Dimensiones: ~16.000 filas y 16 columnas.  
- Variables clave: plataforma, género, año de lanzamiento, calificaciones de críticos y usuarios, ventas por región y globales.  
- Enriquecimiento: se incluyó un ejemplo de integración con la API de RAWG, que puede proveer información adicional de juegos modernos.  

## ❓ Preguntas de Interés
1. ¿Qué géneros concentran mayor cantidad de lanzamientos y ventas globales?  
2. ¿Las calificaciones de críticos y usuarios son buenos predictores del éxito comercial?  
3. ¿Se puede anticipar si un juego venderá más de 1 millón de copias?  

## 🛠️ Metodología
1. **Limpieza de datos:**  
   - Eliminación de nulos en variables críticas.  
   - Conversión de `User_Score` a formato numérico.  
   - Normalización y estandarización básica.  

2. **Análisis Exploratorio (EDA):**  
   - Distribución de géneros y ventas.  
   - Boxplots para comparar géneros en términos de ingresos.  
   - Scatterplots entre calificaciones y ventas.  

3. **Modelado Predictivo:**  
   - **Regresión Lineal y Random Forest Regressor** para estimar ventas globales.  
   - **Random Forest Classifier** para clasificar juegos en dos categorías: ventas altas (>1M) y bajas (≤1M).  

4. **Evaluación de modelos:**  
   - Métricas de regresión: **MSE** y **R²**.  
   - Métricas de clasificación: **accuracy**, matriz de confusión y reporte de clasificación.  

## 📈 Resultados
- La relación entre las calificaciones de críticos/usuarios y las ventas globales es **positiva pero débil**.  
- Los modelos de regresión obtuvieron **R² ≈ 0.04**, lo que indica bajo poder explicativo.  
- El modelo de clasificación con Random Forest alcanzó un desempeño aceptable para predecir si un juego superará el millón de ventas.  
- Los géneros de **acción, deportes y shooters** concentran la mayor parte de los títulos y las ventas, confirmando tendencias históricas.  

## 💡 Insights Ejecutivos
- **Las reseñas influyen, pero no determinan el éxito.** Factores externos como marketing, exclusividad de plataforma o pertenencia a franquicias son decisivos.  
- **Los géneros con mayor retorno (acción, deportes, shooters)** deberían ser prioritarios en estrategias de inversión.  
- **El modelo de clasificación aporta valor práctico**: permite detectar juegos con altas probabilidades de superar un umbral de ventas, útil para estimar riesgos.  
- **La predicción exacta de ventas es limitada** → se recomienda integrar datos de mercado actuales (digital, microtransacciones, F2P) para mejorar el análisis.  

## ⚠️ Limitaciones
- Dataset histórico, no refleja la industria digital moderna.  
- No incluye variables como inversión en marketing, acuerdos de exclusividad o tendencias de plataformas online.  

## ✅ Conclusión General
El análisis confirma que, si bien las calificaciones aportan información útil, **no son suficientes para explicar el éxito de un videojuego en ventas**. El verdadero valor está en combinar métricas de percepción del público con factores comerciales externos.  

Los modelos de clasificación permiten obtener una visión práctica del mercado, ofreciendo a ejecutivos e inversores una herramienta para **detectar de forma temprana títulos con potencial de “ventas altas”** y así optimizar la asignación de recursos.  

## 📌 Entrega Final: Machine Learning

Este proyecto incorpora modelos de Machine Learning para predecir el éxito comercial de videojuegos.

### Modelos utilizados
- Clasificación: Random Forest y XGBoost
- Regresión: Linear Regression

### Ingeniería de atributos
- Normalización de variables numéricas
- Encoding de variables categóricas
- Creación de nuevas variables derivadas

### Optimización
- GridSearchCV aplicado a Random Forest
- Validación cruzada para selección de modelo

### Resultados
- El modelo de clasificación Random Forest optimizado obtuvo mejor desempeño (AUC > 0.75).
- La regresión mostró bajo poder explicativo (R² < 0.3).
- Se recomienda usar clasificación como herramienta práctica para detectar títulos con alto potencial de ventas.

### Conclusiones
Los modelos de Machine Learning permiten anticipar el éxito de un videojuego combinando reseñas y características de género/plataforma. La clasificación es más útil para la audiencia ejecutiva e inversora, mientras que la regresión requiere datos adicionales para mejorar su precisión.
