# ================================================
# Proyecto Final - Segunda Entrega
# Autor: Marcos Romano
# ================================================

# ================================================
# 1. Importación de librerías
# ================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

# ================================================
# 2. Abstracto y motivación
# ================================================
"""
Este proyecto analiza la industria de los videojuegos con el objetivo de identificar
qué factores influyen en las ventas globales. La audiencia esperada son ejecutivos
del sector gaming, inversores y analistas de mercado que buscan comprender qué
géneros, plataformas o calificaciones son más relevantes para el éxito comercial.
"""

# ================================================
# 3. Carga de datos
# ================================================
# Cargar dataset local
df = pd.read_csv("Dataset_final.csv")

print("Dimensiones iniciales:", df.shape)
print(df.head())

# ================================================
# 4. Limpieza y transformación
# ================================================
# Eliminar nulos importantes
df = df.dropna(subset=["Global_Sales", "Critic_Score", "User_Score", "Genre", "Platform"])

# Convertir User_Score a numérico
df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")
df = df.dropna(subset=["User_Score"])

# Reset index
df = df.reset_index(drop=True)

print("Dimensiones luego de la limpieza:", df.shape)

# ================================================
# 5. Enriquecimiento desde API (ejemplo RAWG)
# ================================================
"""
Este bloque puede ser comentado tras ejecutar una vez.
Muestra cómo traer datos de una API y guardarlos en CSV para enriquecer el dataset.
"""
# import requests
# url = "https://api.rawg.io/api/games?key=YOUR_API_KEY&page_size=5"
# response = requests.get(url).json()
# games = response["results"]
# df_api = pd.json_normalize(games)
# df_api.to_csv("rawg_sample.csv", index=False)

# ================================================
# 6. Análisis exploratorio de datos (EDA)
# ================================================
plt.figure(figsize=(10,5))
sns.countplot(x="Genre", data=df, order=df["Genre"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Distribución de juegos por género")
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x="Genre", y="Global_Sales", data=df)
plt.xticks(rotation=45)
plt.title("Ventas globales por género")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(x="Critic_Score", y="Global_Sales", data=df)
plt.title("Relación entre puntuación de críticos y ventas globales")
plt.show()

# ================================================
# 7. Modelos de predicción
# ================================================
# Variables de entrada y salida
X = df[["Critic_Score", "User_Score"]]
y = df["Global_Sales"]

# Separar train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Regresión Lineal ----
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

print("Regresión Lineal")
print("MSE:", mean_squared_error(y_test, y_pred_lin))
print("R²:", r2_score(y_test, y_pred_lin))

# ---- Random Forest Regressor ----
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

print("\nRandom Forest Regressor")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R²:", r2_score(y_test, y_pred_rf))

# ================================================
# 8. Clasificación binaria (ventas altas vs bajas)
# ================================================
# Crear target binario: 1 si ventas > 1 millón, 0 si no
df["High_Sales"] = (df["Global_Sales"] > 1.0).astype(int)

X = df[["Critic_Score", "User_Score"]]
y = df["High_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Random Forest Classifier ----
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_clf = rf_clf.predict(X_test)

print("\nRandom Forest Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_clf))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_clf))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_clf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bajas", "Altas"], yticklabels=["Bajas", "Altas"])
plt.title("Matriz de confusión - Ventas altas vs bajas")
plt.show()

# ================================================
# 9. Conclusiones
# ================================================
"""
Conclusiones preliminares:

- La relación entre las calificaciones de críticos/usuarios y las ventas es positiva,
  pero débil: muchos factores externos (marketing, franquicias, exclusividad) afectan las ventas.
- Los modelos de regresión tienen bajo poder predictivo (R² cercano a 0),
  pero Random Forest muestra una ligera mejora respecto a la regresión lineal.
- El modelo de clasificación logra predecir de manera razonable si un juego
  tendrá ventas altas (>1M) usando solo las calificaciones.
- Los géneros más populares y rentables (acción, deportes, shooters) confirman
  la tendencia histórica de la industria.
"""
