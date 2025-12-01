# 💊 PharmaSeven: Pronóstico de Demanda Inteligente

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Gemini AI](https://img.shields.io/badge/AI-Google%20Gemini-orange)

> **Solución End-to-End para el Challenge Grupo Collins 2025B - Matemáticas Aplicadas a Ciencia de Datos.**

## 📖 Descripción del Proyecto

**PharmaSeven** es una herramienta analítica diseñada para optimizar la Cadena de Suministro de **Grupo Collins**, una farmacéutica líder en México. El objetivo principal es resolver la incertidumbre en la planificación de la demanda de antibióticos, mitigando problemas críticos como el **sobreinventario (obsolescencia)** y el **desabasto (pérdida de ventas)**.

La solución no es solo un modelo estático, sino una **Aplicación Web Interactiva** que:
1.  Entrena múltiples modelos de Series de Tiempo en paralelo.
2.  Selecciona automáticamente el mejor modelo basándose en métricas de error (RMSE/MAPE).
3.  Utiliza **Inteligencia Artificial Generativa (Google Gemini)** para explicar los resultados a los tomadores de decisiones en lenguaje natural.

---

## 🚀 Características Principales

* **Multi-Model Forecasting:** Implementación competitiva de algoritmos robustos:
    * 📈 **SARIMA:** Para capturar estacionalidad compleja y autocorrelación.
    * 🌲 **LightGBM:** Machine Learning basado en árboles para relaciones no lineales.
    * 🔮 **Prophet:** Modelo aditivo robusto ante valores atípicos y cambios de tendencia.
    * 📉 **Holt-Winters:** Suavización exponencial triple para tendencias estacionales.
* **AutoML Logic:** El sistema evalúa el rendimiento en un conjunto de prueba (2025) y despliega solo el modelo ganador por producto/cliente.
* **AI Insights 🤖:** Integración vía API con Google Gemini para interpretar *por qué* ganó un modelo (ej. "Detectó mejor la estacionalidad invernal") y ofrecer recomendaciones de negocio.
* **Interfaz Interactiva:** Dashboard en Streamlit para visualización de ventas históricas (2019-2024) y predicciones futuras (2025).

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.9+
* **Frontend:** Streamlit
* **Ciencia de Datos:** Pandas, NumPy, Scikit-Learn, Statsmodels
* **Modelado:** `prophet`, `lightgbm`, `pmdarima`
* **GenAI:** Google Generative AI (Gemini Pro)

---

