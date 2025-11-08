import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from pmdarima import auto_arima
from prophet import Prophet
import google.generativeai as genai
import time
import warnings

# --- Configuración de la Página y Advertencias ---
st.set_page_config(layout="wide", page_title="Dashboard de Predicción de Ventas")
warnings.filterwarnings('ignore')

# --- Constantes y Nombres ---
# ¡ASEGÚRATE DE QUE ESTA RUTA SEA CORRECTA EN GITHUB!
NOMBRE_ARCHIVO_DATOS = 'data/datos_finales_listos_para_modelo.csv' 
COLUMNA_PRODUCTO = 'Producto - Descripción'
COLUMNA_CLIENTE = 'Cliente - Descripción'
COLUMNA_FECHA = 'Fecha'
METRICAS_PREDICCION = ['Pedido_piezas', 'Pedido_MXN', 'Factura_piezas', 'Factura_MXN']

# --- Funciones de Carga y Preparación de Datos ---

# La carga de datos SÍ la dejamos en caché. Esto es seguro y rápido.
@st.cache_data
def cargar_datos(nombre_archivo):
    """Carga y pre-procesa los datos desde el CSV."""
    try:
        df = pd.read_csv(nombre_archivo)
        df[COLUMNA_FECHA] = pd.to_datetime(df[COLUMNA_FECHA], format='%d/%m/%Y')
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{nombre_archivo}'.")
        st.error("Verifica que la ruta sea correcta (ej. 'data/archivo.csv').")
        return None
    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo: {e}")
        return None

def preparar_series_de_tiempo(df_filtrado, metrica_seleccionada):
    """Agrupa los datos filtrados y los prepara como serie de tiempo mensual."""
    if df_filtrado.empty:
        return None
    
    ts_data = df_filtrado.groupby(COLUMNA_FECHA)[metrica_seleccionada].sum().reset_index()
    ts_data = ts_data.set_index(COLUMNA_FECHA)
    ts_data = ts_data.asfreq('MS', fill_value=0)
    ts_data = ts_data[metrica_seleccionada]
    
    if len(ts_data) < 24: # Aumentamos el mínimo para modelos estacionales
        st.warning("Advertencia: Se tienen menos de 24 meses de datos. Las predicciones estacionales pueden no ser fiables.")
        if len(ts_data) < 12:
             return None
        
    return ts_data

# --- Funciones de Métricas ---

def calcular_metricas(y_true, y_pred):
    """Calcula RMSE y MAPE para un modelo."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true[y_true != 0], y_pred[y_true != 0])
    return {'RMSE': rmse, 'MAPE': mape}

# --- Funciones de Modelos ---

def run_model(model_name, model_func, ts_train, ts_test, ts_full, n_forecast):
    """
    Función genérica para correr un modelo.
    Ahora devuelve la predicción Y el intervalo de confianza.
    """
    start_time = time.time()
    
    # 1. Entrenar en 'train' y predecir en 'test' para métricas
    # No necesitamos el intervalo para el test, así que tomamos el primer valor [0]
    pred_test, _ = model_func(ts_train, len(ts_test)) 
    metrics = calcular_metricas(ts_test, pred_test)
    
    # 2. Entrenar en 'full' y predecir el futuro
    pred_future, conf_int = model_func(ts_full, n_forecast)
    
    end_time = time.time()
    st.write(f"Modelo '{model_name}' completado en {end_time - start_time:.2f}s")
    
    return {
        'name': model_name,
        'metrics': metrics,
        'forecast': pred_future,
        'interval': conf_int,  # <-- ¡NUEVO!
        'test_prediction': pred_test
    }

# --- Modelos Potentes (No planos) ---

def model_holt_winters(ts_data, n_steps):
    """Suavizamiento Exponencial Triple (Holt-Winters)."""
    try:
        model = ExponentialSmoothing(
            ts_data, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=12
        ).fit()
    except Exception:
        # Fallback si falla el modelo completo
        try:
            model = ExponentialSmoothing(
                ts_data, 
                trend=None, 
                seasonal='add', 
                seasonal_periods=12
            ).fit()
        except Exception:
            # Fallback final
            model = SimpleExpSmoothing(ts_data, initialization_method="estimated").fit()

    forecast = model.forecast(n_steps)
    return forecast, None # No devuelve intervalo de confianza

def model_arima(ts_data, n_steps):
    """Auto-ARIMA (SARIMA) optimizado y con intervalo de confianza."""
    model = auto_arima(ts_data, 
                       seasonal=True, m=12,
                       stepwise=True, suppress_warnings=True, error_action='ignore',
                       max_p=2, max_q=2, max_P=1, max_Q=1,
                       d=None, D=1)
    
    # Pedimos la predicción Y el intervalo de confianza (alpha=0.1 es 90% confianza)
    forecast, conf_int_df = model.predict(n_periods=n_steps, return_conf_int=True, alpha=0.1)
    
    # Convertir el array de numpy a un DataFrame con nombres
    conf_int_result = pd.DataFrame(conf_int_df, index=forecast.index, columns=['lower', 'upper'])
    
    return forecast, conf_int_result

def model_prophet(ts_data, n_steps):
    """Modelo Prophet con intervalo de confianza."""
    df_prophet = ts_data.reset_index()
    df_prophet.columns = ['ds', 'y']
    
    # Nivel de intervalo de 0.90 (90%)
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                    daily_seasonality=False, interval_width=0.90)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=n_steps, freq='MS')
    forecast_df = model.predict(future)
    
    # Extraer solo la predicción futura
    future_results = forecast_df.iloc[-n_steps:]
    
    # Extraer predicción puntual
    forecast = future_results['yhat']
    forecast.index = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=n_steps, freq='MS')
    
    # Extraer intervalos y renombrar columnas
    conf_int_result = future_results[['yhat_lower', 'yhat_upper']]
    conf_int_result.columns = ['lower', 'upper']
    conf_int_result.index = forecast.index
    
    return forecast, conf_int_result

# --- Función de Gemini AI ---

def get_gemini_analysis(metrics_summary, n_meses, metrica_nombre):
    """Llama a la API de Gemini para analizar los resultados."""
    if not api_key:
        return "Error: No se encontró la API Key. Asegúrate de que esté configurada en los 'Secrets' de Streamlit."
    
    try:
        genai.configure(api_key=api_key)
        # Usamos el modelo más rápido y moderno
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        Eres un analista de datos senior especializado en pronósticos de ventas.
        Quiero predecir '{metrica_nombre}' para los próximos {n_meses} meses.
        
        He corrido 3 modelos (SARIMA, Prophet, Holt-Winters) y he calculado sus métricas de error (RMSE y MAPE) en un conjunto de prueba. 
        Un valor más bajo es mejor para ambas métricas.

        Estos son los resultados:
        {metrics_summary.to_string()}

        Por favor, responde con lo siguiente en un formato markdown claro:
        
        1.  **Recomendación del Modelo:** ¿Cuál es el modelo más eficiente y por qué? (Prioriza MAPE).
        2.  **Análisis de Resultados:** Explica brevemente por qué este modelo pudo haber ganado (ej. "SARIMA/Prophet capturó bien la estacionalidad...") y por qué otros pudieron fallar.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error al contactar la API de Gemini: {e}")
        return "No se pudo generar el análisis. Verifica tu API Key o el nombre del modelo."

# --- == APLICACIÓN PRINCIPAL (STREAMLIT) == ---

st.title("📈 Dashboard de Predicción de Ventas")

# --- Carga de Datos ---
df = cargar_datos(NOMBRE_ARCHIVO_DATOS)

if df is not None:
    
    # --- Barra Lateral (Filtros) ---
    st.sidebar.header("⚙️ Configuración de la Predicción")
    
    # Lee la API Key desde los "Secrets" de Streamlit
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    # --- Lógica de Filtros Simple (¡SIN CACHÉ!) ---
    # Esta lógica es simple. Streamlit recuerda la selección del
    # multiselect entre ejecuciones mientras no se refresque la página.
    
    productos_lista = df[COLUMNA_PRODUCTO].unique().tolist()
    # Inicia con todo seleccionado
    productos_seleccionados = st.sidebar.multiselect(
        "Selecciona Productos:", 
        options=productos_lista, 
        default=productos_lista
    )
    
    clientes_lista = df[COLUMNA_CLIENTE].unique().tolist()
    # Inicia con todo seleccionado
    clientes_seleccionados = st.sidebar.multiselect(
        "Selecciona Clientes:", 
        options=clientes_lista, 
        default=clientes_lista
    )
    
    # --- Filtros de Métrica y Horizonte ---
    metrica_seleccionada = st.sidebar.selectbox("Selecciona la Métrica a Predecir:", METRICAS_PREDICCION)
    n_meses_prediccion = st.sidebar.slider("Meses a Predecir:", min_value=1, max_value=24, value=12)
    
    # --- Botón para ejecutar ---
    if st.sidebar.button("🚀 Generar Predicción", type="primary"):
        
        if not productos_seleccionados or not clientes_seleccionados:
            st.warning("Por favor, selecciona al menos un producto y un cliente.")
        elif not api_key:
            st.error("Error: No se encontró la 'GEMINI_API_KEY'.")
            st.error("Por favor, agrégala en 'Settings > Secrets' en Streamlit Cloud y reinicia la app.")
        else:
            # ¡SIN CACHÉ! Esto se ejecuta siempre
            with st.spinner(f"Ejecutando predicción para {n_meses_prediccion} meses... Esto puede tardar unos minutos..."):
                
                # --- 1. Preparación de Datos (se hace siempre) ---
                df_filtrado = df[
                    (df[COLUMNA_PRODUCTO].isin(productos_seleccionados)) &
                    (df[COLUMNA_CLIENTE].isin(clientes_seleccionados))
                ]
                
                ts_full = preparar_series_de_tiempo(df_filtrado, metrica_seleccionada)
                
                if ts_full is not None:
                    
                    # --- 2. División Train/Test (se hace siempre) ---
                    # A. División FIJA 80/20 para métricas y gráfico de evaluación
                    split_point = int(len(ts_full) * 0.8)
                    ts_train = ts_full.iloc[:split_point]
                    ts_test = ts_full.iloc[split_point:]

                    # B. El slider (n_meses_prediccion) es solo para el pronóstico FUTURO.
                    #    (Ya lo tenemos en la variable n_meses_prediccion)
                    
                    # C. Validación (asegurarnos de que el 20% no sea muy pequeño)
                    if len(ts_test) < 2 or len(ts_train) < 12:
                        st.error(f"Error: No hay suficientes datos para una división 80/20 válida (Train: {len(ts_train)}, Test: {len(ts_test)}).")
                        st.stop() # Detener la ejecución si no hay datos
                    else:
                        
                        # --- 3. Ejecución de Modelos (se hace siempre) ---
                        st.write("Entrenando modelos...")
                        
                        model_pipeline = [
                            ('SARIMA', model_arima),
                            ('Prophet', model_prophet),
                            ('Holt-Winters', model_holt_winters)
                        ]
                        
                        all_metrics = {}
                        all_forecasts = {}
                        all_intervals = {} # ¡NUEVO! Para guardar los intervalos
                        all_test_preds = {}
                        
                        for name, func in model_pipeline:
                            try:
                                resultado = run_model(name, func, ts_train, ts_test, ts_full, n_meses_prediccion)
                                all_metrics[name] = resultado['metrics']
                                all_forecasts[name] = resultado['forecast']
                                all_intervals[name] = resultado['interval']
                                all_test_preds[name] = resultado['test_prediction'] # <--- ¡GUARDA EL RESULTADO!
                            except Exception as e:
                                st.error(f"Error al ejecutar el modelo '{name}': {e}")
                        
                        if not all_metrics:
                            st.error("No se pudieron ejecutar los modelos. Revisa los datos o filtros.")
                            st.stop()
                            
                        df_metrics = pd.DataFrame(all_metrics).T.sort_values(by='MAPE')
                        df_forecast = pd.DataFrame(all_forecasts)
                        df_forecast.index.name = "Fecha"

                        # --- 4. Análisis con Gemini ---
                        st.write("Enviando resultados a Gemini para análisis...")
                        with st.spinner("🧠 Gemini está pensando..."):
                            analisis_gemini = get_gemini_analysis(df_metrics, n_meses_prediccion, metrica_seleccionada)
                        
                        st.subheader("🤖 Análisis y Recomendación (Gemini AI)")
                        st.markdown(analisis_gemini)

                        
                        # --- 5. Gráfico de Comparación (Todos los modelos) ---
                        st.subheader("📊 Gráfico de Comparación (Todos los Modelos)")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=ts_full.index, y=ts_full.values,
                            mode='lines+markers', name='Datos Históricos'
                        ))
                        for model_name in df_forecast.columns:
                            fig.add_trace(go.Scatter(
                                x=df_forecast.index, y=df_forecast[model_name],
                                mode='lines', name=f'Predicción: {model_name}'
                            ))
                        fig.update_layout(title=f"Comparación de Modelos - '{metrica_seleccionada}'")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # --- 6. GRÁFICO DEL MEJOR MODELO (¡NUEVO!) ---
                        best_model_name = df_metrics.index[0]
                        best_forecast = all_forecasts[best_model_name]
                        best_interval = all_intervals.get(best_model_name) # .get() no da error si no existe

                        st.subheader(f"📈 Análisis Detallado del Mejor Modelo: {best_model_name}")
                        
                        fig_best = go.Figure()

                        # Intervalo de confianza (se dibuja primero para que quede de fondo)
                        if best_interval is not None:
                            fig_best.add_trace(go.Scatter(
                                x=best_interval.index, y=best_interval['upper'],
                                mode='lines', name='Máximo (Intervalo 90%)',
                                line=dict(width=0.5, color='gray')
                            ))
                            fig_best.add_trace(go.Scatter(
                                x=best_interval.index, y=best_interval['lower'],
                                mode='lines', name='Mínimo (Intervalo 90%)',
                                line=dict(width=0.5, color='gray'),
                                fill='tonexty', # Rellena el área entre 'lower' y 'upper'
                                fillcolor='rgba(150,150,150,0.2)' # Color de relleno gris claro
                            ))

                        # Dato Histórico
                        fig_best.add_trace(go.Scatter(
                            x=ts_full.index, y=ts_full.values,
                            mode='lines', name='Datos Históricos',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Predicción del mejor modelo
                        fig_best.add_trace(go.Scatter(
                            x=best_forecast.index, y=best_forecast.values,
                            mode='lines', name='Predicción (Mejor Modelo)',
                            line=dict(color='green', width=3, dash='dash')
                        ))
                        
                        fig_best.update_layout(title=f"Predicción e Intervalo de Confianza - {best_model_name}")
                        st.plotly_chart(fig_best, use_container_width=True)

                        # --- 5. GRÁFICO DE EVALUACIÓN (¡NUEVO!) ---
                        st.subheader(f"🛠️ Gráfico de Evaluación del Modelo (Train/Test)")
                        
                        best_model_name_eval = df_metrics.index[0] # Tomamos el mejor modelo
                        best_test_pred = all_test_preds[best_model_name_eval]

                        fig_eval = go.Figure()

                        # 1. Datos de Entrenamiento
                        fig_eval.add_trace(go.Scatter(
                            x=ts_train.index, y=ts_train.values,
                            mode='lines', name='1. Datos de Entrenamiento (80%)', # <-- CAMBIO DE TEXTO
                            line=dict(color='blue')
                        ))

                        # 2. Datos Reales de Prueba
                        fig_eval.add_trace(go.Scatter(
                            x=ts_test.index, y=ts_test.values,
                            mode='lines+markers', name='2. Datos Reales (Test - 20%)', # <-- CAMBIO DE TEXTO
                            line=dict(color='black', width=3)
                        ))

                        # 3. Predicción sobre el Test
                        fig_eval.add_trace(go.Scatter(
                            x=best_test_pred.index, y=best_test_pred.values,
                            mode='lines', name=f'3. Predicción ({best_model_name_eval})',
                            line=dict(color='red', width=3)
                        ))
                        
                        fig_eval.update_layout(
                            title=f"Comparación: Real vs. Predicción en el set de Prueba (20%)", # <-- CAMBIO DE TEXTO
                            xaxis_title="Fecha",
                            yaxis_title=metrica_seleccionada,
                            legend_title="Series"
                        )
                        st.plotly_chart(fig_eval, use_container_width=True)
                        st.caption(f"Este gráfico muestra qué tan bien el modelo '{best_model_name_eval}'... en el set de Prueba (20%).") # <-- CAMBIO DE TEXTO
                        
                        # --- 7. Tablas de Resultados ---
                        st.header("Detalle de Resultados")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(f"🗓️ Tabla de Predicciones")
                            st.dataframe(df_forecast.style.format("{:,.2f}"))
                        
                        with col2:
                            st.subheader("🏆 Métricas de Desempeño (Test Set)")
                            st.dataframe(df_metrics.style.format("{:,.2f}"))
                            st.caption("Valores más bajos son mejores.")
else:
    st.info("Cargando datos... Si el error persiste, revisa el nombre/ruta del archivo.")





