import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from pmdarima import auto_arima
from prophet import Prophet
import google.generativeai as genai
import time
import warnings
import lightgbm as lgb

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

def crear_features(df, target_col):
    """
    Crea características (features) a partir del índice de fecha
    para un DataFrame que contiene la columna objetivo.
    """
    df = df.copy()
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    
    # Creamos un lag de 12 meses (estacionalidad)
    # Este es el feature más importante
    df['lag_12'] = df[target_col].shift(12)
    
    features = ['month', 'year', 'quarter', 'lag_12']
    
    return df, features

# --- Funciones de Métricas ---

def calcular_metricas(y_true, y_pred):
    """
    Calcula métricas completas para evaluación de modelos.
    
    - RMSE: Penaliza mucho los errores grandes (picos).
    - MAPE: Error porcentual (cuidado con los ceros).
    - MAE: El error promedio en unidades reales (muy interpretable).
    - R2: Qué tan bien se ajusta el modelo (1.0 es perfecto, negativo es pésimo).
    """
    
    # 1. RMSE (Ya lo tenías)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 2. MAPE (Con protección para ceros)
    # Solo calculamos MAPE donde el valor real NO es 0 para evitar divisiones por infinito
    mask = y_true != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
    else:
        mape = np.nan # O 0.0 si prefieres

    # 3. MAE (Error Absoluto Medio) - ¡NUEVA!
    # Es más noble que el RMSE, no se asusta tanto con outliers.
    mae = mean_absolute_error(y_true, y_pred)

    # 4. R2 Score - ¡NUEVA!
    # Nos dice qué porcentaje de la varianza explica el modelo.
    r2 = r2_score(y_true, y_pred)

    return {
        'RMSE': rmse, 
        'MAPE': mape, 
        'MAE': mae, 
        'R2': r2
    }

# --- Funciones de Modelos ---

def run_model(model_name, model_func, ts_train_log, ts_test_log, ts_full_log, n_forecast):
    """
    Ejecuta el modelo usando datos logarítmicos, pero revierte la transformación
    para calcular métricas y entregar resultados en la escala real.
    """
    start_time = time.time()
    
    # --- A. FASE DE PRUEBA (Train/Test) ---
    # 1. Entrenar con datos LOG y predecir LOG
    # Nota: Pasamos len(ts_test_log) para que sepa cuántos pasos predecir
    pred_test_log, _ = model_func(ts_train_log, len(ts_test_log))
    
    # 2. Revertir transformación (Log -> Real)
    pred_test_real = np.expm1(pred_test_log)
    actual_test_real = np.expm1(ts_test_log) # Revertimos también los datos reales de prueba
    
    # 3. Corrección de Negativos (Clipping)
    pred_test_real = pred_test_real.clip(lower=0)
    
    # 4. Calcular métricas en la escala REAL (Importante para que el RMSE tenga sentido)
    metrics = calcular_metricas(actual_test_real, pred_test_real)
    
    # --- B. FASE DE FUTURO (Full Data) ---
    # 1. Entrenar con TODOS los datos LOG
    pred_future_log, conf_int_log = model_func(ts_full_log, n_forecast)
    
    # 2. Revertir predicción futura
    pred_future_real = np.expm1(pred_future_log).clip(lower=0)
    
    # 3. Revertir intervalos de confianza (si existen)
    conf_int_real = None
    if conf_int_log is not None:
        conf_int_real = np.expm1(conf_int_log).clip(lower=0)
    
    end_time = time.time()
    # st.write(f"Modelo '{model_name}' completado en {end_time - start_time:.2f}s") # Opcional: comentar para limpiar UI
    
    return {
        'name': model_name,
        'metrics': metrics,
        'forecast': pred_future_real,
        'interval': conf_int_real,
        'test_prediction': pred_test_real
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

def model_lightgbm(ts_data, n_steps):
    """Modelo de Machine Learning (LightGBM) con Feature Engineering."""
    
    target_col = 'y'
    df = ts_data.to_frame(name=target_col)
    
    # 1. Crear features para los datos de entrenamiento
    df_con_features, feature_names = crear_features(df, target_col)
    
    # 2. Eliminar filas con NaN (creadas por el lag)
    df_train = df_con_features.dropna()
    
    X_train = df_train[feature_names]
    y_train = df_train[target_col]
    
    # Si no hay datos después de los lags, no se puede entrenar
    if X_train.empty:
        st.warning("Advertencia (LGBM): No hay suficientes datos para el lag de 12 meses.")
        forecast = pd.Series(np.zeros(n_steps), index=pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=n_steps, freq='MS'))
        return forecast, None
        
    # 3. Entrenar el modelo LGBM
    model = lgb.LGBMRegressor(
        objective='regression_l1', # MAE es más robusto a outliers que MSE (L2)
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 4. Crear features para las fechas futuras
    last_date = ts_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_steps, freq='MS')
    
    df_future = pd.DataFrame(index=future_dates)
    
    # Concatenar los datos históricos + el esqueleto futuro
    df_full = pd.concat([df, df_future])
    
    # Aplicar feature engineering a todo el conjunto
    df_full_con_features, _ = crear_features(df_full, target_col)
    
    # Seleccionar solo las filas futuras para predecir
    X_future = df_full_con_features.iloc[-n_steps:][feature_names]
    
    # 5. Predecir
    forecast_values = model.predict(X_future)
    
    forecast = pd.Series(forecast_values, index=future_dates)
    
    return forecast, None # LGBM no da intervalos de confianza por defecto

# --- Función de Gemini AI ---

def get_gemini_analysis(metrics_summary, n_meses, metrica_nombre):
    """Llama a la API de Gemini con reintentos automáticos (Backoff)."""
    if not api_key:
        return "Error: No se encontró la API Key."
    
    # Configuración inicial
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    Eres un analista de datos experto. Predicción para: '{metrica_nombre}' ({n_meses} meses).
    
    Resultados de métricas (RMSE y MAPE, menor es mejor):
    {metrics_summary.to_string()}

    Responde brevemente en markdown:
    1. **Mejor Modelo:** Cuál elegir y por qué (basado en MAPE).
    2. **Análisis:** Por qué funcionó mejor que los otros.
    """

    # --- Lógica de Reintento (Retry Logic) ---
    max_retries = 3
    wait_time = 2 # Segundos iniciales de espera

    for attempt in range(max_retries):
        try:
            # Intentamos llamar a la API
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            # Si el error es 429 (Resource exhausted), esperamos y reintentamos
            if "429" in error_msg or "Resource exhausted" in error_msg:
                if attempt < max_retries - 1: # Si no es el último intento
                    time.sleep(wait_time)
                    wait_time *= 2 # Duplicamos el tiempo de espera (2s -> 4s -> 8s)
                    continue # Volvemos al inicio del loop
                else:
                    return "⚠️ La API de Gemini está saturada en este momento. Intenta en 1 minuto."
            else:
                # Si es otro error (ej. API Key inválida), fallamos inmediatamente
                st.error(f"Error de Gemini: {e}")
                return "Error al procesar la solicitud."

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
                
                # --- ¡FILTRO DE FECHA NUEVO! ---
                # Ajusta esta fecha al inicio del 'Régimen 3' (la nueva estabilidad)
                FECHA_INICIO_NUEVA = '2023-01-01' 
                df_filtrado = df_filtrado[df_filtrado[COLUMNA_FECHA] >= FECHA_INICIO_NUEVA]
                
                ts_full = preparar_series_de_tiempo(df_filtrado, metrica_seleccionada)
                
                if ts_full is not None:
                        
                        # --- 2. División Train/Test (en ESCALA REAL) ---
                            # ¡Creamos la división en escala real PRIMERO para los gráficos!
                            split_point = int(len(ts_full) * 0.8)
                            ts_train = ts_full.iloc[:split_point]  # <--- ¡ESTA ES LA VARIABLE QUE FALTABA!
                            ts_test = ts_full.iloc[split_point:]   # <--- ¡Y ESTA!

                            # --- ¡TRANSFORMACIÓN LOGARÍTMICA AQUÍ! ---
                            # Ahora transformamos TODO
                            ts_full_log = np.log1p(ts_full)
                            
                            # --- División Train/Test (en ESCALA LOG) ---
                            # Y volvemos a dividir los datos LOG para los modelos
                            ts_train_log = ts_full_log.iloc[:split_point]
                            ts_test_log = ts_full_log.iloc[split_point:]

                            # C. Validación (usamos los splits reales para el conteo)
                            if len(ts_test) < 2 or len(ts_train) < 12:
                                st.error(f"Error: No hay suficientes datos para una división 80/20 válida (Train: {len(ts_train)}, Test: {len(ts_test)}).")
                                st.stop()
                            else:
                                # --- 3. Ejecución de Modelos ---
                                st.write("Entrenando modelos con transformación Log-Normal...")
                            
                            model_pipeline = [
                                ('SARIMA', model_arima),
                                ('Prophet', model_prophet),
                                ('Holt-Winters', model_holt_winters),
                                ('LightGBM', model_lightgbm)
                            ]
                            # (Tu código para inicializar diccionarios ya está aquí)
                            all_metrics = {}
                            all_forecasts = {}
                            all_intervals = {}
                            all_test_preds = {}

                            for name, func in model_pipeline:
                                try:
                                    # CAMBIO: Pasamos las versiones _log de los datos
                                    resultado = run_model(name, func, ts_train_log, ts_test_log, ts_full_log, n_meses_prediccion)
                                    
                                    # El resto del código sigue igual, porque run_model ya devuelve
                                    # los datos convertidos a la escala real en 'forecast' y 'metrics'
                                    all_metrics[name] = resultado['metrics']
                                    all_forecasts[name] = resultado['forecast']
                                    all_intervals[name] = resultado['interval']
                                    all_test_preds[name] = resultado['test_prediction']
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





