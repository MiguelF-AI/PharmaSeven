import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from pmdarima import auto_arima
from prophet import Prophet
from prophet.plot import plot_plotly
import google.generativeai as genai
import time
import warnings

# --- Configuración de la Página y Advertencias ---
st.set_page_config(layout="wide", page_title="Dashboard de Predicción de Ventas")
warnings.filterwarnings('ignore') # Ocultar advertencias de modelos

# --- Constantes y Nombres ---
NOMBRE_ARCHIVO_DATOS = 'data/datos_finales_listos_para_modelo.csv'
COLUMNA_PRODUCTO = 'Producto - Descripción'
COLUMNA_CLIENTE = 'Cliente - Descripción'
COLUMNA_FECHA = 'Fecha'
METRICAS_PREDICCION = ['Pedido_piezas', 'Pedido_MXN', 'Factura_piezas', 'Factura_MXN']

# --- Funciones de Carga y Preparación de Datos ---

@st.cache_data
def cargar_datos(nombre_archivo):
    """Carga y pre-procesa los datos desde el CSV."""
    try:
        df = pd.read_csv(nombre_archivo)
        df[COLUMNA_FECHA] = pd.to_datetime(df[COLUMNA_FECHA], format='%d/%m/%Y')
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{nombre_archivo}'.")
        st.error("Asegúrate de que el archivo esté en la misma carpeta que 'dashboard.py'.")
        return None
    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo: {e}")
        return None

def preparar_series_de_tiempo(df_filtrado, metrica_seleccionada):
    """Agrupa los datos filtrados y los prepara como serie de tiempo mensual."""
    if df_filtrado.empty:
        return None
    
    # Agrupar por fecha (sumar ventas si se seleccionan múltiples productos/clientes)
    ts_data = df_filtrado.groupby(COLUMNA_FECHA)[metrica_seleccionada].sum().reset_index()
    ts_data = ts_data.set_index(COLUMNA_FECHA)
    
    # Asegurar frecuencia mensual (MS = Month Start) y rellenar ceros
    ts_data = ts_data.asfreq('MS', fill_value=0)
    ts_data = ts_data[metrica_seleccionada] # Convertir a Series
    
    # Asegurarnos de tener suficientes datos para los modelos
    if len(ts_data) < 12:
        st.warning("Advertencia: Se tienen menos de 12 meses de datos. Las predicciones pueden no ser fiables.")
        return None
        
    return ts_data

# --- Funciones de Métricas ---

def calcular_metricas(y_true, y_pred):
    """Calcula RMSE y MAPE para un modelo."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Evitar división por cero en MAPE
    mape = mean_absolute_percentage_error(y_true[y_true != 0], y_pred[y_true != 0])
    return {'RMSE': rmse, 'MAPE': mape}

# --- Funciones de Modelos ---
# Cada función entrena, predice en test, y luego re-entrena y predice el futuro.

def run_model(model_name, model_func, ts_train, ts_test, ts_full, n_forecast):
    """Función genérica para correr un modelo y capturar resultados."""
    start_time = time.time()
    
    # 1. Entrenar en 'train' y predecir en 'test' para métricas
    pred_test = model_func(ts_train, len(ts_test))
    metrics = calcular_metricas(ts_test, pred_test)
    
    # 2. Entrenar en 'full' y predecir el futuro
    pred_future = model_func(ts_full, n_forecast)
    
    end_time = time.time()
    st.write(f"Modelo '{model_name}' completado en {end_time - start_time:.2f}s")
    
    return {
        'name': model_name,
        'metrics': metrics,
        'forecast': pred_future
    }

# Modelos específicos
def model_linear_regression(ts_data, n_steps):
    """Modelo de Regresión Lineal simple sobre el tiempo."""
    X_train = np.arange(len(ts_data)).reshape(-1, 1)
    y_train = ts_data.values
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    X_future = np.arange(len(ts_data), len(ts_data) + n_steps).reshape(-1, 1)
    forecast = model.predict(X_future)
    return pd.Series(forecast, index=pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=n_steps, freq='MS'))

def model_moving_average(ts_data, n_steps):
    """Predicción simple usando el promedio móvil de los últimos 6 meses."""
    window = 6
    if len(ts_data) < window:
        window = len(ts_data)
        
    forecast_value = ts_data.rolling(window=window).mean().iloc[-1]
    forecast = np.repeat(forecast_value, n_steps)
    return pd.Series(forecast, index=pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=n_steps, freq='MS'))

def model_exp_smoothing(ts_data, n_steps):
    """Suavizamiento Exponencial Simple (Holt)."""
    model = SimpleExpSmoothing(ts_data, initialization_method="estimated").fit()
    forecast = model.forecast(n_steps)
    return forecast

def model_arima(ts_data, n_steps):
    """
    Auto-ARIMA (SARIMA) optimizado para velocidad en el dashboard.
    """
    
    # --- Parámetros de velocidad ---
    # stepwise=True (que ya teníamos) es la clave principal.
    # Los siguientes parámetros 'max_' limitan el espacio de búsqueda.
    
    model = auto_arima(ts_data, 
                       seasonal=True,        # Activar SARIMA
                       m=12,                 # Estacionalidad de 12 meses
                       stepwise=True,        # Búsqueda "inteligente" (¡la más rápida!)
                       suppress_warnings=True,
                       error_action='ignore',
                       
                       # --- ¡NUEVOS LÍMITES PARA VELOCIDAD! ---
                       max_p=2,              # Máximo orden P (de 3 a 2)
                       max_q=2,              # Máximo orden Q (de 3 a 2)
                       max_P=1,              # Máximo orden P estacional (de 2 a 1)
                       max_Q=1,              # Máximo orden Q estacional (de 2 a 1)
                       
                       # Parámetros que ya teníamos
                       start_p=1, start_q=1,
                       start_P=0,
                       d=None,               # Dejar que determine 'd'
                       D=1,                  # Asumir una diferencia estacional
                       trace=False
                       )
    
    forecast = model.predict(n_periods=n_steps)
    return forecast

def model_prophet(ts_data, n_steps):
    """Modelo Prophet de Facebook."""
    df_prophet = ts_data.reset_index()
    df_prophet.columns = ['ds', 'y']
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=n_steps, freq='MS')
    forecast_df = model.predict(future)
    
    # Extraer solo la predicción futura
    forecast = forecast_df.iloc[-n_steps:]['yhat']
    forecast.index = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=n_steps, freq='MS')
    return forecast

def model_croston(ts_data, n_steps):
    """Modelo de Croston para demanda intermitente."""
    
    # 1. Separar la serie
    non_zero_data = ts_data[ts_data > 0]
    
    # Si no hay ventas, la predicción es 0
    if non_zero_data.empty:
        return pd.Series(np.zeros(n_steps), index=pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=n_steps, freq='MS'))
    
    # 2. Calcular los intervalos entre ventas
    indices = np.where(ts_data > 0)[0]
    
    # Si solo hay 1 venta, no podemos calcular intervalos
    if len(indices) < 2:
        return pd.Series(np.zeros(n_steps), index=pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=n_steps, freq='MS'))
    
    # El primer intervalo es desde el inicio + 1
    first_interval = indices[0] + 1
    other_intervals = np.diff(indices)
    
    # Combinar en una Serie de pandas
    intervals = pd.Series(np.concatenate(([first_interval], other_intervals)), index=non_zero_data.index)

    # 3. Aplicar Suavizamiento Exponencial Simple (SES) a ambas series
    # Usamos 'estimated' para que encuentre el mejor alpha
    ses_demand = SimpleExpSmoothing(non_zero_data, initialization_method="estimated").fit()
    ses_interval = SimpleExpSmoothing(intervals, initialization_method="estimated").fit()
    
    # 4. Pronosticar el siguiente valor de cada uno
    forecast_demand = ses_demand.forecast(1).iloc[0]
    forecast_interval = ses_interval.forecast(1).iloc[0]
    
    # 5. Calcular la predicción de Croston
    # Evitar división por cero
    if forecast_interval == 0:
        forecast_value = 0
    else:
        forecast_value = forecast_demand / forecast_interval
        
    # El pronóstico de Croston es un valor constante (la tasa promedio)
    forecast = np.repeat(forecast_value, n_steps)
    
    return pd.Series(forecast, index=pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=n_steps, freq='MS'))

@st.cache_data # ¡Se mantiene el caché!
def run_model_pipeline(_df_completo, _productos_sel, _clientes_sel, _metrica_sel, _n_forecast):
    """
    Ejecuta el pipeline completo.
    Ahora recibe los FILTROS como argumentos para que el caché funcione correctamente.
    """
    
    # --- 1. Filtrar datos (DENTRO del caché) ---
    # Si las tuplas de filtros están vacías, no hacer nada
    if not _productos_sel or not _clientes_sel:
        st.warning("Advertencia: No hay productos o clientes seleccionados.")
        return pd.DataFrame(), pd.DataFrame(), None # Devolver DFs vacíos y ts_full=None
        
    df_filtrado = _df_completo[
        (_df_completo[COLUMNA_PRODUCTO].isin(_productos_sel)) &
        (_df_completo[COLUMNA_CLIENTE].isin(_clientes_sel))
    ]
    
    # --- 2. Preparar Series de Tiempo (DENTRO del caché) ---
    ts_full = preparar_series_de_tiempo(df_filtrado, _metrica_sel)
    
    if ts_full is None:
        st.warning("Advertencia: No hay suficientes datos para la serie de tiempo con esos filtros.")
        return pd.DataFrame(), pd.DataFrame(), None

    # --- 3. División Train/Test (DENTRO del caché) ---
    test_size = _n_forecast
    if len(ts_full) <= test_size:
        st.error(f"Error: Se necesitan más de {test_size} meses para la validación.")
        return pd.DataFrame(), pd.DataFrame(), ts_full # Devolver ts_full para el gráfico

    ts_train = ts_full.iloc[:-test_size]
    ts_test = ts_full.iloc[-test_size:]
    
    # --- 4. Pipeline de Modelos (Esto ya estaba) ---
    st.write("Ejecutando pipeline de modelos (Esto se cacheará la primera vez)...")
    
    model_pipeline = [
        ('Regresión Lineal', model_linear_regression),
        ('Promedio Móvil (6m)', model_moving_average),
        ('Suavizamiento Exponencial', model_exp_smoothing),
        ('ARIMA', model_arima),
        ('Prophet', model_prophet),
        ('Método Croston', model_croston)
    ]
    
    all_metrics = {}
    all_forecasts = {}

    for name, func in model_pipeline:
        try:
            # Usamos las variables locales que acabamos de crear
            resultado = run_model(name, func, ts_train, ts_test, ts_full, _n_forecast)
            all_metrics[name] = resultado['metrics']
            all_forecasts[name] = resultado['forecast']
        except Exception as e:
            st.error(f"Error al ejecutar el modelo '{name}': {e}")

    df_metrics = pd.DataFrame(all_metrics).T.sort_values(by='MAPE')
    df_forecast = pd.DataFrame(all_forecasts)
    df_forecast.index.name = "Fecha"
    
    # Devolvemos también ts_full para poder graficarlo fuera
    return df_metrics, df_forecast, ts_full

# --- Función de Gemini AI ---

def get_gemini_analysis(metrics_summary, n_meses, metrica_nombre):
    """Llama a la API de Gemini para analizar los resultados."""
    if not api_key:
        return "Por favor, introduce una API Key de Gemini en la barra lateral para obtener el análisis."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Eres un analista de datos senior especializado en pronósticos de ventas.
        Quiero predecir '{metrica_nombre}' para los próximos {n_meses} meses.
        
        He corrido 5 modelos y he calculado sus métricas de error (RMSE y MAPE) en un conjunto de prueba. 
        Un valor más bajo es mejor para ambas métricas.

        Estos son los resultados:
        {metrics_summary.to_string()}

        Por favor, responde con lo siguiente en un formato markdown claro:
        
        1.  **Recomendación del Modelo:** ¿Cuál es el modelo más eficiente y por qué? (Considera ambas métricas, pero da prioridad a MAPE ya que es un error porcentual y más fácil de comparar).
        2.  **Análisis de Resultados:** Explica brevemente por qué este modelo pudo haber ganado (ej. "Probablemente capturó bien la tendencia/estacionalidad...") y por qué otros pudieron fallar (ej. "La regresión lineal es muy simple...").
        3.  **Advertencia:** Termina con una breve advertencia sobre la confianza en las predicciones.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error al contactar la API de Gemini: {e}")
        return "No se pudo generar el análisis. Verifica tu API Key o la configuración."

# --- == APLICACIÓN PRINCIPAL (STREAMLIT) == ---

st.title("📈 Dashboard de Predicción de Ventas")

# --- Carga de Datos ---
df = cargar_datos(NOMBRE_ARCHIVO_DATOS)

if df is not None:
    
    # --- Barra Lateral (Filtros) ---
    st.sidebar.header("⚙️ Configuración de la Predicción")
    
    api_key = st.sidebar.text_input("🔑 API Key de Google Gemini", type="password", help="Necesaria para el análisis de IA")
    
    # Filtros de Producto y Cliente
    productos_unicos = [st.sidebar.checkbox("Seleccionar Todos los Productos", value=True)] + df[COLUMNA_PRODUCTO].unique().tolist()
    if productos_unicos[0]: # Si "Todos" está marcado
        productos_seleccionados = df[COLUMNA_PRODUCTO].unique().tolist()
    else:
        productos_seleccionados = st.sidebar.multiselect("Selecciona Productos:", df[COLUMNA_PRODUCTO].unique())

    clientes_unicos = [st.sidebar.checkbox("Seleccionar Todos los Clientes", value=True)] + df[COLUMNA_CLIENTE].unique().tolist()
    if clientes_unicos[0]:
        clientes_seleccionados = df[COLUMNA_CLIENTE].unique().tolist()
    else:
        clientes_seleccionados = st.sidebar.multiselect("Selecciona Clientes:", df[COLUMNA_CLIENTE].unique())
    
    # Filtros de Métrica y Horizonte
    metrica_seleccionada = st.sidebar.selectbox("Selecciona la Métrica a Predecir:", METRICAS_PREDICCION)
    n_meses_prediccion = st.sidebar.slider("Meses a Predecir:", min_value=1, max_value=24, value=12)
    
    # Botón para ejecutar
    if st.sidebar.button("🚀 Generar Predicción", type="primary"):
        
        if not productos_seleccionados or not clientes_seleccionados:
            st.warning("Por favor, selecciona al menos un producto y un cliente.")
        elif not api_key:
            st.error("Por favor, introduce tu API Key de Gemini en la barra lateral.")
        else:
            with st.spinner(f"Ejecutando predicción para {n_meses_prediccion} meses... Esto puede tardar unos minutos..."):
                
                # --- 1. Conversión de filtros a Tuplas ---
                # Las tuplas son "hashables" y garantizan que el caché funcione.
                # sorted() asegura que ['A', 'B'] y ['B', 'A'] se traten como el mismo filtro.
                productos_tuple = tuple(sorted(productos_seleccionados))
                clientes_tuple = tuple(sorted(clientes_seleccionados))

                # --- 2. Ejecución del Pipeline Cacheado ---
                # Pasamos el DataFrame original (df) y los filtros (tuplas)
                df_metrics, df_forecast, ts_full = run_model_pipeline(
                    df, 
                    productos_tuple,
                    clientes_tuple,
                    metrica_seleccionada,
                    n_meses_prediccion
                )

                # --- 3. Mostrar Resultados ---
                # Comprobar si la ejecución fue exitosa (si ts_full no es None)
                if ts_full is not None:
                    
                    # --- 4. Análisis con Gemini ---
                    # Comprobar si se generaron métricas (si no hubo error en train/test)
                    if not df_metrics.empty:
                        st.write("Enviando resultados a Gemini para análisis...")
                        with st.spinner("🧠 Gemini está pensando..."):
                            analisis_gemini = get_gemini_analysis(df_metrics, n_meses_prediccion, metrica_seleccionada)
                        
                        st.subheader("🤖 Análisis y Recomendación (Gemini AI)")
                        st.markdown(analisis_gemini)
                    else:
                        st.info("No se generó análisis de IA (datos insuficientes para validación).")

                    # --- 5. Gráfico ---
                    st.subheader("📊 Gráfico de Predicción vs Histórico")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ts_full.index, y=ts_full.values,
                        mode='lines+markers',
                        name='Datos Históricos'
                    ))
                    # Predicciones
                    for model_name in df_forecast.columns:
                        fig.add_trace(go.Scatter(
                            x=df_forecast.index, y=df_forecast[model_name],
                            mode='lines',
                            name=f'Predicción: {model_name}'
                        ))
                    
                    fig.update_layout(
                        title=f"Predicción de '{metrica_seleccionada}'",
                        xaxis_title="Fecha",
                        yaxis_title=metrica_seleccionada,
                        legend_title="Series"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- 6. Tablas de Resultados ---
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"🗓️ Tabla de Predicciones")
                        st.dataframe(df_forecast.style.format("{:,.2f}"))
                    
                    with col2:
                        st.subheader("🏆 Métricas de Desempeño (Test Set)")
                        st.dataframe(df_metrics.style.format("{:,.2f}"))
                        st.caption("Valores más bajos son mejores.")
                
                else:
                    st.warning("No se pudieron generar predicciones con los filtros seleccionados.")
else:

    st.info("Cargando datos... Si el error persiste, revisa el nombre del archivo.")





