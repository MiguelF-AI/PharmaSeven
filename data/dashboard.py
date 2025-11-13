import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px # Necesitamos express para los gráficos de Top N
import warnings

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Dashboard de Predicción (Rápido)")
warnings.filterwarnings('ignore')

# --- Constantes ---
RUTA_HISTORICO = 'data/datos_finales_listos_para_modelo.csv'
RUTAS_PREDICCIONES = {
    'ARIMA': 'data/predicciones_precalculadas (ARIMA)_sin_decimales.csv',
    'Holt-Winters': 'data/predicciones_precalculadas (holt_winters)_sin_decimales.csv',
    'Red Neuronal (LSTM)': 'data/predicciones_lstm_sin_decimales.csv'
}
COL_PRODUCTO = 'Producto - Descripción'
COL_CLIENTE = 'Cliente - Descripción'
COL_FECHA = 'Fecha'
METRICAS = ['Pedido_piezas', 'Pedido_MXN', 'Factura_piezas', 'Factura_MXN']
# --- Eliminadas métricas KPI fijas ---

# --- Funciones de Carga de Datos (Cacheada) ---
@st.cache_data
def cargar_datos_completos():
    """
    Carga el histórico y todos los archivos de predicciones.
    Une las predicciones en un solo DataFrame con una columna 'Modelo'.
    """
    try:
        # Cargar histórico
        df_hist = pd.read_csv(RUTA_HISTORICO)
        df_hist[COL_FECHA] = pd.to_datetime(df_hist[COL_FECHA], format='%d/%m/%Y')
        
        # Cargar y unir predicciones
        lista_dfs_pred = []
        for nombre_modelo, ruta in RUTAS_PREDICCIONES.items():
            df_pred = pd.read_csv(ruta)
            df_pred[COL_FECHA] = pd.to_datetime(df_pred[COL_FECHA], format='%d/%m/%Y')
            df_pred['Modelo'] = nombre_modelo
            lista_dfs_pred.append(df_pred)
            
        df_pred_total = pd.concat(lista_dfs_pred)
        
        return df_hist, df_pred_total
        
    except FileNotFoundError as e:
        st.error(f"Error fatal: No se encontró el archivo {e.filename}.")
        st.error("Asegúrate de que los 4 archivos (1 histórico, 3 de predicción) estén en la carpeta 'data/' de GitHub.")
        return None, None
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None, None

# --- Cargar TODOS los datos al inicio ---
df_hist, df_pred = cargar_datos_completos()

if df_hist is None:
    st.stop()

# --- Listas de Filtros (del histórico) ---
productos_lista_completa = df_hist[COL_PRODUCTO].unique().tolist()
clientes_lista_completa = df_hist[COL_CLIENTE].unique().tolist()
modelos_lista_completa = df_pred['Modelo'].unique().tolist()

# --- Títulos Principales ---
st.title("📈 Dashboard de Predicción (Versión Pre-calculada)")
st.info("Esta versión es instantánea. Los modelos se pre-calcularon offline.")

# --- Inicializar "Memoria" (Session State) ---
if 'productos_seleccionados' not in st.session_state:
    st.session_state.productos_seleccionados = productos_lista_completa
if 'clientes_seleccionados' not in st.session_state:
    st.session_state.clientes_seleccionados = clientes_lista_completa

# --- Funciones de Callback ---
def callback_select_all():
    st.session_state.productos_seleccionados = productos_lista_completa
    st.session_state.clientes_seleccionados = clientes_lista_completa

def callback_deselect_all():
    st.session_state.productos_seleccionados = []
    st.session_state.clientes_seleccionados = []

# --- Controles y Filtros en la Página Principal ---
st.header("⚙️ Configuración del Dashboard")

st.write("Control de Filtros:")
col1, col2, _ = st.columns([1, 1, 3]) 
with col1:
    st.button("Seleccionar Todos", on_click=callback_select_all, use_container_width=True)
with col2:
    st.button("Limpiar Todo", on_click=callback_deselect_all, use_container_width=True)

st.divider()

# --- Filtros (Conectados a la memoria) ---
col_filtros1, col_filtros2 = st.columns(2)

with col_filtros1:
    productos_seleccionados = st.multiselect(
        "Selecciona Productos:", 
        options=productos_lista_completa, 
        key='productos_seleccionados'
    )
    metrica_seleccionada = st.selectbox(
        "Selecciona la Métrica (para Gráficos):", 
        METRICAS
    )

with col_filtros2:
    clientes_seleccionados = st.multiselect(
        "Selecciona Clientes:", 
        options=clientes_lista_completa, 
        key='clientes_seleccionados'
    )
    modelo_seleccionado = st.selectbox(
        "Selecciona el Modelo (para Predicción):", 
        modelos_lista_completa
    )

st.divider() 

# --- Lógica Principal ---
if not productos_seleccionados or not clientes_seleccionados:
    st.warning("Por favor, selecciona al menos un producto y un cliente.")
else:
    # 1. Filtrar AMBOS dataframes primero
    df_hist_filtrado = df_hist[
        (df_hist[COL_PRODUCTO].isin(productos_seleccionados)) &
        (df_hist[COL_CLIENTE].isin(clientes_seleccionados))
    ]
    
    df_pred_filtrado = df_pred[
        (df_pred[COL_PRODUCTO].isin(productos_seleccionados)) &
        (df_pred[COL_CLIENTE].isin(clientes_seleccionados)) &
        (df_pred['Modelo'] == modelo_seleccionado)
    ]

    # 2. Variable de formato (para MXN o piezas)
    is_mxn = "MXN" in metrica_seleccionada

    # 3. SECCIÓN HISTÓRICA (KPIs y Gráficos)
    # Solo se muestra si hay datos históricos
    if df_hist_filtrado.empty:
        st.warning("No se encontraron datos históricos para la selección actual.")
    else:
        st.subheader("Resumen de Datos Históricos (Selección Actual)")
        
        # Calcular KPIs Históricos (dinámicos)
        top_producto_serie = df_hist_filtrado.groupby(COL_PRODUCTO)[metrica_seleccionada].sum()
        top_cliente_serie = df_hist_filtrado.groupby(COL_CLIENTE)[metrica_seleccionada].sum()
        
        kpi_hist_total = top_producto_serie.sum()
        kpi_top_producto = top_producto_serie.idxmax() if not top_producto_serie.empty else "N/A"
        kpi_top_cliente = top_cliente_serie.idxmax() if not top_cliente_serie.empty else "N/A"
        
        # Formato dinámico para el KPI
        kpi_hist_total_str = f"${kpi_hist_total:,.0f}" if is_mxn else f"{kpi_hist_total:,.0f} pz"

        # Mostrar KPIs Históricos (3 columnas)
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric(f"Total Histórico ({metrica_seleccionada})", kpi_hist_total_str)
        kpi_col2.metric(f"Producto Principal", kpi_top_producto)
        kpi_col3.metric(f"Cliente Principal", kpi_top_cliente)

        st.divider()

        # --- Gráficos de Desglose Histórico ---
        st.subheader(f"Desglose Histórico por '{metrica_seleccionada}'")
        
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            # Gráfico de Barras: Top 5 Productos
            st.write(f"**Top 5 Productos**")
            top_5_prod = top_producto_serie.nlargest(5).sort_values(ascending=True)
            fig_bar = px.bar(
                top_5_prod, 
                x=top_5_prod.values, 
                y=top_5_prod.index, 
                orientation='h', 
                title=f"Top 5 Productos por {metrica_seleccionada}",
                labels={'x': metrica_seleccionada, 'y': 'Producto'}
            )
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with g_col2:
            # Gráfico de Donut: Distribución por Cliente
            st.write(f"**Distribución por Cliente**")
            fig_donut = px.pie(
                top_cliente_serie, 
                values=top_cliente_serie.values, 
                names=top_cliente_serie.index, 
                title=f"Distribución por {metrica_seleccionada}",
                hole=.4 # Esto lo hace un gráfico de Donut
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_donut, use_container_width=True)

        st.divider()

    # 4. SECCIÓN DE PREDICCIÓN (KPIs, Gráfico TS, Tabla)
    # Solo se muestra si hay datos de predicción
    if df_pred_filtrado.empty:
        st.warning(f"No se encontraron datos de predicción para el modelo '{modelo_seleccionado}' con los filtros actuales.")
    else:
        # --- NUEVA SECCIÓN: KPIs de PREDICCIÓN (dinámicos al modelo) ---
        st.subheader(f"Resumen de Predicción ({modelo_seleccionado})")
        
        # Calcular KPIs de Predicción
        ts_pred_sum = df_pred_filtrado.groupby(COL_FECHA)[metrica_seleccionada].sum()
        
        kpi_pred_total = ts_pred_sum.sum()
        kpi_pred_avg = ts_pred_sum.mean()
        kpi_pred_meses = ts_pred_sum.count()
        
        kpi_pred_total_str = f"${kpi_pred_total:,.0f}" if is_mxn else f"{kpi_pred_total:,.0f} pz"
        kpi_pred_avg_str = f"${kpi_pred_avg:,.0f}" if is_mxn else f"{kpi_pred_avg:,.0f} pz"
        
        # Mostrar KPIs de Predicción
        pkpi_col1, pkpi_col2, pkpi_col3 = st.columns(3)
        pkpi_col1.metric(f"Total Pronosticado ({metrica_seleccionada})", kpi_pred_total_str)
        pkpi_col2.metric("Promedio Mensual Pronosticado", kpi_pred_avg_str)
        pkpi_col3.metric("Meses Pronosticados", f"{kpi_pred_meses} meses")
        
        st.divider()

        # --- Gráfico de Serie de Tiempo (requiere histórico) ---
        if not df_hist_filtrado.empty:
            # Preparar datos históricos para el gráfico
            ts_hist_sum = df_hist_filtrado.groupby(COL_FECHA)[metrica_seleccionada].sum()
            
            st.subheader(f"Gráfico de Predicción ({modelo_seleccionado}) - {metrica_seleccionada}")
            fig_ts = go.Figure()

            # Histórico
            fig_ts.add_trace(go.Scatter(
                x=ts_hist_sum.index, y=ts_hist_sum.values,
                mode='lines+markers', name='Datos Históricos (Total)'
            ))
            
            # Predicción (ya calculada como ts_pred_sum)
            fig_ts.add_trace(go.Scatter(
                x=ts_pred_sum.index, y=ts_pred_sum.values,
                mode='lines', name=f'Predicción ({modelo_seleccionado})',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig_ts.update_layout(xaxis_title="Fecha", yaxis_title=metrica_seleccionada, legend_title="Series")
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.warning("No se puede mostrar el gráfico de serie de tiempo porque faltan datos históricos para comparar.")


        # --- Tabla de Predicción (Desglosada) ---
        st.subheader(f"Tabla de Predicciones Desglosada ({modelo_seleccionado})")
        
        columnas_tabla = [COL_PRODUCTO, COL_CLIENTE, COL_FECHA, metrica_seleccionada]
        df_tabla = df_pred_filtrado[columnas_tabla]
        
        st.dataframe(
            df_tabla.style.format({
                metrica_seleccionada: "{:,.0f}"
            }),
            height=400,
            use_container_width=True
        )
