import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Campañas de Marketing",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores profesional
PRIMARY_COLOR = "#2E86C1"
SECONDARY_COLOR = "#AED6F1"
BACKGROUND_COLOR = "#F4F6F7"

# Estilos personalizados
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background-color: {BACKGROUND_COLOR};
        }}
        .sidebar .sidebar-content {{
            background-color: {SECONDARY_COLOR};
        }}
        h1, h2, h3, h4 {{
            color: {PRIMARY_COLOR};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Título principal
st.title("Análisis Profesional de Campañas de Marketing")

# Breve introducción
st.markdown("""
Este panel interactivo permite visualizar y analizar los datos de campañas de marketing, explorando características de diferentes canales, patrones de compra y segmentación de clientes para obtener insights clave que apoyen la toma de decisiones estratégicas.
""")
    
# Inicializar variables globales
filtered_df = None
df = None

# Cargar datos
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\Usuario\Documents\GitHub\analisis_campa-a_marketing\Data\marketingcampaigns_limpia.csv")        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

   # Cargar los datos
try:
    with st.spinner("Cargando datos..."):
        df = load_data() 

    if df is not None and not df.empty:

        # Función para asegurar tamaños positivos
        def ensure_positive(values, min_size=3):
            if isinstance(values, (pd.Series, np.ndarray, list)):
                return np.maximum(np.abs(values), min_size)
            else:
                return max(abs(values), min_size)
            
        if len(df) == 0:
            st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
        else:
            main_tabs = st.tabs(["Información General", "Análisis de Datos", "Conclusiones y Recomendaciones"])

        # Filtros de fecha usando campos start_date y end_date
        if 'start_date' in df.columns and 'end_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
            min_date = df['start_date'].min()
            max_date = df['end_date'].max()
            fecha_inicio, fecha_fin = st.sidebar.date_input(
            "Filtrar por rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
            )
            if fecha_inicio and fecha_fin:
                mask = (df['start_date'] >= pd.to_datetime(fecha_inicio)) & (df['end_date'] <= pd.to_datetime(fecha_fin))
                filtered_df = df.loc[mask].copy()
            else:
                filtered_df = df.copy()
        else:
            filtered_df = df.copy()
        df = filtered_df



        # Filtros adicionales por 'channel', 'type' y 'target_audience'
        filtro_cols = []
        if 'channel' in df.columns:
            canales = df['channel'].dropna().unique()
            canal_sel = st.sidebar.multiselect("Filtrar por canal", opciones := sorted(canales))
            if canal_sel:
                filtro_cols.append(df['channel'].isin(canal_sel))

                   # Aplicar filtro silenciosamente (sin mostrar los datos)
            filtered_df = filtered_df[filtered_df['channel'].isin(canal_sel)]


        if 'type' in df.columns:
            tipos = df['type'].dropna().unique()
            tipo_sel = st.sidebar.multiselect("Filtrar por tipo", opciones := sorted(tipos))
            if tipo_sel:
                filtro_cols.append(df['type'].isin(tipo_sel))

            # Aplicar filtro silenciosamente (sin mostrar los datos)
            filtered_df = filtered_df[filtered_df['type'].isin(tipo_sel)]


        if 'target_audience' in df.columns:
            targets = df['target_audience'].dropna().unique()
            target_sel = st.sidebar.multiselect("Filtrar por audiencia objetivo", opciones := sorted(targets))
            if target_sel:
                filtro_cols.append(df['target_audience'].isin(target_sel))

                # Aplicar filtro silenciosamente (sin mostrar los datos)
            filtered_df = filtered_df[filtered_df['target_audience'].isin(target_sel)]

        if filtro_cols:
            mask = np.logical_and.reduce(filtro_cols)
            df = df[mask]

               # Mostrar información general en la primera pestaña
        with main_tabs[0]:
            st.header("Información General de la Base de Datos")
            col1, col2, col3 = st.columns(3)
            with col1:
                 st.metric("Número de campañas", filtered_df['campaign_id'].nunique() if 'campaign_id' in filtered_df.columns else filtered_df.shape[0])
            with col2:
                st.metric("Presupuesto total", f"{filtered_df['budget'].sum():,.2f}€" if 'budget' in filtered_df.columns else "N/A")
            with col3:
                st.metric("días de campaña", f"{(filtered_df['end_date'] - filtered_df['start_date']).dt.days.max() if 'start_date' in filtered_df.columns and 'end_date' in filtered_df.columns else 'N/A'} días")

        with main_tabs[1]:
            st.header("Análisis de Datos")

        with main_tabs[2]:
            st.header("Conclusiones y Recomendaciones")

    else:
        st.error("No se pudieron cargar los datos. Verifica que el archivo existe y tiene el formato correcto.")

except Exception as e:
    st.error(f"Error al cargar o procesar los datos: {e}")
    st.info("Verifica que el archivo 'marketingcampaigns_limpia.csv' esté disponible y tenga el formato correcto.")
