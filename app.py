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
df_original = None

# Cargar datos
@st.cache_data(ttl=3600)
def load_data():
    try:
        df_original = pd.read_csv(r"C:\Users\Usuario\Documents\GitHub\analisis_campa-a_marketing\Data\marketingcampaigns.csv", sep=',',on_bad_lines='skip')
        df = pd.read_csv(r"C:\Users\Usuario\Documents\GitHub\analisis_campa-a_marketing\Data\marketingcampaigns_limpia.csv", sep=',', on_bad_lines='skip')
        return df_original, df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Cargar los datos
try:
    with st.spinner("Cargando datos..."):
        df_original, df = load_data()  # Desempaquetar los dos DataFrames

        if df_original is None or len(df_original) == 0:
            st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
        else:
            main_tabs = st.tabs(["Información General"])   
        
    if df_original is not None and not df_original.empty:
         # Asegurar que las columnas de fecha estén en formato datetime
        if 'start_date' in df_original.columns:
            df_original['start_date'] = pd.to_datetime(df_original['start_date'], errors='coerce')
        if 'end_date' in df_original.columns:
            df_original['end_date'] = pd.to_datetime(df_original['end_date'], errors='coerce')

        with main_tabs[0]:
            st.header("Información General de la Base de Datos")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Número de campañas", df_original['campaign_name'].nunique() if 'campaign_name' in df_original.columns else df_original.shape[0])
            with col2:
                st.metric("Valores nulos", df_original.isnull().sum().sum())
            with col3:
                st.metric("Registros duplicados", df_original.duplicated().sum())

            st.subheader("Muestra de los datos")
            st.dataframe(df_original)
    else:
        st.error("No se pudieron cargar los datos. Verifica que el archivo existe y tiene el formato correcto.")
    
    if df is not None and not df.empty:
        if df_original is None or len(df_original) == 0:
            st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
        else:
            main_tabs = st.tabs([" "])
        
        with main_tabs[0]:
            st.header("Información de la Base de Datos tratados")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Número de campañas", df['campaign_name'].nunique() if 'campaign_name' in df.columns else df.shape[0])
            with col2:
                st.metric("Valores nulos", df.isnull().sum().sum())
            with col3:
                st.metric("Registros duplicados", df.duplicated().sum())

            st.subheader("Muestra de los datos")
            st.dataframe(df)
    else:
        st.error("No se pudieron cargar los datos. Verifica que el archivo existe y tiene el formato correcto.")
except Exception as e:
    st.error(f"Error al cargar o procesar los datos: {e}")
    st.info("Verifica que el archivo 'marketingcampaigns_limpia.csv' esté disponible y tenga el formato correcto.")


try:
    with st.spinner("Cargando datos..."):
        df = load_data()  # Solo obtener el DataFrame limpio

        if df is None or len(df) == 0:
            st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
        else:
            main_tabs = st.tabs(["Análisis de Campañas de Marketing"])

    if df is not None and not df.empty:

        # Función para asegurar tamaños positivos
        def ensure_positive(values, min_size=3):
            if isinstance(values, (pd.Series, np.ndarray, list)):
                return np.maximum(np.abs(values), min_size)
            else:
                return max(abs(values), min_size)
        st.sidebar.header("Filtros")   
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
            canal_sel = st.sidebar.multiselect("Canal", opciones := sorted(canales), format_func=lambda x: str(x))
            if canal_sel:
                filtro_cols.append(df['channel'].isin(canal_sel))
                filtered_df = filtered_df[filtered_df['channel'].isin(canal_sel)]

        if 'type' in df.columns:
            tipos = df['type'].dropna().unique()
            tipo_sel = st.sidebar.multiselect("Tipo", opciones := sorted(tipos), format_func=lambda x: str(x))
            if tipo_sel:
                filtro_cols.append(df['type'].isin(tipo_sel))
                filtered_df = filtered_df[filtered_df['type'].isin(tipo_sel)]

        if 'target_audience' in df.columns:
            targets = df['target_audience'].dropna().unique()
            target_sel = st.sidebar.multiselect("Audiencia", opciones := sorted(targets), format_func=lambda x: str(x))
            if target_sel:
                filtro_cols.append(df['target_audience'].isin(target_sel))
                filtered_df = filtered_df[filtered_df['target_audience'].isin(target_sel)]

        if filtro_cols:
            mask = np.logical_and.reduce(filtro_cols)
            df = df[mask]

                # Filtro por presupuesto (budget)
        if 'budget' in df.columns:
            min_budget, max_budget = st.sidebar.slider(
            "Rango de presupuesto",
            min_value=float(df['budget'].min()),
            max_value=float(df['budget'].max()),
            value=(float(df['budget'].min()), float(df['budget'].max())),
            step=100.0
            )
        filtered_df = filtered_df[(filtered_df['budget'] >= min_budget) & (filtered_df['budget'] <= max_budget)]
        
                 # Filtro por roi_calculado
        if 'roi_calculado' in df.columns:
            min_budget, max_budget = st.sidebar.slider(
            "Rango de roi",
            min_value=float(df['roi_calculado'].min()),
            max_value=float(df['roi_calculado'].max()),
            value=(float(df['roi_calculado'].min()), float(df['roi_calculado'].max())),
            step=100.0
            )
        filtered_df = filtered_df[(filtered_df['roi_calculado'] >= min_budget) & (filtered_df['roi_calculado'] <= max_budget)]

                         # Filtro por duration_Day
        if 'duration_day' in df.columns:
            min_budget, max_budget = st.sidebar.slider(
            "Rango de día de duración",
            min_value=float(df['duration_day'].min()),
            max_value=float(df['duration_day'].max()),
            value=(float(df['duration_day'].min()), float(df['duration_day'].max())),
            step=100.0
            )
        filtered_df = filtered_df[(filtered_df['duration_day'] >= min_budget) & (filtered_df['duration_day'] <= max_budget)]
            
        if len(df) == 0:
            st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
        else:
            main_tabs = st.tabs(["Análisis de Datos", "Interación con los datos", "Conclusiones y Recomendaciones"])

        with main_tabs[4]:
            st.header("Análisis de Datos")
            col1, col2 = st.columns(2)

            with col1:
                # Pregunta y análisis: ¿Qué canal de marketing se utiliza con mayor frecuencia y cuál genera mejor ROI?
                st.subheader("¿Qué canal de marketing se utiliza con mayor frecuencia y cuál genera mejor ROI?")

                # Calcular la frecuencia de cada canal
                if 'channel' in df.columns and 'roi_calculado' in df.columns:
                    channel_counts = df['channel'].value_counts()
                    roi_calculado = df.groupby('channel')['roi_calculado'].mean().sort_values(ascending=False)

                    canal_mayor_frecuencia = channel_counts.idxmax()
                    canal_mejor_roi = roi_calculado.idxmax()

                    st.markdown(f"**Canal con mayor frecuencia:** {canal_mayor_frecuencia} ({channel_counts.max()} campañas)")
                    st.markdown(f"**Canal con mejor ROI promedio:** {canal_mejor_roi} ({roi_calculado.max():.2f})")

                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    color1 = '#1f77b4'
                    color2 = '#ff7f0e'

                    ax1.set_xlabel('Canal de Marketing')
                    ax1.set_ylabel('Frecuencia', color=color1)
                    sns.barplot(x=channel_counts.index, y=channel_counts.values, ax=ax1, alpha=0.7, color=color1)
                    ax1.tick_params(axis='y', labelcolor=color1)
                    ax1.set_xticklabels(channel_counts.index, rotation=45)

                    ax2 = ax1.twinx()
                    ax2.set_ylabel('ROI Promedio', color=color2)
                    sns.pointplot(x=roi_calculado.index, y=roi_calculado.values, ax=ax2, color=color2)
                    ax2.tick_params(axis='y', labelcolor=color2)

                    plt.title('Frecuencia y ROI Promedio por Canal de Marketing')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("No se encuentra la columna 'channel' o 'roi_calculado' en los datos.")

            with col2:
                # Pregunta y análisis: ¿Qué tipo de campaña genera más ingresos en promedio y cuál tiene mejor conversión?
                st.subheader("¿Qué tipo de campaña genera más ingresos en promedio y cuál tiene mejor conversión?")

      
        with main_tabs[5]:
            st.header("Interación con los Datos")

        with main_tabs[6]:
            st.header("Conclusiones y Recomendaciones")

    else:
        st.error("No se pudieron cargar los datos. Verifica que el archivo existe y tiene el formato correcto.")

except Exception as e:
    st.error(f"Error al cargar o procesar los datos: {e}")
    st.info("Verifica que el archivo 'marketingcampaigns_limpia.csv' esté disponible y tenga el formato correcto.")




