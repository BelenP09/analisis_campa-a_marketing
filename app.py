import streamlit as st
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Campañas de Marketing",
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
Bienvenido al panel interactivo de análisis de campañas de marketing. Aquí podrás explorar los principales resultados obtenidos tras el preprocesamiento y análisis de la base de datos **marketingcampaigns_limpia**.
""")

# Global variables for safe initialization
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

df = load_data()

# Mostrar información general
st.header("Información General de la Base de Datos")
st.write(f"Total de registros: {df.shape[0]}")
st.write(f"Total de variables: {df.shape[1]}")
st.dataframe(df.head())

# Sidebar para navegación
st.sidebar.title("Navegación")
section = st.sidebar.radio(
    "Selecciona una sección:",
    ("Resumen Demográfico", "Análisis de Compras", "Segmentación de Clientes")
)

# Sección 1: Resumen Demográfico
if section == "Resumen Demográfico":
    st.header("Resumen Demográfico")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución por Género")
        if 'Gender' in df.columns:
            gender_counts = df['Gender'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=gender_counts.index, y=gender_counts.values, palette="Blues", ax=ax)
            ax.set_ylabel("Cantidad")
            st.pyplot(fig)
            st.markdown("La mayoría de los clientes pertenecen al género más representado en la gráfica.")

    with col2:
        st.subheader("Distribución de Edad")
        if 'Age' in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df['Age'], bins=20, kde=True, color=PRIMARY_COLOR, ax=ax)
            ax.set_xlabel("Edad")
            st.pyplot(fig)
            st.markdown("La distribución de edad muestra el rango predominante de nuestros clientes.")

# Sección 2: Análisis de Compras
elif section == "Análisis de Compras":
    st.header("Análisis de Compras")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gasto Total por Cliente")
        if 'TotalSpent' in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df['TotalSpent'], bins=30, color=PRIMARY_COLOR, ax=ax)
            ax.set_xlabel("Gasto Total (€)")
            st.pyplot(fig)
            st.markdown("La mayoría de los clientes gastan menos de la media, con algunos clientes destacados por su alto gasto.")

    with col2:
        st.subheader("Compras por Canal")
        canales = [col for col in df.columns if "Purchases" in col]
        if canales:
            compras = df[canales].sum()
            fig, ax = plt.subplots()
            compras.plot(kind='bar', color=SECONDARY_COLOR, ax=ax)
            ax.set_ylabel("Total de Compras")
            st.pyplot(fig)
            st.markdown("El canal más utilizado para las compras es el que presenta mayor altura en la gráfica.")

# Sección 3: Segmentación de Clientes
elif section == "Segmentación de Clientes":
    st.header("Segmentación de Clientes")
    st.subheader("Distribución por Segmentos")
    if 'Segment' in df.columns:
        segment_counts = df['Segment'].value_counts()
        fig, ax = plt.subplots()
        segment_counts.plot.pie(autopct='%1.1f%%', colors=sns.color_palette("Blues"), ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)
        st.markdown("La segmentación permite identificar grupos clave de clientes para campañas personalizadas.")

    st.subheader("Relación entre Segmento y Gasto")
    if 'Segment' in df.columns and 'TotalSpent' in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x='Segment', y='TotalSpent', data=df, palette="Blues", ax=ax)
        st.pyplot(fig)
        st.markdown("Algunos segmentos presentan un gasto significativamente mayor, lo que sugiere oportunidades de focalización.")

# Nota sobre notebooks y preprocesamiento
st.sidebar.markdown("---")
st.sidebar.markdown("**¿Quieres ver el código de preprocesamiento o análisis?**")
if st.sidebar.button("Mostrar resumen de notebooks"):
    st.markdown("""
    ### Resumen del Preprocesamiento y Análisis
    - **Limpieza de datos:** Se eliminaron registros incompletos y se corrigieron valores atípicos.
    - **Transformaciones:** Se crearon variables como `TotalSpent` y se agruparon canales de compra.
    - **Análisis exploratorio:** Se identificaron patrones demográficos y de comportamiento de compra.
    - **Segmentación:** Se aplicaron técnicas de clustering para identificar grupos de clientes.
    """)