import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

if df is not None and not df.empty:
    # Mostrar información general en columnas
    st.header("Información General de la Base de Datos")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de registros", df.shape[0])
    with col2:
        st.metric("Total de variables", df.shape[1])
    with col3:
        st.metric("Valores nulos", df.isnull().sum().sum())
    with col4:
        st.metric("Registros duplicados", df.duplicated().sum())
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
        st.markdown("Aquí puedes agregar análisis de segmentación de clientes.")

else:
    st.error("No se pudieron cargar los datos. Verifica que el archivo existe y tiene el formato correcto.")
