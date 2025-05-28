import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

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
Este panel interactivo permite visualizar y analizar los datos de campañas de marketing, explorando características de diferentes canales, patrones de compra y segmentación de clientes para obtener conocimiento clave que apoyen la toma de decisiones estratégicas.
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
        _, df = load_data()  # Solo obtener el DataFrame limpio

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

        # Paleta de colores profesional
        color1 = PRIMARY_COLOR
        color2 = SECONDARY_COLOR
        color3 = "#27ae60"
        color4 = "#e67e22"
        color5 = "orchid"

        if len(df) == 0:
            st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
        else:
            main_tabs = st.tabs(["Análisis de Datos", "Interación con los datos", "Conclusiones y Recomendaciones"])

        with main_tabs[0]:
            st.header("Análisis de Datos")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("¿Qué canal de marketing se utiliza con mayor frecuencia y cuál genera mejor ROI?")

                if 'channel' in df.columns and 'roi_calculado' in df.columns:
                    channel_counts = df['channel'].value_counts()
                    roi_calculado = df.groupby('channel')['roi_calculado'].mean().sort_values(ascending=False)

                    canal_mayor_frecuencia = channel_counts.idxmax()
                    canal_mejor_roi = roi_calculado.idxmax()

                    st.markdown(f"**Canal con mayor frecuencia:** {canal_mayor_frecuencia} ({channel_counts.max()} campañas)")
                    st.markdown(f"**Canal con mejor ROI promedio:** {canal_mejor_roi} ({roi_calculado.max():.2f})")

                    fig, ax1 = plt.subplots(figsize=(10, 6))
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
                st.subheader("¿Qué tipo de campaña genera más ingresos en promedio y cuál tiene mejor conversión?")
                if 'type' in df.columns and 'revenue' in df.columns and 'conversion_rate' in df.columns:
                    ingresos_promedio = df.groupby('type')['revenue'].mean().sort_values(ascending=False)
                    conversion_promedio = df.groupby('type')['conversion_rate'].mean().sort_values(ascending=False)

                    tipo_mas_ingresos = ingresos_promedio.idxmax()
                    tipo_mejor_conversion = conversion_promedio.idxmax()

                    st.markdown(f"**Tipo de campaña con más ingresos promedio:** {tipo_mas_ingresos} ({ingresos_promedio.max():.2f})")
                    st.markdown(f"**Tipo de campaña con mejor conversión promedio:** {tipo_mejor_conversion} ({conversion_promedio.max():.2%})")

                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.set_xlabel('Tipo de Campaña')
                    ax1.set_ylabel('Ingresos Promedio', color=color3)
                    sns.barplot(x=ingresos_promedio.index, y=ingresos_promedio.values, ax=ax1, alpha=0.7, color=color3)
                    ax1.tick_params(axis='y', labelcolor=color3)
                    ax1.set_xticklabels(ingresos_promedio.index, rotation=45)

                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Conversión Promedio', color=color4)
                    sns.pointplot(x=conversion_promedio.index, y=conversion_promedio.values, ax=ax2, color=color4)
                    ax2.tick_params(axis='y', labelcolor=color4)

                    plt.title('Ingresos y Conversión Promedio por Tipo de Campaña')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("No se encuentra la columna 'type', 'revenue' o 'conversion_rate' en los datos.")

            with col1:
                st.subheader("¿Cómo se distribuye el ROI entre las campañas? ¿Qué factores están asociados con un ROI alto?")

                if 'roi_calculado' in df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df['roi_calculado'], bins=30, kde=True, color=color1, ax=ax)
                    ax.set_xscale('log')
                    ax.set_title('Distribución del ROI entre las Campañas (Escala Logarítmica)', fontsize=16, fontweight='bold')
                    ax.set_xlabel('ROI (escala log)', fontsize=14)
                    ax.set_ylabel('Número de Campañas', fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No hay suficientes variables numéricas para analizar correlaciones.")

            with col2:
                st.subheader("¿Hay diferencias significativas en la tasa de conversión entre audiencias B2B y B2C?")

                if 'target_audience' in df.columns and 'conversion_rate' in df.columns:
                    conversion_b2b = df[df['target_audience'] == 'B2B']['conversion_rate'].dropna()
                    conversion_b2c = df[df['target_audience'] == 'B2C']['conversion_rate'].dropna()

                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(x='target_audience', y='conversion_rate', data=df, palette=[color1, color2], ax=ax)
                    ax.set_title('Tasa de Conversión por Audiencia (B2B vs B2C)')
                    ax.set_xlabel('Audiencia')
                    ax.set_ylabel('Tasa de Conversión')
                    plt.tight_layout()
                    st.pyplot(fig)

                    if len(conversion_b2b) > 1 and len(conversion_b2c) > 1:
                        t_stat, p_value = ttest_ind(conversion_b2b, conversion_b2c, equal_var=False)
                        st.markdown(f"**t-statistic:** {t_stat:.3f}, **p-value:** {p_value:.4f}")
                        if p_value < 0.05:
                            st.success("Hay diferencias significativas en la tasa de conversión entre B2B y B2C.")
                        else:
                            st.info("No hay diferencias significativas en la tasa de conversión entre B2B y B2C.")
                    else:
                        st.warning("No hay suficientes datos para realizar la prueba estadística.")
                else:
                    st.warning("No se encuentra la columna 'target_audience' o 'conversion_rate' en los datos.")

            with col1:
                st.subheader("¿Qué campaña tiene el mayor beneficio neto (net_profit)? ¿Qué características la hacen exitosa?")

                if 'benefit' in df.columns:
                    idx_max = df['benefit'].idxmax()
                    max_campaign = df.loc[idx_max]

                    st.markdown(f"**Campaña con mayor beneficio neto:** {max_campaign['campaign_name'] if 'campaign_name' in max_campaign else idx_max}")
                    st.markdown(f"**Beneficio neto máximo:** {max_campaign['benefit']:.2f}")

                    st.markdown("**Características de la campaña exitosa:**")
                    st.write(max_campaign)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[df['benefit'] > 0]['benefit'], bins=30, kde=True, color=color5, ax=ax)
                    ax.set_title('Distribución del Beneficio Neto de las Campañas (solo beneficio positivo)', fontsize=16, fontweight='bold')
                    ax.set_xlabel('Beneficio Neto', fontsize=14)
                    ax.set_ylabel('Número de Campañas', fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("No se encuentra la columna 'benefit' en los datos.")

            with col2:
                    st.subheader("¿Existe correlación entre el presupuesto (budget) y los ingresos (revenue)?")
                    if 'budget' in df.columns and 'revenue' in df.columns:
                        correlacion_budget_revenue = df[['budget', 'revenue']].corr().loc['budget', 'revenue']
                        st.markdown(f"**Correlación entre presupuesto e ingresos:** {correlacion_budget_revenue:.3f}")

                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.scatterplot(x='budget', y='revenue', data=df, alpha=0.7, color=color1, ax=ax)
                        ax.set_title('Relación entre Presupuesto e Ingresos', fontsize=16, fontweight='bold')
                        ax.set_xlabel('Presupuesto (budget)', fontsize=14)
                        ax.set_ylabel('Ingresos (revenue)', fontsize=14)
                        ax.set_xscale('log')
                        ax.grid(True, linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("No se encuentra la columna 'budget' o 'revenue' en los datos.")

            with col2:
                st.subheader("¿Qué campañas tienen un ROI mayor a 0.5 y ingresos encima de 500,000?")

                if 'channel' in df.columns and 'revenue' in df.columns:
                    campanias_filtradas = df[(df['roi_calculado'] > 0.5) & (df['revenue'] > 500000)]

                    if not campanias_filtradas.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = sns.scatterplot(
                            data=campanias_filtradas,
                            x='revenue',
                            y='roi_calculado',
                            hue='channel',
                            style='type',
                            s=100,
                            palette=[color1, color2, color3, color4, color5],
                            ax=ax
                        )
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                        ax.set_yscale('log')
                        ax.set_title('Campañas con ROI > 0.5 e Ingresos > 500,000')
                        ax.set_xlabel('Ingresos (revenue)')
                        ax.set_ylabel('ROI Calculado')
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.dataframe(campanias_filtradas)
                    else:
                        st.info("No hay campañas que cumplan con los criterios seleccionados.")
                else:
                    st.warning("No se encuentra la columna 'roi' o 'revenue' en los datos.")

            with col1:
                st.subheader("¿Existen patrones estacionales o temporales en el rendimiento de las campañas?")

                if 'start_date' in df.columns and 'roi_calculado' in df.columns:
                    if 'year' not in df.columns or 'month' not in df.columns:
                        df['date'] = pd.to_datetime(df['start_date'])
                        df['year'] = df['date'].dt.year
                        df['month'] = df['date'].dt.month

                        df['period'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
                        roi_mensual = df.groupby(['year', 'month', 'period'])['roi_calculado'].mean().reset_index()

                        fig, ax = plt.subplots(figsize=(14,6))
                        sns.lineplot(x='period', y='roi_calculado', data=roi_mensual, marker='o', color=color1, ax=ax)
                        ax.set_xticklabels(roi_mensual['period'], rotation=45)
                        ax.set_title('Evolución Temporal del ROI Promedio Mensual de las Campañas')
                        ax.set_xlabel('Periodo (Año-Mes)')
                        ax.set_ylabel('ROI Promedio')
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.warning("No se encuentra la columna 'channel' o 'revenue' en los datos.")

        with main_tabs[1]:
            st.header("Interación con los Datos")

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

            # Gráfica: Presupuesto promedio por canal
            if 'budget' in filtered_df.columns and 'channel' in filtered_df.columns:
                st.subheader("Presupuesto Promedio por Canal")
                budget_channel = filtered_df.groupby('channel')['budget'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=budget_channel.index, y=budget_channel.values, palette=[PRIMARY_COLOR, SECONDARY_COLOR, color3, color4, color5], ax=ax)
                ax.set_ylabel("Presupuesto Promedio")
                ax.set_xlabel("Canal")
                ax.set_title("Presupuesto Promedio por Canal")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No hay columnas 'budget' y 'channel' en los datos filtrados.")

            # Gráfica: ROI promedio por tipo de campaña
            if 'type' in filtered_df.columns and 'roi_calculado' in filtered_df.columns:
                st.subheader("ROI Promedio por Tipo de Campaña")
                roi_tipo = filtered_df.groupby('type')['roi_calculado'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=roi_tipo.index, y=roi_tipo.values, palette=[PRIMARY_COLOR, SECONDARY_COLOR, color3, color4, color5], ax=ax)
                ax.set_ylabel("ROI Promedio")
                ax.set_xlabel("Tipo de Campaña")
                ax.set_title("ROI Promedio por Tipo de Campaña")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No hay columnas 'type' y 'roi_calculado' en los datos filtrados.")
                
            # Gráfica: Relación entre duración en días y audiencia
            if 'duration_day' in filtered_df.columns and 'target_audience' in filtered_df.columns:
                st.subheader("Relación entre Duración de la Campaña (días) y Audiencia")
                tipo_grafica = st.radio(
                    "Selecciona el tipo de gráfico para visualizar la relación:",
                    ("Boxplot", "Violinplot", "Swarmplot"),
                    horizontal=True,
                    key="grafica_duracion_audiencia"
                )
                fig, ax = plt.subplots(figsize=(8, 4))
                if tipo_grafica == "Boxplot":
                    sns.boxplot(x='target_audience', y='duration_day', data=filtered_df, palette=[PRIMARY_COLOR, SECONDARY_COLOR], ax=ax)
                elif tipo_grafica == "Violinplot":
                    sns.violinplot(x='target_audience', y='duration_day', data=filtered_df, palette=[PRIMARY_COLOR, SECONDARY_COLOR], ax=ax)
                else:  # Swarmplot
                    # Swarmplot puede fallar si hay muchos datos, así que limitamos el tamaño
                    sample_df = filtered_df.copy()
                    if len(sample_df) > 500:
                        sample_df = sample_df.sample(500, random_state=42)
                    sns.swarmplot(x='target_audience', y='duration_day', data=sample_df, palette=[PRIMARY_COLOR, SECONDARY_COLOR], ax=ax)
                ax.set_xlabel("Audiencia")
                ax.set_ylabel("Duración (días)")
                ax.set_title("Duración de la Campaña por Audiencia")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No hay columnas 'duration_day' y 'target_audience' en los datos filtrados.")

        with main_tabs[2]:
            st.header("Conclusiones y Recomendaciones")

            st.markdown("## Resúmenes Analíticos para la Dirección del Proyecto")

            if 'campaign_name' in df.columns and 'roi_calculado' in df.columns:
                st.subheader("Top 10 Mejores Campañas por ROI Calculado")
                top10 = df.sort_values('roi_calculado', ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(
                    x='roi_calculado',
                    y='campaign_name',
                    data=top10,
                    palette=[color1, color2, color3, color4, color5]*2,
                    ax=ax
                )
                ax.set_xlabel("ROI Calculado")
                ax.set_ylabel("Nombre de Campaña")
                ax.set_title("Top 10 Campañas con Mejor ROI")
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(top10)

            if 'channel' in df.columns and 'benefit' in df.columns:
                st.subheader("Beneficio Neto Promedio por Canal")
                benefit_channel = df.groupby('channel')['benefit'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=benefit_channel.index, y=benefit_channel.values, palette=[color1, color2, color3, color4, color5], ax=ax)
                ax.set_ylabel("Beneficio Neto Promedio")
                ax.set_xlabel("Canal")
                ax.set_title("Beneficio Neto Promedio por Canal")
                plt.tight_layout()
                st.pyplot(fig)

            if 'roi_calculado' in df.columns and 'conversion_rate' in df.columns:
                st.subheader("Relación entre ROI y Tasa de Conversión")
                fig, ax = plt.subplots(figsize=(8, 4))
                hue_col = 'type' if 'type' in df.columns else None
                palette = [color1, color2, color3, color4, color5] if hue_col else color1
                sns.scatterplot(x='roi_calculado', y='conversion_rate', data=df, hue=hue_col, palette=palette, ax=ax)
                ax.set_xlabel("ROI Calculado")
                ax.set_ylabel("Tasa de Conversión")
                ax.set_title("ROI vs Tasa de Conversión")
                plt.tight_layout()
                st.pyplot(fig)

                numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.subheader("Correlación entre Variables Numéricas")
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap=sns.color_palette([color1, color2], as_cmap=True), ax=ax)
                ax.set_title("Matriz de Correlación")
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("""
            **Conclusiones principales del análisis:**

            - **Canales de marketing:** El canal más utilizado no siempre es el que genera el mejor ROI. Es importante analizar ambos aspectos para optimizar la asignación de recursos.
            - **Tipos de campaña:** Existen diferencias claras entre los tipos de campaña en cuanto a ingresos y tasas de conversión. Identificar el tipo más rentable y con mejor conversión ayuda a enfocar futuras estrategias.
            - **ROI y factores asociados:** El ROI presenta una distribución variable y está correlacionado con ciertas variables numéricas, lo que permite identificar palancas de mejora.
            - **Audiencia objetivo:** Hay diferencias significativas en la tasa de conversión entre audiencias B2B y B2C, lo que sugiere adaptar los mensajes y canales según el público.
            - **Campañas exitosas:** Analizar las características de las campañas con mayor beneficio neto permite replicar buenas prácticas y evitar errores.
            - **Presupuesto e ingresos:** Existe correlación positiva entre presupuesto e ingresos, aunque no necesariamente lineal, lo que indica que invertir más puede generar mayores retornos, pero requiere optimización.
            - **Campañas destacadas:** Identificar campañas con ROI alto e ingresos elevados ayuda a reconocer patrones de éxito y oportunidades de escalabilidad.
            - **Estacionalidad:** Se observan patrones temporales en el rendimiento de las campañas, lo que sugiere planificar acciones clave en los periodos de mayor efectividad.

            **Recomendaciones:**
            - Priorizar los canales y tipos de campaña con mejor desempeño.
            - Segmentar estrategias según la audiencia (B2B vs B2C).
            - Analizar periódicamente el ROI y ajustar presupuestos en función de resultados históricos.
            - Replicar las características de las campañas más exitosas y aprender de las menos efectivas.
            - Aprovechar los periodos estacionales identificados para maximizar el impacto de las campañas.
            """)

    else:
        st.error("No se pudieron cargar los datos. Verifica que el archivo existe y tiene el formato correcto.")

except Exception as e:
    st.error(f"Error al cargar o procesar los datos: {e}")
    st.info("Verifica que el archivo 'marketingcampaigns_limpia.csv' esté disponible y tenga el formato correcto.")




