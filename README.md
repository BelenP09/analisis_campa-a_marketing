# 📊 Análisis Profesional de Campañas de Marketing
<p align="center">
    <img src="img/marketing.jpg" alt="Panel de Análisis de Marketing" width="600"/>
</p>
Bienvenido al panel interactivo para el **Análisis de Campañas de Marketing**. Esta aplicación, desarrollada en Streamlit, permite explorar, visualizar y analizar datos de campañas de marketing, facilitando la toma de decisiones estratégicas basadas en datos reales.

---

## 🚀 Características Principales

- **Carga y limpieza de datos**: Importa automáticamente los datasets originales y tratados.
- **Visualización avanzada**: Gráficas interactivas con Matplotlib y Seaborn para analizar canales, tipos de campaña, ROI, ingresos y más.
- **Filtros dinámicos**: Filtra por fechas, canales, tipos, audiencias, presupuesto, ROI y duración.
- **Análisis estadístico**: Pruebas de hipótesis y correlaciones para identificar diferencias y relaciones clave.
- **Resumen ejecutivo**: Conclusiones y recomendaciones automáticas para la dirección del proyecto.
- **Interfaz profesional**: Diseño personalizado y responsivo, con paleta de colores corporativa.

---

## 🛠️ Estructura del Proyecto

```
.
├── app.py
├── Data/
│   ├── marketingcampaigns.csv
│   └── marketingcampaigns_limpia.csv
├── img/
│   └── marketing.jpg
├── Notebook/
│   ├── notebook.ipynb
│   └── preprocesamiento.ipynb
└── README.md
```

---

## ⚙️ Instalación y Ejecución

1. **Clona el repositorio:**
    ```sh
    git clone https://github.com/tu_usuario/analisis_campa-a_marketing.git
    cd analisis_campa-a_marketing
    ```

2. **Instala las dependencias:**
    ```sh
    pip install -r requirements.txt
    ```

---

## 🧩 Esquema de la Aplicación

### Carga de datos
Lee los archivos CSV originales y limpios desde la carpeta `Data`.

### Panel General
- Métricas clave: campañas, nulos, duplicados.
- Vista previa de los datos.

### Análisis de Datos
- Frecuencia y ROI por canal.
- Ingresos y conversión por tipo de campaña.
- Distribución y factores asociados al ROI.
- Comparativa B2B vs B2C.
- Beneficio neto y campañas destacadas.
- Correlaciones y patrones temporales.

### Interacción con los Datos
- Filtros avanzados por múltiples variables.
- Gráficas dinámicas según los filtros aplicados.

### Conclusiones y Recomendaciones
- Top campañas por ROI.
- Beneficio neto por canal.
- Matriz de correlación.
- Resumen ejecutivo y recomendaciones accionables.

---

## 📈 Ejemplo de Visualizaciones

- Barras y líneas para comparar canales y tipos.
- Histogramas y boxplots para analizar distribuciones.
- Scatterplots para correlaciones.
- Heatmaps para matrices de correlación.

---

## 📝 Notebooks y Preprocesamiento

El directorio `Notebook` contiene los notebooks de análisis exploratorio y preprocesamiento, donde se documenta la limpieza y transformación de los datos.

---

## 💡 Recomendaciones de Uso

- Asegúrate de tener los archivos `marketingcampaigns.csv` y `marketingcampaigns_limpia.csv` en la carpeta `Data`.
- Personaliza los filtros en la barra lateral para obtener insights específicos.
- Consulta la sección de conclusiones para recomendaciones estratégicas basadas en los datos.

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT.

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Abre un issue o pull request para sugerencias, mejoras o correcciones.

