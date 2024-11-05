# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

# <h2 align=center> Nombre del Proyecto:</h2>
<h2 align=center>Sistema de recomendación para una Plataforma de Streaming </h2>

## **Descripción**
Primer proyecto individual de la etapa de labs de Henry, en este proyecto he desarrollado un sistema de recomendación de películas, para una Start-up que provee servicios de agregación de plataformas de streaming. El objetivo fue crear un modelo inteligente que sugiera películas personalizadas para los usuarios, basándose en sus gustos y preferencias. Para hacer esto posible, construí un flujo de trabajo que comenzó con la limpieza y preparación de los datos (ETL), seguido de un análisis exploratorio para entender mejor el conjunto de información. Luego, desarrollé un modelo de machine learning que combina datos como géneros, actores, directores y colecciones para ofrecer recomendaciones precisas.

El modelo se basa en técnicas de procesamiento de texto como TF-IDF (Frecuencia Inversa de Documento-Término) y usa la similitud de coseno para identificar las películas más similares a las que disfruta el usuario. Además, este sistema se implementó en una API que permite realizar consultas y obtener recomendaciones de manera rápida y eficiente. El proyecto también incluye una interfaz fácil de acceder para realizar consulta detalladas y recibir sugerencias de películas en tiempo real, brindando una experiencia más personalizada y agradable.

## **Conjunto de datos**
Se trabajó con dos [Datasets](https://drive.google.com/drive/folders/1X_LdCoGTHJDbD28_dJTxaD4fVuQC9Wt5?usp=drive_link), que contiene múltiples columnas y datos que fueron transformados y procesados, a fin de obtener variables relevantes para el sistema de recomendación, como: title, overview, genres, director, actor, belongs_to_collection, entre otras. 

## **Objetivo**
Desarrollar un sistema de recomendación de películas utilizando técnicas avanzadas de Machine Learning y MLOps, asegurando que el modelo pueda ser integrado en una API robusta y eficiente.

## **Etapas del Proyecto**

### 1. Ingeniería de Datos (ETL)
- Extracción: Se cargaron y limpiaron los datos de diversas fuentes, aqui también se trataron datos duplicados.
- Transformación: Se procesaron las columnas relevantes, como la extracción de datos anidados y la creación de nuevas variables para mejorar el modelo.
- Carga: Los datos transformados se almacenaron en un formato adecuado para facilitar el análisis y la creación del modelo.

### 2. Análisis Exploratorio de Datos (EDA)
- Análisis de distribuciones, correlaciones y patrones en los datos.
- Identificación de valores atípicos y datos faltantes.
- Visualizaciones para comprender las relaciones clave entre las variables.

### 3. Modelo de Mchine Learning 
- Se desarrolló un sistema de recomendación basado en TF-IDF para capturar similitudes entre películas.
- Se aplicaron técnicas como la vectorización de características de texto y la similitud del coseno para recomendar películas.
- Se optimizó el modelo para mejorar la precisión de las recomendaciones.

### 4. Implementación del modelo
- Se desarrolló una API utilizando FastAPI para mostrar las recomendaciones.
- La API se probó y se optimizó para manejar solicitudes y proporcionar respuestas rápidas.
- Se incluyó un flujo de pruebas para asegurar la fiabilidad del sistema en producción.
- Se utilizó una muestra representativa del conjunto de datos usados en el modelo, a fin de obtener una rápidas visualización del Deployment. 

<br/>

**`Deployment`**: Link en [Render](https://py-recomendacion.onrender.com/docs) 

<br/>

### 5. Video Explicativo
- Se preparó un video donde se explica todo el proceso seguido en el proyecto, desde la extracción de datos hasta la implementación del modelo en la API.

## **Autores**
Jossy Romero 
<br/>
Contacto: [![LinkedIn](https://img.shields.io/badge/linkedin-%231DA1F2.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jossy-romero-villanueva-31b11657/)
