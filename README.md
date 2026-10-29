# Análisis del catálogo de Netflix con Apache Spark

Proyecto de la Tarea 3 – Big Data (UNAD). Procesamiento en batch con PySpark.

## Problema
¿Cómo está compuesto el catálogo de Netflix (películas vs series, países,
géneros, clasificaciones) y cómo ha cambiado con los años?

## Dataset
Netflix Movies and TV Shows (Kaggle, autor Shivam Bansal): `netflix_titles.csv`.
Columnas: show_id, type, title, director, cast, country, date_added,
release_year, rating, duration, listed_in, description.

## Qué hace el código
1. Carga el CSV con Spark.
2. Limpieza: nulos, duplicados, fechas, tipos y duración.
3. EDA con DataFrames (y un ejemplo con RDD).
4. Genera 5 gráficas en la carpeta `graficas/`.

## Requisitos
- Python 3 y PySpark (ya instalado en la máquina virtual)
- `pip install matplotlib pandas`

## Cómo ejecutarlo
```bash
spark-submit batch_netflix.py
```
El archivo `netflix_titles.csv` debe estar en la misma carpeta del script.

## Resultados
Las gráficas quedan en `graficas/`:
1_peliculas_vs_series, 2_top_paises, 3_top_generos, 4_clasificaciones, 5_titulos_por_anio.
