# -*- coding: utf-8 -*-
"""
Análisis en batch del catálogo de Netflix con Apache Spark (PySpark)
Dataset: Netflix Movies and TV Shows (Kaggle) -> netflix_titles.csv

Pregunta del proyecto:
¿Cómo está compuesto el catálogo de Netflix (películas vs series, países,
géneros, clasificaciones) y cómo ha cambiado con los años?

Ejecutar:  spark-submit batch_netflix.py
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, split, explode, to_date, year, regexp_extract,
    count, avg, when, desc, lower
)
import matplotlib
matplotlib.use("Agg")  # para guardar imágenes sin pantalla (SSH)
import matplotlib.pyplot as plt

RUTA_CSV = "netflix_titles.csv"
CARPETA_GRAFICAS = "graficas"
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

# 1. Sesión de Spark
spark = (SparkSession.builder
         .appName("AnalisisNetflixBatch")
         .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# 2. Carga del dataset
# multiLine y escape porque las descripciones traen comas y comillas
df = (spark.read
      .option("header", True)
      .option("multiLine", True)
      .option("escape", '"')
      .csv(RUTA_CSV))

print("\n===== 1. CARGA DE DATOS =====")
print("Filas:", df.count(), "| Columnas:", len(df.columns))
df.printSchema()
df.show(5, truncate=40)

# 3. Limpieza
print("\n===== 2. LIMPIEZA =====")

# 3.1 Nulos por columna
print("Nulos por columna:")
df.select([count(when(col(c).isNull() | (trim(col(c)) == ""), c)).alias(c)
           for c in df.columns]).show(truncate=False)

# 3.2 Duplicados
total = df.count()
sin_dup = df.dropDuplicates(["title", "type", "release_year"])
print("Duplicados eliminados:", total - sin_dup.count())

# 3.3 Rellenar vacíos y ajustar tipos
limpio = (sin_dup
          .fillna({"director": "Desconocido", "cast": "Desconocido",
                   "country": "Desconocido"})
          .withColumn("release_year", col("release_year").cast("int"))
          .withColumn("date_added", to_date(trim(col("date_added")), "MMMM d, yyyy"))
          .withColumn("year_added", year(col("date_added")))
          .filter(col("type").isin("Movie", "TV Show")))

# 3.4 Duración: número de minutos (películas) o de temporadas (series)
limpio = (limpio
          .withColumn("duracion_num",
                      regexp_extract(col("duration"), r"(\d+)", 1).cast("int"))
          .withColumn("minutos",
                      when(col("type") == "Movie", col("duracion_num")))
          .withColumn("temporadas",
                      when(col("type") == "TV Show", col("duracion_num"))))

limpio.cache()
print("Filas después de limpiar:", limpio.count())

# 4. Análisis exploratorio (EDA)
print("\n===== 3. ANÁLISIS EXPLORATORIO =====")

# 4.1 Películas vs series
por_tipo = limpio.groupBy("type").count().orderBy(desc("count"))
print("Películas vs Series:")
por_tipo.show()

# 4.2 Países (un título puede tener varios países)
paises = (limpio.filter(col("country") != "Desconocido")
          .withColumn("pais", explode(split(col("country"), ",\\s*")))
          .groupBy("pais").count().orderBy(desc("count")).limit(10))
print("Top 10 países con más títulos:")
paises.show(truncate=False)

# 4.3 Géneros
generos = (limpio
           .withColumn("genero", explode(split(col("listed_in"), ",\\s*")))
           .groupBy("genero").count().orderBy(desc("count")).limit(10))
print("Top 10 géneros:")
generos.show(truncate=False)

# 4.4 Clasificaciones por edad
ratings = (limpio.filter(col("rating").rlike("^(TV|PG|R|G|NC|NR|UR)"))
           .groupBy("rating").count().orderBy(desc("count")).limit(8))
print("Clasificaciones más comunes:")
ratings.show()

# 4.5 Títulos agregados por año
por_anio = (limpio.filter(col("year_added").isNotNull())
            .groupBy("year_added", "type").count()
            .orderBy("year_added"))
print("Títulos agregados por año:")
por_anio.show(30)

# 4.6 Duración promedio de películas y temporadas promedio de series
print("Duración promedio de las películas (min) y temporadas de las series:")
limpio.agg(avg("minutos").alias("min_promedio_peliculas"),
           avg("temporadas").alias("temporadas_promedio_series")).show()

# 4.7 Ejemplo con RDD: palabras más usadas en los títulos
print("Ejemplo con RDD - palabras más repetidas en los títulos:")
palabras = (limpio.select("title").rdd
            .flatMap(lambda fila: fila[0].lower().split())
            .filter(lambda p: len(p) > 3)
            .map(lambda p: (p, 1))
            .reduceByKey(lambda a, b: a + b)
            .sortBy(lambda x: -x[1]))
for palabra, veces in palabras.take(10):
    print(f"  {palabra}: {veces}")

# 5. Visualizaciones (se guardan como PNG)
print("\n===== 4. GRÁFICAS =====")

# Pasamos a pandas solo los resultados ya resumidos (son pequeños)
pd_tipo = por_tipo.toPandas()
pd_paises = paises.toPandas()
pd_generos = generos.toPandas()
pd_ratings = ratings.toPandas()
pd_anio = por_anio.toPandas()

# Gráfica 1: películas vs series
plt.figure(figsize=(6, 6))
plt.pie(pd_tipo["count"], labels=pd_tipo["type"], autopct="%1.1f%%",
        colors=["#E50914", "#221F1F"])
plt.title("Películas vs Series en Netflix")
plt.savefig(f"{CARPETA_GRAFICAS}/1_peliculas_vs_series.png", bbox_inches="tight")
plt.close()

# Gráfica 2: países
plt.figure(figsize=(9, 5))
plt.barh(pd_paises["pais"][::-1], pd_paises["count"][::-1], color="#E50914")
plt.title("Top 10 países con más títulos")
plt.xlabel("Cantidad de títulos")
plt.savefig(f"{CARPETA_GRAFICAS}/2_top_paises.png", bbox_inches="tight")
plt.close()

# Gráfica 3: géneros
plt.figure(figsize=(9, 5))
plt.barh(pd_generos["genero"][::-1], pd_generos["count"][::-1], color="#B20710")
plt.title("Top 10 géneros")
plt.xlabel("Cantidad de títulos")
plt.savefig(f"{CARPETA_GRAFICAS}/3_top_generos.png", bbox_inches="tight")
plt.close()

# Gráfica 4: clasificaciones
plt.figure(figsize=(8, 5))
plt.bar(pd_ratings["rating"], pd_ratings["count"], color="#221F1F")
plt.title("Clasificaciones por edad más comunes")
plt.ylabel("Cantidad de títulos")
plt.savefig(f"{CARPETA_GRAFICAS}/4_clasificaciones.png", bbox_inches="tight")
plt.close()

# Gráfica 5: títulos agregados por año
tabla = pd_anio.pivot(index="year_added", columns="type", values="count").fillna(0)
tabla.plot(kind="line", marker="o", figsize=(9, 5),
           color={"Movie": "#E50914", "TV Show": "#221F1F"})
plt.title("Títulos agregados al catálogo por año")
plt.xlabel("Año")
plt.ylabel("Cantidad de títulos")
plt.grid(alpha=0.3)
plt.savefig(f"{CARPETA_GRAFICAS}/5_titulos_por_anio.png", bbox_inches="tight")
plt.close()

print(f"Listo. Gráficas guardadas en la carpeta '{CARPETA_GRAFICAS}/'")

# Pausa para poder ver la interfaz web de Spark en http://IP_DE_LA_VM:4040
# (la página solo existe mientras el programa está corriendo)
input("\nSpark UI disponible en el puerto 4040. Presiona ENTER para terminar...")
spark.stop()
