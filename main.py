from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
import nltk
nltk.download('stopwords')

app = FastAPI()


# Cargamos los datos
df_movies = pd.read_parquet('D:/2024/HenryData/Py_Individual/PI_Recomendacion/Datasets/df_limpio.parquet')

# Cargamos las matrices y el DF para el modelo
df = joblib.load('D:/2024/HenryData/Py_Individual/PI_Recomendacion/Datasets/df.pkl')
# D:\2024\HenryData\Py_Individual\PI_Recomendacion\Datasets\df.pkl

combined_matrix = joblib.load('D:/2024/HenryData/Py_Individual/PI_Recomendacion/Datasets/combined_matrix.pkl')
#D:\2024\HenryData\Py_Individual\PI_Recomendacion\Datasets\combined_matrix.pkl

cosine_sim = joblib.load('D:/2024/HenryData/Py_Individual/PI_Recomendacion/Datasets/cosine_sim.pkl')
# D:\2024\HenryData\Py_Individual\PI_Recomendacion\Datasets\cosine_sim.pkl



# Función auxiliar para traducir meses y días a español
meses_esp = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
             "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}

dias_esp = {"lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3, "viernes": 4, "sabado": 5, "domingo": 6}

# 1. Función para la cantidad de filmaciones en un mes específico
def cantidad_filmaciones_mes(mes: str):
    mes = mes.lower()
    if mes in meses_esp:
        mes_num = meses_esp[mes]
        count = df_movies[df_movies['release_date'].dt.month == mes_num].shape[0]
        return {"message": f"{count} películas fueron estrenadas en el mes de {mes.capitalize()}"}
    else:
        return {"error": "Mes no válido. Use un mes en español"}

# ruta para devolver la cantidad de filmaciones por mes desde el archivo parquet

@app.get('/cantidad_filmaciones_mes/{mes}', name = 'Cantidad de Películas por mes',
                    description=
                    """ <font color="black">
                    INSTRUCCIONES<br>
                    1. Haga clik en "Try it out".<br>
                    2. Ingrese un mes. Ejemplo: Marzo <br>
                    3. Execute para ver el resultado.
                    </font>"""
                    , tags=['Consultas'])

async def get_cantidad_filmaciones_mes(mes: str):
    return cantidad_filmaciones_mes(mes)


# 2. Función para la cantidad de filmaciones en un día específico

def cantidad_filmaciones_dia(dia: str):
    dia = dia.lower()
    if dia in dias_esp:
        dia_num = dias_esp[dia]
        count = df_movies[df_movies['release_date'].dt.weekday == dia_num].shape[0]
        return {"message": f"{count} películas fueron estrenadas en los días {dia.capitalize()}"}
    else:
        return {"error": "Día no válido. Use un día en español"}

# Ruta para devolver la cantidad de filmaciones por día
@app.get('/cantidad_filmaciones_dia/{dia}', name = 'Cantidad de Películas por día',
                    description=
                    """ <font color="black">
                    INSTRUCCIONES<br>
                    1. Haga clik en "Try it out".<br>
                    2. Ingrese un día. Ejemplo: Martes <br>
                    3. Execute para ver el resultado.
                    </font>"""
                    , tags=['Consultas'])
async def get_cantidad_filmaciones_dia(dia:str):
    return cantidad_filmaciones_dia(dia)


# 3. Función para obtener el score de una película por título
def score_titulo(titulo: str):
    row = df_movies[df_movies['title'].str.lower() == titulo.lower()]
    if not row.empty:
        year = row.iloc[0]['release_year']
        score = row.iloc[0]['popularity']
        return {"message": f"La película {titulo} fue estrenada en el año {year} con un score de {score}"}
    else:
        return {"error": "Película no encontrada"}

# Ruta para devolver el score
@app.get('/score_titulo/{titulo}', name = 'Popularidad de la Película por título',
                    description=
                    """ <font color="black">
                    INSTRUCCIONES<br>
                    1. Haga clik en "Try it out".<br>
                    2. Ingrese título de una película. Ejemplo: Toy Story <br>
                    3. Execute para ver el resultado.
                    </font>"""
                    , tags=['Consultas'])
async def get_score_titulo(titulo: str):
    return score_titulo(titulo) 


# 4. Función para obtener votos de una película por título

def votos_titulo(titulo: str):
    row = df_movies[df_movies['title'].str.lower() == titulo.lower()]
    if not row.empty:
        votes = row.iloc[0]['vote_count']
        average_vote = row.iloc[0]['vote_average']
        year = row.iloc[0]['release_year']
        if votes >= 2000:
            return {"message": f"La película {titulo} fue estrenada en el año {year}. La misma cuenta con un total de {votes} valoraciones, con un voto promedio de {average_vote}"}
        else:
            return {"message": "La película no cumple con la condición de al menos 2000 valoraciones"}
    else:
        return {"error": "Película no encontrada"}

# Ruta para devolver votos
@app.get('/votos_titulo/{titulo}', name = 'Votos de la película por título',
                    description=
                    """ <font color="black">
                    INSTRUCCIONES<br>
                    1. Haga clik en "Try it out".<br>
                    2. Ingrese título de una película. Ejemplo: Titanic <br>
                    3. Execute para ver el resultado.
                    </font>"""
                    , tags=['Consultas'])
async def get_votos_titulo(titulo: str):
    return votos_titulo(titulo)


# 5. Función para obtener datos de un actor

def get_actor(nombre_actor: str):
    # Filtrar el DataFrame para obtener películas en las que el actor ha participado
    df_actor = df_movies[df_movies['actor'].str.contains(nombre_actor, case=False, na=False)]

    # Verificar si se encontraron resultados
    if df_actor.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado.")

    # Calcular el retorno total
    total_return = df_actor['revenue'].sum()

    # Calcular el número de películas en las que el actor ha participado
    num_movies = df_actor.shape[0]

    # Calcular el retorno promedio por película
    avg_return = df_actor['return'].mean()

    return {
        "message": f"El actor/actriz {nombre_actor} ha participado en {num_movies} filmaciones, consiguiendo un retorno total de {total_return} y un promedio de {avg_return:.2f} por filmación"
    }

# Ruta para devolver valores
@app.get('/get_actor/{nombre_actor}', name = 'Actor',
                    description=
                    """ <font color="black">
                    INSTRUCCIONES<br>
                    1. Haga clik en "Try it out".<br>
                    2. Ingrese nombre de un actor. Ejemplo: Emma Watson <br>
                    3. Execute para ver el resultado.
                    </font>"""
                    , tags=['Consultas'])
async def actor_get(nombre_actor: str):
    return get_actor(nombre_actor)


# 6. Función para obtener datos de un director

def get_director(nombre_director: str):
    # Filtrar el DataFrame para obtener películas dirigidas por el director especificado
    df_director = df_movies[df_movies['director'].str.contains(nombre_director, case=False, na=False)]

    # Verificar si se encontraron resultados
    if df_director.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado o sin filmaciones en el dataset.")

    # Inicializar variables
    total_return = 0
    details = []

    # Iterar sobre las películas del director para calcular el retorno individual y recopilar detalles
    for _, row in df_director.iterrows():
        return_individual = row['revenue'] - row['budget']
        total_return += return_individual
        details.append({
            "title": row['title'],
            "release_date": row['release_date'],
            "return": return_individual,
            "budget": row['budget'],
            "revenue": row['revenue']
        })

    # Retornar la información
    return {
        "message": f"El director {nombre_director} tiene un retorno total de {total_return} con el siguiente detalle:",
        "details": details
    }

# Ruta para devolver valores
@app.get('/get_director/{nombre_director}', name = 'Director',
                    description=
                    """ <font color="black">
                    INSTRUCCIONES<br>
                    1. Haga clik en "Try it out".<br>
                    2. Ingrese nombre de un director. Ejemplo: Matt Reeves <br>
                    3. Execute para ver el resultado.
                    </font>"""
                    , tags=['Consultas'])
async def director_get(nombre_director: str):
    return get_director(nombre_director)

"""
Modelo de Recomendación
"""
# Modelo
# Previos
# Inicializa el vectorizador TF-IDF
#tfidf = TfidfVectorizer(stop_words='english')

# Genera la matriz de TF-IDF para 'overview'
#tfidf_matrix = tfidf.fit_transform(df['overview'])

# Función para extraer y unificar la información de género, actor, director y coleccion
#def combine_features(row):
    #return row['genres'] + " " + row['actor'] + " " + row['director']+ " " + row['belongs_to_collection']

# Aplica la función de combinación de características
#df['combined_features'] = df.apply(combine_features, axis=1)
#tfidf_combined = TfidfVectorizer(max_features=5000, stop_words='english')
#combined_matrix = tfidf_combined.fit_transform(df['combined_features'])

# Calcula la similitud del coseno sobre la matriz TF-IDF de la combinación de características
#cosine_sim = cosine_similarity(combined_matrix)


# Función para el modelo 

def recommend_movies(title: str, cosine_sim=cosine_sim):
      # Normalizar el título 
    title = title.strip().lower()

    # Filtra el DataFrame con una comparación insensible a las mayúsculas
    matches = df[df['title'].str.lower() == title]

    # Verifica si se encontraron coincidencias
    if matches.empty:
        return f"No se encontró la película '{title.capitalize()}' en el DataFrame."

    try:
        # Obtiene el índice de la primera coincidencia
        idx = matches.index[0]

        # Calcula las puntuaciones de similitud de todas las películas
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Ordena las películas según los puntajes de similitud
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Obtén los índices de las 5 películas más similares (excluyendo la misma)
        top_indices = [i[0] for i in sim_scores[1:6]]

        # Retorna los títulos de las películas recomendadas
        return df['title'].iloc[top_indices].tolist()
    
    except Exception as e:
        return f"Error: {str(e)}"

@app.get('/recommend/{title}', name= 'Sistema de Recomendación',
                    description=
                    """ <font color="black">
                    INSTRUCCIONES<br>
                    1. Haga clik en "Try it out".<br>
                    2. Ingrese título de una película. Ejemplo: Pitch Perfect <br>
                    3. Execute para ver el resultado.
                    </font>"""
                    , tags=['Modelo'])
async def recomendacion_mov(titulo: str):
    return recommend_movies(titulo)