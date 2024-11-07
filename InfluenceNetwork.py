from typing import Dict

import pandas as pd
from fastapi import FastAPI, UploadFile
from pysentimiento import create_analyzer
from sentence_transformers import SentenceTransformer

from influenceHeuristics import build_mentions_matrix, identify_nodes
from processData import preprocess_dataframe, build_users_tweet_text, get_links_matrix

app = FastAPI()

# Inicialización del modelo y el analizador de sentimiento
similarity_model = SentenceTransformer('jaimevera1107/all-MiniLM-L6-v2-similarity-es')
sentiment_analyzer = create_analyzer(task="sentiment", lang="es")


def build_influence_networks(df: pd.DataFrame, users_tweet_text, user_index, index_user) -> Dict[str, dict]:
    """
    Construye las redes de influencia a partir de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con los datos de tweets.

    Returns:
        Dict[str, dict]: Diccionario que contiene las matrices y redes de influencia.
    """

    # Construir la matriz de menciones
    mentions_matrix = build_mentions_matrix(df, user_index)
    """
    # Construir la matriz de influencia global y local
    global_influence_matrix, global_influence = build_global_influence_matrix(df, user_index, mentions_matrix)
    local_influence_matrix = build_local_influence_matrix(df, user_index, global_influence)



    # Construir matrices de afinidad global y local
    affinities_global_matrix = build_affinities_matrix(global_influence_matrix, user_index, users_tweet_text,
                                                       similarity_model, sentiment_analyzer)
    affinities_local_matrix = build_affinities_matrix(local_influence_matrix, user_index, users_tweet_text,
                                                      similarity_model, sentiment_analyzer)
    """

    # Convertir las matrices a listas para poder devolverlas en formato JSON

    return {
        "mentions_links": get_links_matrix(mentions_matrix, index_user),
        # "global_influence_matrix": global_influence_matrix.tolist(),
        # "local_influence_matrix": local_influence_matrix.tolist(),
        # "affinities_global_matrix": affinities_global_matrix.tolist(),
        # "affinities_local_matrix": affinities_local_matrix.tolist(),
    }


@app.post("/influenceGraph/")
async def process_csv(file: UploadFile):
    df = pd.read_csv(file.file)  # Leer el CSV desde la consulta HTTP

    # Preprocesar el DataFrame
    df = preprocess_dataframe(df)

    # Identificar los nodos (usuarios) únicos y crear índices de usuario
    users = identify_nodes(df)
    user_index: Dict[str, int] = {user: idx for idx, user in enumerate(users)}
    index_user: Dict[int, str] = {index: user for user, index in user_index.items()}
    # Construir textos de los tweets por usuario
    users_tweet_text = build_users_tweet_text(df, user_index)

    result = build_influence_networks(df, users_tweet_text, user_index, index_user)  # Llamar a la función que procesa los datos
    return result  # Devolver el resultado en formato JSON
