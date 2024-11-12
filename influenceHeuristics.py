from ast import literal_eval
from typing import List, Dict, Set, Tuple
import numpy as np
import pandas as pd
from numpy import ndarray
from time import time
from datetime import timedelta

from processData import get_embeddings, get_polarity


def get_mentions_list(tweet_entities: str) -> List[str]:
    """
    Extrae las menciones de un tweet a partir de los datos de entidades.

    Parámetros:
        tweet_entities (str): Cadena JSON que representa las entidades de un tweet.

    Devuelve:
        List[str]: Lista de nombres de usuario mencionados en el tweet.
    """
    return [mention['username'] for mention in literal_eval(tweet_entities).get("mentions", [])]


def identify_nodes(df: pd.DataFrame) -> List[str]:
    """
    Identifica los nodos únicos en el DataFrame a partir de los autores y menciones en los tweets.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos de los tweets.

    Devuelve:
        Set[str]: Conjunto de nombres de usuario únicos que representan los nodos.
    """
    # Crear conjunto de usuarios únicos a partir de los autores y autores referenciados
    users: Set[str] = set(np.concatenate((df['author_username'].unique(), df['ref_author'].unique())))

    # Añadir menciones de cada tweet a los nodos
    for tweet_entities in df['entities']:
        users.update(get_mentions_list(tweet_entities))

    # Remover posibles valores vacíos del conjunto
    users.discard("")

    return list(users)

def normalization_min_max(matrix: ndarray) -> ndarray:
    min_value = matrix.min()
    max_value = matrix.max()
    if max_value > min_value:  # Evitar división por cero
        matrix = (matrix - min_value) / (max_value - min_value)
    return matrix
def build_mentions_matrix(
        df: pd.DataFrame,
        user_index: Dict[str, int],
        normalize: bool = True
) -> ndarray[int]:
    """
    Construye una matriz de menciones que cuantifica las interacciones entre usuarios.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos de los tweets.
        user_index (Dict[str, int]): Diccionario que asigna un índice único a cada usuario.
        index_user (Dict[int, str]): Diccionario que asigna un usuario a cada índice.

    Devuelve:
        ndarray: Matriz de menciones (n x n), donde cada posición [i, j] representa
        la cantidad de veces que el usuario i ha sido mencionado por el usuario j.
    """
    # Inicializar la matriz de menciones con ceros
    n: int = len(user_index)
    mentions_matrix: ndarray[int] = np.zeros((n, n), dtype=int)

    # Iterar sobre cada fila del DataFrame para procesar menciones
    for index, row in df.iterrows():
        author: str = row['author_username']  # Usuario que escribió el tweet
        author_idx: int = user_index.get(author)

        # Obtener el conjunto de usuarios mencionados en el tweet
        mentions: Set[str] = set(get_mentions_list(row['entities']))

        # Incluir el usuario referenciado si no está ya en las menciones
        if row['ref_author'] and row['ref_author'] not in mentions:
            mentions.add(row['ref_author'])

        # Actualizar la matriz de menciones para cada usuario mencionado
        for mentioned_user in mentions:
            mentioned_user_idx: int = user_index.get(mentioned_user)
            if mentioned_user_idx is not None and author_idx is not None:
                mentions_matrix[mentioned_user_idx, author_idx] += 1

    # Normalización Min-Max
    if normalize:
        mentions_matrix = normalization_min_max(mentions_matrix)
    return np.round(mentions_matrix, 3)


def build_global_influence_matrix(
        df: pd.DataFrame,
        user_index: Dict[str, int]
) -> Tuple[ndarray[ndarray[int]], ndarray[int]]:
    """
    Calcula la influencia global de cada usuario en función de métricas ponderadas de interacción
    y la matriz de menciones.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos de los tweets.
        user_index (Dict[str, int]): Diccionario que asigna un índice único a cada usuario.

    Devuelve:
        np.ndarray: Matriz de influencia global (n x n) ajustada por la influencia global de cada usuario.
    """
    # Crear matriz de menciones utilizando la función build_mentions_matrix
    n: int = len(user_index)
    global_influence: ndarray[int] = np.ones(n, dtype=int)

    metrics_keys = ['retweet_count', 'reply_count', 'like_count', 'quote_count', 'bookmarks_count', 'impressions_count']
    for _, row in df.iterrows():
        author = row['author_username']
        author_idx = user_index.get(author)
        if author_idx is not None:
            # Extraer métricas de interacción del tweet
            tweet_metrics = literal_eval(row['public_metrics'])
            n_retweets, n_replies, n_likes, n_quotes, n_bookmarks, n_impressions = (
                tweet_metrics.get(key, 0) for key in metrics_keys
            )

            # Calcular la influencia global ponderada para el usuario actual
            global_influence[author_idx] += (
                    .4 * (n_retweets + n_quotes + n_replies) +
                    .3 * (n_likes + n_bookmarks) +
                    .2 * n_impressions + .1
            )
    mentions_matrix = build_mentions_matrix(df, user_index, normalize=False)
    # Multiplicar la matriz de menciones por la influencia global de cada usuario
    global_influence_matrix: ndarray[ndarray[float]] = mentions_matrix * global_influence[:, np.newaxis]

    # Normalización Min-Max
    global_influence_matrix = normalization_min_max(global_influence_matrix)
    return np.round(global_influence_matrix, 3), global_influence


def build_local_influence_matrix(
        df: pd.DataFrame,
        user_index: Dict[str, int],
        global_influence: np.ndarray[float]
) -> np.ndarray[ndarray[float]]:
    """
    Calcula la influencia local de cada usuario en función de su influencia global y las conexiones con
    su comunidad directa.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos de los tweets.
        user_index (Dict[str, int]): Diccionario que asigna un índice único a cada usuario.
        global_influence (np.ndarray): Vector de influencia global calculado para cada usuario.

    Devuelve:
        np.ndarray: Matriz de influencia local (n x n) ajustada por la influencia local de cada usuario.
    """
    mentions_matrix = build_mentions_matrix(df, user_index, normalize=False)

    n = len(user_index)
    local_influence: ndarray[float] = np.ones(n, dtype=float)

    # Calcular la influencia local de cada usuario en su comunidad
    for i in range(n):
        mentions_matrix[i, i] = 0
        community_influence = np.sum(global_influence[mentions_matrix[i] != 0])

        # Calcular la proporción de influencia local
        local_influence[i] = (
            global_influence[i] / (community_influence + global_influence[i])
            if community_influence + global_influence[i] else 0
        )

    # Crear la matriz de influencia local
    local_influence_matrix: ndarray[ndarray[float]] = mentions_matrix * local_influence[:, np.newaxis]

    # Normalización Min-Max
    local_influence_matrix = normalization_min_max(local_influence_matrix)
    return np.round(local_influence_matrix, 3)


def build_affinities_matrix(
        user_index: Dict[str, int],
        users_tweet_text: List[set[str]],
        similarity_model
) -> ndarray[ndarray[float]]:
    """
    Calcula la afinidad entre usuarios basándose en similitudes de opiniones y polaridades.

    Parámetros:
        globalInfluence_df (pd.DataFrame): DataFrame con la influencia global de cada usuario.
        user_index (Dict[str, int]): Diccionario que asigna un índice único a cada usuario.
        users_tweet_text (List[set[str]]): Lista de opiniones preprocesadas de cada usuario.

    Devuelve:
        np.ndarray: Matriz de afinidad entre usuarios.
        Dict[Tuple[str, str], str]: Diccionario con las relaciones de postura ("A favor" o "En contra") entre pares de usuarios.
    """
    n = len(user_index)
    users_affinity: ndarray[ndarray[float]] = np.zeros((n, n), float)

    for i in range(n):
        #user_idx_i: int = user_index[i]
        for j in range(n):
                user_i_opinions = list(users_tweet_text[i])
                user_j_opinions = list(users_tweet_text[j])
                embeddings_user_i = np.array([get_embeddings(opinion, similarity_model) for opinion in
                                                  user_i_opinions])
                embeddings_user_j = np.array([get_embeddings(opinion, similarity_model) for opinion in
                                                  user_j_opinions])
                affinity_ij = []

                if embeddings_user_i.size > 0 and embeddings_user_j.size > 0:
                    for index_i, opinions_similarity_i in enumerate(
                            similarity_model.similarity(embeddings_user_i, embeddings_user_j)):
                        opinion_i = user_i_opinions[index_i]
                        #opinion_polarity_i = get_polarity(opinion_i, sentiment_analyzer)
                        for index_j, opinion_similarity_ij in enumerate(opinions_similarity_i):
                            opinion_j = user_j_opinions[index_j]
                            #opinion_polarity_j = get_polarity(opinion_j, sentiment_analyzer)
                            #polarity_similarity = (2 - abs(opinion_polarity_i - opinion_polarity_j)) / 2
                            opinion_affinity = opinion_similarity_ij #* polarity_similarity
                            affinity_ij.append(opinion_affinity)

                    affinity_users_ij: float = sum(affinity_ij) / len(affinity_ij)
                    users_affinity[i, j] = affinity_users_ij

    return users_affinity
