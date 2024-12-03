import time
from ast import literal_eval
from typing import List, Dict, Set, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from sentence_transformers import SentenceTransformer

from processData import get_embeddings, normalization_min_max

similarity_model = SentenceTransformer('jaimevera1107/all-MiniLM-L6-v2-similarity-es')


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

def build_mentions_matrix(
        df: pd.DataFrame,
        user_index: Dict[str, int]
) -> Tuple[ndarray[ndarray[int]], ndarray[ndarray[List[str]]], ndarray[ndarray[float]]]:
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
    mentions_matrix_date: ndarray[ndarray[List[str]]] = np.empty_like(mentions_matrix, dtype=list)

    # Iterar sobre cada fila del DataFrame para procesar menciones
    for index, row in df.iterrows():
        author: str = row['author_username']  # Usuario que escribió el tweet
        author_idx: int = user_index.get(author)

        # Obtener el conjunto de usuarios mencionados en el tweet
        mentions: Set[str] = set(get_mentions_list(row['entities']))

        # Incluir el usuario referenciado si no está ya en las menciones
        if row['ref_author']:
            mentions.add(row['ref_author'])

        # Actualizar la matriz de menciones para cada usuario mencionado
        for mentioned_user in mentions:
            mentioned_user_idx: int = user_index.get(mentioned_user)
            if mentioned_user_idx is not None and author_idx is not None:
                mentions_matrix[mentioned_user_idx, author_idx] += 1
                if not mentions_matrix_date[mentioned_user_idx, author_idx]:
                    mentions_matrix_date[mentioned_user_idx, author_idx] = [row['created_at']]
                else:
                    mentions_matrix_date[mentioned_user_idx, author_idx].append(row['created_at'])
    # Normalización Min-Max
    return normalization_min_max(mentions_matrix), mentions_matrix_date, np.round(mentions_matrix, 3)


def build_global_influence_matrix(
        df: pd.DataFrame,
        user_index: Dict[str, int],
        mentions_matrix_nonNorm: ndarray[ndarray[int]],
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
    global_influence: ndarray[int] = np.zeros(n, dtype=int)

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
                    .44 * (n_retweets + n_quotes + n_replies) +
                    .33 * (n_likes + n_bookmarks) +
                    .23 * n_impressions
            )
    # Multiplicar la matriz de menciones por la influencia global de cada usuario
    global_influence_matrix: ndarray[ndarray[float]] = mentions_matrix_nonNorm * global_influence[:, np.newaxis]

    # Normalización Min-Max
    global_influence_matrix = normalization_min_max(global_influence_matrix)
    return np.round(global_influence_matrix, 3), global_influence


def build_local_influence_matrix(
        df: pd.DataFrame,
        user_index: Dict[str, int],
        global_influence: np.ndarray[float],
        mentions_matrix_nonNorm: np.ndarray[ndarray[int]]
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

    n = len(user_index)
    local_influence: ndarray[float] = np.zeros(n, dtype=float)

    # Calcular la influencia local de cada usuario en su comunidad
    for i in range(n):
        community_influence = np.sum(global_influence[mentions_matrix_nonNorm[i] != 0])

        # Calcular la proporción de influencia local
        local_influence[i] = (
            global_influence[i] / (community_influence + global_influence[i])
            if community_influence + global_influence[i] else 0
        )

    # Crear la matriz de influencia local
    local_influence_matrix: ndarray[ndarray[float]] = mentions_matrix_nonNorm * local_influence[:, np.newaxis]

    # Normalización Min-Max
    local_influence_matrix = normalization_min_max(local_influence_matrix)
    return np.round(local_influence_matrix, 3)


def build_affinities_matrix(
        users_tweet_text: List[set[str]],
        users_stances: Dict[str, float],
        index_user: Dict[str, int],
        mentions_matrix_nonNorm: ndarray[ndarray[int]]
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
    a = time.time()
    n = len(users_tweet_text)

    users_affinity: ndarray[ndarray[float]] = np.zeros((n, n), float)
    users_embeddings = np.array([None] * n)
    similarity_cache = dict()

    def calculate_embeddings(index):
        if users_embeddings[index] is not None:
            return users_embeddings[index]
        opinions = users_tweet_text[index]
        embeddings = np.array([get_embeddings(opinion, similarity_model) for opinion in opinions])
        users_embeddings[index] = embeddings
        return embeddings

    for i in range(n):
        if len(users_tweet_text[i]) == 0 or users_stances[index_user[i]] is None:
            continue

        embeddings_user_i = calculate_embeddings(i)
        for j in range(n):
            if i == j or users_stances[index_user[j]] is None:
                continue
            mentions_ij = mentions_matrix_nonNorm[i, j]

            if abs(users_stances[index_user[i]] - users_stances[index_user[j]]) < 0.2:
                embeddings_user_j = calculate_embeddings(j)
                if not (j,i) in similarity_cache:
                    similarity_opinions_ij: float = similarity_model.similarity(embeddings_user_i,
                                                                                embeddings_user_j).mean()
                    similarity_opinions_ij = round(float(similarity_opinions_ij),3)
                    similarity_cache[(i,j)] = similarity_opinions_ij
                    #print(i, j)
                else:
                    similarity_opinions_ij = similarity_cache[(j,i)]
                    #print(f"cache: {i}, {j}, {similarity_opinions_ij}")
                users_affinity[i, j] = similarity_opinions_ij * mentions_ij
    #users_affinity[users_affinity < 1] = 0
    b = time.time()
    print(b - a)
    users_affinity = normalization_min_max(users_affinity)
    return users_affinity


def build_agreement_clique_matrix(
        users_tweet_text: ndarray[set[str]],
        users_stances: Dict[str, float],
        index_user: Dict[str, int],
        agreement_threshold=0.15,
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
    n = len(users_tweet_text)
    users_agreement: ndarray[ndarray[float]] = np.zeros((n, n), float)

    for i in range(n - 1):
        if len(users_tweet_text[i]) == 0:
            continue
        for j in range(i + 1, n):
            if len(users_tweet_text[j]) == 0:
                continue
            users_ij_agreement = abs(
                users_stances[index_user[i]] - users_stances[index_user[j]]) < agreement_threshold
            users_agreement[i, j] = 1 if users_ij_agreement else -1
    return users_agreement


def build_agreement_matrix(
        mentions_matrix: ndarray[ndarray[float]],
        users_stances: Dict[str, float],
        index_user: Dict[str, int],
        agreement_threshold=0.2,
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

    n = len(users_stances)
    users_agreement: ndarray[ndarray[float]] = np.zeros((n, n), float)

    for i in range(n - 1):
        if users_stances[index_user[i]] is None:
            continue
        for j in range(i + 1, n):
            if mentions_matrix[i, j] == 0  or users_stances[index_user[j]] is None:
                continue
            #users_ij_agreement = abs(
            #    users_stances[index_user[i]] - users_stances[index_user[j]]) < agreement_threshold
            #users_agreement[i, j] = 1 if users_ij_agreement else -1
            users_agreement[i, j] = abs(users_stances[index_user[i]] - users_stances[index_user[j]])
    return users_agreement
