import time
from ast import literal_eval
from typing import List, Dict, Set, Tuple
import json
import numpy as np
import pandas as pd
from numpy import ndarray
from sentence_transformers import SentenceTransformer

from processData import get_embeddings, normalization_min_max

similarity_model = SentenceTransformer('jaimevera1107/all-MiniLM-L6-v2-similarity-es')


def get_mentions_list(tweet_entities: str) -> List[str]:
    """
    Extracts the list of mentioned usernames from a tweet's entity data.

    Parameters:
        tweet_entities (str): JSON string representing the tweet's entities.

    Returns:
        List[str]: List of mentioned usernames in the tweet.
    """
    try:
        return [mention['username'] for mention in literal_eval(tweet_entities).get("mentions", [])]
    except:
        raise ValueError(f"Invalid tweet_entities format")

def identify_nodes(df: pd.DataFrame) -> List[str]:
    """
    Identifies unique nodes in the DataFrame based on tweet authors and mentions.

    Parameters:
        df (pd.DataFrame): DataFrame containing tweet data with the following columns:
            - 'author_username': The username of the tweet's author.
            - 'ref_author': The username of the referenced author.
            - 'entities': A JSON string containing tweet entities.

    Returns:
        List[str]: A list of unique usernames representing the nodes.

    """
    users: Set[str] = set(np.concatenate((
        df['author_username'].dropna().unique(),
        df['ref_author'].dropna().unique()
    )))

    for tweet_entities in df['entities'].dropna():
        try:
            tweet_mentions = get_mentions_list(tweet_entities)
            users.update(tweet_mentions)
        except ValueError:
            continue

    users.discard("")

    return list(users)

def build_mentions_matrix(
    df: pd.DataFrame,
    user_index: Dict[str, int]
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Builds a mentions matrix quantifying interactions between users, along with a matrix of dates.

    Parameters:
        df (pd.DataFrame): DataFrame containing tweet data with the following columns:
            - 'author_username': The username of the tweet's author.
            - 'entities': A JSON string representing tweet entities.
            - 'ref_author': The username of the referenced author (if any).
            - 'created_at': Timestamp of the tweet's creation.
        user_index (Dict[str, int]): Dictionary mapping usernames to unique indices.

    Returns:
        Tuple[ndarray, ndarray, ndarray]:
            - Normalized mentions matrix (n x n), where [i, j] represents the interaction count from user j to user i.
            - Matrix of lists containing timestamps for mentions.
            - Rounded mentions matrix before normalization.
    """
    n = len(user_index)
    mentions_matrix = np.zeros((n, n), dtype=int)
    mentions_matrix_date = np.empty((n, n), dtype=object)

    for row in df.itertuples(index=False):
        author = row.author_username
        author_idx = user_index.get(author)
        mentions = set(get_mentions_list(row.entities))

        if row.ref_author:
            mentions.add(row.ref_author)

        for mentioned_user in mentions:
            mentioned_user_idx = user_index.get(mentioned_user)
            if mentioned_user_idx is not None and author_idx is not None:
                mentions_matrix[mentioned_user_idx, author_idx] += 1
                if mentions_matrix_date[mentioned_user_idx, author_idx] is None:
                    mentions_matrix_date[mentioned_user_idx, author_idx] = [row.created_at]
                else:
                    mentions_matrix_date[mentioned_user_idx, author_idx].append(row.created_at)

    normalized_matrix = normalization_min_max(mentions_matrix)
    rounded_matrix = np.round(mentions_matrix, 3)

    return normalized_matrix, mentions_matrix_date, rounded_matrix


def build_global_influence_matrix(
    df: pd.DataFrame,
    user_index: Dict[str, int],
    mentions_matrix_nonNorm: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Computes the global influence matrix by weighting user interactions and mentions.

    Parameters:
        df (pd.DataFrame): DataFrame containing tweet data with the following columns:
            - 'author_username': The username of the tweet's author.
            - 'public_metrics': JSON string with metrics such as retweets, replies, likes, etc.
        user_index (Dict[str, int]): Dictionary mapping usernames to unique indices.
        mentions_matrix_nonNorm (ndarray): Non-normalized mentions matrix.

    Returns:
        Tuple[ndarray, ndarray]:
            - Global influence matrix (n x n), normalized using Min-Max.
            - Global influence vector for each user.
    """
    n = len(user_index)
    global_influence = np.zeros(n, dtype=float)

    metrics_keys = [
        'retweet_count', 'reply_count', 'like_count',
        'quote_count', 'bookmarks_count', 'impressions_count'
    ]

    for row in df.itertuples(index=False):
        author = row.author_username
        author_idx = user_index.get(author)
        if author_idx is not None:
            tweet_metrics = literal_eval(row.public_metrics)
            n_retweets, n_replies, n_likes, n_quotes, n_bookmarks, n_impressions = (
                tweet_metrics.get(key, 0) for key in metrics_keys
            )

            global_influence[author_idx] += (
                0.43 * (n_retweets + n_quotes + n_replies) +
                0.33 * (n_likes + n_bookmarks) +
                0.23 * n_impressions +
                0.01
            )

    global_influence_matrix = mentions_matrix_nonNorm * global_influence[:, np.newaxis]
    global_influence_matrix = normalization_min_max(global_influence_matrix)

    return global_influence_matrix, global_influence



def build_local_influence_matrix(
    df: pd.DataFrame,
    user_index: Dict[str, int],
    global_influence: ndarray,
    mentions_matrix_nonNorm: ndarray
) -> ndarray:
    """
    Computes the local influence matrix based on global influence and community connections.

    Parameters:
        df (pd.DataFrame): DataFrame containing tweet data.
        user_index (Dict[str, int]): Dictionary mapping usernames to unique indices.
        global_influence (ndarray): Array of global influence values for each user.
        mentions_matrix_nonNorm (ndarray): Non-normalized mentions matrix.

    Returns:
        ndarray: Local influence matrix (n x n), normalized using Min-Max scaling.
    """
    n = len(user_index)
    local_influence = np.zeros(n, dtype=float)

    for i in range(n):
        community_influence = np.sum(global_influence[mentions_matrix_nonNorm[i] != 0])
        total_influence = community_influence + global_influence[i]
        local_influence[i] = global_influence[i] / total_influence if total_influence else 0

    local_influence_matrix = mentions_matrix_nonNorm * local_influence[:, np.newaxis]
    return normalization_min_max(local_influence_matrix)



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

    for i in range(n-1):
        if len(users_tweet_text[i]) == 0 or users_stances[index_user[i]] is None:
            continue

        embeddings_user_i = calculate_embeddings(i)
        for j in range(i+1,n):
            if users_stances[index_user[j]] is None:
                continue
            mentions_ij = mentions_matrix_nonNorm[i, j]

            if abs(users_stances[index_user[i]] - users_stances[index_user[j]]) < 0.2:
                embeddings_user_j = calculate_embeddings(j)

                similarity_opinions_ij: float = similarity_model.similarity(embeddings_user_i,
                                                                                embeddings_user_j).mean()
                similarity_opinions_ij = round(float(similarity_opinions_ij),3)

                users_affinity[i, j] = similarity_opinions_ij * (mentions_ij + 1)
    #users_affinity[users_affinity < 1] = 0
    b = time.time()
    print(b - a)
    users_affinity = normalization_min_max(users_affinity)
    return users_affinity

"""
def build_agreement_clique_matrix(
        users_tweet_text: ndarray[set[str]],
        users_stances: Dict[str, float],
        index_user: Dict[str, int],
        agreement_threshold=0.15,
) -> ndarray[ndarray[float]]:
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
"""

def build_agreement_matrix(
    mentions_matrix: ndarray,
    users_stances: Dict[str, float],
    index_user: Dict[int, str],
    agreement_threshold: float = 0.2
) -> ndarray:
    """
    Computes an agreement matrix based on users' stance similarities and mentions.

    Parameters:
        mentions_matrix (ndarray): Matrix indicating mentions between users (n x n).
        users_stances (Dict[str, float]): Dictionary mapping user identifiers to their stances.
        index_user (Dict[int, str]): Dictionary mapping indices to user identifiers.
        agreement_threshold (float): Threshold for determining agreement (default is 0.2).

    Returns:
        ndarray: Agreement matrix (n x n) where:
                 - 1 indicates agreement,
                 - -1 indicates disagreement,
                 - 0 indicates no interaction or invalid stances.
    """
    n = len(users_stances)
    users_agreement = np.zeros((n, n), dtype=float)

    for i in range(n - 1):
        user_i = index_user.get(i)
        stance_i = users_stances.get(user_i)
        if stance_i is None:
            continue

        for j in range(i + 1, n):
            user_j = index_user.get(j)
            stance_j = users_stances.get(user_j)
            if stance_j is None or (mentions_matrix[i, j] == 0 and mentions_matrix[j, i] == 0):
                continue

            agreement = abs(stance_i - stance_j) < agreement_threshold
            users_agreement[i, j] = 1 if agreement else -1

    return users_agreement

