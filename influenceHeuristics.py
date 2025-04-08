import time
from ast import literal_eval
from typing import List, Dict, Set, Tuple, Callable

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.threshold import betweenness_sequence
from numpy import ndarray
from sentence_transformers import SentenceTransformer

from processData import normalization_min_max, get_embeddings, get_mentions_list

similarity_model = SentenceTransformer('jaimevera1107/all-MiniLM-L6-v2-similarity-es')


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


def build_interaction_matrix(
        df: pd.DataFrame,
        user_to_index: Dict[str, int]
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Builds multiple interaction matrices quantifying interactions between users.

    Parameters:
        df (pd.DataFrame): DataFrame containing tweet data with the following columns:
            - 'author_username': The username of the tweet's author.
            - 'entities': A JSON string representing tweet entities (mentions).
            - 'ref_author': The username of the original author (if retweeted).
            - 'created_at': Timestamp of the tweet's creation.
            - 'text': The full text of the tweet.

        user_to_index (Dict[str, int]): Dictionary mapping usernames to unique indices.

    Returns:
        Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
            - Normalized interaction matrix (mentions + retweets).
            - Matrix of lists containing timestamps for mentions.
            - Rounded interaction matrix before normalization.
            - Retweets-only matrix.
            - Direct mentions-only matrix.
    """
    n = len(user_to_index)
    interactions_matrix = np.zeros((n, n), dtype=int)
    mentions_matrix = np.zeros((n, n), dtype=int)
    retweets_matrix = np.zeros((n, n), dtype=int)
    interactions_matrix_date = np.empty((n, n), dtype=object)

    for row in df.itertuples(index=False):
        author = row.author_username
        author_idx = user_to_index.get(author)

        if author_idx is None:
            continue

        is_retweet = row.ref_type == 'retweeted'
        if is_retweet:
            ref_author_idx = user_to_index.get(row.ref_author, None)
            if ref_author_idx is not None:
                interactions_matrix[ref_author_idx, author_idx] += 1
                retweets_matrix[ref_author_idx, author_idx] += 1

                if interactions_matrix_date[ref_author_idx, author_idx] is None:
                    interactions_matrix_date[ref_author_idx, author_idx] = [row.created_at]
                else:
                    interactions_matrix_date[ref_author_idx, author_idx].append(row.created_at)

            continue

        is_reply = row.ref_type == 'replied_to'
        mentions = set(get_mentions_list(row.entities))
        for mentioned_user in mentions:
            mentioned_user_idx = user_to_index.get(mentioned_user, None)
            if mentioned_user_idx is not None:
                interactions_matrix[mentioned_user_idx, author_idx] += 1
                if interactions_matrix_date[mentioned_user_idx, author_idx] is None:
                    interactions_matrix_date[mentioned_user_idx, author_idx] = [row.created_at]
                else:
                    interactions_matrix_date[mentioned_user_idx, author_idx].append(row.created_at)

                if is_reply:
                    first_space_idx = row.text.find(" ")
                    cleaned_text = row.text[first_space_idx + 1:] if first_space_idx != -1 else row.text

                    if f"@{mentioned_user}" in cleaned_text:
                        mentions_matrix[mentioned_user_idx, author_idx] += 1

    normalized_matrix = normalization_min_max(interactions_matrix)
    retweets_matrix = normalization_min_max(retweets_matrix)
    mentions_matrix = normalization_min_max(mentions_matrix)
    rounded_matrix = np.round(interactions_matrix, 3)

    return normalized_matrix, interactions_matrix_date, rounded_matrix, retweets_matrix, mentions_matrix


def build_global_influence_matrix(
        df: pd.DataFrame,
        user_to_index: Dict[str, int],
        interactions_matrix_nonNorm: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Computes the global influence matrix by weighting user interactions and mentions.

    Parameters:
        df (pd.DataFrame): DataFrame containing tweet data with the following columns:
            - 'author_username': The username of the tweet's author.
            - 'public_metrics': JSON string with metrics such as retweets, replies, likes, etc.
        user_to_index (Dict[str, int]): Dictionary mapping usernames to unique indices.
        interactions_matrix_nonNorm (ndarray): Non-normalized mentions matrix.

    Returns:
        Tuple[ndarray, ndarray]:
            - Global influence matrix (n x n), normalized using Min-Max.
            - Global influence vector for each user.
    """
    n = len(user_to_index)
    global_influence = np.zeros(n, dtype=float)
    betweenness_centrality = np.zeros(n, dtype=float)

    metrics_keys = [
        'retweet_count', 'reply_count', 'like_count',
        'quote_count', 'bookmarks_count', 'impressions_count'
    ]

    for row in df.itertuples(index=False):
        author = row.author_username
        author_idx = user_to_index.get(author)
        if author_idx is not None:
            tweet_metrics = literal_eval(row.public_metrics)
            n_retweets, n_replies, n_likes, n_quotes, n_bookmarks, n_impressions = (
                tweet_metrics.get(key, 0) for key in metrics_keys
            )
            global_influence[author_idx] += (n_retweets + n_quotes + n_replies + n_likes)

    global_influence_matrix = interactions_matrix_nonNorm * global_influence[:, np.newaxis]
    global_influence_matrix = normalization_min_max(global_influence_matrix)

    G = nx.from_numpy_array(interactions_matrix_nonNorm, create_using=nx.DiGraph)
    betweenness = nx.betweenness_centrality(G)
    for index, user_betweenness in betweenness.items():
        betweenness_centrality[index] = user_betweenness
    betweenness_influence_matrix = interactions_matrix_nonNorm * betweenness_centrality[:, np.newaxis]
    betweenness_influence_matrix = normalization_min_max(betweenness_influence_matrix)

    return global_influence_matrix, betweenness_influence_matrix


def build_affinities_matrix(
        users_tweet_text: List[Set[str]],
        users_stances: Dict[str, float],
        index_user: Dict[int, str],
        mentions_matrix_nonNorm: np.ndarray,
        affinityEmit: Callable[[str, int], None]
) -> np.ndarray:
    """
    Calculates the affinity between users based on similarity of opinions and polarities.

    Params:
        users_tweet_text (List[set[str]]): Preprocessed opinions for each user.
        users_stances (Dict[str, float]): Dictionary of user stances on a particular topic.
        index_user (Dict[str, int]): Mapping of user identifiers to their index.
        mentions_matrix_nonNorm (np.ndarray): Matrix of user interactions/mentions.

    Returns:
        np.ndarray: Affinity matrix between users.
    """
    print("working...")
    a = time.time()
    n = len(users_tweet_text)
    users_affinity = np.zeros((n, n), float)

    user_with_stances = {i: users_stances[user] for i, user in enumerate(index_user.values()) if
                         users_stances[user] is not None}

    def calculate_embeddings(index):
        opinions = users_tweet_text[index]
        embeddings = np.array([get_embeddings(opinion, similarity_model) for opinion in opinions])
        return embeddings

    embeddings = {}
    affinityEmit("affinity_work_info", n)
    for i in range(n - 1):
        if i not in user_with_stances:
            continue
        stance_i = user_with_stances[i]
        embeddings_i = calculate_embeddings(i) if i not in embeddings else embeddings.pop(i)
        for j in range(i + 1, n):
            if j not in user_with_stances:
                continue
            stance_j = user_with_stances[j]
            embeddings_j = calculate_embeddings(j) if j not in embeddings else embeddings[j]
            embeddings[j] = embeddings_j
            stance_diff = abs(stance_i - stance_j)
            if stance_diff < 0.2:
                similarity_opinions: float = similarity_model.similarity(embeddings_i,
                                                                         embeddings_j).mean()

                affinity_value_ij = similarity_opinions * mentions_matrix_nonNorm[i, j]
                affinity_value_ji = similarity_opinions * mentions_matrix_nonNorm[j, i]
                users_affinity[i, j] = affinity_value_ij if affinity_value_ij != 0 else (
                    similarity_opinions if similarity_opinions > .7 else 0)
                users_affinity[j, i] = affinity_value_ji if affinity_value_ji != 0 else (
                    similarity_opinions if similarity_opinions > .7 else 0)
        b = time.time()
        print(f"affinity work: {i=}, {(i + 1) / (n - 1):.1%}, {b - a:.1f}s", end="\r")
        if i % 5 == 0:
            affinityEmit("affinity_work", (i + 1) / (n - 1))

    users_affinity = normalization_min_max(users_affinity)
    users_affinity[users_affinity < 0.01] = 0
    print("\naffinity work: done")
    return users_affinity


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
