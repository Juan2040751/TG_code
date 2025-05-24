import time
from ast import literal_eval
from typing import List, Dict, Set, Tuple, Callable, Optional

import networkx as nx
import numpy as np
import pandas as pd
from numpy import ndarray
from sentence_transformers import SentenceTransformer

from beliefHeuristic import users_with_unique_opinions
from processData import normalization_min_max, get_embeddings, get_mentions_list

similarity_model = SentenceTransformer('jaimevera1107/all-MiniLM-L6-v2-similarity-es')


def build_interaction_matrix(
        df: pd.DataFrame,
        user_to_index: Dict[str, int],
        send_feedback: Callable[[Dict[str, any]], None]
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
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
        send_feedback (Callable[[Dict[str, any]], None]): Function to send feedback to the UI.

    Returns:
        Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
            - Normalized interaction matrix (mentions + retweets).
            - Rounded interaction matrix before normalization.
            - Retweets-only matrix.
            - Direct mentions-only matrix.
    """
    n = len(user_to_index)
    interactions_matrix = np.zeros((n, n), dtype=int)
    mentions_matrix = np.zeros((n, n), dtype=int)
    retweets_matrix = np.zeros((n, n), dtype=int)
    send_feedback({"open": True, "message": "Calculando influencia basada en interacciones", "progress": 0.0})
    count_rows = len(df)
    for row in df.itertuples():
        index = row.Index
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

        is_reply = row.ref_type == 'replied_to'
        mentions = set(get_mentions_list(row.entities))
        for mentioned_user in mentions:
            if is_retweet and mentioned_user == row.ref_author:
                continue
            mentioned_user_idx = user_to_index.get(mentioned_user, None)
            if mentioned_user_idx is not None:
                interactions_matrix[mentioned_user_idx, author_idx] += 1

                if is_reply:
                    first_space_idx = row.text.find(" ")
                    cleaned_text = row.text[first_space_idx + 1:] if first_space_idx != -1 else row.text

                    if f"@{mentioned_user}" in cleaned_text:
                        mentions_matrix[mentioned_user_idx, author_idx] += 1
        if index % 20 == 0:
            send_feedback({"open": True, "message": "Calculando influencia basada en interacciones",
                           "progress": index / count_rows})

    normalized_matrix = normalization_min_max(interactions_matrix)
    retweets_matrix = normalization_min_max(retweets_matrix)
    mentions_matrix = normalization_min_max(mentions_matrix)
    rounded_matrix = np.round(interactions_matrix, 3)
    send_feedback({"open": False, "message": "", "progress": 0})
    return normalized_matrix, rounded_matrix, retweets_matrix, mentions_matrix


def build_popularity_influence_matrix(
        df: pd.DataFrame,
        user_to_index: Dict[str, int],
        interactions_matrix_nonNorm: ndarray,
        send_feedback: Callable[[Dict[str, any], Optional[str]], None]
) -> Tuple[ndarray, ndarray]:
    """
    Computes the global influence matrix by weighting user interactions and mentions.

    Params:
        df (pd.DataFrame): Tweet data with 'author_username' and 'public_metrics' as JSON strings.
        user_to_index (Dict[str, int]): Maps usernames to unique row indices.
        interactions_matrix_nonNorm (ndarray): Raw mentions or interaction matrix.
        send_feedback (Callable[[Dict[str, any], Optional[str]], None]): Function to send feedback to the UI.


    Returns:
        Tuple[ndarray, ndarray]: (normalized global influence matrix, normalized betweenness matrix)
    """
    n = len(user_to_index)
    global_influence = np.zeros(n, dtype=float)
    send_feedback({"message":"Calculando influencia basada en popularidad", "open": True}, event="progress_feedback")
    df = df[df['author_username'].isin(user_to_index)]
    df['author_index'] = df['author_username'].map(user_to_index)
    df['metrics'] = df['public_metrics'].map(literal_eval)

    df['influence_score'] = df['metrics'].map(
        lambda m: sum(m.get(k, 0) for k in ['retweet_count', 'reply_count', 'like_count', 'quote_count'])
    )

    influence_per_author = df.groupby('author_index')['influence_score'].sum()
    global_influence[influence_per_author.index] = influence_per_author.values

    popularity_influence_matrix = interactions_matrix_nonNorm * global_influence[:, None]
    popularity_influence_matrix = normalization_min_max(popularity_influence_matrix)

    betweenness_centrality = np.zeros(n, dtype=float)
    G = nx.from_numpy_array(interactions_matrix_nonNorm, create_using=nx.DiGraph)
    betweenness = nx.betweenness_centrality(G)
    for index, user_betweenness in betweenness.items():
        betweenness_centrality[index] = user_betweenness
    betweenness_influence_matrix = interactions_matrix_nonNorm * betweenness_centrality[:, np.newaxis]
    betweenness_influence_matrix = normalization_min_max(betweenness_influence_matrix)
    send_feedback({"open": False, "message": "",
                   "progress": 0})
    return popularity_influence_matrix, betweenness_influence_matrix


def build_affinities_matrix(
        users_tweet_text: List[Set[str]],
        users_stances: Dict[str, float],
        user_to_index: Dict[str, int],
        mentions_matrix_nonNorm: np.ndarray,
        affinityEmit: Callable[[str, int | float], None]
) -> np.ndarray:
    print("working...")
    n = len(users_tweet_text)
    users_affinity = np.zeros((n, n), dtype=float)
    a = time.time()

    users = np.array(list(user_to_index.keys()))
    users_with_opinions = np.array(
        [(user, frozenset(opinions))
         for user, opinions in zip(users, users_tweet_text)
         if users_stances[user] is not None and opinions]
    )
    users_with_unique_opinions_, users_with_same_opinions = users_with_unique_opinions(users_with_opinions)

    len_users_with_unique_opinions = len(users_with_unique_opinions_)
    embeddings = {}
    affinityEmit("affinity_work_info", n)
    for index, (user, opinions) in enumerate(users_with_unique_opinions_):
        i = user_to_index[user]
        embeddings[i] = np.mean([get_embeddings(op, similarity_model) for op in opinions], axis=0)
        if index % 30 == 0:
            affinityEmit("affinity_work_buffer", index / len_users_with_unique_opinions)
            print(f"affinity buffer: ({index / len_users_with_unique_opinions:.1%})", end="\r")

    rep_users = np.array([user for user, _ in users_with_unique_opinions_])

    for idx_i, user_i in enumerate(rep_users):
        i = user_to_index[user_i]
        stance_i = users_stances.get(user_i)
        emb_i = embeddings[i]

        for idx_j, user_j in enumerate(rep_users[idx_i + 1:], start=idx_i + 1):
            j = user_to_index[user_j]
            stance_j = users_stances.get(user_j)

            if abs(stance_i - stance_j) > 0.2:
                continue

            emb_j = embeddings[j]
            sim = similarity_model.similarity(emb_i, emb_j).mean()
            aff_ij = sim * mentions_matrix_nonNorm[i, j]
            aff_ji = sim * mentions_matrix_nonNorm[j, i]
            val_ij = aff_ij if aff_ij != 0 else (sim if sim > 0.8 else 0)
            val_ji = aff_ji if aff_ji != 0 else (sim if sim > 0.8 else 0)

            all_i = [i] + [user_to_index[u] for u in users_with_same_opinions.get(user_i, [])]
            all_j = [j] + [user_to_index[u] for u in users_with_same_opinions.get(user_j, [])]

            for u_i in all_i:
                for u_j in all_j:
                    if u_i != u_j:
                        users_affinity[u_i, u_j] = val_ij
                        users_affinity[u_j, u_i] = val_ji

        if idx_i % 10 == 0:
            affinityEmit("affinity_work", (idx_i + 1) / (len(rep_users) - 1))
            b = time.time()
            print(f"affinity work: {int((b - a) // 60)}:{(b - a) % 60:.0f}, ({(idx_i + 1) / (len(rep_users) - 1):.1%})",
                  end="\r")

    users_affinity = normalization_min_max(users_affinity)
    print("\naffinity work: done")
    return users_affinity
