import re
from functools import cache
from typing import Dict, List, Optional

import emoji
import numpy as np
import pandas as pd
from numpy import ndarray


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a DataFrame by filling missing values in specific columns with default values.

    Columns processed:
        - 'geo': Fills missing values with "{}".
        - 'entities': Fills missing values with "{}".
        - 'in_reply_to_user_id': Fills missing values with 0.
        - 'ref_id': Fills missing values with 0.
        - 'ref_type': Fills missing values with "tweeted".
        - 'ref_author_id': Fills missing values with 0.
        - 'ref_author': Fills missing values with an empty string "".
        - 'ref_text': Fills missing values with an empty string "".
        - 'ref_note_tweet': Fills missing values with an empty string "".

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data to process.

    Returns:
        pd.DataFrame: Processed DataFrame with missing values replaced.

    Raises:
        ValueError: If one or more expected columns are missing from the DataFrame.
    """

    default_values = {
        "text": "",
        "created_at": 0,
        "public_metrics": "{}",
        'entities': "{}",
        'author_username': "",
        'ref_type': "tweeted",
        'ref_author': "",
        'ref_text': "",
        'ref_note_tweet': ""
    }
    missing_columns = [col for col in default_values if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"El Dataset no contiene las siguientes columnas necesarias: {', '.join(missing_columns)}"
        )
    df = df[["text", "created_at", "public_metrics", "entities", "author_username", "ref_type",
            "ref_author", "ref_text", "ref_note_tweet"]]
    for col, default in default_values.items():
        df[col] = df[col].fillna(default)
    return df


def clean_text(text: str) -> str:
    """
    Preprocesses a text by replacing URLs and emojis, and removing extra spaces.

    Parameters:
        text (str): Original text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    text = re.sub(r'http\S+|www\S+', ':url:', text)  # Replace URLs with ':url:'
    text = emoji.demojize(text, language="es")  # Replace emojis with text descriptions
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and trim
    return text


def build_users_tweet_text(
        df: pd.DataFrame,
        user_index: Dict[str, int]
) -> ndarray:
    """
    Builds a list of sets containing preprocessed opinions or interactions for each user.

    Parameters:
        df (pd.DataFrame): DataFrame containing tweet data.
        user_index (Dict[str, int]): Dictionary mapping each username to a unique index.

    Returns:
        ndarray: Array of sets with preprocessed opinions or interactions for each user.
    """
    n = len(user_index)
    users_tweet_text = [set() for _ in range(n)]

    for row in df.itertuples(index=False):
        author_idx = user_index[row.author_username]
        ref_author = row.ref_author if row.ref_author else None
        ref_text = row.ref_note_tweet if row.ref_note_tweet else row.ref_text

        if row.ref_type == "tweeted":
            users_tweet_text[author_idx].add(clean_text(row.text))
        elif row.ref_type == "retweeted" and ref_text:
            clean_ref_text = clean_text(ref_text)
            users_tweet_text[author_idx].add(clean_ref_text)

            if ref_author:
                ref_author_idx = user_index[ref_author]
                users_tweet_text[ref_author_idx].add(clean_ref_text)
        elif row.ref_type in ["quoted", "replied_to"] and ref_text:
            interaction_text = f'{clean_text(row.text)}\n[{row.ref_type} @{ref_author}: "{clean_text(ref_text)[:200]}"]'
            users_tweet_text[author_idx].add(interaction_text)

            if ref_author and ref_author in user_index:
                ref_author_idx = user_index[ref_author]
                users_tweet_text[ref_author_idx].add(clean_text(ref_text))

    return np.array(users_tweet_text, dtype=object)


@cache
def get_embeddings(text: str, similarity_model):
    """
    Computes and caches the embeddings for a given opinion, removing additional information
    after a newline and within square brackets.

    Parameters:
        text (str): The opinion text to compute embeddings for.

    Returns:
        Embeddings of the cleaned opinion.
    """
    cleaned_text = re.sub(r'\n\[.*?\]', '', text).strip()
    return similarity_model.encode(cleaned_text, clean_up_tokenization_spaces=True, normalize_embeddings=True)

@cache
def get_polarity(text: str, sentiment_analyzer) -> float:
    """
    Calcula y guarda en caché la polaridad de una opinión.

    Parámetros:
        text (str): Opinión a calcular polaridad.

    Devuelve:
        float: Valor de polaridad de la opinión.
    """
    prediction = sentiment_analyzer.predict(text)
    NEG, NEU, POS = prediction.probas["NEG"], prediction.probas["NEU"], prediction.probas["POS"]
    polarity = round(POS - NEG, 3)
    return polarity


def create_link_processor(index_user: Dict[int, str]):
    """
    Creates a processor function to convert an adjacency matrix into a list of links for a network.

    Parameters:
        index_user (Dict[int, str]): Dictionary mapping node indices to their unique identifiers.

    Returns:
        Callable[[ndarray, Optional[ndarray]], List[Dict[str, Any]]]: A function that processes an adjacency matrix
        and an optional mentions date matrix into a list of links.
    """
    def get_links_matrix(
        adjacency_matrix: ndarray,
        mentions_matrix_date: Optional[ndarray] = None
    ) -> List[Dict[str, float]]:
        """
        Converts an adjacency matrix into a list of links (edges) with source, target, and influence values.

        Parameters:
            adjacency_matrix (ndarray): A 2D array representing influence relationships.
                                        Each cell contains the influence value from a source node to a target node.
            mentions_matrix_date (Optional[ndarray]): A 2D array containing dates for mentions between nodes.
                                                      Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the network's links.
                                  Each dictionary includes:
                                    - "source" (str): Source node identifier.
                                    - "target" (str): Target node identifier.
                                    - "influenceValue" (float): Influence value from source to target.
                                    - "date" (str): Date of mention (if mentions_matrix_date is provided).
        """
        links = []

        for influencer_id, user_influences in enumerate(adjacency_matrix):
            source_name = index_user[influencer_id]

            for influenced_user_id, interpersonal_influence in enumerate(user_influences):
                if abs(interpersonal_influence) >= 0.01:
                    target_name = index_user[influenced_user_id]
                    link = {
                        "source": source_name,
                        "target": target_name,
                        "influenceValue": round(float(interpersonal_influence), 3)
                    }
                    if mentions_matrix_date is not None:
                        link["date"] = mentions_matrix_date[influencer_id, influenced_user_id]
                    links.append(link)

        return links

    return get_links_matrix



def normalization_min_max(matrix: ndarray) -> ndarray:
    """
    Applies log transformation followed by Min-Max normalization to a matrix.

    Parameters:
        matrix (ndarray): A numerical 2D array to be normalized.

    Returns:
        ndarray: A transformed and normalized matrix, where values are scaled to the range [0, 1].
    """
    matrix_transformed = np.log1p(matrix)

    # Min-Max normalization
    min_value = matrix_transformed.min()
    max_value = matrix_transformed.max()
    if max_value > min_value:
        matrix_transformed = (matrix_transformed - min_value) / (max_value - min_value)
    matrix_transformed = np.round(matrix_transformed,3)
    return matrix_transformed

