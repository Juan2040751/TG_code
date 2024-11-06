import re
from functools import cache
from typing import List, Dict, Set

import emoji
import pandas as pd


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa el DataFrame para rellenar valores nulos en columnas específicas con valores predeterminados.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos a procesar.

    Devuelve:
        pd.DataFrame: DataFrame procesado con valores nulos reemplazados.
    """
    # Rellenar valores nulos en columnas con valores predeterminados
    df['geo'] = df['geo'].fillna("{}")
    df['entities'] = df['entities'].fillna("{}")
    df['in_reply_to_user_id'] = df['in_reply_to_user_id'].fillna(0)
    df['ref_id'] = df['ref_id'].fillna(0)
    df['ref_type'] = df['ref_type'].fillna("tweeted")
    df['ref_author_id'] = df['ref_author_id'].fillna(0)
    df['ref_author'] = df['ref_author'].fillna("")
    df['ref_text'] = df['ref_text'].fillna("")
    df['ref_note_tweet'] = df['ref_note_tweet'].fillna("")

    return df


def clean_text(text: str) -> str:
    """
    Realiza el preprocesamiento de un texto, reemplazando URL y emojis, y eliminando espacios extras.

    Parámetros:
        text (str): Texto original a preprocesar.

    Devuelve:
        str: Texto preprocesado.
    """
    text = re.sub(r'http\S+|www\S+', ':url:', text)  # Reemplaza URL
    text = emoji.demojize(text, language="es")  # Reemplaza emojis por su descripción en texto
    text = re.sub(' +', ' ', text)  # Elimina espacios extras
    return text


def build_users_tweet_text(
        df: pd.DataFrame,
        user_index: Dict[str, int]
) -> List[Set[str]]:
    """
    Construye una lista de sets de opiniones de cada usuario preprocesadas.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos de los tweets.
        user_index (Dict[str, int]): Diccionario que asigna un índice único a cada usuario.

    Devuelve:
        List[Set[str]]: Lista de sets de opiniones preprocesadas de cada usuario.
    """
    n = len(user_index)
    users_tweet_text = [set() for _ in range(n)]

    for _, row in df.iterrows():
        author = row['author_username']
        author_idx = user_index[author]

        if row['ref_type'] == "tweeted":
            users_tweet_text[author_idx].add(clean_text(row['text']))
            continue

        ref_author = row['ref_author']
        ref_author_idx = user_index[ref_author] if ref_author else None

        if row['ref_type'] == "retweeted":
            tweet_text = row['ref_note_tweet'] if row['ref_note_tweet'] else row['ref_text']
            users_tweet_text[author_idx].add(clean_text(tweet_text))
            if ref_author_idx is not None:
                users_tweet_text[ref_author_idx].add(clean_text(tweet_text))
            continue

        if row['ref_type'] in ["quoted", "replied_to"]:
            users_tweet_text[author_idx].add(clean_text(row['text']))
            ref_tweet_text = row['ref_note_tweet'] if row['ref_note_tweet'] else row['ref_text']
            if ref_author_idx is not None:
                users_tweet_text[ref_author_idx].add(clean_text(ref_tweet_text))
            users_tweet_text[author_idx].add(clean_text(row['text']))
            continue

    return users_tweet_text


@cache
def get_embeddings(text: str, similarity_model):
    """
    Calcula y guarda en caché los embeddings de una opinión dada.

    Parámetros:
        text (str): Opinión a calcular embeddings.

    Devuelve:
        Embeddings de la opinión.
    """
    return similarity_model.encode(text)


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
