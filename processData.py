import concurrent.futures
import csv
import re
import time
from ast import literal_eval
from functools import cache
from typing import Dict, List, Set, Union

import emoji
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from numpy import ndarray
from openai import OpenAI, OpenAIError, APIConnectionError
from pydantic import BaseModel, Field, model_validator


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
) -> ndarray:
    """
    Construye una lista de sets de opiniones o interacciones para cada usuario preprocesadas.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos de los tweets.
        user_index (Dict[str, int]): Diccionario que asigna un índice único a cada usuario.

    Devuelve:
        ndarray: Matriz de listas de opiniones o interacciones preprocesadas de cada usuario.
    """
    n = len(user_index)
    users_tweet_text = [set() for _ in range(n)]

    for row in df.itertuples(index=False):
        author_idx = user_index[row.author_username]
        ref_author = row.ref_author if row.ref_author else None
        ref_text = row.ref_note_tweet if row.ref_note_tweet else row.ref_text

        if row.ref_type == "tweeted":
            # Opinión propia del autor
            users_tweet_text[author_idx].add(clean_text(row.text))
        elif row.ref_type == "retweeted" and ref_text:
            # Opinión propia al retweetear
            clean_ref_text = clean_text(ref_text)
            users_tweet_text[author_idx].add(clean_ref_text)

            # Si hay un autor referenciado, también se asocia el texto al autor original
            if ref_author and ref_author in user_index:
                ref_author_idx = user_index[ref_author]
                users_tweet_text[ref_author_idx].add(clean_ref_text)
        elif row.ref_type in ["quoted", "replied_to"] and ref_text:
            # Construir el texto como interacción
            interaction_text = f'{clean_text(row.text)}\n[{row.ref_type} @{ref_author}: "{clean_text(ref_text)}"]'

            # Se registra como interacción para el autor del tweet
            users_tweet_text[author_idx].add(interaction_text)

            # Si hay un autor referenciado, también se asocia la interacción al autor original
            if ref_author and ref_author in user_index:
                ref_author_idx = user_index[ref_author]
                users_tweet_text[ref_author_idx].add(clean_text(ref_text))

    # Convertir conjuntos en listas para mantener consistencia
    return np.array([user_texts for user_texts in users_tweet_text])


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


def create_link_processor(index_user: Dict[float | int, str]):
    def get_links_matrix(adjacency_matrix: ndarray, mentions_matrix_date: ndarray[ndarray[List[str]]] = None) -> List[
        Dict[str, float]]:
        """
        Convierte una matriz de adyacencia en una lista de enlaces (aristas) con los nodos fuente y destino,
        incluyendo el valor de influencia.

        Args:
            adjacency_matrix (np.ndarray): Matriz de adyacencia que representa las relaciones de influencia entre usuarios.
                                            Cada celda contiene el valor de la influencia de un nodo fuente a un nodo destino.
            index_user (Dict[int, str]): Diccionario que mapea el índice de un nodo (entero) a su identificador (cadena).

        Returns:
            List[Dict[str, float]]: Lista de diccionarios que representa los enlaces de la red.
                                    Cada diccionario contiene:
                                        - "source_id" (str): Identificador del nodo fuente.
                                        - "target_id" (str): Identificador del nodo destino.
                                        - "influence_value" (float): Valor de la influencia del nodo fuente sobre el nodo destino.
        """
        links = []

        for influencer_id, user_influences in enumerate(adjacency_matrix):
            source_id = index_user[influencer_id]  # Obtener el identificador del nodo fuente

            for influenced_user_id, interpersonal_influence in enumerate(user_influences):
                if interpersonal_influence != 0:  # Solo considerar influencias no nulas
                    target_id = index_user[influenced_user_id]  # Obtener el identificador del nodo destino
                    link = {
                        "source": source_id,
                        "target": target_id,
                        "influenceValue": round(float(interpersonal_influence), 3)  # Convertir el valor a float
                    }
                    if mentions_matrix_date is not None:
                        link["date"] = mentions_matrix_date[influencer_id, influenced_user_id]
                    links.append(link)

        return links

    return get_links_matrix






def normalization_min_max(matrix: ndarray) -> ndarray:
    matrix_transformed = np.log1p(matrix)

    # Normalización Min-Max después de la transformación
    min_value = matrix_transformed.min()
    max_value = matrix_transformed.max()
    if max_value > min_value:  # Evitar división por cero
        matrix_transformed = (matrix_transformed - min_value) / (max_value - min_value)
    return matrix_transformed
