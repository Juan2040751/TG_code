import csv
import re
import time
from ast import literal_eval
from functools import cache
from typing import List, Dict, Set

import emoji
import openai
import pandas as pd
from dotenv import dotenv_values
from numpy import ndarray
from openai import OpenAI
from pydantic import BaseModel
from openai import OpenAIError, APIConnectionError

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
) -> List[List[str]]:
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

    return list(map(lambda user_texts: list(user_texts), users_tweet_text))


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

def create_link_processor(index_user: Dict[float| int, str]):
    def get_links_matrix(adjacency_matrix: ndarray, mentions_matrix_date: ndarray[ndarray[List[str]]] = None) -> List[Dict[str, float]]:
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
                if interpersonal_influence > 0:  # Solo considerar influencias no nulas
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

openIAKey = dotenv_values(".env")["OPENAI_API_KEY"]
client = OpenAI(api_key=openIAKey)


class Stance(BaseModel):
    value: float





def calculate_stance(users_tweet_text: list[List[str]], users: List[str], prompt: str) -> Dict[str, float]:
    def stanceDetection(opinions: List[str]) -> float:
        max_retries = 3
        delay = 2  # segundos entre reintentos

        for attempt in range(max_retries):
            try:
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": str(opinions)
                        }
                    ],
                    response_format=Stance,
                )
                return literal_eval(completion.choices[0].message.content)["value"]
            except APIConnectionError as e:
                print(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                time.sleep(delay)
            except OpenAIError as e:
                print(f"OpenAI API error: {e}")
                break
        return 0.0

    def save_stance_to_cache(user: str, stance: float):
        """Guarda la postura de un usuario en el archivo CSV."""
        with open(CACHE_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([user, stance])

    CACHE_FILE = "user_stances.csv"
    cached_stances = {}
    with open(CACHE_FILE, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            user, stance = row
            cached_stances[user] = float(stance)

    print(cached_stances.__len__())
    stances = {}

    for index, user in enumerate(users):
        if user in cached_stances:
            # Usar la postura en caché si existe
            stances[user] = cached_stances[user]
        else:
            # Obtener la postura y guardarla en caché
            stance = stanceDetection(users_tweet_text[index])
            stances[user] = stance
            save_stance_to_cache(user, stance)
            print(stances.__len__())
    return stances
