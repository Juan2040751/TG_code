from typing import Dict
from influenceHeuristics import identify_nodes, build_mentions_matrix
from processData import preprocess_dataframe, build_users_tweet_text, create_link_processor
from ast import literal_eval
import networkx as nx
import numpy as np
import pandas as pd
from numpy import ndarray

df = pd.read_csv('ReformaPensional.csv')
df = preprocess_dataframe(df)
users = identify_nodes(df)
user_index: Dict[str, int] = {user: idx for idx, user in enumerate(users)}
index_user: Dict[int, str] = {index: user for user, index in user_index.items()}
users_tweet_text = build_users_tweet_text(df, user_index)
get_links_matrix = create_link_processor(index_user)

from processData import calculate_stance
from influenceHeuristics import build_agreement_clique_matrix
topic = "La reforma pensional en Colombia"
topic_context = (
        "El gobierno del presidente Gustavo Petro presentó un proyecto de ley para reformar el sistema pensional, "
        "el cual fue aprobado con el apoyo de algunos sectores y cuestionado por otros.")
prompt = (
        f"Eres un analista experto e imparcial en {topic}. Tu tarea es medir la postura de un usuario frente a este tema en un rango continuo entre 0 y 1, donde: "
        f"0 = Férreamente en contra, 1 = Reciamente a favor, 0.5 = Neutral, "
        f"0.25 = Oposición moderada (reconoce o no está enfáticamente indispuesto a reconocer aspectos positivos), "
        f"0.75 = Apoyo moderado (reconoce o está predispuesto a reconocer objeciones)."
        f"Contexto: \"{topic_context}\" "
        "Entrada: Opiniones del usuario desde una red social"
        "Instrucciones: "
        "1. Analiza la postura general en cada opinión"
        "2. Calcula la proporción de frases a favor vs. en contra del tema"
        "3. Emociones explícitas directas hacia el tema refuerzan la postura (más emoción positiva = postura cercana a 1; negativa = cercana a 0). "
        "4. Pondera adjetivos y adverbios utilizados para expresar alguna postura frente al tema"
        "5. Conectores de oposición indican postura moderada. Ej: 'aunque no es perfecta, la reforma pensional es un paso positivo' o 'a pesar de las críticas, tiene aspectos rescatables'"
        "6. Los adverbios de cantidad indican posturas extremas, y los de duda, posturas moderadas"
        f"Salida esperada: Un número decimal entre 0 y 1, con dos decimales, como 0.21, 0.75 o 0.03, representando la postura general del usuario frente a {topic}.")

stances = calculate_stance(users_tweet_text, users, prompt)
agreement_matrix= build_agreement_clique_matrix(users_tweet_text, stances, index_user, 0.08)

print(len(agreement_matrix[agreement_matrix!=0]))