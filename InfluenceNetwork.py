import base64
from io import StringIO
from typing import Dict, Set, List

import pandas as pd
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from numpy import ndarray

from influenceHeuristics import (
    build_mentions_matrix, identify_nodes, build_global_influence_matrix,
    build_local_influence_matrix, build_affinities_matrix, build_agreement_matrix
)
from processData import preprocess_dataframe, create_link_processor, build_users_tweet_text
from stanceHeuristics import calculate_stance

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=15000000, always_connect=True, ping_interval=1000, ping_timeout=1000)


def build_influence_networks(df: pd.DataFrame, user_index, index_user, sid):
    get_links_matrix = create_link_processor(index_user)
    mentions_matrix, mentions_matrix_date, mentions_matrix_nonNorm = build_mentions_matrix(df, user_index)

    socketio.emit("influence_heuristic", {"mentions_links": get_links_matrix(mentions_matrix, mentions_matrix_date)},
                  to=sid)

    global_influence_matrix, global_influence = build_global_influence_matrix(df, user_index, mentions_matrix_nonNorm)
    socketio.emit("influence_heuristic",
                  {"global_influence_links": get_links_matrix(global_influence_matrix, mentions_matrix_date)}, to=sid)

    local_influence_matrix = build_local_influence_matrix(user_index, global_influence, mentions_matrix_nonNorm)
    socketio.emit("influence_heuristic",
         {"local_influence_links": get_links_matrix(local_influence_matrix, mentions_matrix_date)},
         to=sid)
    socketio.sleep(0.01)
    return mentions_matrix, mentions_matrix_date, get_links_matrix, mentions_matrix_nonNorm


def build_influence_networks_with_stances(stances, index_user, users_tweet_text, mentions_matrix, mentions_matrix_date,
                                          get_links_matrix, mentions_matrix_nonNorm):
    agreement_matrix = build_agreement_matrix(mentions_matrix, stances, index_user)
    emit("influence_heuristic",
         {"agreement_links": get_links_matrix(agreement_matrix, mentions_matrix_date)},
         broadcast=False)

    affinities_matrix = build_affinities_matrix(users_tweet_text, stances, index_user, mentions_matrix_nonNorm)
    emit("influence_heuristic", {"affinities_links": get_links_matrix(affinities_matrix)},
         broadcast=False)



def calc_stances(users_tweet_text: ndarray[Set[str]], users: List[str], sid) -> Dict[str, float | None]:
    topic = "La reforma pensional en Colombia"
    topic_context = (
        "El gobierno del presidente Gustavo Petro presentó un proyecto de ley para reformar el sistema pensional, "
        "el cual fue aprobado con el apoyo de algunos sectores y cuestionado por otros.")
    prompt = (
        f"Eres un analista experto e imparcial en {topic}. Tu tarea es medir la postura de un grupo de usuarios frente a este tema en un rango continuo entre 0 y 1, donde: "
        f"0 = Férreamente en contra, 1 = Reciamente a favor, 0.5 = Neutral, "
        f"0.25 = Oposición moderada (reconoce o no está enfáticamente indispuesto a reconocer aspectos positivos), "
        f"0.75 = Apoyo moderado (reconoce o está predispuesto a reconocer objeciones)."
        f"Contexto: \"{topic_context}\" "
        "Entrada: Un JSON donde cada clave es el nombre de un usuario y su valor es una lista de opiniones obtenidas de redes sociales sobre este tema."
        "Instrucciones: "
        "1. Para cada usuario, analiza la postura general en todas sus opiniones."
        "2. Calcula la proporción de frases a favor vs. en contra del tema."
        "3. Emociones explícitas directas hacia el tema refuerzan la postura (más emoción positiva = postura cercana a 1; negativa = cercana a 0)."
        "4. Pondera adjetivos y adverbios utilizados para expresar alguna postura frente al tema."
        "5. Conectores de oposición indican postura moderada. Ej: 'aunque no es perfecta, la reforma pensional es un paso positivo' o 'a pesar de las críticas, tiene aspectos rescatables'."
        "6. Los adverbios de cantidad indican posturas extremas, y los de duda, posturas moderadas."
        f"Y cualquier otro criterio que consideres relevante para estimar la postura de cada usuario frente a {topic} en el rango [0,1]"
        f"Salida esperada: Un JSON donde cada clave es el nombre de un usuario a analizar sus opiniones y su correspondiente valor es un número de dos decimales, como 0.21, 0.75 o 0.03, representando la postura general del usuario frente a {topic}. No retornes nada más."
    )
    stances = calculate_stance(users_tweet_text, users, prompt, testing=False)

    socketio.emit("stance_heuristic", stances, to=sid, callback=lambda: print("send..."))

    socketio.sleep(1)
    return stances

def calculate_heuristics(users, df, sid):
    user_index: Dict[str, int] = {user: idx for idx, user in enumerate(users)}
    index_user: Dict[int, str] = {index: user for user, index in user_index.items()}


    mentions_matrix, mentions_matrix_date, get_links_matrix, mentions_matrix_nonNorm = build_influence_networks(df, user_index, index_user, sid)

    users_tweet_text = build_users_tweet_text(df, user_index)
    calc_stances(users_tweet_text, users, sid)
    # stances = calc_stances(users_tweet_text, users)

    # build_influence_networks_with_stances(stances, index_user, users_tweet_text, mentions_matrix, mentions_matrix_date, get_links_matrix, mentions_matrix_nonNorm)


@socketio.on('influenceGraph')
def process_csv(message):
    """
    Processes the received CSV data, extracts user information, and builds influence networks.

    Parameters:
        message (dict): Dictionary containing the CSV data in base64 format under the key 'csv_data'.

    Emits:
        - "preprocess_error": If the preprocessing step fails due to missing required columns.
        - "users": List of identified users after processing the CSV.

    Workflow:
        1. Decodes and reads the CSV data into a DataFrame.
        2. Preprocesses the DataFrame to handle missing values.
        3. Identifies unique nodes (users) from the data.
        4. Constructs user interactions and opinions.
        5. Builds influence networks based on the provided context and prompt.
    """
    try:
        # Decode CSV from base64
        csv_data = message.get('csv_data')
        if not csv_data:
            emit("preprocess_error", "CSV data is missing", broadcast=False)
            return

        csv_data = base64.b64decode(csv_data).decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))
        df = preprocess_dataframe(df)

        users = identify_nodes(df)
        emit("users", list(users), broadcast=False)

        sid = request.sid
        socketio.start_background_task(calculate_heuristics, users, df, sid)
    except ValueError as e:
        print(e)
        emit("preprocess_error", f"Error processing CSV: {str(e)}", broadcast=False)


if __name__ == '__main__':
    socketio.run(app, debug=True)
