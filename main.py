import base64
from io import StringIO
from typing import Dict, Set, List, Tuple, Callable, Optional

import pandas as pd
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from numpy import ndarray

from beliefHeuristic import calculate_stance
from confidenceHeuristic import estimate_confidence
from influenceHeuristics import (
    build_interaction_matrix, identify_nodes, build_global_influence_matrix, build_affinities_matrix,
    build_agreement_matrix
)
from processData import preprocess_dataframe, create_link_processor, build_users_tweet_text

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=15000000, always_connect=True,
                    ping_timeout=2000, async_mode='threading')


def build_influence_networks(df: pd.DataFrame, user_to_index, index_to_user, sid) -> Tuple[
    ndarray[ndarray[float]], Callable[
        [ndarray, Optional[ndarray]], List[Dict[str, float]]], ndarray[ndarray[int]]]:
    """
    Constructs and emits the first three influence networks, independent of belief estimation.

    Params:
        df (pd.DataFrame): DataFrame containing tweet interactions.
        user_to_index (dict[str, int]): Mapping of usernames to their corresponding index.
        index_to_user (dict[int, str]): Mapping of indices to their corresponding usernames.
        sid (str): Session ID for emitting socket messages.

    Returns:
        tuple:
            - interactions_matrix (ndarray): Matrix representing user mentions.
            - get_links_matrix (Callable): Function to extract links from matrices.
            - interactions_matrix_nonNorm (ndarray): Non-normalized mentions matrix.

    Emits:
        - "influence_heuristic" with mentions-based influence links.
        - "influence_heuristic" with global influence links.
        - "influence_heuristic" with local influence links.
    """

    get_links_matrix = create_link_processor(index_to_user)
    interactions_matrix, interactions_matrix_nonNorm, retweets_matrix, mentions_matrix = build_interaction_matrix(df,
                                                                                                                  user_to_index)

    interactions_links = get_links_matrix(interactions_matrix, "Interacciones")
    retweets_links = get_links_matrix(retweets_matrix, "Retweets")
    del retweets_matrix
    mentions_links = get_links_matrix(mentions_matrix, "Menciones")
    del mentions_matrix

    interactions_links.extend(retweets_links)
    interactions_links.extend(mentions_links)
    socketio.emit("influence_heuristic", {"mentions_links": interactions_links},
                  to=sid)

    global_influence_matrix, betwenness_influence_matrix = build_global_influence_matrix(df, user_to_index,
                                                                                         interactions_matrix_nonNorm)
    popularity_links = get_links_matrix(global_influence_matrix, "Popularidad")
    betwenness_links = get_links_matrix(betwenness_influence_matrix, "Betwenness")
    popularity_links.extend(betwenness_links)
    socketio.emit("influence_heuristic",
                  {"global_influence_links": popularity_links}, to=sid)

    socketio.sleep(0.01)
    return interactions_matrix, get_links_matrix, interactions_matrix_nonNorm


def build_influence_networks_with_stances(stances: Dict[str, float | None], index_to_user: Dict[int, str],
                                          users_tweet_text: ndarray[Set[str]],
                                          interactions_matrix: ndarray[ndarray[float]],
                                          get_links_matrix: Callable[
                                              [ndarray, Optional[ndarray]], List[Dict[str, float]]],
                                          interactions_matrix_nonNorm: ndarray[ndarray[int]], sid: str) -> None:
    """
    Constructs influence networks considering stance similarity, generating both agreement
    and affinity-based networks, and emits the results.

    Params:
        stances: Dictionary mapping users to their stance values.
        index_to_user: Dictionary mapping matrix indices to user IDs.
        users_tweet_text: Dictionary mapping users to their tweets.
        interactions_matrix: Normalized matrix representing user mentions.
        get_links_matrix: Function to extract link representations from a given matrix.
        interactions_matrix_nonNorm: Non-normalized mentions matrix.
        sid: Session ID for emitting results via SocketIO.

    Returns:
        None. The function emits the computed agreement and affinity networks.
    """
    agreement_matrix = build_agreement_matrix(interactions_matrix, stances, index_to_user)
    agreement_links = get_links_matrix(agreement_matrix, "Acuerdo/Desacuerdo")

    def affinityEmit(event: str, val: Dict[str, int]) -> None:
        """ Helper function to emit affinity-related events. """
        socketio.emit(event, val, to=sid)

    affinities_matrix = build_affinities_matrix(users_tweet_text, stances, index_to_user, interactions_matrix_nonNorm,
                                                affinityEmit)
    affinities_links = get_links_matrix(affinities_matrix, "Afinidad")

    affinities_links.extend(agreement_links)
    socketio.emit("influence_heuristic",
                  {"affinities_links": affinities_links},
                  to=sid)


def calculate_beliefs(users_tweet_text: ndarray[Set[str]], users: List[str], sid: str, topic_info: Dict[str, str]) -> \
        Dict[str, float | None]:
    """
    Calculates and emits users' belief estimations based on their textual content.

    Args:
        users_tweet_text (ndarray[Set[str]]): Array containing sets of tweets for each user.
        users (List[str]): List of user IDs.
        sid (str): Session ID for emitting socket messages.

    Returns:
        Dict[str, float | None]: Dictionary mapping each user to a belief score in the range [0,1],
        where 0 indicates strong opposition, 1 strong support, and intermediate values represent
        varying degrees of neutrality or moderation.

    Emits:
        - "belief_heuristic" with the calculated belief scores.

    Notes:
        - Uses a predefined prompt to guide the stance estimation process.
        - Incorporates linguistic and contextual analysis to determine the belief score.
    """
    topic = topic_info["topic"]
    topic_context = topic_info["topic_context"]
    prompt = f"""
        Eres un analista experto e imparcial de opiniones sobre {topic}.
        Tu tarea es medir la postura de un grupo de usuarios frente a este tema en un rango continuo entre 0 y 1, donde: 
        0 = Férreamente en contra, 1 = Reciamente a favor, 0.5 = Neutral, 
        0.25 = Oposición moderada (reconoce o no está enfáticamente indispuesto a reconocer aspectos positivos), 
        0.75 = Apoyo moderado (reconoce o está predispuesto a reconocer objeciones).
        Contexto: \"{topic_context}\"
        Entrada: Un JSON donde cada clave es el nombre de un usuario y su valor es una lista de opiniones obtenidas de redes sociales sobre este tema.
        Instrucciones: 
        1. Para cada usuario, analiza la postura general en todas sus opiniones.
        2. Calcula la proporción de frases a favor vs. en contra del tema.
        3. Emociones explícitas directas hacia el tema refuerzan la postura (más emoción positiva = postura cercana a 1; negativa = cercana a 0).
        4. Pondera adjetivos y adverbios utilizados para expresar alguna postura frente al tema.
        5. Conectores de oposición indican postura moderada.
         Ej: 'aunque no es perfecta, la reforma pensional es un paso positivo' o 'a pesar de las críticas, tiene aspectos rescatables'.
        6. Los adverbios de cantidad indican posturas extremas, y los de duda, posturas moderadas.
        Y cualquier otro criterio que consideres relevante para estimar la postura de cada usuario frente a {topic} en el rango [0,1]
        Salida esperada: Un JSON donde cada clave es el nombre de un usuario a analizar sus opiniones y su correspondiente valor es un número de dos decimales, 
        como 0.21, 0.75 o 0.03, representando la postura general del usuario frente a {topic}.
        O si las opiniones de un usuario no son suficientes para determinar una postura con certeza, devuelve 'None'.
        No retornes nada más.
        """

    def stanceEmit(event: str, val: Dict[str, int]) -> None:
        socketio.emit(event, val, to=sid)

    stances = calculate_stance(users_tweet_text, users, prompt, stanceEmit, testing=True)
    socketio.emit("belief_heuristic", stances, to=sid)
    socketio.sleep(0.01)
    return stances


def calculate_confidences(stances: Dict[str, float | None], sid: str) -> None:
    """
    Calculates and emits the confidence estimation for users based on their stance values.

    Params:
        stances (Dict[str, float | None]): A dictionary mapping each user to their estimated stance value.
        sid (str): The session ID to which the confidence estimation will be emitted.

    Returns:
        None
    """
    confidences = estimate_confidence(stances)
    socketio.emit("confidence_heuristic", confidences, to=sid)


def calculate_heuristics(users: List[str], df: pd.DataFrame, sid: str, topic_info: Dict[str, str]) -> None:
    """
    Calculates and emits estimates for stance, confidence, and multiple influence heuristics.

    This function runs as a background task to avoid blocking the main thread handling requests.

    Params:
        users (List[str]): List of user identifiers.
        df (pd.DataFrame): DataFrame containing interaction data.
        sid (str): Session ID to track progress and emit results.

    Returns:
        None

    Raises:
        Exception: If any of the calculations fail.
    """

    user_to_index: Dict[str, int] = {user: idx for idx, user in enumerate(users)}
    index_to_user: Dict[int, str] = {index: user for user, index in user_to_index.items()}

    interactions_matrix, get_links_matrix, interactions_matrix_nonNorm = build_influence_networks(df,
                                                                                                  user_to_index,
                                                                                                  index_to_user,
                                                                                                  sid)

    users_tweet_text = build_users_tweet_text(df, user_to_index)
    stances = calculate_beliefs(users_tweet_text, users, sid, topic_info)

    calculate_confidences(stances, sid)

    build_influence_networks_with_stances(stances, index_to_user, users_tweet_text, interactions_matrix,
                                          get_links_matrix, interactions_matrix_nonNorm, sid)


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
        4. Start the heuristics calculation
    """
    try:
        # Decode CSV from base64
        csv_data = message.get('csv_data')
        topic_info = message.get('topic_info')
        if not csv_data:
            emit("preprocess_error", "CSV data is missing", broadcast=False)
            return

        csv_data = base64.b64decode(csv_data).decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))
        df = preprocess_dataframe(df)

        users = identify_nodes(df)
        emit("users", list(users), broadcast=False)

        sid = request.sid
        socketio.start_background_task(calculate_heuristics, users, df, sid, topic_info)
    except ValueError as e:
        print(e)
        emit("preprocess_error", f"Error processing CSV: {str(e)}", broadcast=False)


if __name__ == '__main__':
    socketio.run(app, debug=True)
