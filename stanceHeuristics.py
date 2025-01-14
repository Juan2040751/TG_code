import concurrent.futures
import json
import os
import re
import threading
import time
from typing import Dict, List, Set

from dotenv import dotenv_values
from numpy import ndarray
from openai import OpenAI, OpenAIError, APIConnectionError
from pydantic import BaseModel

openIAKey = dotenv_values(".env")["OPENAI_API_KEY"]
client = OpenAI(api_key=openIAKey)


class Stance(BaseModel):
    value: float | None


"""
def stanceDetection(user_opinions: Set[str], prompt: str, user: str) -> (str, float | None):
    if not user_opinions:
        return user, None

    max_retries = 10  # Máximo número de intentos
    initial_delay = 0.5  # Tiempo inicial de espera en segundos

    for attempt in range(max_retries):
        try:

            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(user_opinions)}
                ],
                response_format=Stance,
            )
            # La salida esperada es un JSON con el formato {usuario: postura}
            response = completion.choices[0].message.content
            print(f"processed {user=} by {threading.current_thread().name}, {response}")
            time.sleep(initial_delay)
            return user,literal_eval(response)["value"]
        except OpenAIError as e:
            if hasattr(e, "code") and e.code == "rate_limit_exceeded":
                # Extraer tiempo de espera desde el mensaje del error
                wait_time_match = re.search(r"Please try again in (\d+\.?\d*)(ms|s)", str(e))
                if wait_time_match:
                    wait_value = float(wait_time_match.group(1))
                    unit = wait_time_match.group(2)
                    wait_time = wait_value / 1000 if unit == "ms" else wait_value
                    print(f"Rate limit reached. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Rate limit reached. Retrying with default wait time of 1 second...")
                    time.sleep(1)
            else:
                print(f"OpenAIError: {e}")
                break
        except APIConnectionError as e:
            print(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(initial_delay * (2 ** attempt))  # Espera exponencial
        except SyntaxError as e:
            print(f"SyntaxError: ", e)
            print(user_opinions, response)
        except ValueError:
            return user, None

    return user, None


def calculate_stance(
        users_tweet_text: ndarray[Set[str]],
        users: List[str],
        prompt: str,
        output_file: str = "results_preliminares.txt"
) -> Dict[str, float | None]:
    a = time.time()

    stances = {}

    # Abrir el archivo para escribir resultados preliminares
    with  open(output_file, "w") as file, concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        batch_futures = []
        for user, opinions in zip(users, users_tweet_text):
            batch_futures.append(
                executor.submit(stanceDetection, opinions, prompt, user)
            )

        for future in concurrent.futures.as_completed(batch_futures):
            b=time.time()
            user, stance = future.result()
            stances[user] = stance
            print(f"{user}: {stance}, {b-a:.2f}s")
            file.write(f"{user}: {stance}\n")
            file.flush()

    return stances
"""


def stanceDetectionBatch(user_opinions: Dict[str, List[str]], prompt: str) -> Dict[str, float | None]:
    if not user_opinions:
        return {}

    max_retries = 10  # Máximo número de intentos
    initial_delay = 0.5  # Tiempo inicial de espera en segundos

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(user_opinions)}
                ],

                response_format={"type": "json_object"}
            )
            # La salida esperada es un JSON con el formato {usuario: postura}
            response: str = completion.choices[0].message.content
            response = response.replace("None", "null")
            response = response.replace("json", "")
            response = response.replace("```", "")
            response = response.replace("\n", "")
            response = response.replace("  ", "")
            response: Dict[str, float | None] = json.loads(response)
            if all([response.get(user, None) is None for user in user_opinions]):
                raise AssertionError(response)
            response = {user: response.get(user, None) for user in user_opinions}

            # print("X:", response)
            time.sleep(initial_delay)
            return response
        except OpenAIError as e:
            if hasattr(e, "code") and e.code == "rate_limit_exceeded":
                # Extraer tiempo de espera desde el mensaje del error
                wait_time_match = re.search(r"Please try again in (\d+\.?\d*)(ms|s)", str(e))
                if wait_time_match:
                    wait_value = float(wait_time_match.group(1))
                    unit = wait_time_match.group(2)
                    wait_time = wait_value / 1000 if unit == "ms" else wait_value
                    print(f"Rate limit reached. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Rate limit reached. Retrying with default wait time of 1 second...")
                    time.sleep(1)
            else:
                print(f"OpenAIError: {e}")
                break
        except APIConnectionError as e:
            print(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(initial_delay * (2 ** attempt))  # Espera exponencial
        except SyntaxError as e:
            print(f"SyntaxError: ", e)
            print(user_opinions, response)
        except ValueError as e:
            print(f"ValueError: {e}, {response}")

    return {user: None for user in user_opinions}


def calculate_stance(
        users_tweet_text: ndarray[Set[str]],
        users: List[str],
        prompt: str,
        batch_size: int = 3,
        testing: bool = True
) -> Dict[str, float | None]:
    a = time.time()
    output_file: str = "testing_result.json"

    if testing and os.path.exists(output_file):
        with open(output_file, "r") as f:
            # opinions = users_tweet_text[users.index("petrogustavo")]
            # print(opinions)
            # print(stanceDetectionBatch({"petrogustavo": opinions}, prompt))
            return json.load(f)

    def process_batch(start_index: int, end_index: int) -> Dict[str, float | None]:
        batch_users = filtered_users[start_index:end_index]
        temp_batch_opinions = {
            user: str(filtered_tweet_text[i]) for i, user in enumerate(batch_users, start=start_index)
        }
        batch_opinions: Dict[str, str] = {}
        answer: Dict[str, float | None] = {}
        for user, opinions in temp_batch_opinions.items():
            if len(opinions) > 3000:
                answer.update(stanceDetectionBatch({user: opinions}, prompt))
            else:
                batch_opinions[user] = opinions
        answer.update(stanceDetectionBatch(batch_opinions, prompt))
        b = time.time()
        print(
            f"processed users {start_index}-{end_index} de {len(filtered_users)}, {b - a:.2f} seconds, {answer} by {threading.current_thread().name}")

        return answer

    stances = {}

    # Filtrar usuarios que tienen opiniones no vacías
    filtered_users = []
    filtered_tweet_text = []
    for user, tweet_set in zip(users, users_tweet_text):
        if len(tweet_set) != 0:  # Si el conjunto de opiniones no está vacío
            filtered_users.append(user)
            filtered_tweet_text.append(tweet_set)
        else:
            stances[user] = None

    # Abrir el archivo para escribir resultados preliminares
    with  concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        batch_futures = []
        for i in range(0, len(filtered_users), batch_size):
            batch_futures.append(
                executor.submit(process_batch, i, i + batch_size)
            )

        for future in concurrent.futures.as_completed(batch_futures):
            batch_result = future.result()
            stances.update(batch_result)

            # Escribir resultados preliminares en el archivo
    if testing:
        with open(output_file, "w") as f:
            json.dump(stances, f, indent=4)

    return stances
