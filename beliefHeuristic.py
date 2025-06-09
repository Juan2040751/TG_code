import json
import os
import re
import time
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple, Callable

import numpy as np
from dotenv import dotenv_values
from numpy import ndarray
from openai import OpenAI, OpenAIError, APIConnectionError

openIAKey = dotenv_values(".env")["OPENAI_API_KEY"]
client = OpenAI(api_key=openIAKey)


def stance_detection(users_batch: Dict[str, Set[str]], prompt: str, a: time) -> Dict[str, float | None]:
    """
    Detects stance for a batch of users based on their textual content.

    Params:
        users_batch (Dict[str, Set[str]]): Dictionary mapping user IDs to their set of opinions.
        prompt (str): Instructional prompt for stance estimation.
        a (time): Start time for performance measurement.

    Returns:
        Dict[str, float | None]: Dictionary mapping each user to a stance score in the range [0,1].
                                 Returns None for users where stance could not be determined.

    Notes:
        - Implements retry logic for handling API rate limits and connection errors.
        - Parses JSON response and ensures robust error handling.
    """
    max_retries = 5
    initial_delay = 0.5
    pid = os.getpid()
    print(f"working on {len(users_batch)} users {pid=}")
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(users_batch)}
                ],
                response_format={"type": "json_object"}
            )
            response = completion.choices[0].message.content
            response = json.loads(response)
            b = time.time()

            answer = {user: float(response[user]) if response.get(user, None) is not None and response[user] !="None" else None for user in users_batch}
            print(f"{len(answer)} users processed, {pid=} ({b - a:.1f}s)")
            return answer

        except OpenAIError as e:
            if hasattr(e, "code") and e.code == "rate_limit_exceeded":
                wait_time_match = re.search(r"Please try again in (\d+\.?\d*)(ms|s)", str(e))
                wait_time = float(wait_time_match.group(1)) / 1000 if wait_time_match and wait_time_match.group(
                    2) == "ms" else 1
                print(f"\rRate limit reached. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"OpenAIError: {e}")
                break
        except APIConnectionError as e:
            print(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}", "\r")
            time.sleep(initial_delay * (2 ** attempt))
        except (SyntaxError, ValueError) as e:
            print(f"Parsing error: {e}")
            break
        except Exception as e:
            print(e)

    return {user: None for user in users_batch.keys()}


def split_batches(users_with_opinions: ndarray[Tuple[str, Set[str]]], prompt: str, max_tokens: int = 60000) -> List[
    Dict[str, Set[str]]]:
    """
    Splits user opinion data into manageable batches to fit within token constraints.

    Params:
        users_with_opinions (ndarray[Tuple[str, Set[str]]]): Array of user-opinion tuples.
        prompt (str): Instructional prompt for stance estimation.
        max_tokens (int, optional): Maximum allowed tokens per batch. Defaults to 30000.

    Returns:
        List[Dict[str, Set[str]]]: List of batches, each being a dictionary mapping users to opinions.
    """
    batches = []
    current_batch = {}

    def estimate_tokens(user_batch: Dict[str, Set[str]]) -> int:
        """
        Estimates the number of tokens for a given prompt and user batch.
        """
        user_batch_str = json.dumps(user_batch, ensure_ascii=False)
        total_content = f"{prompt}\n{user_batch_str}"
        return len(total_content.split())

    current_tokens = estimate_tokens(current_batch)

    for user, opinions in users_with_opinions:
        user_data = {user: list(opinions)}
        additional_tokens = estimate_tokens(user_data)

        if current_tokens + additional_tokens > max_tokens:
            batches.append(current_batch)
            current_batch = user_data
            current_tokens = additional_tokens
        else:
            current_batch.update(user_data)
            current_tokens += additional_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def users_with_unique_opinions(users_with_opinions: ndarray) -> Tuple[ndarray, Dict[str, List[str]]]:
    """
    Identifies unique opinion groups and tracks users sharing the same opinions.

    Params:
        users_with_opinions (ndarray): Array of tuples containing user IDs and their set of opinions.

    Returns:
        Tuple:
            - ndarray: Array of unique opinion representatives and their opinions.
            - Dict[str, List[str]]: Mapping of representative users to a list of users sharing the same opinions.
    """
    unique_opinions = {}
    users_with_same_opinions = {}

    for user, opinions in users_with_opinions:
        if opinions not in unique_opinions:
            unique_opinions[opinions] = user
        else:
            representative_user = unique_opinions[opinions]
            users_with_same_opinions.setdefault(representative_user, []).append(user)

    users_with_unique_opinions_ = np.array([(user, opinions) for opinions, user in unique_opinions.items()])
    return users_with_unique_opinions_, users_with_same_opinions


def calculate_stance(users_tweet_text: ndarray[Set[str]], users: List[str], prompt: str,
                     stanceEmit: Callable[[str, Dict[str, int]], None],
                     output_file: str = "testing_result.json", testing: bool = False) -> Dict[str, float | None]:
    """
    Computes and emits stance estimation for users based on their textual content.

    Params:
        users_tweet_text (ndarray[Set[str]]): Array containing sets of tweets per user.
        users (List[str]): List of user IDs.
        prompt (str): Instructional prompt for stance estimation.
        stanceEmit (Callable): Function to emit stance-related events.
        output_file (str, optional): File path to store cached results. Defaults to "porteDeArmas_result.json".
        testing (bool, optional): If True, loads cached results when available. Defaults to True.

    Returns:
        Dict[str, float | None]: Dictionary mapping each user to a stance score in the range [0,1].
                                 Returns None for users where stance could not be determined.

    Emits:
        - "stance_time" with batch processing statistics.
    """
    if testing and os.path.exists(output_file):
        with open(output_file, "r") as f:
            return json.load(f)

    stances = {user: None for user, opinions in zip(users, users_tweet_text) if not opinions}

    users_with_opinions = np.array(
        [(user, frozenset(opinions)) for user, opinions in zip(users, users_tweet_text) if opinions])
    users_with_opinions, users_with_same_opinions = users_with_unique_opinions(users_with_opinions)
    batches = split_batches(users_with_opinions, prompt)

    stanceEmit("stance_time",
               {"n_users": len(users), "null_stances": len(stances), "estimated_time": 50 * len(batches) // 8,
                "n_batch": len(batches)})

    print("op4:", len(batches))
    a = time.time()
    with Pool(8) as p:
        stance_batches = p.map(partial(stance_detection, prompt=prompt, a=a), batches)

    for stance_batch in stance_batches:
        batch_copy = stance_batch.copy()
        for user, stance in batch_copy.items():
            if user in users_with_same_opinions:
                for same_user in users_with_same_opinions[user]:
                    stance_batch[same_user] = stance
        stances.update(stance_batch)

    if testing:
        with open(output_file, "w") as f:
            json.dump(stances, f, indent=4)
    print(f"{len(stances)}, {len(users)}")
    return stances
