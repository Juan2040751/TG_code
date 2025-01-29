import json
import os
import re
import time
from functools import partial
from multiprocessing import Pool, Manager, Process
from typing import Dict, List, Set, Tuple

import numpy as np
from dotenv import dotenv_values
from numpy import ndarray
from openai import OpenAI, OpenAIError, APIConnectionError

openIAKey = dotenv_values(".env")["OPENAI_API_KEY"]
client = OpenAI(api_key=openIAKey)





def stanceDetection(users_batch: Dict[str, Set[str]], prompt: str, a: time) -> Dict[str, float | None]:
    max_retries = 10
    initial_delay = 0.5
    pid = os.getpid()
    print(f"working on {len(users_batch)} users {pid=}")
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(users_batch)}
                ],
                response_format={"type": "json_object"}
            )
            response: str = completion.choices[0].message.content
            b = time.time()

            answer = {user: json.loads(response).get(user, None) for user in users_batch.keys()}
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

    return {user: None for user in users_batch.keys()}


def split_batches(
        users_with_opinions: ndarray[Tuple[str, Set[str]]],
        prompt: str,
        max_tokens: int = 22500
) -> List[Dict[str, Set[str]]]:
    """
    Dynamically split user data into batches that fit within the max_tokens limit.
    """
    batches = []
    current_batch = {}

    def estimate_tokens(prompt: str, user_batch: Dict[str, Set[str]]) -> int:
        """
        Estimates the number of tokens for a given prompt and user batch.
        Estimates the number of tokens for a given prompt and user batch.
        """
        user_batch_str = json.dumps(user_batch, ensure_ascii=False)
        total_content = f"{prompt}\n{user_batch_str}"
        return len(total_content.split())

    current_tokens = estimate_tokens(prompt, current_batch)

    for user, opinions in users_with_opinions:
        user_data = {user: list(opinions)}
        additional_tokens = estimate_tokens(prompt, user_data)

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


def users_with_unique_opinions(users_with_opinions: ndarray
                               ) -> Tuple[ndarray, Dict[str, List[str]]]:
    """
    Groups users by unique opinions and tracks users sharing the same opinions.

    Parameters:
        users_with_opinions (list[tuple[str, set[str]]]): A list of tuples where each tuple contains
                                                          a user (str) and their opinions (set[str]).

    Returns:
        tuple:
            - dict[str, set[str]]: A dictionary where each key is a user representing unique opinions
                                   and the value is the corresponding set of unique opinions.
            - dict[str, list[str]]: A dictionary where each key is a user representing a unique opinion
                                    group, and the value is a list of users sharing the same opinions.
    """
    unique_opinions = {}
    users_with_same_opinions = {}

    for user, opinions in users_with_opinions:
        if opinions not in unique_opinions:
            unique_opinions[opinions] = user
        else:
            # Retrieve the representative user for these opinions
            representative_user = unique_opinions[opinions]
            users_with_same_opinions.setdefault(representative_user, []).append(user)

    users_with_unique_opinions = np.array([(user, opinions) for opinions, user in unique_opinions.items()])
    return users_with_unique_opinions, users_with_same_opinions


def calculate_stance(
        users_tweet_text: ndarray[Set[str]],
        users: List[str],
        prompt: str,
        stanceEmit,
        output_file: str = "testing_result.json",
        testing: bool = True
) -> Dict[str, float | None]:
    if testing and os.path.exists(output_file):
        with open(output_file, "r") as f:
            return json.load(f)

    stances = {user: None for user, opinions in zip(users, users_tweet_text) if not opinions}

    users_with_opinions = np.array(
        [(user, frozenset(opinions)) for user, opinions in zip(users, users_tweet_text) if opinions])
    users_with_opinions, users_with_same_opinions = users_with_unique_opinions(users_with_opinions)
    batches = split_batches(users_with_opinions, prompt)


    stanceEmit("stance_time", {"n_users": len(users), "null_stances": len(stances), "estimated_time": 30*len(batches)//8, "n_batch": len(batches) })


    print("op4:", len(batches))
    a = time.time()
    with Pool(8) as p:
        stance_batches = p.map(partial(stanceDetection, prompt=prompt, a=a,), batches)



    for stance_batch in stance_batches:
        batch_copy = stance_batch.copy()
        for user, stance in batch_copy.items():
            if user in users_with_same_opinions:
                for same_user in users_with_same_opinions[user]:
                    stance_batch[same_user] = stance
        stances.update(stance_batch)

    with open(output_file, "w") as f:
        json.dump(stances, f, indent=4)
    print(f"{len(stances)}, {len(users)}")
    return stances
