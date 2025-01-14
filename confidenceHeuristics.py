import numpy as np
from numpy import ndarray


def calculate_extremism_confidence(stances: ndarray):
    """
    Calculates confidence based on how extreme the stance is.
    Extreme stances (close to 0 or 1) are assigned higher confidence.
    """
    stance_confidence = lambda stance: -1 if stance < 0 or stance > 1 else abs(stance - 0.5) / 0.5
    return np.vectorize(stance_confidence)(stances)


def calculate_deviation_confidence(stances: ndarray[float | None]) -> dict[str, float]:
    """
    Calculates confidence based on deviations from the mean stance.
    """
    mean_stance = np.mean(stances)
    deviations = np.abs(stances - mean_stance)
    max_deviation = np.max(deviations)
    if max_deviation == 0:  # If all stances are identical
        return {username: None for username in stances}

    normalized_confidences = deviations / max_deviation
    return normalized_confidences


def calculate_combined_confidence(users_stances: dict[str, float], weight_extremism: float = 0.4) -> dict[str, float]:
    """
    Combines confidence based on deviations from the mean and extremism.

    :param users_stances: Dictionary where keys are usernames and values are their stances (float).
    :param weight_extremism: Weight for the extremism-based confidence. Default is 0.5.
    :return: Dictionary where keys are usernames and values are combined confidence scores (float between 0 and 1).
    """



    # Calculate extremism-based confidence
    stances = np.array(list(users_stances.values()))
    extremism_confidence = calculate_extremism_confidence(stances)
    deviation_confidence = calculate_deviation_confidence(stances)

    combined_confidence = weight_extremism * extremism_confidence + (1 - weight_extremism) * deviation_confidence
    users_confidence = dict(zip(users_stances.keys(), combined_confidence))

    return users_confidence

