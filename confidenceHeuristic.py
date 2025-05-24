import numpy as np
from numpy import ndarray


def calculate_extremism_confidence(stances: ndarray[float | None]) -> ndarray[float | None]:
    """
    Computes confidence scores based on the extremism of stance values.

    Params:
        stances (ndarray): An array of stance values ranging from 0 to 1.

    Returns:
        ndarray: An array of confidence scores, where extreme stances (close to 0 or 1)
                 have higher confidence.
    """
    stance_confidence = lambda stance: np.nan if stance is None or (0 > stance > 1) else abs(stance - 0.5) / 0.5
    return np.vectorize(stance_confidence)(stances)


def calculate_deviation_confidence(stances: ndarray[float | None]) -> ndarray[float | None]:
    """
    Computes confidence scores based on deviation from the mean stance.

    Params:
        stances (ndarray[float | None]): An array of stance values, where None represents missing values.

    Returns:
        ndarray: An array of confidence scores, where stances deviating more from the mean have higher confidence.
    """
    stances = np.array(stances, dtype=float)
    stances[np.isnan(stances)] = np.nan

    mean_stance = np.nanmean(stances)
    deviations = np.abs(stances - mean_stance)
    max_deviation = np.nanmax(deviations)

    if max_deviation == 0 or np.isnan(max_deviation):  # If all stances are identical
        return np.full_like(stances, np.nan)

    return deviations / max_deviation


def estimate_confidence(users_stances: dict[str, float], weight_extremism: float = 0.4) -> dict[str, float | None]:
    """
    Estimates users' confidence in expressing their opinion based on extremism and deviation.

    Params:
        users_stances (dict[str, float]): Dictionary mapping users to their stance values (float in range 0-1).
        Weight_extremism (float): Weight assigned to the extremism factor in the confidence calculation. Default is 0.4.

    Returns:
        dict[str, float]: Dictionary mapping users to their estimated confidence scores (float between 0 and 1).
                          Returns None for users where opinion could not be determined.
    """
    stances = np.array(list(users_stances.values()))
    extremism_confidence = calculate_extremism_confidence(stances)
    deviation_confidence = calculate_deviation_confidence(stances)

    combined_confidence = weight_extremism * extremism_confidence + (1 - weight_extremism) * deviation_confidence
    combined_confidence = np.round(combined_confidence, 3)
    combined_confidence= np.clip(combined_confidence, 0, 1)
    combined_confidence = np.where(np.isnan(combined_confidence), None, combined_confidence)


    return dict(zip(users_stances.keys(), combined_confidence))
