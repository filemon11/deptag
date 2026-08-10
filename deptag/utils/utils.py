import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1)[..., np.newaxis])
    return e_x / e_x.sum(axis=-1)[..., np.newaxis]


def neg_log10_softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=-1, keepdims=True)
    shifted = x - x_max

    logsumexp = (
        x_max
        + np.log(
            np.exp(shifted).sum(
                axis=-1,
                keepdims=True,
            )
        )
    )

    return (logsumexp - x) / np.log(10.0)
