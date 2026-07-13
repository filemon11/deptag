import numpy as np
import torch.nn.functional as F
from ufal.chu_liu_edmonds import chu_liu_edmonds  # type: ignore


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1)[..., np.newaxis])
    return e_x / e_x.sum(axis=-1)[..., np.newaxis]


def mst_single(
        score_matrix: np.ndarray
        ) -> np.ndarray:
    """Computes a maximum spanning tree for a (non)-stochastic
    adjacency matrix containing transition probabilities.
    Expects probabilities between 0 and 1. A row may
    sum to more than 1.

    Parameters
    ----------
    score_matrix : np.ndarray
        Stochastic matrix

    Returns
    -------
    np.ndarray
        Identified head positions of the nodes.
    """
    # TODO: x vs y axis unclear

    # Convert to logspace (addition corresponds to
    # multiplication of probabilities)
    with np.errstate(divide='ignore'):
        # 0-probability entries may throw a division error
        # which we ignore
        score_matrix = np.log(score_matrix)
    assert isinstance(score_matrix, np.ndarray)
    heads, _ = chu_liu_edmonds(score_matrix.astype(np.double))
    return np.array(heads)


def mst(
        score_matrix: np.ndarray,
        ignore_deprels: np.ndarray,
        ) -> np.ndarray:
    """Computes a maximum spanning tree for a batch of
    (non)-stochastic adjacency matrices containing
    transition probabilities.
    Expects probabilities between 0 and 1. A row may
    sum to more than 1.

    Parameters
    ----------
    score_matrix : np.ndarray
        batch of stochastic matrices

    Returns
    -------
    np.ndarray
        batch of identified head positions of the nodes.
    """
    # TODO: x vs y axis unclear

    # print(score_matrix.shape)

    prefix = np.zeros((1, score_matrix.shape[-1]))
    prefix[0, 0] = 1

    def process(
            prefix: np.ndarray, ma: np.ndarray, ig: np.ndarray, pad_len: int
            ) -> np.ndarray:
        ignore = ig == -1
        # temp = ma.copy()
        # temp[ignore][:, ignore] = float("-inf")
        # temp = temp.argmax(-1)
        # temp[ignore] = -1

        ma = ma[~ignore][:, np.logical_or(
            ~ignore, prefix[0][:ma.shape[-1]])]
        prefix = prefix[:, :ma.shape[-1]]

        inpt = np.concatenate([prefix, ma], axis=0)
        inpt = softmax(inpt)
        result = mst_single(inpt)
        # shape: [unpadded_len]
        # The indices refer to the head positions without ignored tokens.
        # Now: add cumulative sum of ignore occurrences to the respective
        # references

        # print("heads_long1", heads_long)
        cumsum = np.cumsum(ignore) - 1  # [0, (0, 0), 1, (1,)]
        # print("cumsum", cumsum)
        head_cumsum = cumsum[~ignore]
        # [1, (1, 1), 2, (2,)][False, True, True, False, True] -1 = [0, 0, 1]
        head_cumsum = head_cumsum[result[1:]-1]
        head_cumsum[result[1:] == 0] = 0
        # [h_c[x], h_c[y], h_c[z]]

        # print("result", result)
        # print("h_cums", head_cumsum)
        result[1:] += head_cumsum

        # heads = np.full((pad_len,), -1)
        # heads[ig != -1] = result[1:]
        # ignore_sum = np.cumsum(ignore)-1
        # return heads + ignore_sum
        heads = np.full((pad_len,), -1)
        heads[~ignore] = result[1:]
        # print("mst   ", heads)
        # print("argmax", temp)
        # print("gold  ", ig)
        return heads

    stack = [
        process(prefix, ma, ig, score_matrix.shape[1])
        for ma, ig in zip(score_matrix, ignore_deprels)]
    # for i, j in zip(stack, ignore_deprels):
    #    print("compare", i, j)
    return np.stack(stack)
