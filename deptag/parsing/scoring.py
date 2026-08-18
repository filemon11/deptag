import numpy as np


def las(predicted_heads: np.ndarray, predicted_deprels: np.ndarray,
        gold_heads: np.ndarray, gold_deprels: np.ndarray,
        gold_pos_tags: np.ndarray | None = None,
        ignore_pos_tag_id: int | None = None,
        id2deprel=None) -> float:
    if ignore_pos_tag_id is not None:
        assert gold_pos_tags is not None
        gold_heads = gold_heads.copy()
        gold_heads[gold_pos_tags == ignore_pos_tag_id] = -1

    predicted_heads = predicted_heads[gold_heads != -1]
    # if predicted_head_matrix:
    #     predicted_heads.argmax(-1)

    # if predicted_deprels_matrix:
    #     # shape: [batch, S, S, ]
    #     # TODO: Select highscore deprels
    #     hds = predicted_heads[:, np.newaxis, np.newaxis, :]
    #     # [batch, 1, 1, sent_len]

    #     hds = np.tile(hds, (1, predicted_deprels.shape[-1], 1, 1))
    #     # [batch, n_labels, 1, sent_len]

    #     deprel_logits = np.take(predicted_deprels, hds, 2).squeeze(2)
    #     # [batch, n_labels, sent_len]

    #     # TODO: this makes no sense

    #     predicted_deprels = deprel_logits.transpose(-1, -2).argmax(-1)
    #     # [batch, sent_len]

    predicted_deprels = predicted_deprels[gold_heads != -1]
    gold_deprels = gold_deprels[gold_heads != -1]

    gold_heads = gold_heads[gold_heads != -1]

    # TODO
    # correct_head = predicted_heads == gold_heads
    # print(
    #     "dependency-label accuracy only for arcs "
    #     "whose predicted head is correct",
    #     (
    #         predicted_deprels[correct_head]
    #         == gold_deprels[correct_head]
    #     ).mean()
    # )

    # for gold, pred in zip(
    #         gold_deprels[correct_head],
    #         predicted_deprels[correct_head],
    # ):
    #     print(
    #         id2deprel[gold],
    #         "->",
    #         id2deprel[pred],
    #     )

    return np.logical_and(
        predicted_heads == gold_heads,
        predicted_deprels == gold_deprels).mean().item()


def uas(
        predicted_heads: np.ndarray, gold_heads: np.ndarray,
        gold_pos_tags: np.ndarray | None = None,
        ignore_pos_tag_id: int | None = None) -> float:
    if ignore_pos_tag_id is not None:
        assert gold_pos_tags is not None
        gold_heads = gold_heads.copy()
        gold_heads[gold_pos_tags == ignore_pos_tag_id] = -1

    predicted_heads = predicted_heads[gold_heads != -1]
    # if predicted_head_matrix:
    #     predicted_heads = predicted_heads.argmax(-1)

    gold_heads = gold_heads[gold_heads != -1]

    return (predicted_heads == gold_heads).mean().item()
