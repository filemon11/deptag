import numpy as np
import torch
import logging
import tqdm

from . import model, factorisation
from .. import parsing, utils, extraction

from collections import defaultdict

from typing import (
    overload, Literal, DefaultDict, Sequence, Mapping,
)


def deprel_func(
        deprels_from_pred_head: bool,
        deprels_matrix: bool,
        pred_deprels: torch.Tensor | None,
        gold_deprels: torch.Tensor,
        pred_heads: torch.Tensor | None,
        gold_heads: torch.Tensor | None,
        ) -> tuple[np.ndarray | None, np.ndarray]:

    deprel_predictions: np.ndarray | None = None
    if pred_deprels is not None:
        if deprels_from_pred_head:
            assert not deprels_matrix
            assert pred_heads is not None, (
                "Cannot use deprels from predicted head "
                "without head prediction."
            )

            heads_for_deprel = pred_heads

        else:
            assert gold_heads is not None
            # Use gold heads.
            root_heads = torch.full(
                (gold_heads.shape[0], 1),
                -1,
                dtype=gold_heads.dtype,
                device=gold_heads.device,
            )

            heads_for_deprel = torch.cat(
                [root_heads, gold_heads],
                dim=1,
            )

        if deprels_matrix:
            # outputs[7]: [B, L, H, D]
            # -> [B, D, H, L]
            deprel_predictions = (
                pred_deprels
                .permute(0, 3, 2, 1)
                .float()
                .cpu()
                .numpy()
            )

        else:
            # Replace -1 by 0 solely to make gather legal.
            safe_heads = heads_for_deprel.clamp_min(0)
            # [B, D]

            hds = safe_heads[:, None, None, :]
            # [B, 1, 1, D]

            hds = hds.expand(
                -1,
                pred_deprels.size(1),
                -1,
                -1,
            )
            # [B, L, 1, D]

            deprel_logits = torch.gather(
                pred_deprels,
                dim=2,
                index=hds,
            ).squeeze(2)
            # [B, L, D]

            deprel_logits = deprel_logits.transpose(-1, -2)
            # [B, D, L]

            # max_parse_len = max(
            #     max_parse_len,
            #     deprel_logits.shape[1],
            # )

            deprel_predictions = (
                deprel_logits.float().cpu().numpy()
            )

    root_deprel = torch.full(
        (gold_deprels.shape[0], 1),
        -1,
        dtype=gold_deprels.dtype,
        device=gold_deprels.device,
    )

    deprel_labels = torch.cat(
        [root_deprel, gold_deprels],
        dim=1,
    )

    return deprel_predictions, deprel_labels.int().cpu().numpy()


def pad_deprel_pred(
        deprel_predictions: Sequence[np.ndarray],
        max_parse_len: int, deprels_matrix: bool) -> np.ndarray:
    if deprels_matrix:
        return np.concatenate([
            np.pad(
                deprel_logits,
                (
                    (0, 0),
                    (0, max_parse_len - deprel_logits.shape[1]),
                    (0, max_parse_len - deprel_logits.shape[2]),
                    (0, 0)),
                'constant', constant_values=-np.inf)
            for deprel_logits in deprel_predictions], axis=0)
    else:
        return np.concatenate([
            np.pad(
                deprel_logits,
                (
                    (0, 0),
                    (0, max_parse_len - deprel_logits.shape[1]),
                    (0, 0)),
                'constant', constant_values=0)
            for deprel_logits in deprel_predictions], axis=0)


def pad_deprel_gold(
        eval_deprel_labels: Sequence[np.ndarray],
        max_parse_len: int) -> np.ndarray:
    return np.concatenate([
        np.pad(
            deprel_labels,
            ((0, 0), (0, max_parse_len - deprel_labels.shape[1])),
            'constant', constant_values=-1)
        for deprel_labels in eval_deprel_labels], axis=0)


@overload
def predict(
        tagging_model: model.ModelForTagging, eval_dataloader, dataset_size,
        num_tags, batch_size, device,
        report_loss: Literal[True], deprels_from_pred_head: bool = False,
        deprels_matrix: bool = False
        ) -> tuple[
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            dict[str, np.ndarray], dict[str, np.ndarray],
            np.ndarray | None, np.ndarray | None,
            dict[str, np.ndarray], dict[str, np.ndarray],
            dict[str, np.ndarray], dict[str, np.ndarray],
            np.ndarray,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            dict[str, np.ndarray],
            np.ndarray | None,
            dict[str, np.ndarray],
            dict[str, np.ndarray],]:
    ...


@overload
def predict(
        tagging_model: model.ModelForTagging, eval_dataloader, dataset_size,
        num_tags, batch_size, device,
        report_loss: Literal[False] = False,
        deprels_from_pred_head: bool = False, deprels_matrix: bool = False
        ) -> tuple[
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            dict[str, np.ndarray], dict[str, np.ndarray],
            np.ndarray | None, np.ndarray | None,
            dict[str, np.ndarray], dict[str, np.ndarray],
            dict[str, np.ndarray], dict[str, np.ndarray],
            np.ndarray,
            None, None,
            None, None,
            dict[str, np.ndarray],
            None,
            dict[str, np.ndarray],
            dict[str, np.ndarray],]:
    ...


def predict(
        tagging_model: model.ModelForTagging, eval_dataloader,
        dataset_size, num_tags, batch_size, device,
        report_loss: bool = False, deprels_from_pred_head: bool = False,
        deprels_matrix: bool = False
        ) -> tuple[
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            dict[str, np.ndarray], dict[str, np.ndarray],
            np.ndarray | None, np.ndarray | None,
            dict[str, np.ndarray], dict[str, np.ndarray],
            dict[str, np.ndarray], dict[str, np.ndarray],
            np.ndarray,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            dict[str, np.ndarray],
            np.ndarray | None,
            dict[str, np.ndarray],
            dict[str, np.ndarray]]:

    tagging_model.eval()
    predictions = []
    eval_labels = []
    max_len = 0
    max_parse_len = 0
    idx = 0

    pos_predictions = []
    xpos_predictions = []
    eval_pos_labels = []
    eval_xpos_labels = []
    arc_predictions = []
    eval_heads = []
    deprel_predictions = []
    eval_deprel_labels = []
    factorised_predictions: DefaultDict[
        str, list[np.ndarray]] = defaultdict(list)
    feats_predictions: DefaultDict[
        str, list[np.ndarray]] = defaultdict(list)
    subtypes_predictions: DefaultDict[
        str, list[np.ndarray]] = defaultdict(list)
    eval_factorised_labels: DefaultDict[
        str, list[np.ndarray]] = defaultdict(list)
    eval_feats_labels: DefaultDict[
        str, list[np.ndarray]] = defaultdict(list)
    eval_subtypes_labels: DefaultDict[
        str, list[np.ndarray]] = defaultdict(list)

    sup_losses: list[np.ndarray] | np.ndarray = []
    pos_losses: list[np.ndarray] | np.ndarray = []
    xpos_losses: list[np.ndarray] | np.ndarray = []
    arc_losses: list[np.ndarray] | np.ndarray = []
    deprel_losses: list[np.ndarray] | np.ndarray = []
    factorised_losses: DefaultDict[str, list[np.ndarray]] = defaultdict(list)
    feats_losses: DefaultDict[str, list[np.ndarray]] = defaultdict(list)
    subtypes_losses: DefaultDict[str, list[np.ndarray]] = defaultdict(list)
    for batch in tqdm.tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad(), torch.amp.autocast(
                "cpu" if device == torch.device("cpu") else "cuda",
                enabled=True, dtype=torch.float16
                ):

            logits: model.TaggingLogits
            word_mask: torch.Tensor
            logits, word_mask = tagging_model(**batch)

            losses: model.TaggingLosses
            losses = tagging_model.calc_losses(
                logits, word_mask, **batch
            )

        idx += 1
        if logits["sup"] is not None:
            sup_logits = logits["sup"].float().cpu().numpy()
            max_len = max(max_len, sup_logits.shape[1])
            predictions.append(sup_logits)
        labels = batch['labels'].int().cpu().numpy()
        eval_labels.append(labels)

        if logits["pos"] is not None:
            pos_logits = logits["pos"].float().cpu().numpy()
            pos_predictions.append(pos_logits)
            max_len = max(max_len, pos_logits.shape[1])
        pos_labels = batch['pos_ids'].int().cpu().numpy()
        eval_pos_labels.append(pos_labels)

        pred_heads = None
        parse_mask = None

        if logits["S_arc"] is not None:
            # batch["heads"]: [B, W]
            # True for actual words, False for padding.
            word_mask = batch["heads"].ne(-1)

            # Parser sequence is ROOT + words.
            parse_mask = torch.cat(
                [
                    torch.ones(
                        (word_mask.shape[0], 1),
                        dtype=torch.bool,
                        device=word_mask.device,
                    ),
                    word_mask,
                ],
                dim=1,
            )
            # [B, W + 1]

            # outputs[5]: [B, H, D]
            arc_logits = logits["S_arc"].masked_fill(
                ~parse_mask.unsqueeze(-1),
                float("-inf"),
            )

            # Compute predicted head once, while scores are still tensors.
            pred_heads = arc_logits.argmax(dim=1)
            # [B, D]

            # ROOT and padding are not dependents.
            dependent_mask = parse_mask.clone()
            dependent_mask[:, 0] = False

            pred_heads = pred_heads.masked_fill(
                ~dependent_mask,
                -1,
            )
            # [B, D]

            # Store arc scores for later evaluation.
            arc_logits_np = (
                arc_logits
                .transpose(-1, -2)
                .float()
                .cpu()
                .numpy()
            )
            # [B, D, H]

            arc_predictions.append(arc_logits_np)

            max_parse_len = max(
                max_parse_len,
                arc_logits_np.shape[1],
            )

            # Gold heads: prepend ignored ROOT dependent.
            root_heads = torch.full(
                (batch["heads"].shape[0], 1),
                -1,
                dtype=batch["heads"].dtype,
                device=batch["heads"].device,
            )

            gold_heads = torch.cat(
                [root_heads, batch["heads"]],
                dim=1,
            )
            # [B, D]

            eval_heads.append(
                gold_heads.int().cpu().numpy()
            )

        s_preds, s_labels = deprel_func(
            deprels_from_pred_head,
            deprels_matrix,
            logits["S_lab"],
            batch["deprel_ids"],
            pred_heads,
            batch["heads"]
        )
        if s_preds is not None:
            deprel_predictions.append(s_preds)
        eval_deprel_labels.append(s_labels)

        # deprel_labels = batch['deprel_ids'].int().cpu().numpy()
        # eval_deprel_labels.append(deprel_labels)

        for f_name, f_logits in logits["factorised"].items():
            f_logits_ = f_logits.float().cpu().numpy()
            factorised_predictions[f_name].append(f_logits_)
            max_len = max(max_len, f_logits_.shape[1])
            f_labels = batch[f_name].int().cpu().numpy()
            eval_factorised_labels[f_name].append(f_labels)

        if logits["xpos"] is not None:
            xpos_logits = logits["xpos"].float().cpu().numpy()
            xpos_predictions.append(xpos_logits)
            max_len = max(max_len, xpos_logits.shape[1])
        xpos_labels = batch['xpos_ids'].int().cpu().numpy()
        eval_xpos_labels.append(xpos_labels)

        for f_name, f_logits in logits["feats"].items():
            f_logits_ = f_logits.float().cpu().numpy()
            feats_predictions[f_name].append(f_logits_)
            max_len = max(max_len, f_logits_.shape[1])
            f_labels = batch[f_name].int().cpu().numpy()
            eval_feats_labels[f_name].append(f_labels)

        for f_name, f_logits in logits["S_extra_lab"].items():

            s_preds, s_labels = deprel_func(
                deprels_from_pred_head,
                deprels_matrix,
                f_logits,
                batch[f_name],
                pred_heads,
                batch["heads"]
            )
            if s_preds is not None:
                subtypes_predictions[f_name].append(s_preds)
            eval_subtypes_labels[f_name].append(s_labels)

        # losses
        if losses["sup"] is not None:
            assert isinstance(sup_losses, list)
            sup_losses.append(losses["sup"].cpu().numpy())

        if losses["pos"] is not None:
            assert isinstance(pos_losses, list)
            pos_losses.append(losses["pos"].cpu().numpy())

        if losses["arc"] is not None:
            assert isinstance(arc_losses, list)
            arc_losses.append(losses["arc"].cpu().numpy())

        if losses["deprel"] is not None:
            assert isinstance(deprel_losses, list)
            deprel_losses.append(losses["deprel"].cpu().numpy())

        if losses["factorised"] is not None:
            for f_name, f_loss in losses["factorised"].items():
                factorised_losses[f_name].append(f_loss.cpu().numpy())

        if losses["xpos"] is not None:
            assert isinstance(xpos_losses, list)
            xpos_losses.append(losses["xpos"].cpu().numpy())

        if losses["feats"] is not None:
            for f_name, f_loss in losses["feats"].items():
                feats_losses[f_name].append(f_loss.cpu().numpy())

        if losses["subtypes"] is not None:
            for f_name, f_loss in losses["subtypes"].items():
                subtypes_losses[f_name].append(f_loss.cpu().numpy())

    if len(predictions) > 0:
        predictions_ = np.concatenate([
            np.pad(
                logits, ((0, 0), (0, max_len - logits.shape[1]), (0, 0)),
                'constant', constant_values=0)
            for logits in predictions], axis=0)
    else:
        predictions_ = None

    if len(eval_labels) > 0:
        eval_labels_ = np.concatenate([
            np.pad(
                labels, ((0, 0), (0, max_len - labels.shape[1])),
                'constant', constant_values=-1)
            for labels in eval_labels], axis=0)
    else:
        eval_labels_ = None

    if len(pos_predictions) > 0:
        pos_predictions_ = np.concatenate([
            np.pad(
                pos_logits,
                ((0, 0), (0, max_len - pos_logits.shape[1]), (0, 0)),
                'constant', constant_values=0)
            for pos_logits in pos_predictions], axis=0)
        eval_pos_labels_ = np.concatenate([
            np.pad(
                pos_labels, ((0, 0), (0, max_len - pos_labels.shape[1])),
                'constant', constant_values=-1)
            for pos_labels in eval_pos_labels], axis=0)
    else:
        pos_predictions_ = None
        eval_pos_labels_ = None

    if len(xpos_predictions) > 0:
        xpos_predictions_ = np.concatenate([
            np.pad(
                xpos_logits,
                ((0, 0), (0, max_len - xpos_logits.shape[1]), (0, 0)),
                'constant', constant_values=0)
            for xpos_logits in xpos_predictions], axis=0)
        eval_xpos_labels_ = np.concatenate([
            np.pad(
                xpos_labels, ((0, 0), (0, max_len - xpos_labels.shape[1])),
                'constant', constant_values=-1)
            for xpos_labels in eval_xpos_labels], axis=0)
    else:
        xpos_predictions_ = None
        eval_xpos_labels_ = None

    if len(arc_predictions) > 0:
        arc_predictions_ = np.concatenate([
            np.pad(
                arc_logits,
                (
                    (0, 0),
                    (0, max_parse_len - arc_logits.shape[1]),
                    (0, max_parse_len - arc_logits.shape[2])),
                'constant', constant_values=-np.inf)
            for arc_logits in arc_predictions], axis=0)
        eval_heads_ = np.concatenate([
            np.pad(
                heads, ((0, 0), (0, max_parse_len - heads.shape[1])),
                'constant', constant_values=-1)
            for heads in eval_heads], axis=0)
    else:
        arc_predictions_ = None
        eval_heads_ = None

    if len(deprel_predictions) > 0:
        deprel_predictions_ = pad_deprel_pred(
            deprel_predictions, max_parse_len, deprels_matrix
        )
    else:
        deprel_predictions_ = None

    if len(eval_deprel_labels) > 0:
        eval_deprel_labels_ = pad_deprel_gold(
            eval_deprel_labels, max_parse_len)
    else:
        eval_deprel_labels_ = None

    factorised_predictions_: dict[str, np.ndarray] = dict()
    eval_factorised_labels_: dict[str, np.ndarray] = dict()
    for f_name, f_predictions in factorised_predictions.items():
        factorised_predictions_[f_name] = np.concatenate([
            np.pad(
                logits, ((0, 0), (0, max_len - logits.shape[1]), (0, 0)),
                'constant', constant_values=0)
            for logits in f_predictions], axis=0)
        eval_factorised_labels_[f_name] = np.concatenate([
            np.pad(
                labels, ((0, 0), (0, max_len - labels.shape[1])),
                'constant', constant_values=-1)
            for labels in eval_factorised_labels[f_name]], axis=0)

    feats_predictions_: dict[str, np.ndarray] = dict()
    eval_feats_labels_: dict[str, np.ndarray] = dict()
    for f_name, f_predictions in feats_predictions.items():
        feats_predictions_[f_name] = np.concatenate([
            np.pad(
                logits, ((0, 0), (0, max_len - logits.shape[1]), (0, 0)),
                'constant', constant_values=0)
            for logits in f_predictions], axis=0)
        eval_feats_labels_[f_name] = np.concatenate([
            np.pad(
                labels, ((0, 0), (0, max_len - labels.shape[1])),
                'constant', constant_values=-1)
            for labels in eval_feats_labels[f_name]], axis=0)

    subtypes_predictions_: dict[str, np.ndarray] = dict()
    eval_subtypes_labels_: dict[str, np.ndarray] = dict()
    for f_name, f_predictions in subtypes_predictions.items():
        subtypes_predictions_[f_name] = pad_deprel_pred(
            f_predictions, max_parse_len, deprels_matrix
        )
        eval_subtypes_labels_[f_name] = pad_deprel_gold(
            eval_subtypes_labels[f_name], max_parse_len)

    losses: np.ndarray = np.zeros(tuple())
    num_losses = 0
    if len(sup_losses) > 0:
        sup_losses_ = sum(sup_losses)/len(sup_losses)
        losses += sup_losses_
        num_losses += 1
    else:
        sup_losses_ = None

    if len(pos_losses) > 0:
        pos_losses_ = sum(pos_losses)/len(pos_losses)
        losses += pos_losses_
        num_losses += 1
    else:
        pos_losses_ = None

    if len(xpos_losses) > 0:
        xpos_losses_ = sum(xpos_losses)/len(xpos_losses)
        losses += xpos_losses_
        num_losses += 1
    else:
        xpos_losses_ = None

    if len(arc_losses) > 0:
        arc_losses_ = sum(arc_losses)/len(arc_losses)
        losses += arc_losses_
        num_losses += 1
    else:
        arc_losses_ = None

    if len(deprel_losses) > 0:
        deprel_losses_ = sum(deprel_losses)/len(deprel_losses)
        losses += deprel_losses_
        num_losses += 1
    else:
        deprel_losses_ = None

    factorised_losses_: dict[str, np.ndarray] = dict()
    for f_name, f_losses in factorised_losses.items():
        factorised_losses_[f_name] = np.sum(f_losses)/len(f_losses)
        losses += factorised_losses_[f_name]
        num_losses += 1

    feats_losses_: dict[str, np.ndarray] = dict()
    for f_name, f_losses in feats_losses.items():
        feats_losses_[f_name] = np.sum(f_losses)/len(f_losses)
        losses += feats_losses_[f_name]
        num_losses += 1

    subtypes_losses_: dict[str, np.ndarray] = dict()
    for f_name, f_losses in subtypes_losses.items():
        subtypes_losses_[f_name] = np.sum(f_losses)/len(f_losses)
        losses += subtypes_losses_[f_name]
        num_losses += 1

    tagging_model.train()
    return (
        predictions_, eval_labels_,
        pos_predictions_, eval_pos_labels_,
        arc_predictions_, eval_heads_,
        deprel_predictions_, eval_deprel_labels_,
        factorised_predictions_, eval_factorised_labels_,
        xpos_predictions_, eval_xpos_labels_,
        feats_predictions_, eval_feats_labels_,
        subtypes_predictions_, eval_subtypes_labels_,
        losses / num_losses, sup_losses_, pos_losses_, arc_losses_, deprel_losses_,
        factorised_losses_, xpos_losses_, feats_losses_,
        subtypes_losses_,)


def calc_tag_accuracy(
        predictions, eval_labels, writer, use_tensorboard, step: int,
        typ: Literal["pos", "sup", "arc", "deprel"] = "sup",
        printinfo: bool = True,
        ) -> float:

    acc = calc_tag_accuracy_k(
        predictions, eval_labels, writer, use_tensorboard, step, k=1,
        typ=typ, printinfo=printinfo
    )
    if use_tensorboard:
        label = ""
        if typ == "sup":
            label = 'pos_tags_pr_curve'
        elif typ == "pos":
            label = 'sup_tags_pr_curve'
        elif typ == "arc":
            label = 'arc_pr_curve'
        elif typ == "deprel":
            label = 'deprel_pr_curve'

        writer.add_pr_curve(
            label,
            eval_labels[eval_labels != -1],
            predictions[eval_labels != -1].argmax(-1),
            global_step=step)
    return acc


def calc_tag_accuracy_k(
        predictions, eval_labels, writer, use_tensorboard,
        step: int, k: int = 1,
        typ: Literal["sup", "pos", "arc", "deprel"] | str = "sup",
        printinfo: bool = True,
        ) -> float:

    mask = eval_labels != -1
    eval_labels = eval_labels[mask]
    predictions = predictions[mask]

    if len(eval_labels) == 0:
        return float("nan")

    n_classes = predictions.shape[-1]
    k_eff = min(k, n_classes)

    if k_eff == 1:
        predictions = predictions.argmax(-1)
        acc = (predictions == eval_labels).mean()

    else:
        predictions = np.argpartition(
            predictions, -k_eff, axis=-1)[..., -k_eff:]
        # predictions = np.top_k(predictions[eval_labels != -1], k=k, dim=-1)

        acc = (predictions == eval_labels[..., None]).any(-1).mean()

    if printinfo:
        label = f"{typ}_accuracy"
        logging.info('{} {} best: {}'.format(label, k_eff, acc))

    return acc


def calc_tag_accuracy_upto_k(
            predictions, eval_labels, writer, use_tensorboard,
            step: int, k: int = 1,
            typ: Literal["sup", "pos", "arc", "deprel"] = "sup",
            printinfo: bool = True,
        ) -> list[float]:
    accs = [
        calc_tag_accuracy_k(
            predictions, eval_labels, writer, use_tensorboard, step, k=m,
            typ=typ, printinfo=printinfo
        )
        for m in range(1, k+1)
    ]

    if use_tensorboard:
        label = ""
        if typ == "sup":
            label = 'pos_tags_pr_curve'
        elif typ == "pos":
            label = 'sup_tags_pr_curve'
        elif typ == "arc":
            label = 'arc_pr_curve'
        elif typ == "deprel":
            label = 'deprel_pr_curve'

        writer.add_pr_curve(
            label,
            eval_labels[eval_labels != -1],
            predictions[eval_labels != -1].argmax(-1),
            global_step=step)

    return accs




def select_deprel_logits(
        deprel_predictions: np.ndarray,
        heads: np.ndarray,
        ) -> np.ndarray:
    """Select dependency-relation logits for chosen heads.

    deprel_predictions:
        [B, D, H, L]

    heads:
        [B, D], with -1 for ROOT/padding.

    Returns:
        [B, D, L]
    """
    safe_heads = np.maximum(heads, 0).astype(np.intp, copy=False)
    # [B, D]

    head_indices = safe_heads[..., None, None]
    # [B, D, 1, 1]

    # Explicitly broadcast across the label dimension.
    head_indices = np.broadcast_to(
        head_indices,
        (
            *safe_heads.shape,
            1,
            deprel_predictions.shape[-1],
        ),
    )
    # [B, D, 1, L]

    selected = np.take_along_axis(
        deprel_predictions,
        head_indices,
        axis=2,
    )
    # [B, D, 1, L]

    return selected.squeeze(2)
    # [B, D, L]


def get_eval_metric(
        eval_metric_type: Literal[
            "cacc", "a*-las", "a*-uas", "mst-las", "mst-uas"],
        factorised: Literal["complete", "structural", "seen", False],
        deprels_from_supertags: bool,
        combined_acc: float,
        sup_predictions: np.ndarray | None,
        arc_predictions: np.ndarray | None,
        pos_predictions: np.ndarray | None,
        deprel_predictions: np.ndarray | None,
        factorised_predictions: Mapping[str, np.ndarray],
        seen_supertag_logps: np.ndarray | None,
        eval_sup_labels: np.ndarray,
        eval_arc_labels: np.ndarray | None,
        eval_deprel_labels: np.ndarray | None,
        id2pos: Mapping[int, str],
        id2deprel: Mapping[int, str],
        deprel2id: Mapping[str, int],
        id2sup: Mapping[int, str],
        sup2id: Mapping[str, int],
        id2sup_relative: Mapping[int, extraction.RelativeTag],
        valid_id2sup: Mapping[int, str] | None,
        valid_id2sup_relative: Mapping[int, extraction.RelativeTag] | None,
        valid_factors: None | factorisation.SupertagFactors,
        max_l: int,
        max_r: int,
        k_supertag: int,
        k_head_scores: int,
        t_sup: float = 1,
        t_arc: float = 1,
        sup_score_scale: float = 1.0,
        ) -> float:
    eval_metric: float
    match eval_metric_type:
        case "cacc":
            eval_metric = combined_acc

        case "a*-las" | "a*-uas":
            root_supertag = "*+root"

            assert arc_predictions is not None
            assert eval_arc_labels is not None
            # assert pos_predictions is not None

            chart_id2sup: Mapping[int, str]
            chart_id2sup_relative: Mapping[int, extraction.RelativeTag]
            if factorised == "complete":
                argument_logps = {
                    f_name: -utils.neg_log10_softmax(f_pred / t_sup)
                    for f_name, f_pred in
                    factorised_predictions.items() if
                    f_name.startswith("left") or
                    f_name.startswith("right")
                }
                candidates = factorisation.top_k_valid_supertags_batch(
                    argument_logps,
                    -utils.neg_log10_softmax(
                        factorised_predictions["l_arg_nums"] / t_sup),
                    -utils.neg_log10_softmax(
                        factorised_predictions["r_arg_nums"] / t_sup),
                    -utils.neg_log10_softmax(
                        factorised_predictions["aux_positions"] / t_sup),
                    -utils.neg_log10_softmax(
                        factorised_predictions["aux_rel_ids"] / t_sup),
                    id2deprel,
                    max_l, max_r, k=k_supertag,
                    projective_only=True,
                    valid_mask=eval_sup_labels != -1,
                )

                (
                    supertag_scores,
                    chart_id2sup,
                    chart_sup2id,
                ) = factorisation.make_batch_supertag_scores(
                    candidates,
                    root_supertag,
                )
                chart_id2sup_relative = {
                    i: extraction.convert_string_to_relative_relation(
                        tag)
                    for i, tag in chart_id2sup.items()}
                root_sup_id = chart_sup2id[root_supertag]
                chart_deprel_dict = deprel2id
            elif factorised == "structural":
                assert valid_factors is not None
                supertag_scores = (
                    -factorisation.score_structural_supertags_batch(
                        valid_factors,
                        -utils.neg_log10_softmax(
                            factorised_predictions["l_arg_nums"] / t_sup),
                        -utils.neg_log10_softmax(
                            factorised_predictions["r_arg_nums"] / t_sup),
                        -utils.neg_log10_softmax(
                            factorised_predictions["aux_positions"] / t_sup),
                    ))
                assert valid_id2sup is not None
                assert valid_id2sup_relative is not None
                chart_id2sup = valid_id2sup
                chart_id2sup_relative = valid_id2sup_relative
                chart_sup2id = {
                    sup: i for i, sup in chart_id2sup.items()}
                chart_deprel_dict = {"_": 0, "dep": 0, "root": 0}
                root_sup_id = chart_sup2id["*+_"]
            elif factorised == "seen":
                assert seen_supertag_logps is not None
                supertag_scores = -seen_supertag_logps
                # supertag_scores = utils.neg_log10_softmax(
                #     seen_supertag_scores / t_sup)
                chart_id2sup = id2sup
                chart_id2sup_relative = id2sup_relative

                chart_deprel_dict = deprel2id
                root_sup_id = sup2id[root_supertag]
            else:
                assert sup_predictions is not None
                supertag_scores = utils.neg_log10_softmax(
                    sup_predictions / t_sup)
                chart_id2sup = id2sup
                chart_id2sup_relative = id2sup_relative
                chart_deprel_dict = deprel2id
                root_sup_id = sup2id[root_supertag]

            # if epo > -1:
            head_preds_astar, deprel_preds_astar = parsing.chart(
                arc_predictions,
                eval_arc_labels,
                supertag_scores,
                chart_id2sup_relative,
                id2pos,
                chart_deprel_dict,
                pos_predictions.argmax(
                    -1) if pos_predictions is not None else None,
                max_l,
                max_r,
                root_sup_id=root_sup_id,
                k_supertag=k_supertag,
                k_head_scores=k_head_scores,
                t_arc=t_arc,
                sup_score_scale=sup_score_scale,
            )

            assert eval_deprel_labels is not None

            if eval_metric_type == "a*-las":

                if not deprels_from_supertags:
                    assert deprel_predictions is not None

                    # deprel_predictions: [B, D, H, L]
                    # head_preds_astar:   [B, D]
                    deprel_logits_astar = select_deprel_logits(
                        deprel_predictions,
                        head_preds_astar,
                    )
                    # [B, D, L]

                    deprel_preds_astar = (
                        deprel_logits_astar.argmax(-1)
                    )
                    # [B, D]

                eval_metric = parsing.las(
                    head_preds_astar,
                    deprel_preds_astar,
                    eval_arc_labels,
                    eval_deprel_labels,
                    id2deprel=id2deprel
                )

            else:  # a*-uas
                eval_metric = parsing.uas(
                    head_preds_astar,
                    eval_arc_labels,
                )

            # else:
            #     eval_metric = 0
            #     tol = 99999

        case "mst-las" | "mst-uas":
            assert arc_predictions is not None
            assert eval_arc_labels is not None

            mst = parsing.mst(
                arc_predictions,
                eval_arc_labels,
            )
            # mst: [B, D]

            if eval_metric_type == "mst-las":
                assert deprel_predictions is not None
                assert eval_deprel_labels is not None

                # deprel_predictions: [B, D, H, L]
                deprel_logits_mst = select_deprel_logits(
                    deprel_predictions,
                    mst,
                )
                # [B, D, L]

                deprel_predictions_mst = (
                    deprel_logits_mst.argmax(-1)
                )
                # [B, D]

                eval_metric = parsing.las(
                    mst,
                    deprel_predictions_mst,
                    eval_arc_labels,
                    eval_deprel_labels,
                )

            else:  # mst-uas
                eval_metric = parsing.uas(
                    mst,
                    eval_arc_labels,
                )

        case _:
            raise Exception(
                f"args.tagging.eval_metric "
                f"'{eval_metric_type}' unknown"
            )

    return eval_metric
