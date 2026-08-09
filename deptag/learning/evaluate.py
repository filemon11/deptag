import numpy as np
import torch
import logging
import tqdm


from typing import overload, Literal


@overload
def predict(
        model, eval_dataloader, dataset_size, num_tags, batch_size, device,
        report_loss: Literal[True], deprels_from_pred_head: bool = False,
        deprels_matrix: bool = False
        ) -> tuple[
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None]:
    ...


@overload
def predict(
        model, eval_dataloader, dataset_size, num_tags, batch_size, device,
        report_loss: Literal[False] = False,
        deprels_from_pred_head: bool = False, deprels_matrix: bool = False
        ) -> tuple[
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            None,
            None, None,
            None, None]:
    ...


def predict(
        model, eval_dataloader, dataset_size, num_tags, batch_size, device,
        report_loss: bool = False, deprels_from_pred_head: bool = False,
        deprels_matrix: bool = False
        ) -> tuple[
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None, np.ndarray | None,
            np.ndarray | None, np.ndarray | None]:

    model.eval()
    predictions = []
    eval_labels = []
    max_len = 0
    idx = 0

    pos_predictions = []
    eval_pos_labels = []
    arc_predictions = []
    eval_heads = []
    deprel_predictions = []
    eval_deprel_labels = []

    sup_losses: None | list[np.ndarray] | np.ndarray = []
    pos_losses: None | list[np.ndarray] | np.ndarray = []
    arc_losses: None | list[np.ndarray] | np.ndarray = []
    deprel_losses: None | list[np.ndarray] | np.ndarray = []
    for batch in tqdm.tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad(), torch.amp.autocast(
                "cpu" if device == torch.device("cpu") else "cuda",
                enabled=True, dtype=torch.float16
                ):
            outputs = model(
                **batch, report_loss=report_loss,
                printinfo=False)

        idx += 1
        if outputs[1] is not None:
            logits = outputs[1].float().cpu().numpy()
            max_len = max(max_len, logits.shape[1])
            predictions.append(logits)
            labels = batch['labels'].int().cpu().numpy()
            eval_labels.append(labels)

        if outputs[3] is not None:
            pos_logits = outputs[3].float().cpu().numpy()
            pos_predictions.append(pos_logits)
            max_len = max(max_len, pos_logits.shape[1])
            pos_labels = batch['pos_ids'].int().cpu().numpy()
            eval_pos_labels.append(pos_labels)

        if outputs[5] is not None:
            # [batch, sent_len (head preds), sent_len]
            # outputs[5][:, 1:][batch['heads'][:, 1:] == -1] = float("-inf")
            valid_head = batch['heads'].ne(-1).clone()
            valid_head[:, 0] = True
            arc_logits = outputs[5].masked_fill(
                ~valid_head.unsqueeze(-1),  # [batch, candidate_head, 1]
                float("-inf"),
            )
            # set impossible likelihood to predicted heads that are masked

            arc_logits = arc_logits.transpose(-1, -2).float().cpu().numpy()
            # [batch, sent_len, sent_len (head preds)]
            arc_predictions.append(arc_logits)
            max_len = max(max_len, arc_logits.shape[1])
            heads = batch['heads'].int().cpu().numpy()
            eval_heads.append(heads)

        if outputs[7] is not None:
            # This reports the accuracy on the label predictions for the
            # correct arc
            if deprels_from_pred_head:
                assert not deprels_matrix
                assert outputs[5] is not None, (
                    "Cannot use deprels from predicted head in model without"
                    " head prediction."
                )
                heads = eval_heads[-1].argmax(-1)
            else:
                heads = batch['heads']
            # from format [B, lab, Sl, S]
            # into format [B, S, Sl, lab]
            if deprels_matrix:
                deprel_predictions.append(
                    outputs[7].permute(0, 3, 2, 1).float().cpu().numpy())
            else:
                # heads with index -1 are padding and are treated as
                # index 0 here (to be disregarded later)
                # print(heads[1])
                hds = heads + (heads < 0)

                hds = hds.unsqueeze(1).unsqueeze(2)
                # [batch, 1, 1, sent_len]

                hds = hds.expand(-1, outputs[7].size(1), -1, -1)
                # [batch, n_labels, 1, sent_len]

                deprel_logits = torch.gather(outputs[7], 2, hds).squeeze(2)
                # [batch, n_labels, sent_len]
                max_len = max(max_len, deprel_logits.shape[1])

                deprel_logits = deprel_logits.transpose(-1, -2)
                # [batch, sent_len, n_labels]

                deprel_predictions.append(deprel_logits.float().cpu().numpy())
        deprel_labels = batch['deprel_ids'].int().cpu().numpy()
        eval_deprel_labels.append(deprel_labels)

        if outputs[0] is not None:
            assert isinstance(sup_losses, list)
            sup_losses.append(outputs[0].cpu().numpy())

        if outputs[2] is not None:
            assert isinstance(pos_losses, list)
            pos_losses.append(outputs[2].cpu().numpy())

        if outputs[4] is not None:
            assert isinstance(arc_losses, list)
            arc_losses.append(outputs[4].cpu().numpy())

        if outputs[6] is not None:
            assert isinstance(deprel_losses, list)
            deprel_losses.append(outputs[6].cpu().numpy())

    if len(predictions) > 0:
        predictions_ = np.concatenate([
            np.pad(
                logits, ((0, 0), (0, max_len - logits.shape[1]), (0, 0)),
                'constant', constant_values=0)
            for logits in predictions], axis=0)
        eval_labels_ = np.concatenate([
            np.pad(
                labels, ((0, 0), (0, max_len - labels.shape[1])),
                'constant', constant_values=-1)
            for labels in eval_labels], axis=0)
    else:
        predictions_ = None
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

    if len(arc_predictions) > 0:
        arc_predictions_ = np.concatenate([
            np.pad(
                arc_logits,
                (
                    (0, 0),
                    (0, max_len - arc_logits.shape[1]),
                    (0, max_len - arc_logits.shape[2])),
                'constant', constant_values=-np.inf)
            for arc_logits in arc_predictions], axis=0)
        eval_heads_ = np.concatenate([
            np.pad(
                heads, ((0, 0), (0, max_len - heads.shape[1])),
                'constant', constant_values=-1)
            for heads in eval_heads], axis=0)
    else:
        arc_predictions_ = None
        eval_heads_ = None

    if len(deprel_predictions) > 0:
        if deprels_matrix:
            deprel_predictions_ = np.concatenate([
                np.pad(
                    deprel_logits,
                    (
                        (0, 0),
                        (0, max_len - deprel_logits.shape[1]),
                        (0, max_len - deprel_logits.shape[2]),
                        (0, 0)),
                    'constant', constant_values=-np.inf)
                for deprel_logits in deprel_predictions], axis=0)
        else:
            deprel_predictions_ = np.concatenate([
                np.pad(
                    deprel_logits,
                    ((0, 0), (0, max_len - deprel_logits.shape[1]), (0, 0)),
                    'constant', constant_values=0)
                for deprel_logits in deprel_predictions], axis=0)
    else:
        deprel_predictions_ = None

    if len(eval_deprel_labels) > 0: 
        eval_deprel_labels_ = np.concatenate([
            np.pad(
                deprel_labels, ((0, 0), (0, max_len - deprel_labels.shape[1])),
                'constant', constant_values=-1)
            for deprel_labels in eval_deprel_labels], axis=0)
    else:
        eval_deprel_labels_ = None

    losses = 0
    num_losses = 0
    if len(sup_losses) > 0:
        sup_losses = sum(sup_losses)/len(sup_losses)
        losses += sup_losses
        num_losses += 1
    else:
        sup_losses = None

    if len(pos_losses) > 0:
        pos_losses = sum(pos_losses)/len(pos_losses)
        losses += pos_losses
        num_losses += 1
    else:
        pos_losses = None

    if len(arc_losses) > 0:
        arc_losses = sum(arc_losses)/len(arc_losses)
        losses += arc_losses
        num_losses += 1
    else:
        arc_losses = None

    if len(deprel_losses) > 0:
        deprel_losses = sum(deprel_losses)/len(deprel_losses)
        losses += deprel_losses
        num_losses += 1
    else:
        deprel_losses = None

    return (
        predictions_, eval_labels_,
        pos_predictions_, eval_pos_labels_,
        arc_predictions_, eval_heads_,
        deprel_predictions_, eval_deprel_labels_,
        losses, sup_losses, pos_losses, arc_losses, deprel_losses)


def calc_tag_accuracy(
        predictions, eval_labels, writer, use_tensorboard, step: int,
        typ: Literal["pos", "sup", "arc", "deprel"] = "sup"
        ) -> float:

    acc = calc_tag_accuracy_k(
        predictions, eval_labels, writer, use_tensorboard, step, k=1,
        typ=typ
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
        typ: Literal["sup", "pos", "arc", "deprel"] = "sup",
        ) -> float:

    if k == 1:
        predictions = predictions[eval_labels != -1].argmax(-1)

        eval_labels = eval_labels[eval_labels != -1]

        acc = (predictions == eval_labels).mean()

    else:
        predictions = np.argpartition(
            predictions[eval_labels != -1], -k, axis=-1)[..., -k:]
        # predictions = np.top_k(predictions[eval_labels != -1], k=k, dim=-1)

        eval_labels = eval_labels[eval_labels != -1]

        acc = (predictions == eval_labels[..., None]).any(-1).mean()

    label = ""
    if typ == "pos":
        label = 'pos_tags_accuracy'
    elif typ == "sup":
        label = 'sup_tags_accuracy'
    elif typ == "arc":
        label = 'arc_accuracy'
    elif typ == "deprel":
        label = 'deprel_accuracy'
    logging.info('{} {} best: {}'.format(label, k, acc))

    return acc


def calc_tag_accuracy_upto_k(
            predictions, eval_labels, writer, use_tensorboard,
            step: int, k: int = 1,
            typ: Literal["sup", "pos", "arc", "deprel"] = "sup",
        ) -> list[float]:
    accs = [
        calc_tag_accuracy_k(
            predictions, eval_labels, writer, use_tensorboard, step, k=m,
            typ=typ
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
