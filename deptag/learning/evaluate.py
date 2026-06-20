import numpy as np
import torch
import logging
import tqdm


from typing import overload, Literal


@overload
def predict(
        model, eval_dataloader, dataset_size, num_tags, batch_size, device,
        report_loss: Literal[True],
        ) -> tuple[
            np.ndarray, np.ndarray,
            np.ndarray, np.ndarray,
            np.ndarray, np.ndarray | None]:
    ...


@overload
def predict(
        model, eval_dataloader, dataset_size, num_tags, batch_size, device,
        report_loss: Literal[False] = False,
        ) -> tuple[
            np.ndarray, np.ndarray,
            np.ndarray, np.ndarray,
            None, np.ndarray | None]:
    ...


def predict(
        model, eval_dataloader, dataset_size, num_tags, batch_size, device,
        report_loss: bool = False,
        ) -> tuple[
            np.ndarray, np.ndarray,
            np.ndarray, np.ndarray,
            np.ndarray | None, np.ndarray | None]:

    model.eval()
    predictions = []
    eval_labels = []
    max_len = 0
    idx = 0

    pos_predictions = []
    eval_pos_labels = []

    losses: None | list[np.ndarray] | np.ndarray = []
    pos_losses: None | list[np.ndarray] | np.ndarray = []
    for batch in tqdm.tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad(), torch.amp.autocast(
                "cpu" if device == torch.device("cpu") else "cuda",
                enabled=True, dtype=torch.bfloat16
                ):
            outputs = model(
                **batch, report_loss=report_loss,
                printinfo=False)

        logits = outputs[1].float().cpu().numpy()
        max_len = max(max_len, logits.shape[1])
        predictions.append(logits)
        labels = batch['labels'].int().cpu().numpy()
        eval_labels.append(labels)
        idx += 1

        if outputs[3] is not None:
            pos_logits = outputs[3].float().cpu().numpy()
            pos_predictions.append(pos_logits)
            pos_labels = batch['pos_ids'].int().cpu().numpy()
            eval_pos_labels.append(pos_labels)

        if outputs[0] is not None:
            losses.append(outputs[0].cpu().numpy())

        if outputs[2] is not None:
            pos_losses.append(outputs[2].cpu().numpy())

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

    if len(pos_predictions) > 0:
        pos_predictions_ = np.concatenate([
            np.pad(
                pos_logits, ((0, 0), (0, max_len - pos_logits.shape[1]), (0, 0)),
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

    if len(losses) > 0:
        losses = sum(losses)/len(losses)
    else:
        losses = None
    if len(pos_losses) > 0:
        pos_losses = sum(pos_losses)/len(pos_losses)
    else:
        pos_losses = None
    return (
        predictions_, eval_labels_,
        pos_predictions_, eval_pos_labels_,
        losses, pos_losses)


def calc_tag_accuracy(
        predictions, eval_labels, writer, use_tensorboard, step: int,
        is_pos: bool = False
        ) -> float:

    acc = calc_tag_accuracy_k(
        predictions, eval_labels, writer, use_tensorboard, step, k=1,
        is_pos=is_pos
    )
    if use_tensorboard:
        if is_pos:
            writer.add_pr_curve(
                'pos_tags_pr_curve',
                eval_labels[eval_labels != -1],
                predictions[eval_labels != -1].argmax(-1),
                global_step=step)
        else:
            writer.add_pr_curve(
                'tags_pr_curve',
                eval_labels[eval_labels != -1],
                predictions[eval_labels != -1].argmax(-1),
                global_step=step)

    return acc


def calc_tag_accuracy_k(
        predictions, eval_labels, writer, use_tensorboard,
        step: int, k: int = 1, is_pos: bool = False,
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

    if is_pos:
        logging.info('pos_tags_accuracy {} best: {}'.format(k, acc))
    else:
        logging.info('tags_accuracy {} best: {}'.format(k, acc))

    return acc


def calc_tag_accuracy_upto_k(
            predictions, eval_labels, writer, use_tensorboard,
            step: int, k: int = 1, is_pos: bool = False,
        ) -> list[float]:
    accs = [
        calc_tag_accuracy_k(
            predictions, eval_labels, writer, use_tensorboard, step, k=m,
            is_pos=is_pos
        )
        for m in range(1, k+1)
    ]

    if use_tensorboard:
        if is_pos:
            writer.add_pr_curve(
                'pos_tags_pr_curve',
                eval_labels[eval_labels != -1],
                predictions[eval_labels != -1].argmax(-1),
                global_step=step)
        else:
            writer.add_pr_curve(
                'tags_pr_curve',
                eval_labels[eval_labels != -1],
                predictions[eval_labels != -1].argmax(-1),
                global_step=step)

    return accs
