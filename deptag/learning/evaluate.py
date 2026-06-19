import numpy as np
import torch
import logging
import tqdm


from typing import overload, Literal


@overload
def predict(
        model, eval_dataloader, dataset_size, num_tags, batch_size, device,
        report_loss: Literal[True],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@overload
def predict(
        model, eval_dataloader, dataset_size, num_tags, batch_size, device,
        report_loss: Literal[False] = False,
        ) -> tuple[np.ndarray, np.ndarray, None]:
    ...


def predict(
        model, eval_dataloader, dataset_size, num_tags, batch_size, device,
        report_loss: bool = False,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:

    model.eval()
    predictions = []
    eval_labels = []
    max_len = 0
    idx = 0

    losses = []
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
        if outputs[0] is not None:
            losses.append(outputs[0].cpu().numpy())

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

    if len(losses) == 0:
        return predictions_, eval_labels_, None
    return predictions_, eval_labels_, sum(losses)/len(losses)


def calc_tag_accuracy(
        predictions, eval_labels, writer, use_tensorboard, step: int,
        ) -> float:

    acc = calc_tag_accuracy_k(
        predictions, eval_labels, writer, use_tensorboard, step, k=1
    )
    if use_tensorboard:
        writer.add_pr_curve(
            'tags_pr_curve',
            eval_labels[eval_labels != -1],
            predictions[eval_labels != -1].argmax(-1),
            global_step=step)

    return acc


def calc_tag_accuracy_k(
        predictions, eval_labels, writer, use_tensorboard,
        step: int, k: int = 1
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

    logging.info('tags_accuracy {} best: {}'.format(k, acc))

    return acc


def calc_tag_accuracy_upto_k(
            predictions, eval_labels, writer, use_tensorboard,
            step: int, k: int = 1
        ) -> list[float]:
    accs = [
        calc_tag_accuracy_k(
            predictions, eval_labels, writer, use_tensorboard, step, k=m
        )
        for m in range(1, k+1)
    ]

    if use_tensorboard:
        writer.add_pr_curve(
            'tags_pr_curve',
            eval_labels[eval_labels != -1],
            predictions[eval_labels != -1].argmax(-1),
            global_step=step)

    return accs