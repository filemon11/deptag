import numpy as np
import torch
import logging
import tqdm


def report_eval_loss(
        model, eval_dataloader, device, n_iter, writer) -> float:
    loss = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(), torch.amp.autocast(
                "cpu" if device == torch.device("cpu") else "cuda",
                enabled=True, dtype=torch.bfloat16):
            outputs = model(**batch)
            loss.append(torch.mean(outputs[0]).cpu())

    mean_loss: float = np.mean(loss).item()
    logging.info("Eval Loss: {}".format(mean_loss))
    if writer is not None:
        writer.add_scalar('eval_loss', mean_loss, n_iter)
    return mean_loss


def predict(
        model, eval_dataloader, dataset_size, num_tags, batch_size, device
        ) -> tuple[np.ndarray, np.ndarray]:

    model.eval()
    predictions = []
    eval_labels = []
    max_len = 0
    idx = 0

    for batch in tqdm.tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad(), torch.amp.autocast(
                "cpu" if device == torch.device("cpu") else "cuda",
                enabled=True, dtype=torch.bfloat16
                ):
            outputs = model(**batch)

        logits = outputs[1].float().cpu().numpy()
        max_len = max(max_len, logits.shape[1])
        predictions.append(logits)
        labels = batch['labels'].int().cpu().numpy()
        eval_labels.append(labels)
        idx += 1

    predictions_ = np.concatenate([
        np.pad(
            logits, ((0, 0), (0, max_len - logits.shape[1]), (0, 0)),
            'constant', constant_values=0)
        for logits in predictions], axis=0)
    eval_labels_ = np.concatenate([
        np.pad(
            labels, ((0, 0), (0, max_len - labels.shape[1])),
            'constant', constant_values=0)
        for labels in eval_labels], axis=0)

    return predictions_, eval_labels_


def calc_tag_accuracy(
        predictions, eval_labels, writer, use_tensorboard
        ) -> float:

    predictions = predictions[eval_labels != -1].argmax(-1)

    eval_labels = eval_labels[eval_labels != -1]

    acc = (predictions == eval_labels).mean()

    logging.info('tags_accuracy: {}'.format(acc))

    if use_tensorboard:
        writer.add_pr_curve(
            'tags_pr_curve', eval_labels, predictions)
    return acc
