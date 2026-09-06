import torch
import torch.nn.functional as F


def calc_loss_helper(
        logits: torch.Tensor, labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        printinfo: bool = False):
    # shape: (batch_size, seq_len, num_tags) -> (batch_size, num_tags, seq_len)
    logits = torch.movedim(logits, -1, 1)

    if (labels != -1).any().item():
        loss = F.cross_entropy(
            logits, labels, ignore_index=-1, reduction="mean",
            label_smoothing=label_smoothing)
    else:
        loss = torch.tensor(0, dtype=logits.dtype, device=logits.device)

    if printinfo:
        logits_wo_ignore = logits.transpose(-1, -2)[labels != -1]

        labels_wo_ignore = labels[labels != -1]

        logits_wo_ignore_mask = (
            logits_wo_ignore.argmax(-1) == labels_wo_ignore)

        logits_correct = logits_wo_ignore[logits_wo_ignore_mask]
        labels_correct = labels_wo_ignore[logits_wo_ignore_mask]
        loss_correct = F.cross_entropy(
            logits_correct, labels_correct, reduction="mean")

        logits_false = logits_wo_ignore[~logits_wo_ignore_mask]
        labels_false = labels_wo_ignore[~logits_wo_ignore_mask]
        loss_false = F.cross_entropy(
            logits_false, labels_false, reduction="mean")

        print("correct item loss:", loss_correct.item())
        print("false item loss:", loss_false.item())
        print("num correct items:", len(labels_correct))
        print("num false items:", len(labels_false))
        print("loss:", loss)
    return loss
