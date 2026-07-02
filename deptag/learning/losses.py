import torch
import torch.nn.functional as F


# @torch.compile
def calc_loss_helper(
        logits, labels, attention_mask, printinfo: bool = False):
    # shape: (batch_size, seq_len, num_tags) -> (batch_size, num_tags, seq_len)
    logits = torch.movedim(logits, -1, 1)

    # # Only keep active parts of the loss
    # active_labels = torch.where(
    #     attention_mask, labels, -1
    # )

    loss = F.cross_entropy(
        logits, labels, ignore_index=-1, reduction="mean")

    logits_wo_ignore = logits.transpose(-1, -2)[labels != -1]
    labels_wo_ignore = labels[labels != -1]

    if printinfo:
        logits_wo_ignore_mask = (
            logits_wo_ignore.argmax(-1) == labels_wo_ignore)

        # loss_wo_ignore = F.cross_entropy(
        #     logits_wo_ignore, labels_wo_ignore, reduce="mean")
        # print(loss, loss_wo_ignore)

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
