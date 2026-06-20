import pathlib
import torch
from torch import nn
import bitsandbytes as bnb
import torch.nn.functional as F

from transformers import AutoModel


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


class ModelForTagging(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_tags: int = config.num_labels
        self.model_path: pathlib.Path = config.task_specific_params[
            'model_path']
        self.use_pos: bool = config.task_specific_params['use_pos']
        self.num_pos_tags: int = config.task_specific_params['num_pos_tags']

        self.pos_emb_dim: int = config.task_specific_params['pos_emb_dim']
        self.dropout_rate: float = config.task_specific_params['dropout']

        self.transformer_layers = config.task_specific_params[
            "transformer_layers"]

        self.bert = AutoModel.from_pretrained(self.model_path, config=config)
        if self.use_pos:
            self.pos_encoder = nn.Sequential(
                bnb.nn.StableEmbedding(
                    self.num_pos_tags, self.pos_emb_dim, padding_idx=0)
            )

        self.endofword_embedding = bnb.nn.StableEmbedding(2, self.pos_emb_dim)

        transformer_input_dim = (
            config.hidden_size
            + self.pos_emb_dim
            + (self.pos_emb_dim if self.use_pos else 0)
        )

        self.input_projection = nn.Linear(
            transformer_input_dim, config.hidden_size)

        if self.transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.task_specific_params["n_heads"],
                dim_feedforward=config.task_specific_params.get(
                    "ffn_dim", 4 * config.hidden_size
                ),
                dropout=self.dropout_rate,
                activation="gelu",
                batch_first=True,   # input/output: (batch, seq, feature)
                norm_first=True,
            )

            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.task_specific_params["transformer_layers"],
            )

        self.dropout = nn.Dropout(self.dropout_rate)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_labels)
        )
        self.pos_projection = None
        if config.task_specific_params["train_pos"]:
            self.pos_projection = nn.Sequential(
                nn.Linear(config.hidden_size, self.num_pos_tags)
            )

    def forward(
            self,
            input_ids=None,
            pos_ids=None,
            end_of_word=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            report_loss: bool = False,
            printinfo: bool = False,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if self.use_pos:
            pos_encodings = self.pos_encoder(pos_ids)
            token_repr = torch.cat([outputs[0], pos_encodings], dim=-1)
        else:
            token_repr = outputs[0]

        token_repr = torch.cat(
            [token_repr, self.endofword_embedding((pos_ids != 0).long())],
            dim=-1)

        token_repr = self.input_projection(token_repr)

        if self.transformer_layers > 0:
            padding_mask = attention_mask == 0
            token_repr = self.transformer(
                token_repr,
                src_key_padding_mask=padding_mask
            )

        tag_logits = self.projection(self.dropout(token_repr))

        pos_logits = None
        if self.pos_projection is not None:
            pos_logits = self.pos_projection(self.dropout(token_repr))

        loss = None
        pos_loss = None
        if labels is not None and (self.training or report_loss):
            loss = calc_loss_helper(
                tag_logits, labels, attention_mask.bool(),
                printinfo=printinfo
            )
            if self.pos_projection is not None:
                pos_loss = calc_loss_helper(
                    pos_logits, pos_ids, attention_mask.bool(),
                    printinfo=printinfo
                )

        return loss, tag_logits, pos_loss, pos_logits
