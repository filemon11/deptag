import pathlib
import torch
from torch import nn
import bitsandbytes as bnb
import torch.nn.functional as F

from transformers import AutoModel


# @torch.compile
def calc_loss_helper(
        logits, labels, attention_mask):
    # shape: (batch_size, seq_len, num_tags) -> (batch_size, num_tags, seq_len)
    logits = torch.movedim(logits, -1, 1)

    # Only keep active parts of the loss
    active_labels = torch.where(
        attention_mask, labels, -1
    )

    print(active_labels)

    loss = F.cross_entropy(logits, active_labels, ignore_index=-1)

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

        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_labels)
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
        padding_mask = attention_mask == 0
        token_repr = self.transformer(
            token_repr,
            src_key_padding_mask=padding_mask
        )

        tag_logits = self.projection(token_repr)

        loss = None
        if labels is not None and self.training:
            loss = calc_loss_helper(
                tag_logits, labels, attention_mask.bool(),
            )
            return loss, tag_logits
        else:
            return loss, tag_logits
