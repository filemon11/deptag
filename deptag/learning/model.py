import pathlib
import torch
from torch import nn
import bitsandbytes as bnb

from . import losses, biaffine

from transformers import AutoModel


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

        self.biaffine = biaffine.make_model(
            config.hidden_size,
            config.task_specific_params["mlp_arc_hidden"],
            config.task_specific_params["mlp_lab_hidden"],
            config.task_specific_params["mlp_dropout"],
            config.task_specific_params["mlp_num_labels"],
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
            heads=None,
            deprel_ids=None,
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

        S_arc, S_lab = self.biaffine(token_repr)

        loss = None
        pos_loss = None
        arc_loss = None
        label_loss = None
        if labels is not None and (self.training or report_loss):
            loss = losses.calc_loss_helper(
                tag_logits, labels, attention_mask.bool(),
                printinfo=printinfo
            )
        if self.pos_projection is not None:
            pos_loss = losses.calc_loss_helper(
                pos_logits, pos_ids, attention_mask.bool(),
                printinfo=printinfo
            )
        if heads is not None:
            arc_loss = self.biaffine.arc_loss(
                S_arc, heads, attention_mask.bool(), printinfo=printinfo)
        if self.biaffine.lab_mlp_d is not None:
            label_loss = self.biaffine.lab_loss(
                S_lab, heads, deprel_ids, attention_mask.bool(),
                printinfo=printinfo)

        return (
            loss, tag_logits, pos_loss, pos_logits,
            arc_loss, S_arc, label_loss, S_lab)
