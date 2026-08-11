import pathlib
import torch
from torch import nn
import bitsandbytes as bnb

from . import losses, biaffine

from transformers import AutoModel, BertModel


class ModelForTagging(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_tags: int = config.num_labels
        self.model_path: pathlib.Path = config.task_specific_params[
            'model_path']
        self.use_pos: bool = config.task_specific_params['use_pos']
        self.train_sup: bool = config.task_specific_params['train_sup']
        self.num_pos_tags: int = config.task_specific_params['num_pos_tags']

        self.pos_emb_dim: int = config.task_specific_params['pos_emb_dim']
        self.dropout_rate: float = config.task_specific_params['dropout']

        self.transformer_layers = config.task_specific_params[
            "transformer_layers"]

        self.bert: BertModel = AutoModel.from_pretrained(
            self.model_path, config=config)

        import transformers.utils.output_capturing as hf_output_capturing

        hf_output_capturing.torch = torch
        hf_output_capturing.maybe_install_capturing_hooks(self.bert)

        # self.bert: BertModel = torch.compile(self.bert)

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

        self.pos_layer = config.task_specific_params["pos_layer"]
        self.supertag_layer = config.task_specific_params["supertag_layer"]
        self.parse_layer = config.task_specific_params["parse_layer"]

        self.input_projection_parse = nn.Linear(
            transformer_input_dim, config.hidden_size)
        self.input_projection_sup = nn.Linear(
            transformer_input_dim, config.hidden_size)
        self.input_projection_pos = nn.Linear(
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
        self.projection = None
        if self.train_sup:
            self.projection = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(config.hidden_size, config.num_labels))
            # self.projection = nn.Sequential(
            #     nn.Linear(config.hidden_size, config.num_labels))
        self.pos_projection = None
        if config.task_specific_params["train_pos"]:
            self.pos_projection = nn.Sequential(
                nn.Linear(config.hidden_size, self.num_pos_tags)
            )

        self.biaffine = None
        if (
                config.task_specific_params["mlp_arc_hidden"] is not None
                or config.task_specific_params["mlp_lab_hidden"] is not None):
            self.biaffine = biaffine.make_model(
                config.hidden_size,
                config.task_specific_params["mlp_arc_hidden"],
                config.task_specific_params["mlp_lab_hidden"],
                config.task_specific_params["mlp_dropout"],
                config.task_specific_params["mlp_num_labels"],
            )
            # self.biaffine.compile()  # (dynamic=True)

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
            report_loss: bool = False,
            printinfo: bool = False,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        # num_layers = len(outputs["hidden_states"])-1
        token_repr_parse = outputs["hidden_states"][self.parse_layer]
        token_repr_sup = outputs["hidden_states"][self.supertag_layer]
        token_repr_pos = outputs["hidden_states"][self.pos_layer]
        # print(num_layers, round(num_layers*(2/3)), round(num_layers*(1/3)))

        if self.use_pos:
            pos_encodings = self.pos_encoder(pos_ids)
            token_repr_parse = torch.cat(
                [token_repr_parse, pos_encodings], dim=-1)
        else:
            token_repr_parse = outputs[0]

        token_repr_parse = torch.cat(
            [token_repr_parse, self.endofword_embedding(
                (pos_ids != 0).long())],
            dim=-1)
        token_repr_sup = torch.cat(
            [token_repr_sup, self.endofword_embedding((pos_ids != 0).long())],
            dim=-1)
        token_repr_pos = torch.cat(
            [token_repr_pos, self.endofword_embedding((pos_ids != 0).long())],
            dim=-1)

        token_repr_parse = self.input_projection_parse(token_repr_parse)
        token_repr_sup = self.input_projection_sup(token_repr_sup)
        token_repr_pos = self.input_projection_pos(token_repr_pos)

        if self.transformer_layers > 0:
            padding_mask = attention_mask == 0
            token_repr_parse = self.transformer(
                token_repr_parse,
                src_key_padding_mask=padding_mask
            )

        tag_logits = None
        if self.projection is not None:
            tag_logits = self.projection(self.dropout(token_repr_sup))

        pos_logits = None
        if self.pos_projection is not None:
            pos_logits = self.pos_projection(self.dropout(token_repr_pos))

        S_arc = None
        S_lab = None
        if self.biaffine is not None:
            S_arc, S_lab = self.biaffine(token_repr_parse.contiguous())

        loss = None
        pos_loss = None
        arc_loss = None
        label_loss = None
        if (
                labels is not None and (self.training or report_loss)
                and tag_logits is not None):
            loss = losses.calc_loss_helper(
                tag_logits, labels, attention_mask.bool(),
                printinfo=printinfo
            )
        if self.pos_projection is not None:
            assert pos_logits is not None
            pos_loss = losses.calc_loss_helper(
                pos_logits, pos_ids, attention_mask.bool(),
                printinfo=printinfo
            )
        if heads is not None:
            if self.biaffine is not None:
                if self.biaffine.arc_mlp_d is not None:
                    arc_loss = self.biaffine.arc_loss(
                        S_arc, heads, attention_mask.bool(),
                        printinfo=printinfo)
                    # print("arc_loss", arc_loss)
                if self.biaffine.lab_mlp_d is not None:
                    label_loss = self.biaffine.lab_loss(
                        S_lab, heads, deprel_ids, attention_mask.bool(),
                        printinfo=printinfo)

        return (
            loss, tag_logits, pos_loss, pos_logits,
            arc_loss, S_arc, label_loss, S_lab)
