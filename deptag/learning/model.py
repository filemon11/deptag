import pathlib
import torch
from torch import nn
import bitsandbytes as bnb

from . import losses, biaffine

from transformers import AutoModel, BertModel

from typing import Literal


class LayerMix(nn.Module):
    # Proposed by https://aclanthology.org/D19-1279.pdf ?
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(
            torch.zeros(num_layers)
        )

    def forward(
            self,
            hidden_states: tuple[torch.Tensor, ...],
            ) -> torch.Tensor:
        if len(hidden_states) != len(self.weights) + 1:
            raise ValueError(
                f"Expected {len(self.weights) + 1} hidden states "
                f"(embedding + {len(self.weights)} layers), "
                f"got {len(hidden_states)}."
            )

        # omit hidden_states[0], the embedding layer
        hs = torch.stack(hidden_states[1:], dim=0)
        weights = torch.softmax(self.weights, dim=0)

        return torch.sum(
            weights[:, None, None, None] * hs,
            dim=0,
        )


class ModelForTagging(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_tags: int = config.num_labels
        self.model_path: pathlib.Path = config.task_specific_params[
            'model_path']
        self.use_pos: bool = config.task_specific_params['use_pos']
        self.train_sup: bool = config.task_specific_params['train_sup']
        self.num_pos_tags: int = config.task_specific_params['num_pos_tags']
        self.num_xpos_tags: int = config.task_specific_params['num_xpos_tags']

        self.num_feats_tags: dict[
            str, int] = config.task_specific_params['num_feats_tags']
        self.train_feats: bool = config.task_specific_params['train_feats']

        self.pos_emb_dim: int = config.task_specific_params['pos_emb_dim']
        self.dropout_rate: float = config.task_specific_params['dropout']

        self.transformer_layers = config.task_specific_params[
            "transformer_layers"]

        self.encoder: BertModel = AutoModel.from_pretrained(
            self.model_path, config=config)

        self.factorised: Literal[
            'structural', 'complete',
            'seen', False] = config.task_specific_params[
                'factorised']
        self.max_l: None | int = config.task_specific_params['max_l']
        self.max_r: None | int = config.task_specific_params['max_r']

        import transformers.utils.output_capturing as hf_output_capturing

        hf_output_capturing.torch = torch
        hf_output_capturing.maybe_install_capturing_hooks(self.encoder)

        # self.encoder: BertModel = torch.compile(self.encoder)

        if self.use_pos:
            self.pos_encoder = nn.Sequential(
                bnb.nn.StableEmbedding(
                    self.num_pos_tags, self.pos_emb_dim, padding_idx=0)
            )

        # self.endofword_embedding = bnb.nn.StableEmbedding(
        #   2, self.pos_emb_dim)

        transformer_input_dim = (
            config.hidden_size
            # + self.pos_emb_dim
            + (self.pos_emb_dim if self.use_pos else 0)
        )

        # self.pos_layer = config.task_specific_params["pos_layer"]
        # self.supertag_layer = config.task_specific_params["supertag_layer"]
        # self.parse_layer = config.task_specific_params["parse_layer"]
        self.pos_mix = None
        if config.task_specific_params["train_pos"]:
            self.pos_mix = LayerMix(config.num_hidden_layers)
        self.xpos_mix = None
        if config.task_specific_params["train_xpos"]:
            self.xpos_mix = LayerMix(config.num_hidden_layers)
        self.arc_mix = LayerMix(config.num_hidden_layers)
        self.rel_mix = None
        if config.task_specific_params["mlp_lab_hidden"] is not None:
            self.rel_mix = LayerMix(config.num_hidden_layers)

        self.feats_mixes = None
        if self.train_feats:
            self.feats_mixes = nn.ModuleDict({
                feat: LayerMix(config.num_hidden_layers)
                for feat in self.num_feats_tags.keys()
            })

        self.sup_mix = None
        self.sup_arg_mix = None
        self.sup_head_mix = None
        if self.train_sup:
            if self.factorised is not False:
                self.sup_arg_mix = LayerMix(config.num_hidden_layers)
                self.sup_head_mix = LayerMix(config.num_hidden_layers)
            else:
                self.sup_mix = LayerMix(config.num_hidden_layers)

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

        def get_sequential(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(transformer_input_dim, config.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(config.hidden_size, out_dim))

        self.dropout = nn.Dropout(self.dropout_rate)
        self.projection = None
        self.left_labels_projection = None
        self.right_labels_projections = None
        if self.train_sup:
            if self.factorised is not False:
                assert self.max_l is not None and self.max_r is not None
                self.l_num_projection = get_sequential(self.max_l + 1)
                self.r_num_projection = get_sequential(self.max_r + 1)
                if self.factorised in ("complete", "seen"):
                    self.left_labels_projections = nn.ModuleList([
                        get_sequential(
                            config.task_specific_params["sup_deprel_num"])
                        for _ in range(self.max_l)
                    ])

                    self.right_labels_projections = nn.ModuleList([
                        get_sequential(
                            config.task_specific_params["sup_deprel_num"])
                        for _ in range(self.max_r)
                    ])
                    self.aux_label_projection = get_sequential(
                        config.task_specific_params["sup_deprel_num"])

                self.aux_position_projection = get_sequential(
                    self.max_l + self.max_r + 3)
            else:
                self.projection = get_sequential(
                    config.num_labels)
        self.pos_projection = None
        if config.task_specific_params["train_pos"]:
            self.pos_projection = get_sequential(self.num_pos_tags)
        self.xpos_projection = None
        if config.task_specific_params["train_xpos"]:
            self.xpos_projection = get_sequential(self.num_xpos_tags)

        self.feats_projections = None
        if self.train_feats:
            self.feats_projections = nn.ModuleDict({
                feat: get_sequential(num)
                for feat, num in self.num_feats_tags.items()
            })

        self.biaffine = None
        self.root_arc = None
        self.root_rel = None
        if (
                config.task_specific_params["mlp_arc_hidden"] is not None
                or config.task_specific_params["mlp_lab_hidden"] is not None):
            self.biaffine = biaffine.make_model(
                transformer_input_dim,
                config.task_specific_params["mlp_arc_hidden"],
                config.task_specific_params["mlp_lab_hidden"],
                config.task_specific_params["mlp_dropout"],
                config.task_specific_params["mlp_num_labels"],
                config.task_specific_params[
                    "extra_num_labels"] if config.task_specific_params[
                        "train_subtypes"] else None
            )
            # self.biaffine.compile()  # (dynamic=True)
            self.root_arc = nn.Parameter(
                torch.empty(config.hidden_size)
            )
            self.root_rel = nn.Parameter(
                torch.empty(config.hidden_size)
            )
            nn.init.normal_(self.root_arc, std=0.02)
            nn.init.normal_(self.root_rel, std=0.02)

            self.extra_num_labels = config.task_specific_params[
                "extra_num_labels"]
            self.train_subtypes = config.task_specific_params[
                "train_subtypes"]
            self.pos_label_smoothing = config.task_specific_params[
                "pos_label_smoothing"]
            self.xpos_label_smoothing = config.task_specific_params[
                "xpos_label_smoothing"]
            self.arc_label_smoothing = config.task_specific_params[
                "arc_label_smoothing"]
            self.deprel_label_smoothing = config.task_specific_params[
                "deprel_label_smoothing"]
            self.sup_label_smoothing = config.task_specific_params[
                "sup_label_smoothing"]
            self.feats_label_smoothing = config.task_specific_params[
                "feats_label_smoothing"]
            self.subtypes_label_smoothing = config.task_specific_params[
                "subtypes_label_smoothing"]

    def forward(
            self,
            input_ids=None,
            pos_ids=None,
            xpos_ids=None,
            word_end_positions=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            heads=None,
            deprel_ids=None,
            output_attentions=None,
            l_arg_nums=None,
            r_arg_nums=None,
            aux_rel_ids=None,
            aux_positions=None,
            report_loss: bool = False,
            printinfo: bool = False,
            **kwargs,
    ):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        # num_layers = len(outputs["hidden_states"])-1
        # token_repr_parse = outputs["hidden_states"][self.parse_layer]
        token_repr_arc = self.arc_mix(outputs["hidden_states"])
        token_repr_rel = None
        if self.rel_mix is not None:
            token_repr_rel = self.rel_mix(outputs["hidden_states"])

        token_repr_sup = None
        token_repr_sup_arg = None
        token_repr_sup_head = None
        if self.train_sup:
            if self.factorised is not False:
                token_repr_sup_arg = self.sup_arg_mix(outputs["hidden_states"])
                token_repr_sup_head = self.sup_head_mix(outputs["hidden_states"])
            else:
                token_repr_sup = self.sup_mix(outputs["hidden_states"])
        token_repr_pos = None
        if self.pos_mix is not None:
            token_repr_pos = self.pos_mix(outputs["hidden_states"])
        token_repr_xpos = None
        if self.xpos_mix is not None:
            token_repr_xpos = self.xpos_mix(outputs["hidden_states"])
        # print(num_layers, round(num_layers*(2/3)), round(num_layers*(1/3)))

        word_repr_arc, word_mask = (
            self._gather_word_representations(
                token_repr_arc,
                word_end_positions,
            )
        )

        word_repr_rel = None
        if token_repr_rel is not None:
            word_repr_rel, _ = (
                self._gather_word_representations(
                    token_repr_rel,
                    word_end_positions,
                )
            )

        word_repr_sup = None
        word_repr_sup_arg = None
        word_repr_sup_head = None

        if self.train_sup:
            if self.factorised is not False:
                assert token_repr_sup_arg is not None
                assert token_repr_sup_head is not None
                word_repr_sup_arg, _ = (
                    self._gather_word_representations(
                        token_repr_sup_arg,
                        word_end_positions,
                    )
                )
                word_repr_sup_head, _ = (
                    self._gather_word_representations(
                        token_repr_sup_head,
                        word_end_positions,
                    )
                )
            else:
                assert token_repr_sup is not None
                word_repr_sup, _ = (
                    self._gather_word_representations(
                        token_repr_sup,
                        word_end_positions,
                    )
                )

        word_repr_pos = None
        if token_repr_pos is not None:
            word_repr_pos, _ = (
                self._gather_word_representations(
                    token_repr_pos,
                    word_end_positions,
                )
            )
        word_repr_xpos = None
        if token_repr_xpos is not None:
            word_repr_xpos, _ = (
                self._gather_word_representations(
                    token_repr_xpos,
                    word_end_positions,
                )
            )

        tag_logits = None
        factorised_logits = {}
        if self.train_sup:
            if self.projection is not None:
                tag_logits = self.projection(self.dropout(word_repr_sup))

            if self.factorised is not False:
                l_num_logits = self.l_num_projection(
                    self.dropout(word_repr_sup_arg))
                r_num_logits = self.r_num_projection(
                    self.dropout(word_repr_sup_arg))
                aux_position_logits = self.aux_position_projection(
                    self.dropout(word_repr_sup_head))

                factorised_logits = {
                    "l_arg_nums": l_num_logits,
                    "r_arg_nums": r_num_logits,
                    "aux_positions": aux_position_logits,
                }
                if self.factorised in ("complete", "seen"):
                    aux_label_logits = self.aux_label_projection(
                        self.dropout(word_repr_sup_head))
                    left_label_logits = [
                        projection(
                            self.dropout(word_repr_sup_arg)) for projection
                        in self.left_labels_projections
                    ]
                    right_label_logits = [
                        projection(
                            self.dropout(word_repr_sup_arg)) for projection
                        in self.right_labels_projections
                    ]
                    factorised_logits["aux_rel_ids"] = aux_label_logits
                    for i, logits in enumerate(left_label_logits):
                        factorised_logits[f"left_{i+1}"] = logits
                    for i, logits in enumerate(right_label_logits):
                        factorised_logits[f"right_{i+1}"] = logits

        pos_logits = None
        if self.pos_projection is not None:
            pos_logits = self.pos_projection(self.dropout(word_repr_pos))
        xpos_logits = None
        if self.xpos_projection is not None:
            xpos_logits = self.xpos_projection(self.dropout(word_repr_xpos))

        S_arc = None
        S_lab = None
        S_extra_lab = {}
        if self.biaffine is not None:
            root_arc = self.root_arc[None, None, :].expand(
                word_repr_arc.shape[0], 1, -1
            )
            root_rel = self.root_rel[None, None, :].expand(
                word_repr_arc.shape[0], 1, -1
            )

            parse_repr_arc = torch.cat(
                [root_arc, word_repr_arc],
                dim=1,
            )

            parse_repr_rel = None
            if word_repr_rel is not None:
                parse_repr_rel = torch.cat(
                    [root_rel, word_repr_rel],
                    dim=1,
                )

            root_mask = torch.ones(
                (word_mask.shape[0], 1),
                dtype=torch.bool,
                device=word_mask.device,
            )

            parse_mask = torch.cat(
                [root_mask, word_mask],
                dim=1,
            )
            # [B, W + 1]

            root_heads = torch.full(
                (heads.shape[0], 1),
                -1,
                dtype=heads.dtype,
                device=heads.device,
            )

            heads = torch.cat(
                [root_heads, heads],
                dim=1,
            )

            deprel_ids = torch.cat(
                [
                    torch.full(
                        (deprel_ids.shape[0], 1),
                        -1,
                        dtype=deprel_ids.dtype,
                        device=deprel_ids.device,
                    ),
                    deprel_ids,
                ],
                dim=1,
            )

            S_arc, S_lab, S_extra_lab = self.biaffine(
                parse_repr_arc,
                parse_repr_rel,
            )

        loss = None
        pos_loss = None
        xpos_loss = None
        arc_loss = None
        label_loss = None
        extra_loss: dict[str, torch.Tensor] = {}
        factorised_losses: dict[str, torch.Tensor] = {}
        if (
                labels is not None and (self.training or report_loss)
                and tag_logits is not None):
            loss = losses.calc_loss_helper(
                tag_logits, labels,  # word_mask,
                label_smoothing=self.sup_label_smoothing,
                printinfo=printinfo
            )

        if len(factorised_logits) > 0:
            assert l_arg_nums is not None
            assert r_arg_nums is not None
            assert aux_positions is not None
            assert aux_rel_ids is not None
            l_num_loss = losses.calc_loss_helper(
                factorised_logits["l_arg_nums"], l_arg_nums,
                label_smoothing=self.sup_label_smoothing,
                printinfo=printinfo
                )
            r_num_loss = losses.calc_loss_helper(
                factorised_logits["r_arg_nums"], r_arg_nums,
                label_smoothing=self.sup_label_smoothing,
                printinfo=printinfo
                )
            aux_position_loss = losses.calc_loss_helper(
                factorised_logits["aux_positions"], aux_positions,
                label_smoothing=self.sup_label_smoothing,
                printinfo=printinfo
                )
            factorised_losses = {
                "l_arg_nums": l_num_loss,
                "r_arg_nums": r_num_loss,
                "aux_positions": aux_position_loss
            }
            if self.factorised in ("complete", "seen"):
                assert self.max_l is not None and self.max_r is not None
                # no_aux_index = math.floor((self.max_l+self.max_r+3)/3)-1
                factorised_losses["aux_rel_ids"] = losses.calc_loss_helper(
                    factorised_logits["aux_rel_ids"], aux_rel_ids,
                    label_smoothing=self.sup_label_smoothing,
                    # aux_labels != no_aux_index,
                    printinfo=printinfo
                    )

                for i, _ in enumerate(left_label_logits):
                    factorised_losses[
                        f"left_{i+1}"] = losses.calc_loss_helper(
                        factorised_logits[
                            f"left_{i+1}"], kwargs[f"left_{i+1}"],
                        label_smoothing=self.sup_label_smoothing,
                        # left_arg_num >= i+1,
                        printinfo=printinfo
                        )
                for i, logits in enumerate(
                        right_label_logits):
                    factorised_losses[
                        f"right_{i+1}"] = losses.calc_loss_helper(
                        factorised_logits[
                            f"right_{i+1}"], kwargs[f"right_{i+1}"],
                        label_smoothing=self.sup_label_smoothing,
                        # right_arg_num >= i+1,
                        printinfo=printinfo
                        )

        if self.pos_projection is not None:
            assert pos_logits is not None
            pos_loss = losses.calc_loss_helper(
                pos_logits, pos_ids,  # word_mask,
                label_smoothing=self.pos_label_smoothing,
                printinfo=printinfo
            )
        if self.xpos_projection is not None:
            assert xpos_logits is not None
            xpos_loss = losses.calc_loss_helper(
                xpos_logits, xpos_ids,  # word_mask,
                label_smoothing=self.xpos_label_smoothing,
                printinfo=printinfo
            )
        if heads is not None:
            if self.biaffine is not None:
                if self.biaffine.arc_mlp_d is not None:
                    arc_loss = self.biaffine.arc_loss(
                        S_arc,
                        heads,
                        parse_mask,
                        label_smoothing=self.arc_label_smoothing,
                        printinfo=printinfo,
                    )
                if S_lab is not None:
                    label_loss = self.biaffine.lab_loss(
                        S_lab, heads, deprel_ids,
                        label_smoothing=self.deprel_label_smoothing,
                        printinfo=printinfo)
                if len(S_extra_lab) > 0:
                    extra_loss = self.biaffine.extra_lab_loss(
                        S_extra_lab, heads, {
                            s_name: torch.cat(
                                [
                                    torch.full(
                                        (kwargs[s_name].shape[0], 1),
                                        -1,
                                        dtype=kwargs[s_name].dtype,
                                        device=kwargs[s_name].device,
                                    ),
                                    kwargs[s_name],
                                ],
                                dim=1,
                            )
                            for s_name in S_extra_lab.keys()},
                        label_smoothing=self.subtypes_label_smoothing,
                        printinfo=printinfo)

        feats_losses: dict[str, torch.Tensor] = {}
        feats_logits: dict[str, torch.Tensor] = {}
        if len(self.num_feats_tags) > 0 and self.train_feats:
            for feat in self.num_feats_tags.keys():

                assert self.feats_mixes is not None
                token_repr = self.feats_mixes[
                    feat](outputs["hidden_states"])

                word_repr = None
                word_repr, _ = (
                    self._gather_word_representations(
                        token_repr,
                        word_end_positions,
                    )
                )
                logits = self.feats_projections[feat](self.dropout(word_repr))
                feats_logits[feat] = logits

                f_loss = losses.calc_loss_helper(
                    logits, kwargs[feat],
                    label_smoothing=self.feats_label_smoothing,
                    printinfo=printinfo
                )
                feats_losses[feat] = f_loss

        return (
            loss, tag_logits, pos_loss, pos_logits,
            arc_loss, S_arc, label_loss, S_lab,
            factorised_losses, factorised_logits,
            xpos_loss, xpos_logits,
            feats_losses, feats_logits,
            extra_loss, S_extra_lab)

    @staticmethod
    def _gather_word_representations(
            token_repr: torch.Tensor,
            word_end_positions: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select one transformer representation per UD word.

        Parameters
        ----------
        token_repr
            [B, S_subword, H]
        word_end_positions
            [B, S_word], padded with -1.

        Returns
        -------
        word_repr
            [B, S_word, H]
        word_mask
            [B, S_word]
        """
        word_mask = word_end_positions >= 0

        # Padding positions are -1 and cannot be passed to gather.
        # Their actual gathered value is irrelevant because word_mask
        # excludes them everywhere downstream.
        safe_positions = word_end_positions.clamp_min(0)

        gather_index = safe_positions.unsqueeze(-1).expand(
            -1,
            -1,
            token_repr.shape[-1],
        )

        word_repr = torch.gather(
            token_repr,
            dim=1,
            index=gather_index,
        )

        # makes padded representations explicitly zero.
        word_repr = word_repr * word_mask.unsqueeze(-1)

        return word_repr, word_mask
