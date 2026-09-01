import pathlib
import torch
from torch import nn
import bitsandbytes as bnb
import transformers
import transformers.utils.output_capturing as hf_output_capturing

from . import losses, biaffine
from .. import settings

from transformers import AutoModel, BertModel

from typing import TypedDict, Mapping, Literal, NotRequired, overload

hf_output_capturing.torch = torch  # type: ignore


class LayerMix(nn.Module):
    # Proposed by https://aclanthology.org/D19-1279.pdf ?
    # Layer dropout by https://nejlt.ep.liu.se/article/view/4932
    def __init__(
            self, num_layers: int,
            layer_dropout: float = 0.1,
            /, one_extra_layer: bool = True,
            ):
        super().__init__()
        self.weights = nn.Parameter(
            torch.zeros(num_layers)
        )
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.layer_dropout = layer_dropout

        self.one_extra_layer: bool = one_extra_layer

    def forward(
            self,
            hidden_states: tuple[torch.Tensor, ...],
            ) -> torch.Tensor:

        if len(hidden_states) != self.weights.numel() + int(
                self.one_extra_layer):
            raise ValueError(
                f"Expected {self.weights.numel() + int(self.one_extra_layer)} "
                + "hidden states"
                + (
                    f" (embedding + {self.weights.numel()} layers), "
                    if self.one_extra_layer
                    else ", ")
                + f"got {len(hidden_states)}."
            )

        # [L, B, S, H]
        hs = torch.stack(hidden_states[int(self.one_extra_layer):], dim=0)

        logits: torch.Tensor = self.weights

        if self.training and self.layer_dropout > 0:
            keep = (
                torch.rand_like(logits) >= self.layer_dropout
            )

            # Avoid pathological case in which every layer is dropped.
            if not keep.any():
                keep[
                    torch.randint(
                        len(keep),
                        (1,),
                        device=keep.device,
                    )
                ] = True

            logits = logits.masked_fill(~keep, float("-inf"))

        # [L]
        weights = torch.softmax(logits, dim=0)

        # [B, S, H]
        return self.gamma * torch.sum(
            weights[:, None, None, None] * hs,
            dim=0,
        )


class Projection(nn.Module):
    def __init__(
            self, input_dim: int, hidden_dim: int, output_dim: int,
            dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.projection(x)


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


class MixedProjection(nn.Module):
    def __init__(
            self,
            num_layers: int,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            mix_drop: float = 0.0,
            proj_drop: float = 0.0,
            /, one_extra_layer: bool = True) -> None:
        super().__init__()

        self.mix: LayerMix = LayerMix(
            num_layers, mix_drop, one_extra_layer=one_extra_layer)
        self.proj: Projection = Projection(
            input_dim, hidden_dim, output_dim, proj_drop)

    def forward(
            self, hidden_states: tuple[torch.Tensor, ...],
            word_end_positions: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
        token_repr = self.mix(hidden_states)
        word_repr, word_mask = _gather_word_representations(
            token_repr, word_end_positions)
        word_repr = self.proj(word_repr)
        return word_repr, word_mask


class FactorisedLogits(TypedDict):
    l_arg_nums: torch.Tensor
    r_arg_nums: torch.Tensor
    aux_positions: torch.Tensor
    aux_rel_ids: NotRequired[torch.Tensor]
    left_1: NotRequired[torch.Tensor]
    left_2: NotRequired[torch.Tensor]
    left_3: NotRequired[torch.Tensor]
    left_4: NotRequired[torch.Tensor]
    right_1: NotRequired[torch.Tensor]
    right_2: NotRequired[torch.Tensor]
    right_3: NotRequired[torch.Tensor]
    right_4: NotRequired[torch.Tensor]
    # Can have more


class FactorisedMixedProjection(nn.Module):
    def __init__(
            self,
            factorised: Literal["structural", "complete", "seen"],
            num_layers: int,
            input_dim: int,
            hidden_dim: int,
            label_num: int,
            max_l: int,
            max_r: int,
            mix_drop: float = 0.0,
            proj_drop: float = 0.0,
            /, one_extra_layer: bool = True) -> None:
        super().__init__()

        self.factorised = factorised

        self.sup_arg_mix = LayerMix(
            num_layers, mix_drop, one_extra_layer=one_extra_layer)
        self.sup_head_mix = LayerMix(
            num_layers, mix_drop, one_extra_layer=one_extra_layer)

        def get_projection(output_dim: int) -> nn.Module:
            return Projection(
                input_dim,
                hidden_dim,
                output_dim,
                proj_drop)

        self.l_num_projection = get_projection(max_l + 1)
        self.r_num_projection = get_projection(max_r + 1)
        if self.factorised in ("complete", "seen"):
            self.left_labels_projections = nn.ModuleList([
                get_projection(
                    label_num)
                for _ in range(max_l)
            ])

            self.right_labels_projections = nn.ModuleList([
                get_projection(
                    label_num)
                for _ in range(max_r)
            ])
            self.aux_label_projection = get_projection(
                label_num)

        self.aux_position_projection = get_projection(
            max_l + max_r + 3)

    def forward(
            self, hidden_states: tuple[torch.Tensor, ...],
            word_end_positions: torch.Tensor
            ) -> tuple[FactorisedLogits, torch.Tensor]:
        token_arg_repr = self.sup_arg_mix(hidden_states)
        token_head_repr = self.sup_head_mix(hidden_states)

        word_repr_arg, word_mask = (
            _gather_word_representations(
                token_arg_repr,
                word_end_positions,
            )
        )
        word_repr_head, _ = (
            _gather_word_representations(
                token_head_repr,
                word_end_positions,
            )
        )

        l_num_logits = self.l_num_projection(word_repr_arg)
        r_num_logits = self.r_num_projection(word_repr_arg)
        aux_position_logits = self.aux_position_projection(word_repr_head)

        factorised_logits = FactorisedLogits(
            l_arg_nums=l_num_logits,
            r_arg_nums=r_num_logits,
            aux_positions=aux_position_logits,
        )
        if self.factorised in ("complete", "seen"):
            aux_label_logits = self.aux_label_projection(word_repr_head)
            left_label_logits = [
                projection(word_repr_arg) for projection
                in self.left_labels_projections
            ]
            right_label_logits = [
                projection(word_repr_arg) for projection
                in self.right_labels_projections
            ]
            factorised_logits["aux_rel_ids"] = aux_label_logits
            for i, logits in enumerate(left_label_logits):
                factorised_logits[f"left_{i+1}"] = logits  # type: ignore
            for i, logits in enumerate(right_label_logits):
                factorised_logits[f"right_{i+1}"] = logits  # type: ignore

        return factorised_logits, word_mask


class MixedBiaffine(nn.Module):
    def __init__(
            self,
            num_layers: int,
            input_dim: int,
            hidden_dim: int,
            arc_hidden: int,
            label_hidden: int,
            label_num: int,
            train_arc: bool,
            train_label: bool,
            train_feats: bool = False,
            extra_feats_num: Mapping[str, int] = {},
            mix_drop: float = 0.0,
            proj_drop: float = 0.0,
            /, one_extra_layer: bool = True) -> None:
        super().__init__()
        self.arc_mix = None
        self.label_mix = None
        if train_arc:
            self.arc_mix = LayerMix(
                num_layers, mix_drop,
                one_extra_layer=one_extra_layer)

        if train_label:
            self.rel_mix = LayerMix(
                num_layers, mix_drop,
                one_extra_layer=one_extra_layer)

        self.feats_mixes = None
        if train_feats:
            self.feats_mixes = nn.ModuleDict({
                feat: LayerMix(hidden_dim)
                for feat in extra_feats_num.keys()
            })

        # Biaffine model
        self.biaffine = biaffine.make_model(
            input_dim,
            arc_hidden,
            label_hidden if train_label else None,
            proj_drop,
            label_num if train_label else None,
            extra_feats_num if train_feats else None,
            single=False,
        )

        self.root_arc = nn.Parameter(
            torch.empty(hidden_dim)
        )
        self.root_rel = nn.Parameter(
            torch.empty(hidden_dim)
        )
        nn.init.normal_(self.root_arc, std=0.02)
        nn.init.normal_(self.root_rel, std=0.02)

    def forward(
            self, hidden_states: tuple[torch.Tensor, ...],
            word_end_positions: torch.Tensor
            ) -> tuple[
                torch.Tensor, torch.Tensor, dict[str, torch.Tensor],
                torch.Tensor]:
        token_repr_arc = None
        token_repr_rel = None
        if self.arc_mix is not None:
            token_repr_arc = self.arc_mix(hidden_states)
        if self.rel_mix is not None:
            token_repr_rel = self.rel_mix(hidden_states)

        word_repr_rel = None
        word_repr_arc = None
        if token_repr_arc is not None:
            word_repr_arc, word_mask = (
                _gather_word_representations(
                    token_repr_arc,
                    word_end_positions,
                )
            )

        word_repr_rel = None
        if token_repr_rel is not None:
            word_repr_rel, _ = (
                _gather_word_representations(
                    token_repr_rel,
                    word_end_positions,
                )
            )

        S_arc = None
        S_lab = None
        S_extra_lab = {}

        root_arc = None
        parse_repr_arc = None
        if word_repr_arc is not None:
            root_arc = self.root_arc[None, None, :].expand(
                word_repr_arc.shape[0], 1, -1
            )
            parse_repr_arc = torch.cat(
                            [root_arc, word_repr_arc],
                            dim=1,
                        )

        root_rel = None
        parse_repr_rel = None
        if word_repr_rel is not None:
            root_rel = self.root_rel[None, None, :].expand(
                word_repr_rel.shape[0], 1, -1
            )
            parse_repr_rel = torch.cat(
                [root_rel, word_repr_rel],
                dim=1,
            )

        S_arc, S_lab, S_extra_lab = self.biaffine(
            parse_repr_arc,
            parse_repr_rel,
        )
        return S_arc, S_lab, S_extra_lab, word_mask


class TaskSpecificParams(TypedDict):
    model_path: pathlib.Path
    encoder_hidden_size: int
    encoder_num_layers: int
    encoder_num_attention_heads: int
    pos_emb_dim: int
    num_pos_tags: int
    num_xpos_tags: int
    extra_num_labels: Mapping[str, int]
    train_subtypes: bool
    dropout: float
    use_pos: bool
    n_heads: int
    transformer_layers: int
    pos_layer: int
    supertag_layer: int
    parse_layer: int
    train_arc: bool
    train_pos: bool
    train_xpos: bool
    mlp_arc_hidden: int
    mlp_lab_hidden: int
    mlp_drop: float
    mlp_num_labels: int
    deprel_num: int
    sup_deprel_num: int
    train_sup: bool
    factorised: settings.Factorised
    max_l: int
    max_r: int
    num_feats_tags: Mapping[str, int]
    train_feats: bool
    pos_label_smoothing: float
    xpos_label_smoothing: float
    arc_label_smoothing: float
    deprel_label_smoothing: float
    sup_label_smoothing: float
    feats_label_smoothing: float
    subtypes_label_smoothing: float
    proj_drop: float
    mix_drop: float


class AutoConfig(transformers.AutoConfig):
    task_specific_params: TaskSpecificParams
    num_labels: int
    hidden_size: int
    num_hidden_layers: int


class TaggingLosses(TypedDict):
    sup: None | torch.Tensor
    pos: None | torch.Tensor
    arc: None | torch.Tensor
    deprel: None | torch.Tensor
    factorised: Mapping[str, torch.Tensor]
    xpos: None | torch.Tensor
    feats: Mapping[str, torch.Tensor]
    subtypes: Mapping[str, torch.Tensor]


class TaggingLogits(TypedDict):
    sup: None | torch.Tensor
    pos: None | torch.Tensor
    S_arc: None | torch.Tensor
    S_lab: None | torch.Tensor
    factorised: Mapping[str, torch.Tensor]
    xpos: None | torch.Tensor
    feats: Mapping[str, torch.Tensor]
    S_extra_lab: Mapping[str, torch.Tensor]


class ModelForTagging(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.num_sup_tags: int = config.num_labels
        self.model_path: pathlib.Path = config.task_specific_params[
            'model_path']
        self.use_pos: bool = config.task_specific_params['use_pos']
        self.train_sup: bool = config.task_specific_params['train_sup']
        self.num_pos_tags: int = config.task_specific_params['num_pos_tags']
        self.num_xpos_tags: int = config.task_specific_params['num_xpos_tags']

        self.num_feats_tags: Mapping[
            str, int] = config.task_specific_params['num_feats_tags']
        self.train_feats: bool = config.task_specific_params['train_feats']

        self.pos_emb_dim: int = config.task_specific_params['pos_emb_dim']
        self.dropout_rate: float = config.task_specific_params['dropout']

        self.transformer_layers = config.task_specific_params[
            "transformer_layers"]
        self.factorised: settings.Factorised = config.task_specific_params[
                'factorised']

        self.max_l: None | int = config.task_specific_params['max_l']
        self.max_r: None | int = config.task_specific_params['max_r']

        # Modules
        # # Encoder
        self.encoder: BertModel = AutoModel.from_pretrained(
            self.model_path, config=config)

        hf_output_capturing.maybe_install_capturing_hooks(self.encoder)

        # if self.use_pos:
        #     self.pos_encoder = nn.Sequential(
        #         bnb.nn.StableEmbedding(
        #             self.num_pos_tags, self.pos_emb_dim, padding_idx=0)
        #     )

        # if self.transformer_layers > 0:
        #     encoder_layer = nn.TransformerEncoderLayer(
        #         d_model=config.hidden_size,
        #         nhead=config.task_specific_params["n_heads"],
        #         dim_feedforward=4*config.hidden_size,
        #         dropout=self.dropout_rate,
        #         activation="gelu",
        #         batch_first=True,   # input/output: (batch, seq, feature)
        #         norm_first=True,
        #     )

        #     self.transformer = nn.TransformerEncoder(
        #         encoder_layer,
        #         num_layers=config.task_specific_params["transformer_layers"],
        #     )

        transformer_input_dim = (
            config.hidden_size
            # + (self.pos_emb_dim if self.use_pos else 0)
        )

        # # Layer mixers
        @overload
        def get_mix_proj(
                out_dim: int, condition: Literal[True] = True) -> nn.Module:
            ...

        @overload
        def get_mix_proj(
                out_dim: int, condition: Literal[False]) -> None:
            ...

        @overload
        def get_mix_proj(
                out_dim: int, condition: bool) -> nn.Module | None:
            ...

        def get_mix_proj(
                out_dim: int, condition: bool = True
                ) -> nn.Module | None:
            if not condition:
                return None
            return MixedProjection(
                config.num_hidden_layers,
                transformer_input_dim,
                config.hidden_size,
                out_dim,
                config.task_specific_params["mix_drop"],
                config.task_specific_params["proj_drop"])

        self.pos_mix_proj = get_mix_proj(
            self.num_pos_tags,
            config.task_specific_params["train_pos"],
            )
        self.xpos_mix_proj = get_mix_proj(
            self.num_xpos_tags,
            config.task_specific_params["train_xpos"],
        )
        self.feats_mixes_proj = nn.ModuleDict({
            feat: get_mix_proj(num)
            for feat, num in self.num_feats_tags.items()
            if self.train_feats
        })

        self.sup_mix_proj = None
        self.factorised_mix_proj = None
        if self.train_sup:
            if self.factorised is not False:
                self.factorised_mix_proj = FactorisedMixedProjection(
                    self.factorised, config.num_hidden_layers,
                    transformer_input_dim, config.hidden_size,
                    config.task_specific_params["deprel_num"],
                    self.max_l, self.max_r,
                    config.task_specific_params["mix_drop"],
                    config.task_specific_params["proj_drop"],
                )
            else:
                self.sup_mix_proj = get_mix_proj(self.num_sup_tags)

        # # Biaffine model
        self.biaffine_mix = MixedBiaffine(
            config.num_hidden_layers,
            transformer_input_dim,
            config.hidden_size,
            config.task_specific_params["mlp_arc_hidden"],
            config.task_specific_params["mlp_lab_hidden"],
            config.task_specific_params["deprel_num"],
            config.task_specific_params["train_arc"],
            config.task_specific_params["mlp_arc_hidden"] is not None,
            config.task_specific_params["train_subtypes"],
            config.task_specific_params["extra_num_labels"],
            config.task_specific_params["mix_drop"],
            config.task_specific_params["mlp_drop"],
        )

        # Label smoothing
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
    ) -> tuple[TaggingLogits, torch.Tensor | None]:
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        hidden_states = outputs["hidden_states"]

        sup_logits = None
        factorised_logits: None | FactorisedLogits = None
        if self.factorised_mix_proj is not None:
            factorised_logits, _ = self.factorised_mix_proj(
                hidden_states, word_end_positions)
        if self.sup_mix_proj is not None:
            sup_logits, _ = self.sup_mix_proj(
                hidden_states, word_end_positions)

        pos_logits = None
        if self.pos_mix_proj is not None:
            pos_logits, _ = self.pos_mix_proj(
                hidden_states, word_end_positions)
        xpos_logits = None
        if self.xpos_mix_proj is not None:
            xpos_logits, _ = self.xpos_mix_proj(
                hidden_states, word_end_positions)

        feats_logits: dict[str, torch.Tensor] = {
            feat: proj(hidden_states, word_end_positions)[0]
            for feat, proj in self.feats_mixes_proj.items()
        }

        S_arc, S_lab, S_extra_lab, word_mask = self.biaffine_mix(
            hidden_states, word_end_positions
        )

        return TaggingLogits(
            sup=sup_logits,
            pos=pos_logits,
            S_arc=S_arc,
            S_lab=S_lab,
            factorised=(
                dict(factorised_logits)  # type: ignore
                if factorised_logits is not None
                else {}),
            xpos=xpos_logits,
            feats=feats_logits,
            S_extra_lab=S_extra_lab,
        ), word_mask

    def calc_losses(
            self,
            logits: TaggingLogits,
            word_mask: torch.Tensor | None,
            pos_ids=None,
            xpos_ids=None,
            labels=None,
            heads=None,
            deprel_ids=None,
            l_arg_nums=None,
            r_arg_nums=None,
            aux_rel_ids=None,
            aux_positions=None,
            report_loss: bool = False,
            printinfo: bool = False,
            **kwargs,) -> TaggingLosses:
        sup_loss = None
        pos_loss = None
        xpos_loss = None
        arc_loss = None
        label_loss = None
        extra_losses: dict[str, torch.Tensor] = {}
        factorised_losses: dict[str, torch.Tensor] = {}
        if (
                labels is not None and (self.training or report_loss)
                and logits["sup"] is not None):
            sup_loss = losses.calc_loss_helper(
                logits["sup"], labels,  # word_mask,
                label_smoothing=self.sup_label_smoothing,
                printinfo=printinfo
            )

        if logits["factorised"] is not None and len(logits["factorised"]) > 0:
            assert l_arg_nums is not None
            assert r_arg_nums is not None
            assert aux_positions is not None
            assert aux_rel_ids is not None
            l_num_loss = losses.calc_loss_helper(
                logits["factorised"]["l_arg_nums"], l_arg_nums,
                label_smoothing=self.sup_label_smoothing,
                printinfo=printinfo
                )
            r_num_loss = losses.calc_loss_helper(
                logits["factorised"]["r_arg_nums"], r_arg_nums,
                label_smoothing=self.sup_label_smoothing,
                printinfo=printinfo
                )
            aux_position_loss = losses.calc_loss_helper(
                logits["factorised"]["aux_positions"], aux_positions,
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
                    logits["factorised"]["aux_rel_ids"], aux_rel_ids,
                    label_smoothing=self.sup_label_smoothing,
                    # aux_labels != no_aux_index,
                    printinfo=printinfo
                    )

                for i in range(self.max_l):
                    factorised_losses[
                        f"left_{i+1}"] = losses.calc_loss_helper(
                            logits["factorised"][
                                f"left_{i+1}"],  # type: ignore
                            kwargs[f"left_{i+1}"],
                        label_smoothing=self.sup_label_smoothing,
                        printinfo=printinfo
                        )
                for i in range(self.max_r):
                    factorised_losses[
                        f"right_{i+1}"] = losses.calc_loss_helper(
                            logits["factorised"][
                                f"right_{i+1}"],  # type: ignore
                            kwargs[f"right_{i+1}"],
                        label_smoothing=self.sup_label_smoothing,
                        printinfo=printinfo
                        )

        if logits["pos"] is not None:
            pos_loss = losses.calc_loss_helper(
                logits["pos"], pos_ids,  # word_mask,
                label_smoothing=self.pos_label_smoothing,
                printinfo=printinfo
            )
        if logits["xpos"] is not None:
            xpos_loss = losses.calc_loss_helper(
                logits["xpos"], xpos_ids,  # word_mask,
                label_smoothing=self.xpos_label_smoothing,
                printinfo=printinfo
            )
        if heads is not None:
            if self.biaffine_mix is not None:
                assert word_mask is not None

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
                if logits["S_arc"] is not None:
                    arc_loss = self.biaffine_mix.biaffine.arc_loss(
                        logits["S_arc"],
                        heads,
                        parse_mask,
                        label_smoothing=self.arc_label_smoothing,
                        printinfo=printinfo,
                    )
                if logits["S_lab"] is not None:
                    label_loss = self.biaffine_mix.biaffine.lab_loss(
                        logits["S_lab"], heads, deprel_ids,
                        label_smoothing=self.deprel_label_smoothing,
                        printinfo=printinfo)
                if len(logits["S_extra_lab"]) > 0:
                    extra_losses = self.biaffine_mix.biaffine.extra_lab_loss(
                        logits["S_extra_lab"], heads, {
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
                            for s_name in logits["S_extra_lab"].keys()},
                        label_smoothing=self.subtypes_label_smoothing,
                        printinfo=printinfo)

        feats_losses = dict()
        if len(logits["feats"]) > 0 and self.train_feats:
            for feat, logits_ in logits["feats"].items():
                f_loss = losses.calc_loss_helper(
                    logits_, kwargs[feat],
                    label_smoothing=self.feats_label_smoothing,
                    printinfo=printinfo
                )
                feats_losses[feat] = f_loss

        return TaggingLosses(
            sup=sup_loss,
            pos=pos_loss,
            arc=arc_loss,
            deprel=label_loss,
            xpos=xpos_loss,
            factorised=factorised_losses,
            feats=feats_losses,
            subtypes=extra_losses
        )
