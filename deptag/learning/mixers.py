import torch
import torch.nn as nn

from . import biaffine

from typing import TypedDict, Mapping, Literal, NotRequired


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
            arc_drop: float = 0.0,
            lab_drop: float = 0.0,
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
            arc_drop,
            lab_drop,
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
