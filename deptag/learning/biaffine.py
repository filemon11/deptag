import torch
from torch import nn
from numpy import prod
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from . import losses

from typing import Self

# Source: https://github.com/daandouwe/biaffine-dependency-parser/blob/master/model.py
# changes made


class MLP(nn.Module):
    """Module for an MLP with dropout."""
    def __init__(self, input_size, layer_size, depth, activation, dropout):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        act_fn = getattr(nn, activation)
        for i in range(depth):
            self.layers.add_module('fc_{}'.format(i),
                                   nn.Linear(input_size, layer_size))
            if activation:
                self.layers.add_module('{}_{}'.format(activation, i),
                                       act_fn())
            if dropout:
                self.layers.add_module('dropout_{}'.format(i),
                                       nn.Dropout(dropout))
            input_size = layer_size

    def forward(self, x):
        return self.layers(x)

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


class BiAffine(nn.Module):
    """Biaffine attention layer."""
    def __init__(self, input_dim, output_dim):
        super(BiAffine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.FloatTensor(
            output_dim, input_dim, input_dim))
        nn.init.xavier_uniform_(self.U)

    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1)

    # TODO: add collumns of ones to Rh and Rd for biases.

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


class BiAffineParser(nn.Module):
    """Biaffine Dependency Parser."""
    def __init__(self,
                 mlp_input: int, mlp_arc_hidden: int | None,
                 mlp_lab_hidden: int | None, mlp_dropout: float,
                 num_labels: int | None):
        super(BiAffineParser, self).__init__()

        # Arc MLPs
        self.arc_mlp_h = None
        self.arc_mlp_d = None
        if mlp_arc_hidden is not None:
            self.arc_mlp_h = MLP(
                mlp_input, mlp_arc_hidden, 2, 'ReLU', mlp_dropout)
            self.arc_mlp_d = MLP(
                mlp_input, mlp_arc_hidden, 2, 'ReLU', mlp_dropout)
        # Label MLPs
        self.lab_mlp_h = None
        self.lab_mlp_d = None
        if mlp_lab_hidden is not None:
            self.lab_mlp_h = MLP(
                mlp_input, mlp_lab_hidden, 2, 'ReLU', mlp_dropout)
            self.lab_mlp_d = MLP(
                mlp_input, mlp_lab_hidden, 2, 'ReLU', mlp_dropout)

        # BiAffine layers
        self.arc_biaffine = None
        if mlp_arc_hidden is not None:
            self.arc_biaffine = BiAffine(mlp_arc_hidden, 1)
        self.lab_biaffine = None
        if mlp_lab_hidden is not None:
            assert num_labels is not None
            self.lab_biaffine = BiAffine(mlp_lab_hidden, num_labels)

    def forward(
            self, h: torch.Tensor, *args, **kwargs
            ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute the score matrices for the arcs and labels."""

        arc_h = None
        arc_d = None
        if self.arc_mlp_h is not None and self.arc_mlp_d is not None:
            arc_h = self.arc_mlp_h(h)
            arc_d = self.arc_mlp_d(h)

        lab_h = None
        lab_d = None
        if self.lab_mlp_h is not None and self.lab_mlp_d is not None:
            lab_h = self.lab_mlp_h(h)
            lab_d = self.lab_mlp_d(h)

        S_arc = None
        if self.arc_biaffine is not None:
            S_arc = self.arc_biaffine(arc_h, arc_d)
        S_lab = None
        if lab_h is not None and lab_d is not None:
            assert self.lab_biaffine is not None
            S_lab = self.lab_biaffine(lab_h, lab_d)
        return S_arc, S_lab

    def arc_loss(
            self, S_arc: torch.Tensor, heads: torch.Tensor,
            attention_mask: torch.Tensor,
            printinfo: bool = False):
        """Compute the loss for the arc predictions."""
        # print(S_arc.isnan().any())
        # print(S_arc.isinf().any())
        batch_size, num_heads, num_dependents = S_arc.shape
        # S_arc
        # [batch, sent_len (head preds), sent_len]

        # S_arc[:, 1:][heads[:, 1:] == -1] = float("-inf")

        # Token positions that may be selected as heads.
        # Valid candidate heads: all real tokens plus artificial ROOT.
        # valid_dependent = heads.ne(-1)
        valid_head = heads.ne(-1).clone()
        valid_head[:, 0] = True

        # valid_targets = heads[valid_dependent]
        # if not torch.all((valid_targets >= 0) & (valid_targets < num_heads)):
        #     raise ValueError(
        #         f"Gold head outside [0, {num_heads - 1}]: "
        #         f"{valid_targets[(
        #             valid_targets < 0) | (valid_targets >= num_heads)]}"
        #     )

        # Check that no gold target is about to be masked.
        # gold_head_is_valid = valid_head.gather(
        #     dim=1,
        #     index=heads.clamp_min(0),
        # )
        # invalid_gold = valid_dependent & ~gold_head_is_valid
        # if invalid_gold.any():
        #     batch_idx, dependent_idx = invalid_gold.nonzero(as_tuple=True)
        #     raise ValueError(
        #         "Some non-padding dependents point "
        #         "to masked candidate heads:\n"
        #         f"batch indices: {batch_idx.tolist()}\n"
        #         f"dependent indices: {dependent_idx.tolist()}\n"
        #         f"gold heads: {heads[invalid_gold].tolist()}"
        #     )

        # Remove padded tokens from the softmax over candidate heads.
        masked_scores = S_arc.masked_fill(
            ~valid_head.unsqueeze(-1),  # [batch, candidate_head, 1]
            float("-inf"),
        )

        # Select only valid dependent positions.
        #
        # S_arc.transpose(1, 2):
        # [batch, dependent, candidate_head]
        # active_scores = masked_scores.transpose(1, 2)[valid_dependent]
        # active_targets = heads[valid_dependent].long()

        # if active_targets.numel() == 0:
        #     return S_arc.sum() * 0.0

        # Verify that the gold logits are finite.
        # gold_scores = active_scores.gather(
        #     dim=1,
        #     index=active_targets.unsqueeze(1),
        # ).squeeze(1)

        # if not torch.isfinite(gold_scores).all():
        #     bad = ~torch.isfinite(gold_scores)
        #     raise ValueError(
        #         "Some gold-head logits are non-finite before cross-entropy:\n"
        #         f"targets: {active_targets[bad].tolist()}\n"
        #         f"scores: {gold_scores[bad].tolist()}"
        #     )

        # heads_ignore = torch.zeros_like(heads)
        # heads_ignore[:, 1:] = heads[:, 1:] == -1
        # heads_ignore = heads_ignore.unsqueeze(1).tile((1, S_arc.shape[-2], 1))
        # set impossible likelihood to predicted heads that are masked

        # S_arc = S_arc.transpose(-1, -2)
        # [batch, sent_len, sent_len (head preds)]

        # S_arc = S_arc.contiguous().view(-1, S_arc.size(-1))
        # # [batch*sent_len, sent_len]

        # heads = heads.view(-1)
        # # [batch*sent_len]
        # return losses.calc_loss_helper(
        #     S_arc, heads, attention_mask.bool(),
        #     printinfo=printinfo)
        # return self.criterion(S_arc, heads)

        # shape: (batch_size, seq_len, num_tags)
        # -> (batch_size, num_tags, seq_len)
        # S_arc = torch.movedim(S_arc, -1, 1)

        # # Only keep active parts of the loss
        # active_labels = torch.where(
        #     attention_mask, labels, -1
        # )

        # print(S_arc.shape)
        loss = F.cross_entropy(
            masked_scores, heads, ignore_index=-1, reduction="mean")
        # print(loss.shape)
        # loss[:, 1:][heads[:, 1:] == -1] = 0
        # print(loss)
        # loss = loss.sum()/(heads.numel() - (heads[:, 1:] == -1).numel())
        # print(loss)
        return loss

    def lab_loss(
            self, S_lab, heads, labels,
            attention_mask: torch.Tensor,
            printinfo: bool = False):
        """Compute the loss for the label predictions
        on the gold arcs (heads)."""
        # ignore = heads == -1
        # ignore[:, 0] = False  # indices don't get shifted by BOS token
        # cumsum: torch.Tensor = torch.cumsum(ignore, dim=-1)  # [B, S]

        # lens_non_ignore = (~ignore).sum(-1)
        # aranges = pad_sequence([
        #     torch.arange(0, le, device=ignore.device)
        #     # does not need to be arange TODO: speed up
        #     for le in lens_non_ignore],
        #     batch_first=True, padding_value=-1)
        # aranges[aranges != -1] = cumsum[~ignore]

        # heads with index -1 are padding and are treated as
        # index 0 here (to be disregarded later)

        heads = heads + (heads < 0)

        # # shift indices to select the correct deprels
        # heads += torch.gather(aranges, 1, heads)

        heads = heads.unsqueeze(1).unsqueeze(2)
        # [batch, 1, 1, sent_len]
 
        heads = heads.expand(-1, S_lab.size(1), -1, -1)
        # [batch, n_labels, 1, sent_len]

        # S_lab: [batch, n_labels, sent_len (potential heads), sent_len]
        S_lab = torch.gather(S_lab, 2, heads).squeeze(2)
        # [batch, n_labels, sent_len]

        S_lab = S_lab.transpose(-1, -2)
        # [batch, sent_len, n_labels]

        # S_lab = S_lab.contiguous().view(-1, S_lab.size(-1))
        # # [batch*sent_len, n_labels]

        # labels = labels.view(-1)
        # # [batch*sent_len]

        return losses.calc_loss_helper(
            S_lab, labels, attention_mask.bool(),
            printinfo=printinfo)
        # return self.criterion(S_lab, labels)

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


def make_model(
        mlp_input: int,
        mlp_arc_hidden: int,
        mlp_lab_hidden: int | None,
        mlp_dropout: float,
        num_labels: int | None) -> BiAffineParser:
    """Initiliaze a the BiAffine parser according to the specs in args."""

    # Initialize the model.
    model = BiAffineParser(
        mlp_input,
        mlp_arc_hidden,
        mlp_lab_hidden,
        mlp_dropout,
        num_labels
    )

    # Initialize parameters with Glorot.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
