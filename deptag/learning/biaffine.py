import torch
from torch import nn
from numpy import prod

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
        nn.init.xavier_uniform(self.U)

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
                 mlp_input: int, mlp_arc_hidden: int,
                 mlp_lab_hidden: int | None, mlp_dropout: float,
                 num_labels: int | None):
        super(BiAffineParser, self).__init__()

        # Arc MLPs
        self.arc_mlp_h = MLP(mlp_input, mlp_arc_hidden, 2, 'ReLU', mlp_dropout)
        self.arc_mlp_d = MLP(mlp_input, mlp_arc_hidden, 2, 'ReLU', mlp_dropout)
        # Label MLPs
        self.lab_mlp_h = None
        self.lab_mlp_d = None
        if mlp_lab_hidden is not None:
            self.lab_mlp_h = MLP(
                mlp_input, mlp_lab_hidden, 2, 'ReLU', mlp_dropout)
            self.lab_mlp_d = MLP(
                mlp_input, mlp_lab_hidden, 2, 'ReLU', mlp_dropout)

        # BiAffine layers
        self.arc_biaffine = BiAffine(mlp_arc_hidden, 1)
        self.lab_biaffine = None
        if mlp_lab_hidden is not None:
            assert num_labels is not None
            self.lab_biaffine = BiAffine(mlp_lab_hidden, num_labels)

    def forward(
            self, h: torch.Tensor, *args, **kwargs
            ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute the score matrices for the arcs and labels."""

        arc_h = self.arc_mlp_h(h)
        arc_d = self.arc_mlp_d(h)

        lab_h = None
        lab_d = None
        if self.lab_mlp_h is not None and self.lab_mlp_d is not None:
            lab_h = self.lab_mlp_h(h)
            lab_d = self.lab_mlp_d(h)

        S_arc = self.arc_biaffine(arc_h, arc_d)
        S_lab = None
        if lab_h is not None and lab_d is not None:
            assert self.lab_biaffine is not None
            S_lab = self.lab_biaffine(lab_h, lab_d)
        return S_arc, S_lab

    def arc_loss(
            self, S_arc, heads, attention_mask: torch.Tensor,
            printinfo: bool = False):
        """Compute the loss for the arc predictions."""
        S_arc = S_arc.transpose(-1, -2)
        # [batch, sent_len, sent_len]

        # S_arc = S_arc.contiguous().view(-1, S_arc.size(-1))
        # # [batch*sent_len, sent_len]

        # heads = heads.view(-1)
        # # [batch*sent_len]
        return losses.calc_loss_helper(
            S_arc, heads, attention_mask.bool(),
            printinfo=printinfo)
        # return self.criterion(S_arc, heads)

    def lab_loss(
            self, S_lab, heads, labels,
            attention_mask: torch.Tensor,
            printinfo: bool = False):
        """Compute the loss for the label predictions
        on the gold arcs (heads)."""

        heads = heads.unsqueeze(1).unsqueeze(2)
        # [batch, 1, 1, sent_len]

        heads = heads.expand(-1, S_lab.size(1), -1, -1)
        # [batch, n_labels, 1, sent_len]

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
            nn.init.xavier_uniform(p)

    return model
