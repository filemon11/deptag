import torch
from torch import nn
from numpy import prod
import torch.nn.functional as F

# Source:
# https://github.com/daandouwe/biaffine-dependency-parser/blob/master/model.py
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
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.U = nn.Parameter(
            torch.empty(output_dim, input_dim, input_dim)
        )
        nn.init.xavier_uniform_(self.U)

    def forward(
        self,
        Rh: torch.Tensor,  # [B, S, D]
        Rd: torch.Tensor,  # [B, S, D]
    ) -> torch.Tensor:
        # [B, O, S_head, S_dependent]
        scores = torch.einsum(
            "bhi,oij,bdj->bohd",
            Rh,
            self.U,
            Rd,
        )

        if self.U.size(0) == 1:
            scores = scores.squeeze(1)

        return scores

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

    # @torch.compile()
    def forward(
            self, h_arc: torch.Tensor | None, h_lab: torch.Tensor | None,
            *args, **kwargs
            ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute the score matrices for the arcs and labels."""

        arc_h = None
        arc_d = None
        if self.arc_mlp_h is not None and self.arc_mlp_d is not None:
            assert h_arc is not None
            arc_h = self.arc_mlp_h(h_arc).contiguous()
            arc_d = self.arc_mlp_d(h_arc).contiguous()

        lab_h = None
        lab_d = None
        if self.lab_mlp_h is not None and self.lab_mlp_d is not None:
            assert h_lab is not None
            lab_h = self.lab_mlp_h(h_lab).contiguous()
            lab_d = self.lab_mlp_d(h_lab).contiguous()

        S_arc = None
        if self.arc_biaffine is not None:
            S_arc = self.arc_biaffine(arc_h, arc_d).contiguous()
        S_lab = None
        if lab_h is not None and lab_d is not None:
            assert self.lab_biaffine is not None
            S_lab = self.lab_biaffine(lab_h, lab_d).contiguous()

        return S_arc, S_lab

    def arc_loss(
            self,
            S_arc: torch.Tensor,
            heads: torch.Tensor,
            head_mask: torch.Tensor,
            label_smoothing: float = 0.0,
            printinfo: bool = False,
            ):
        """Compute arc loss.

        S_arc:
            [B, H, D], where S_arc[b, h, d] is the score
            of h being the head of d.

        heads:
            [B, D], containing gold head indices.
            The artificial root and padding positions contain -1.

        head_mask:
            [B, H], True for valid candidate heads.
            This includes the artificial root.
        """

        if label_smoothing > 0:
            masked_scores = S_arc.masked_fill(
                ~head_mask.unsqueeze(-1),
                float("-inf"),
            )

            log_probs = F.log_softmax(
                masked_scores,
                dim=1,
            )  # [B, H, D]

            valid_dep = heads != -1

            safe_heads = heads.clamp_min(0).long()

            # Standard NLL for the gold head.
            gold_logp = torch.gather(
                log_probs,
                dim=1,
                index=safe_heads.unsqueeze(1),
            ).squeeze(1)  # [B, D]

            nll = -gold_logp

            # Smoothing loss over VALID head classes only.
            #
            # Do not multiply -inf by 0, because that gives NaN.
            valid_log_probs = torch.where(
                head_mask.unsqueeze(-1),
                log_probs,
                0.0,
            )

            num_valid_heads = head_mask.sum(
                dim=1,
                keepdim=True,
            )  # [B, 1]

            smooth_loss = (
                -valid_log_probs.sum(dim=1)
                / num_valid_heads
            )  # [B, D]

            loss = (
                (1.0 - label_smoothing) * nll
                + label_smoothing * smooth_loss
            )

            return loss[valid_dep].mean()

        # Mask padded WORD positions as candidate heads.
        #
        # [B, H] -> [B, H, 1], broadcasting over dependents.
        masked_scores = S_arc.masked_fill(
            ~head_mask.unsqueeze(-1),
            float("-inf"),
        )

        return F.cross_entropy(
            masked_scores,
            heads.long(),
            ignore_index=-1,
            reduction="mean",
            label_smoothing=label_smoothing,
        )

    def lab_loss(
            self, S_lab, heads, labels,
            mask: torch.Tensor | None = None,
            label_smoothing: float = 0.0,
            printinfo: bool = False):
        """Compute the loss for the label predictions
        on the gold arcs (heads)."""

        safe_heads = heads.clamp_min(0).long()

        # [B, 1, 1, D]
        head_indices = safe_heads[:, None, None, :]

        # Broadcasting across the label dimension:
        # [B, L, H, D] -> [B, L, 1, D] -> [B, L, D]
        selected_scores = torch.take_along_dim(
            S_lab,
            head_indices,
            dim=2,
        ).squeeze(2)

        return F.cross_entropy(
            selected_scores,
            labels.long(),
            ignore_index=-1,
            reduction="mean",
            label_smoothing=label_smoothing
        )

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
