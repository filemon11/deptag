import torch

from dataclasses import dataclass, field


def pcgrad_corrections(
        primary_loss: torch.Tensor,
        aux_losses: dict[str, tuple[torch.Tensor, float]],
        shared_params: list[torch.nn.Parameter],
        scaler: torch.amp.GradScaler,
        grad_acc: int,
        global_loss_scale: float = 1.0,
        eps: float = 1e-12,
        ):
    """
    Project each auxiliary-task gradient independently against the
    primary-task gradient.

    aux_losses:
        {
            "SUP": (sup_loss, sup_weight),
            "POS": (pos_loss, pos_weight),
        }

    Returned corrections are AMP-scaled and can therefore be added
    directly to p.grad before scaler.unscale_(optimizer).
    """

    amp_scale = scaler.get_scale()

    # ------------------------------------------------------------
    # Primary gradient -- compute ONCE
    # ------------------------------------------------------------

    scaled_primary = scaler.scale(
        global_loss_scale
        * primary_loss
        / grad_acc
    )

    primary_grads = torch.autograd.grad(
        scaled_primary,
        shared_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    device = shared_params[0].device

    primary_norm_sq = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )

    # Compute diagnostics with AMP scaling removed, to avoid
    # unnecessarily large squared values.
    for g_p in primary_grads:
        if g_p is not None:
            g_p_unscaled = g_p.float() / amp_scale
            primary_norm_sq += g_p_unscaled.square().sum()

    # The correction for every auxiliary task is a scalar multiple
    # of g_primary, so we only need to accumulate the coefficients.
    total_coefficient = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )

    stats = {}

    # ------------------------------------------------------------
    # Auxiliary gradients
    # ------------------------------------------------------------

    for name, (aux_loss, aux_weight) in aux_losses.items():

        if not aux_loss.requires_grad:
            stats[name] = {
                "cosine": None,
                "coefficient": None,
                "aux_norm": None,
                "norm_ratio": None,
            }
            continue

        scaled_aux = scaler.scale(
            global_loss_scale
            * aux_weight
            * aux_loss
            / grad_acc
        )

        aux_grads = torch.autograd.grad(
            scaled_aux,
            shared_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        dot = torch.zeros(
            (),
            device=device,
            dtype=torch.float32,
        )
        aux_norm_sq = torch.zeros_like(dot)

        for g_p, g_a in zip(
                primary_grads,
                aux_grads,
                ):

            if g_a is not None:
                g_a_unscaled = g_a.float() / amp_scale
                aux_norm_sq += g_a_unscaled.square().sum()

            if g_p is not None and g_a is not None:
                g_p_unscaled = g_p.float() / amp_scale
                g_a_unscaled = g_a.float() / amp_scale

                dot += (
                    g_p_unscaled
                    * g_a_unscaled
                ).sum()

        primary_norm = primary_norm_sq.sqrt()
        aux_norm = aux_norm_sq.sqrt()

        cosine = dot / (
            primary_norm * aux_norm + eps
        )

        # PCGrad:
        #
        # g_aux' =
        #     g_aux
        #     - (g_aux . g_primary / ||g_primary||²) g_primary
        #
        # only if the dot product is negative.
        coefficient = (
            torch.minimum(
                dot,
                torch.zeros_like(dot),
            )
            / (primary_norm_sq + eps)
        )
        if not (
            torch.isfinite(cosine)
            and torch.isfinite(coefficient)
        ):
            stats[name] = {
                "cosine": None,
                "coefficient": None,
                "aux_norm": None,
                "norm_ratio": None,
            }
            continue

        total_coefficient += coefficient

        stats[name] = {
            "cosine": cosine,
            "coefficient": coefficient,
            "aux_norm": aux_norm,
            "norm_ratio": (
                aux_norm / (primary_norm + eps)
            ),
        }

    # ------------------------------------------------------------
    # Ordinary backward() will already add every g_aux.
    #
    # We therefore add only:
    #
    #   g_aux' - g_aux
    #
    # for each auxiliary task.
    #
    # Since all corrections are multiples of g_primary, their sum is
    #
    #   -sum(coefficients) * g_primary
    #
    # primary_grads are AMP-scaled, which is exactly what we want
    # because p.grad is also AMP-scaled at this point.
    # ------------------------------------------------------------

    corrections = []

    for g_p in primary_grads:
        if g_p is None:
            corrections.append(None)
        else:
            corrections.append(
                -total_coefficient * g_p
            )

    return corrections, stats


def pcgrad_correction(
        primary_loss: torch.Tensor,
        aux_loss: torch.Tensor,
        shared_params: list[torch.nn.Parameter],
        scaler: torch.amp.GradScaler,
        grad_acc: int,
        aux_weight: float = 1.0,
        eps: float = 1e-12,
        ):
    """
    Compute the correction required to remove the component of
    the auxiliary gradient that conflicts with the primary gradient.

    Returned gradients are AMP-scaled, so they can be added directly
    to p.grad before scaler.unscale_(optimizer).
    """

    # These have the same AMP scaling and grad-accumulation scaling
    # as the ordinary backward below.
    scaled_primary = scaler.scale(
        primary_loss / grad_acc
    )
    scaled_aux = scaler.scale(
        aux_weight * aux_loss / grad_acc
    )

    primary_grads = torch.autograd.grad(
        scaled_primary,
        shared_params,
        retain_graph=True,
        allow_unused=True,
    )

    aux_grads = torch.autograd.grad(
        scaled_aux,
        shared_params,
        retain_graph=True,
        allow_unused=True,
    )

    device = shared_params[0].device

    dot = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    primary_norm_sq = torch.zeros_like(dot)
    aux_norm_sq = torch.zeros_like(dot)

    for g_p, g_a in zip(primary_grads, aux_grads):
        if g_p is not None:
            primary_norm_sq += (
                g_p.float().square().sum()
            )

        if g_a is not None:
            aux_norm_sq += (
                g_a.float().square().sum()
            )

        if g_p is not None and g_a is not None:
            dot += (
                g_p.float() * g_a.float()
            ).sum()

    cosine = dot / (
        primary_norm_sq.sqrt()
        * aux_norm_sq.sqrt()
        + eps
    )

    # PCGrad:
    #
    # g_aux' =
    #     g_aux - (g_aux · g_primary / ||g_primary||²) g_primary
    #
    # but only when the dot product is negative.
    coefficient = (
        torch.minimum(
            dot,
            torch.zeros_like(dot),
        )
        / (primary_norm_sq + eps)
    )
    print("coefficient:", coefficient)

    # Since ordinary backward already adds g_aux, we only need
    # to add:
    #
    # g_aux' - g_aux
    # = -coefficient * g_primary
    corrections = []

    for g_p in primary_grads:
        if g_p is None:
            corrections.append(None)
        else:
            corrections.append(
                -coefficient * g_p
            )

    return corrections, cosine


@dataclass
class PCGradAccumulator:
    primary: list[torch.Tensor | None]
    auxiliaries: dict[
        str,
        list[torch.Tensor | None],
    ] = field(default_factory=dict)


def make_accumulator(
        shared_params: list[torch.nn.Parameter],
        ) -> PCGradAccumulator:

    return PCGradAccumulator(
        primary=[None] * len(shared_params),
    )


def _accumulate_grads(
        buffer: list[torch.Tensor | None],
        grads: tuple[torch.Tensor | None, ...],
        ) -> None:

    with torch.no_grad():
        for i, grad in enumerate(grads):
            if grad is None:
                continue

            grad = grad.detach()

            if buffer[i] is None:
                buffer[i] = grad.clone()
            else:
                buffer[i].add_(grad)


def accumulate_task_gradients(
        accumulator: PCGradAccumulator,
        primary_loss: torch.Tensor,
        aux_losses: dict[
            str,
            tuple[torch.Tensor, float],
        ],
        shared_params: list[torch.nn.Parameter],
        scaler: torch.amp.GradScaler,
        grad_acc: int,
        global_loss_scale: float = 1.0,
        ) -> None:
    """
    Accumulate the unprojected primary and auxiliary gradients.

    The gradients are:
      - AMP-scaled;
      - divided by grad_acc;
      - scaled exactly as in the ordinary loss.

    They are detached before being stored, so the computation graph
    itself is not retained across microbatches.
    """

    # ------------------------------------------------------------
    # Primary task
    # ------------------------------------------------------------

    scaled_primary = scaler.scale(
        global_loss_scale
        * primary_loss
        / grad_acc
    )

    primary_grads = torch.autograd.grad(
        scaled_primary,
        shared_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    _accumulate_grads(
        accumulator.primary,
        primary_grads,
    )

    # ------------------------------------------------------------
    # Auxiliary tasks
    # ------------------------------------------------------------

    for name, (aux_loss, aux_weight) in aux_losses.items():

        if (
            aux_loss is None
            or not aux_loss.requires_grad
        ):
            continue

        scaled_aux = scaler.scale(
            global_loss_scale
            * aux_weight
            * aux_loss
            / grad_acc
        )

        aux_grads = torch.autograd.grad(
            scaled_aux,
            shared_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        if name not in accumulator.auxiliaries:
            accumulator.auxiliaries[name] = [
                None
            ] * len(shared_params)

        _accumulate_grads(
            accumulator.auxiliaries[name],
            aux_grads,
        )


def compute_corrections(
        accumulator: PCGradAccumulator,
        scaler: torch.amp.GradScaler,
        eps: float = 1e-12,
        ) -> tuple[
            list[torch.Tensor | None],
            dict[str, dict[str, torch.Tensor | None]],
        ]:
    """
    Apply asymmetric PCGrad to the accumulated effective-batch
    gradients.

    Returned corrections are still AMP-scaled and can therefore be
    added directly to p.grad before scaler.unscale_(optimizer).
    """

    primary_grads = accumulator.primary

    if not any(
        grad is not None
        for grad in primary_grads
    ):
        return (
            [None] * len(primary_grads),
            {},
        )

    amp_scale = scaler.get_scale()

    corrections: list[torch.Tensor | None] = [
        None
    ] * len(primary_grads)

    stats = {}

    for name, aux_grads in (
            accumulator.auxiliaries.items()
            ):

        dot = None
        primary_norm_sq = None
        aux_norm_sq = None

        for g_p, g_a in zip(
                primary_grads,
                aux_grads,
                ):

            if g_p is None or g_a is None:
                continue

            # Remove AMP scaling for statistics / coefficient
            # calculation.
            g_p_unscaled = (
                g_p.float() / amp_scale
            )
            g_a_unscaled = (
                g_a.float() / amp_scale
            )

            current_dot = (
                g_p_unscaled
                * g_a_unscaled
            ).sum()

            current_p_norm = (
                g_p_unscaled.square().sum()
            )

            current_a_norm = (
                g_a_unscaled.square().sum()
            )

            if dot is None:
                dot = current_dot
                primary_norm_sq = current_p_norm
                aux_norm_sq = current_a_norm
            else:
                dot += current_dot
                primary_norm_sq += current_p_norm
                aux_norm_sq += current_a_norm

        # No common differentiable parameters for this task.
        if dot is None:
            stats[name] = {
                "cosine": None,
                "coefficient": None,
                "aux_norm": None,
                "norm_ratio": None,
            }
            continue

        assert primary_norm_sq is not None
        assert aux_norm_sq is not None

        primary_norm = primary_norm_sq.sqrt()
        aux_norm = aux_norm_sq.sqrt()

        if (
            primary_norm_sq <= eps
            or aux_norm_sq <= eps
        ):
            stats[name] = {
                "cosine": None,
                "coefficient": None,
                "aux_norm": aux_norm,
                "norm_ratio": None,
            }
            continue

        cosine = dot / (
            primary_norm * aux_norm + eps
        )

        coefficient = (
            torch.minimum(
                dot,
                torch.zeros_like(dot),
            )
            / (primary_norm_sq + eps)
        )

        if not (
            torch.isfinite(cosine)
            and torch.isfinite(coefficient)
        ):
            stats[name] = {
                "cosine": None,
                "coefficient": None,
                "aux_norm": None,
                "norm_ratio": None,
            }
            continue

        stats[name] = {
            "cosine": cosine,
            "coefficient": coefficient,
            "aux_norm": aux_norm,
            "norm_ratio": (
                aux_norm
                / (primary_norm + eps)
            ),
        }

        # No conflict -> no correction.
        if coefficient >= 0:
            continue

        # g_aux' - g_aux
        #
        # = -coefficient * g_primary
        #
        # Apply only to parameters that actually occur in both
        # gradients.
        with torch.no_grad():
            for i, (g_p, g_a) in enumerate(
                    zip(primary_grads, aux_grads)
                    ):

                if g_p is None or g_a is None:
                    continue

                correction = (
                    -coefficient * g_p
                )

                if corrections[i] is None:
                    corrections[i] = (
                        correction.clone()
                    )
                else:
                    corrections[i].add_(
                        correction
                    )

    return corrections, stats