import torch


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
