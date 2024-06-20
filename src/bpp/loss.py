from torch import Tensor
from torch.nn import Module


def focal_tversky_loss(
    p: Tensor,
    t: Tensor,
    w: Tensor | None = None,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 4 / 3,
    smooth: float = 1.0,
) -> Tensor:
    """Focal Tversky Loss.

    Arguments:
        p: Predicted probabilities.
        t: Binary targets.
        w: Sample weights.
        alpha: False-positive weight.
        beta: False-negative weight.
        gamma: Focal scaling exponent.
        smooth: Smoothing coefficient.

    Returns:
        Loss between `p` and `t`.
    """

    if w is None:
        tp = p.flatten() @ t.flatten()
        ts = t.sum()
        ps = p.sum()
    else:
        tp = (w * p * t).sum()
        ts = (w * t).sum()
        ps = (w * p).sum()

    tversky = (tp + smooth) / (
        (1 - alpha - beta) * tp + alpha * ts + beta * ps + smooth
    )
    loss = (1 - tversky) ** gamma

    return loss


class FocalTverskyLoss(Module):
    """Focal Tversky Loss."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 4 / 3,
        smooth: float = 1,
    ) -> None:
        """
        Arguments:
            alpha: False-positive weight.
            beta: False-negative weight.
            gamma: Focal scaling exponent.
            smooth: Smoothing coefficient.
        """

        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(
        self,
        p: Tensor,
        t: Tensor,
        w: Tensor | None = None,
    ) -> Tensor:
        """Compute loss.

        Arguments:
            p: Predicted probabilities.
            t: Binary targets.
            w: Sample weights.

        Returns:
            Loss between `p` and `t`.
        """

        return focal_tversky_loss(
            p,
            t,
            w,
            self.alpha,
            self.beta,
            self.gamma,
            self.smooth,
        )
