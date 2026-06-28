"""Fixed-theta optimizer: fit once at the configured theta, with no search."""

from pydacefit.optimizers.base import Optimizer, fit_feasible


class Fixed(Optimizer):
    """No search: fit once at the model's current theta.

    The cheapest strategy -- a single solve. Pass it to ``refit`` to freeze theta
    when appending a few points that should not move it. The theta is kept as given
    (never relocated); if it is infeasible the model's ``max_noise`` ceiling decides
    whether to raise (max_noise=0) or climb the noise to the smallest amount that makes
    R positive-definite at that theta.
    """

    def optimize(self, dace, validation=None):
        """Fit at ``dace.theta`` (no relocation); return ``(model, None)``.

        ``validation`` is accepted for interface uniformity but ignored: there is no
        search to steer, so there is no trajectory to select theta from. When a
        validation mask is given to ``DACE.fit`` with a ``Fixed`` optimizer, the held-
        out rows still rejoin via the append step.
        """
        _, model = fit_feasible(dace, dace.theta, relocate=False)
        return model, None
