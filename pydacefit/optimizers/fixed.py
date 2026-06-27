"""Fixed-theta optimizer: fit once at the configured theta, with no search."""

from pydacefit.optimizers.base import Optimizer, fit_feasible


class Fixed(Optimizer):
    """No search: fit once at the model's current theta.

    The cheapest strategy -- a single solve. Pass it to ``refit`` to freeze theta
    when appending a few points that should not move it. The theta is kept as given
    (never relocated); if it is infeasible the model's ``raise_error`` policy decides
    whether to raise or fall back to a regularized fit at that theta.
    """

    def optimize(self, dace):
        """Fit at ``dace.theta`` (no relocation); return ``(model, None)``."""
        _, model = fit_feasible(dace, dace.theta, relocate=False)
        return model, None
