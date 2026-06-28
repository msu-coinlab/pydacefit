"""Theta-optimization strategies for the DACE model.

End users pick a strategy and pass an instance to ``DACE(optimizer=...)`` or
``model.refit(optimizer=...)``:

- ``Boxmin()``  -- Hooke & Jeeves pattern search (the default; robust, global-ish).
- ``LBFGS()``   -- bounded quasi-Newton with an analytic gradient; fast for refits.
- ``Fixed()``   -- no search, fit at the current theta (the cheapest refit).

All are subclasses of ``Optimizer`` and obtain their *committed* fits through
``fit_feasible``, so they consistently honor the model's noise / ``max_noise`` policy
(the per-step search fits stay strict -- a non-PD theta is simply infeasible).
"""

from pydacefit.optimizers.base import Optimizer, fit_feasible
from pydacefit.optimizers.boxmin import Boxmin
from pydacefit.optimizers.fixed import Fixed
from pydacefit.optimizers.lbfgs import LBFGS, objective_gradient

__all__ = [
    "Optimizer",
    "fit_feasible",
    "Boxmin",
    "Fixed",
    "LBFGS",
    "objective_gradient",
]
