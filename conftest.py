"""Pytest config: pin BLAS to a single thread so many tiny solves don't thrash."""

import os

# DACE fits small (~20-30 row) correlation matrices hundreds of times during the theta
# search. OpenBLAS spawns a per-core thread pool for *every* Cholesky/solve, and on
# matrices this small the spawn/sync overhead dwarfs the arithmetic -- single-threaded
# BLAS is ~10x faster here. Under pytest-xdist the per-worker pools also oversubscribe
# the cores, compounding it. Set the thread caps before numpy/OpenBLAS is imported (and
# so xdist worker subprocesses inherit them).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# Belt-and-suspenders: enforce at runtime too, in case a plugin imported numpy before
# this ran (the env vars only take effect at OpenBLAS load time). Hold the controller at
# module scope so the limit stays active for the whole session. threadpoolctl is a scipy
# transitive dep; degrade to env-vars-only if it is somehow absent.
try:
    from threadpoolctl import threadpool_limits  # noqa: E402
except ImportError:
    _BLAS_LIMIT = None
else:
    _BLAS_LIMIT = threadpool_limits(limits=1)
