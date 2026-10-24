"""
Runs the experiment scripts on one thread. Import this module before NumPy,
SciPy or scikit-learn, and call limit() after those imports:

    import single_thread  # sets the thread variables before NumPy loads
    import numpy as np
    ...
    single_thread.limit()

The thread variables are overwritten, so values already set in the shell
cannot raise the thread count. limit() then caps every BLAS and OpenMP pool
that threadpoolctl finds at one thread, and pools() reports the thread count
each pool actually has, which run_experiment.py writes to environment.txt.
"""
import os

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _var in THREAD_VARIABLES:
    os.environ[_var] = "1"

_limiter = None


def limit():
    """Cap the thread pools of the loaded numerical libraries at one thread."""
    global _limiter
    import numpy  # noqa: F401  (make sure the BLAS and OpenMP libraries are loaded)
    import scipy.linalg  # noqa: F401
    import sklearn.neighbors  # noqa: F401
    from threadpoolctl import threadpool_limits
    _limiter = threadpool_limits(limits=1)


def pools():
    """One line per thread pool: library, API and number of threads."""
    from threadpoolctl import threadpool_info
    return [f"{p['internal_api']} ({p['user_api']}, {os.path.basename(p['filepath'])}): "
            f"{p['num_threads']} thread(s)" for p in threadpool_info()]


def max_threads():
    from threadpoolctl import threadpool_info
    return max((p["num_threads"] for p in threadpool_info()), default=1)
