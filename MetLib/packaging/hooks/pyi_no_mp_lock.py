"""PyInstaller runtime hook for frozen builds.

`tqdm` creates a `multiprocessing.RLock()` the first time a progress bar is
used. Creating a multiprocessing lock starts the `resource_tracker` helper
process, which is spawned as::

    <frozen-exe> -B -S -I -c "from multiprocessing.resource_tracker import main; main(fd)"

A frozen executable cannot act as a Python interpreter, so the helper dies
right away and every run prints an argparse usage/error block together with a
"resource_tracker: process died unexpectedly" warning.

MetDetPy is single-process (all concurrency is thread based) and shares no
resources across processes, so the optional multiprocessing lock is disabled
here; tqdm then falls back to its thread-only lock.
"""

import multiprocessing

try:
    from tqdm.std import TqdmDefaultWriteLock

    # `create_mp_lock()` skips the creation when the attribute already exists,
    # so pre-setting it to None keeps the resource tracker from being spawned.
    TqdmDefaultWriteLock.mp_lock = None
except Exception:  # pragma: no cover - tqdm missing or restructured
    # Fallback: make lock creation fail with OSError, which tqdm handles by
    # falling back to its thread-only lock.
    def _mp_lock_unavailable(*args, **kwargs):
        raise OSError("multiprocessing locks are unavailable in frozen builds")

    multiprocessing.RLock = _mp_lock_unavailable
    multiprocessing.Lock = _mp_lock_unavailable
