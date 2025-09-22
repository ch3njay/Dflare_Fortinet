"""Minimal Cupy stub used when the real library is unavailable.

The real project only relies on a tiny subset of the Cupy API.  When Cupy
isn't installed we provide drop-in functions that simply return the original
Python objects so the rest of the code can fall back to CPU execution without
raising attribute errors.

Dask inspects ``cupy.ndarray`` when optional GPU support is enabled.  The
stub therefore exposes a lightweight placeholder ``ndarray`` type so that
imports touching :mod:`dask.array` keep working even though the actual Cupy
package is missing.
"""


class ndarray:  # pragma: no cover - behaviour is trivial
    """Placeholder for :class:`cupy.ndarray`.

    The class intentionally has no behaviour; it only needs to exist so that
    libraries performing ``hasattr(cupy, "ndarray")`` or ``isinstance``
    checks do not fail when Cupy is stubbed out.
    """


def array(obj):
    return obj


# Alias asarray to array for compatibility with the real Cupy API.
def asarray(obj):
    return obj


def asnumpy(obj):
    return obj


__all__ = ["array", "asarray", "asnumpy", "ndarray"]

